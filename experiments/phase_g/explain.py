from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn

from experiments.phase_g.generate import SyntheticPatient, _make_glyph


CLASS_COUNT = 6
PATCH_SIZE = 16
PATCH_GRID = 14
PATCH_COUNT = PATCH_GRID * PATCH_GRID
MASKED_CATEGORY = 6
IMAGE_METHODS = ("gradcam", "chefer", "attention_rollout", "occlusion")


def _device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _image_tensor(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("images must have shape (N, 3, 224, 224)")
    if images.dtype == torch.uint8:
        result = images.to(device=device, dtype=torch.float32).div_(255.0)
    elif images.is_floating_point():
        result = images.to(device=device, dtype=torch.float32)
        if not torch.isfinite(result).all() or torch.any((result < 0) | (result > 1)):
            raise ValueError("floating-point images must be finite and in [0, 1]")
    else:
        raise TypeError("images must be uint8 or floating-point raw pixels")
    return result


def _categories_tensor(
    categories: np.ndarray | torch.Tensor | None,
    count: int,
    device: torch.device,
    model_kind: str,
) -> torch.Tensor | None:
    if model_kind == "image_only":
        if categories is not None:
            raise ValueError("image-only explanations cannot consume clinical categories")
        return None
    if model_kind != "joint":
        raise ValueError("image maps require model_kind 'image_only' or 'joint'")
    if categories is None:
        raise ValueError("joint image explanations require fixed clinical categories")
    result = torch.as_tensor(categories, dtype=torch.long, device=device)
    if result.shape != (count,) or torch.any((result < 0) | (result >= CLASS_COUNT)):
        raise ValueError("joint clinical categories must align and lie in [0, 5]")
    return result


def _logits(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
) -> torch.Tensor:
    if model_kind == "image_only":
        return model(images)
    if model_kind == "joint":
        if categories is None:
            raise ValueError("joint model requires fixed clinical categories")
        return model(images, categories)
    raise ValueError("model_kind must be 'image_only' or 'joint'")


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    return value


def _normalize_maps(values: torch.Tensor) -> torch.Tensor:
    maps = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).relu()
    flat = maps.flatten(1)
    low = flat.min(dim=1).values[:, None, None]
    high = flat.max(dim=1).values[:, None, None]
    return ((maps - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)


@contextmanager
def _capture_attentions(vit: nn.Module) -> Iterator[list[torch.Tensor]]:
    captured: list[torch.Tensor] = []
    originals = []
    for block in vit.encoder.layers:
        attention = block.self_attention
        original = attention.forward
        originals.append((attention, original))

        def wrapped(
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
            _attention=attention,
            _captured=captured,
        ):
            if (
                query is not key
                or key is not value
                or key_padding_mask is not None
                or attn_mask is not None
                or is_causal
                or _attention.bias_k is not None
                or _attention.bias_v is not None
                or _attention.add_zero_attn
            ):
                raise RuntimeError("unsupported ViT self-attention configuration")
            batch, sequence, embed = query.shape
            heads = _attention.num_heads
            head_dim = embed // heads
            q, k, v = F.linear(
                query, _attention.in_proj_weight, _attention.in_proj_bias
            ).chunk(3, dim=-1)
            q = q.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            q = q * (head_dim**-0.5)
            k = k.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            v = v.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
            weights = F.dropout(weights, p=_attention.dropout, training=_attention.training)
            _captured.append(weights)
            output = (weights @ v).transpose(1, 2).reshape(batch, sequence, embed)
            output = F.linear(output, _attention.out_proj.weight, _attention.out_proj.bias)
            returned = weights.mean(dim=1) if average_attn_weights else weights
            return output, returned if need_weights else None

        attention.forward = wrapped
    try:
        yield captured
    finally:
        for attention, original in originals:
            attention.forward = original


def _vit_for(model: nn.Module) -> nn.Module:
    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is None or not hasattr(image_encoder, "encoder"):
        raise ValueError("explanation model must expose image_encoder.encoder")
    layers = image_encoder.encoder.layers
    if not layers or not hasattr(layers[-1], "ln_1"):
        raise ValueError("ViT must expose a final encoder block with ln_1")
    return image_encoder


def _gradcam(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
    targets: torch.Tensor,
) -> np.ndarray:
    vit = _vit_for(model)
    activations: list[torch.Tensor] = []
    handle = vit.encoder.layers[-1].ln_1.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        logits = _logits(model, model_kind, images, categories)
        score = logits.gather(1, targets[:, None]).sum()
        gradients = torch.autograd.grad(score, activations[0])[0]
    finally:
        handle.remove()
    patch_activations = activations[0][:, 1:, :]
    patch_gradients = gradients[:, 1:, :]
    channel_weights = patch_gradients.mean(dim=1, keepdim=True)
    raw = (patch_activations * channel_weights).sum(dim=-1).reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _attention_rollout(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
) -> np.ndarray:
    vit = _vit_for(model)
    with _capture_attentions(vit) as attentions:
        with torch.no_grad():
            _logits(model, model_kind, images, categories)
    if len(attentions) != len(vit.encoder.layers):
        raise RuntimeError("attention weights were not captured from every ViT block")
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=images.device).unsqueeze(0)
    joint = identity.expand(len(images), -1, -1).clone()
    for attention in attentions:
        matrix = attention.mean(dim=1) + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = matrix @ joint
    raw = joint[:, 0, 1:].reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _chefer(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
    targets: torch.Tensor,
) -> np.ndarray:
    vit = _vit_for(model)
    with _capture_attentions(vit) as attentions:
        logits = _logits(model, model_kind, images, categories)
        score = logits.gather(1, targets[:, None]).sum()
        gradients = torch.autograd.grad(score, tuple(attentions))
    if len(attentions) != len(vit.encoder.layers):
        raise RuntimeError("attention weights were not captured from every ViT block")
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=images.device).unsqueeze(0)
    joint = identity.expand(len(images), -1, -1).clone()
    for attention, gradient in zip(attentions, gradients, strict=True):
        relevance = (attention * gradient).clamp_min(0).mean(dim=1) + identity
        relevance = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = relevance @ joint
    raw = joint[:, 0, 1:].reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _blur(image: torch.Tensor, radius: float = 6.0) -> torch.Tensor:
    pixels = (image.detach().cpu().clamp(0, 1) * 255.0).round().byte()
    array = pixels.permute(1, 2, 0).contiguous().numpy()
    blurred = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=radius))
    return torch.from_numpy(np.asarray(blurred).copy()).permute(2, 0, 1).float().div_(255.0)


def _occlusion_batched(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
    targets: torch.Tensor,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    device = images.device
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("occlusion images must have shape (N, 3, 224, 224)")
    if targets.shape != (len(images),):
        raise ValueError("occlusion targets must align with images")
    if model_kind == "joint" and (
        categories is None or categories.shape != (len(images),)
    ):
        raise ValueError("joint occlusion requires aligned clinical categories")
    if batch_size < 1:
        raise ValueError("occlusion batch size must be positive")

    def target_scores(
        batch_images: torch.Tensor,
        batch_categories: torch.Tensor | None,
        batch_targets: torch.Tensor,
    ) -> np.ndarray:
        context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), context:
            logits = _logits(model, model_kind, batch_images, batch_categories)
            probabilities = torch.softmax(logits / temperature, dim=1)
            return (
                probabilities.gather(1, batch_targets[:, None])
                .squeeze(1)
                .float()
                .cpu()
                .numpy()
            )

    patient_chunk_size = 4
    inference_batch_size = max(batch_size, 128) if device.type == "cuda" else batch_size
    erase_masks = (
        torch.eye(PATCH_COUNT, dtype=torch.bool, device=device)
        .reshape(PATCH_COUNT, PATCH_GRID, PATCH_GRID)
        .repeat_interleave(PATCH_SIZE, dim=-2)
        .repeat_interleave(PATCH_SIZE, dim=-1)
        .unsqueeze(1)
    )
    maps = np.zeros((len(images), PATCH_GRID, PATCH_GRID), dtype=np.float32)
    for start in range(0, len(images), patient_chunk_size):
        stop = min(start + patient_chunk_size, len(images))
        source = _image_tensor(images[start:stop], device)
        blurred = torch.stack(
            [_blur(image) for image in source.detach().cpu()]
        ).to(device)
        variants = torch.where(
            erase_masks.unsqueeze(0),
            blurred[:, None],
            source[:, None],
        ).reshape(-1, 3, 224, 224)
        chunk_targets = targets[start:stop].to(device)
        target_variants = chunk_targets.repeat_interleave(PATCH_COUNT)
        category_variants = (
            categories[start:stop].to(device).repeat_interleave(PATCH_COUNT)
            if categories is not None
            else None
        )
        variant_scores = []
        for batch_start in range(0, len(variants), inference_batch_size):
            batch_stop = min(batch_start + inference_batch_size, len(variants))
            batch_categories = (
                None
                if category_variants is None
                else category_variants[batch_start:batch_stop]
            )
            variant_scores.extend(
                target_scores(
                    variants[batch_start:batch_stop],
                    batch_categories,
                    target_variants[batch_start:batch_stop],
                ).tolist()
            )
        original_categories = (
            None if categories is None else categories[start:stop].to(device)
        )
        original_scores = target_scores(source, original_categories, chunk_targets)
        drops = np.maximum(
            original_scores[:, None] - np.asarray(variant_scores).reshape(-1, PATCH_COUNT),
            0.0,
        ).reshape(-1, PATCH_GRID, PATCH_GRID)
        maximum = drops.max(axis=(1, 2), keepdims=True)
        maps[start:stop] = np.divide(
            drops,
            maximum,
            out=np.zeros_like(drops),
            where=maximum > 1e-8,
        )
    return maps


def _occlusion(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
    targets: torch.Tensor,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    return _occlusion_batched(
        model, model_kind, images, categories, targets, batch_size, temperature
    )


def compute_image_maps(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    targets: np.ndarray,
    categories: np.ndarray | None,
    methods: tuple[str, ...],
    batch_size: int = 16,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    model.eval()
    temperature = _validate_temperature(temperature)
    if model_kind not in {"image_only", "joint"}:
        raise ValueError("model_kind must be 'image_only' or 'joint'")
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (len(images),) or np.any((targets < 0) | (targets >= CLASS_COUNT)):
        raise ValueError("targets must align with image rows and lie in [0, 5]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(set(methods)) != len(methods) or any(method not in IMAGE_METHODS for method in methods):
        raise ValueError(f"methods must be unique values from {IMAGE_METHODS}")
    device = _device(model)
    all_categories = _categories_tensor(categories, len(images), device, model_kind)
    result = {method: np.empty((len(images), 14, 14), dtype=np.float32) for method in methods}
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = _image_tensor(images[start:stop], device)
        batch_categories = (
            None if all_categories is None else all_categories[start:stop]
        )
        batch_targets = torch.as_tensor(targets[start:stop], device=device)
        if "gradcam" in methods:
            result["gradcam"][start:stop] = _gradcam(
                model, model_kind, batch, batch_categories, batch_targets
            )
        if "chefer" in methods:
            result["chefer"][start:stop] = _chefer(
                model, model_kind, batch, batch_categories, batch_targets
            )
        if "attention_rollout" in methods:
            result["attention_rollout"][start:stop] = _attention_rollout(
                model, model_kind, batch, batch_categories
            )
        if "occlusion" in methods:
            result["occlusion"][start:stop] = _occlusion(
                model, model_kind, batch, batch_categories, batch_targets,
                batch_size, temperature,
            )
    return result


def compute_clinical_attributions(
    model: nn.Module,
    clinical_values: np.ndarray,
    targets: np.ndarray,
    reference: np.ndarray,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    model.eval()
    temperature = _validate_temperature(temperature)
    clinical_values = np.asarray(clinical_values, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.float64)
    if clinical_values.ndim != 1 or targets.shape != clinical_values.shape:
        raise ValueError("clinical values and targets must be aligned vectors")
    if np.any((clinical_values < 0) | (clinical_values >= CLASS_COUNT)):
        raise ValueError("clinical values must lie in [0, 5]")
    if np.any((targets < 0) | (targets >= CLASS_COUNT)):
        raise ValueError("targets must lie in [0, 5]")
    if (
        reference.shape != (CLASS_COUNT,)
        or not np.isfinite(reference).all()
        or np.any(reference < 0)
        or not np.isclose(reference.sum(), 1.0)
    ):
        raise ValueError("reference must be a probability distribution over six categories")
    device = _device(model)
    categories = torch.arange(CLASS_COUNT, dtype=torch.long, device=device)
    with torch.no_grad():
        category_probabilities = torch.softmax(
            model(categories) / temperature, dim=1
        ).cpu().numpy()
        observed_probabilities = torch.softmax(
            model(torch.as_tensor(clinical_values, dtype=torch.long, device=device))
            / temperature,
            dim=1,
        ).cpu().numpy()
    target_probabilities = category_probabilities[:, targets].T
    observed_values = observed_probabilities[np.arange(len(targets)), targets]
    reference_values = target_probabilities @ reference
    signed = observed_values - reference_values
    return {
        "category_values": target_probabilities.astype(np.float32),
        "observed_value": observed_values.astype(np.float32),
        "reference_value": reference_values.astype(np.float32),
        "signed_effect": signed.astype(np.float32),
        "absolute_effect": np.abs(signed).astype(np.float32),
        "target_classes": targets,
        "reference_distribution": reference.astype(np.float64),
        "temperature": np.asarray(temperature, dtype=np.float64),
    }


def _patient_tensor(patient: SyntheticPatient, device: torch.device) -> torch.Tensor:
    image = torch.from_numpy(patient.image.transpose(2, 0, 1).copy()).unsqueeze(0)
    return _image_tensor(image, device)


def compute_joint_shapley(
    model: nn.Module,
    patients: Sequence[SyntheticPatient],
    targets: np.ndarray,
    donors: Sequence[SyntheticPatient],
    game: str,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    model.eval()
    temperature = _validate_temperature(temperature)
    if game not in {"modality", "cue"}:
        raise ValueError("game must be 'modality' or 'cue'")
    targets = np.asarray(targets, dtype=np.int64)
    if len(patients) == 0 or len(donors) == 0 or targets.shape != (len(patients),):
        raise ValueError("patients, donors, and targets must be nonempty and aligned")
    if any(not 0 <= int(target) < CLASS_COUNT for target in targets):
        raise ValueError("targets must lie in [0, 5]")
    device = _device(model)
    donor_images = torch.cat(
        [_patient_tensor(donor, torch.device("cpu")) for donor in donors], dim=0
    )
    donor_glyphs = []
    for donor in donors:
        marker_y, marker_x = donor.render_metadata["marker_top_left"]
        donor_glyphs.append(
            _patient_tensor(donor, torch.device("cpu"))[
                :, :, marker_y:marker_y + 16, marker_x:marker_x + 16
            ]
        )
    donor_glyph_tensor = torch.cat(donor_glyphs, dim=0)
    coalitions = np.empty((len(patients), 4), dtype=np.float64)
    with torch.no_grad():
        for row, patient in enumerate(patients):
            sums = np.zeros(4, dtype=np.float64)
            patient_image = _patient_tensor(patient, torch.device("cpu"))[0]
            for start in range(0, len(donors), 16):
                donor_batch = donors[start:start + 16]
                block_count = len(donor_batch)
                target_images = patient_image.unsqueeze(0).expand(block_count, -1, -1, -1)
                if game == "modality":
                    image_00 = donor_images[start:start + block_count]
                    image_10 = target_images
                    image_01 = image_00
                    image_11 = image_10
                    categories_00 = [donor.clinical_value for donor in donor_batch]
                    categories_10 = categories_00
                    categories_01 = [patient.clinical_value] * len(donor_batch)
                    categories_11 = categories_01
                else:
                    marker_y, marker_x = patient.render_metadata["marker_top_left"]
                    image_00 = target_images.clone()
                    image_00[:, :, marker_y:marker_y + 16, marker_x:marker_x + 16] = (
                        donor_glyph_tensor[start:start + block_count]
                    )
                    image_10 = target_images
                    image_01 = image_00
                    image_11 = image_10
                    categories_00 = [donor.clinical_value for donor in donor_batch]
                    categories_10 = categories_00
                    categories_01 = [patient.clinical_value] * len(donor_batch)
                    categories_11 = categories_01
                images = torch.cat((image_00, image_10, image_01, image_11)).to(device)
                categories = torch.tensor(
                    categories_00 + categories_10 + categories_01 + categories_11,
                    dtype=torch.long,
                    device=device,
                )
                probabilities = torch.softmax(
                    model(images, categories) / temperature, dim=1
                )
                scores = probabilities[:, int(targets[row])].cpu().numpy()
                for coalition in range(4):
                    sums[coalition] += float(
                        scores[coalition * block_count:(coalition + 1) * block_count].sum()
                    )
            coalitions[row] = sums / len(donors)

    value_00, value_10, value_01, value_11 = coalitions.T
    image_values = 0.5 * ((value_10 - value_00) + (value_11 - value_01))
    clinical_values = 0.5 * ((value_01 - value_00) + (value_11 - value_10))
    residual = image_values + clinical_values - (value_11 - value_00)
    return {
        "image": image_values.astype(np.float32),
        "clinical": clinical_values.astype(np.float32),
        "baseline_value": value_00.astype(np.float32),
        "full_value": value_11.astype(np.float32),
        "efficiency_residual": residual.astype(np.float64),
        "coalition_values": coalitions.astype(np.float32),
        "game": np.asarray(game),
        "donor_count": np.asarray(len(donors), dtype=np.int64),
        "temperature": np.asarray(temperature, dtype=np.float64),
    }


def cue_mask_sensitivity(
    joint_model: nn.Module,
    patients: Sequence[SyntheticPatient],
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    joint_model.eval()
    if not patients:
        raise ValueError("patients must be nonempty")
    draws = int(config.get("draws_per_test_patient", 20))
    if draws != 20:
        raise ValueError("the approved cue sensitivity uses exactly 20 replacement draws")
    temperature = _validate_temperature(float(config.get("temperature", 1.0)))
    device = _device(joint_model)
    output = {
        name: np.empty(len(patients), dtype=np.float32)
        for name in (
            "original_target_probability",
            "original_nll",
            "original_accuracy",
            "image_replacement_target_probability",
            "clinical_replacement_target_probability",
            "literal_absence_target_probability",
            "image_replacement_nll",
            "clinical_replacement_nll",
            "literal_absence_nll",
            "image_replacement_accuracy",
            "clinical_replacement_accuracy",
            "literal_absence_accuracy",
            "image_replacement_target_probability_change",
            "clinical_replacement_target_probability_change",
            "literal_absence_target_probability_change",
            "image_replacement_nll_change",
            "clinical_replacement_nll_change",
            "literal_absence_nll_change",
            "image_replacement_accuracy_change",
            "clinical_replacement_accuracy_change",
            "literal_absence_accuracy_change",
        )
    }
    with torch.no_grad():
        for row, patient in enumerate(patients):
            original = _patient_tensor(patient, torch.device("cpu"))[0]
            marker_y, marker_x = patient.render_metadata["marker_top_left"]
            original_probability = torch.softmax(
                joint_model(
                    original.unsqueeze(0).to(device),
                    torch.tensor([patient.clinical_value], device=device),
                ) / temperature,
                dim=1,
            )[0].cpu().numpy()
            output["original_target_probability"][row] = original_probability[patient.label]
            output["original_nll"][row] = -np.log(
                np.clip(original_probability[patient.label], 1e-12, 1.0)
            )
            output["original_accuracy"][row] = float(
                int(original_probability.argmax()) == patient.label
            )
            image_values = patient.render_metadata["replacement_image_cues"]
            clinical_values = patient.render_metadata["replacement_clinical_cues"]
            if len(image_values) != draws or len(clinical_values) != draws:
                raise ValueError("test patient is missing the prespecified replacement draws")
            image_variants = []
            for value in image_values:
                variant = patient.image.copy()
                variant[marker_y:marker_y + 16, marker_x:marker_x + 16] = _make_glyph(
                    int(value)
                )
                image_variants.append(
                    torch.from_numpy(variant.transpose(2, 0, 1).copy()).float().div_(255.0)
                )
            base_images = _patient_tensor(patient, torch.device("cpu")).expand(draws, -1, -1, -1)
            no_marker = base_images.clone()
            background = torch.from_numpy(
                patient.background_canvas.transpose(2, 0, 1).copy()
            ).float().div_(255.0)
            no_marker[:, :, marker_y:marker_y + 16, marker_x:marker_x + 16] = background[
                :, marker_y:marker_y + 16, marker_x:marker_x + 16
            ]
            cases = (
                (
                    torch.stack(image_variants),
                    torch.full((draws,), patient.clinical_value, dtype=torch.long),
                    "image_replacement",
                ),
                (
                    base_images,
                    torch.as_tensor(clinical_values, dtype=torch.long),
                    "clinical_replacement",
                ),
                (
                    no_marker,
                    torch.full((draws,), MASKED_CATEGORY, dtype=torch.long),
                    "literal_absence",
                ),
            )
            for images, categories, prefix in cases:
                probabilities = torch.softmax(
                    joint_model(
                        _image_tensor(images, device),
                        categories.to(device),
                    ) / temperature,
                    dim=1,
                ).cpu().numpy()
                target_probs = probabilities[:, patient.label]
                output[f"{prefix}_target_probability"][row] = float(target_probs.mean())
                output[f"{prefix}_nll"][row] = float(-np.log(np.clip(target_probs, 1e-12, 1.0)).mean())
                output[f"{prefix}_accuracy"][row] = float(
                    (probabilities.argmax(axis=1) == patient.label).mean()
                )
                output[f"{prefix}_target_probability_change"][row] = (
                    output[f"{prefix}_target_probability"][row]
                    - output["original_target_probability"][row]
                )
                output[f"{prefix}_nll_change"][row] = (
                    output[f"{prefix}_nll"][row] - output["original_nll"][row]
                )
                output[f"{prefix}_accuracy_change"][row] = (
                    output[f"{prefix}_accuracy"][row] - output["original_accuracy"][row]
                )
    output["outcome_type"] = np.asarray("intervention_sensitivity")
    output["literal_absence_type"] = np.asarray("out_of_support_diagnostic")
    output["replacement_operator"] = np.asarray("marginal_replacement")
    output["temperature"] = np.asarray(temperature, dtype=np.float64)
    return output
