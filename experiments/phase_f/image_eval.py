import argparse
import copy
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import PIL
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity
from torchvision import models

from experiments.phase_f.io import load_frozen_arrays, prepare_run_dir, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
SANITY_CONFIG_PATH = Path(__file__).resolve().parent / "sanity_config.json"
OUTPUT_DIR = ROOT / "outputs" / "phase_f_v1"
CHECKPOINT = ROOT / "outputs" / "vit_final_v1.pt"
CACHE = ROOT / "cache" / "images_uint8_224.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BATCH_SIZE = 32

MAP_KEYS = (
    "gradcam_fused",
    "gradcam_vit_top",
    "attention_rollout_fused",
    "attention_rollout_vit_top",
    "chefer_fused",
    "chefer_vit_top",
    "occlusion_fused",
    "occlusion_vit_top",
)


def map_back_horizontal_flip(maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(maps)
    if maps.ndim < 2 or maps.shape[-2:] != (14, 14):
        raise ValueError("attribution maps must end in a 14x14 grid")
    return np.flip(maps, axis=-1).copy()


def brightness_transform(images: torch.Tensor, factor: float) -> torch.Tensor:
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("brightness factor must be finite and positive")
    if images.dtype == torch.uint8:
        raw = images.to(torch.float32).div(255.0)
    elif images.is_floating_point():
        raw = images.to(torch.float32)
        if not torch.isfinite(raw).all() or torch.any((raw < 0) | (raw > 1)):
            raise ValueError("floating-point images must be finite and in [0, 1]")
    else:
        raise TypeError("images must be uint8 or floating-point raw pixels")
    return (raw * float(factor)).clamp(0.0, 1.0)


def unchanged_prediction_mask(
    original_probabilities: np.ndarray,
    transformed_probabilities: np.ndarray,
) -> np.ndarray:
    original_probabilities = np.asarray(original_probabilities)
    transformed_probabilities = np.asarray(transformed_probabilities)
    if (
        original_probabilities.ndim != 2
        or original_probabilities.shape != transformed_probabilities.shape
        or not np.isfinite(original_probabilities).all()
        or not np.isfinite(transformed_probabilities).all()
    ):
        raise ValueError("prediction arrays must be finite, aligned class-probability matrices")
    return original_probabilities.argmax(axis=1) == transformed_probabilities.argmax(axis=1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_map_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (14, 14) or second.shape != first.shape:
        raise ValueError("Spearman inputs must be 14x14 maps")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Spearman inputs must be finite")
    first_ranks = _average_ranks(first.ravel())
    second_ranks = _average_ranks(second.ravel())
    if first_ranks.std() == 0 or second_ranks.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def ssim_map_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (14, 14) or second.shape != first.shape:
        raise ValueError("SSIM inputs must be 14x14 maps")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("SSIM inputs must be finite")
    if np.any((first < 0) | (first > 1)) or np.any((second < 0) | (second > 1)):
        raise ValueError("SSIM inputs must lie in [0, 1]")
    return float(structural_similarity(
        first,
        second,
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
    ))


def _reset_randomized_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            if child.bias_k is not None:
                nn.init.xavier_normal_(child.bias_k)
            if child.bias_v is not None:
                nn.init.xavier_normal_(child.bias_v)
        elif isinstance(child, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            child.reset_parameters()


def randomize_vit_parameters(model: nn.Module, stage: str, seed: int) -> nn.Module:
    valid_stages = {
        "head",
        "final_encoder_block_and_head",
        "full_encoder_and_head",
    }
    if stage not in valid_stages:
        raise ValueError(f"unknown ViT randomization stage: {stage}")
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise ValueError("model must expose encoder.layers")
    if not hasattr(model, "heads") or not hasattr(model.heads, "head"):
        raise ValueError("model must expose heads.head")
    layers = model.encoder.layers
    if len(layers) == 0:
        raise ValueError("model encoder must contain at least one layer")

    original_device = next(model.parameters()).device
    randomized = copy.deepcopy(model).cpu()
    selected_modules = [randomized.heads.head]
    if stage == "final_encoder_block_and_head":
        selected_modules.append(randomized.encoder.layers[-1])
    elif stage == "full_encoder_and_head":
        selected_modules.extend(randomized.encoder.layers)
        if hasattr(randomized.encoder, "ln"):
            selected_modules.append(randomized.encoder.ln)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for module in selected_modules:
            _reset_randomized_modules(module)
        if stage == "full_encoder_and_head" and hasattr(
            randomized.encoder, "pos_embedding"
        ):
            with torch.no_grad():
                randomized.encoder.pos_embedding.normal_(mean=0.0, std=0.02)

    return randomized.to(original_device).eval()


def make_patch_schedule() -> np.ndarray:
    return np.round(np.linspace(0, 196, 21)).astype(np.int64)


def map_to_patch_order(map_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(map_scores)
    if scores.shape != (14, 14) or not np.isfinite(scores).all():
        raise ValueError("map scores must be a finite 14x14 array")
    return np.argsort(-scores.reshape(-1), kind="stable").astype(np.int64)


def apply_ranked_patches(
    original: torch.Tensor,
    baseline: torch.Tensor,
    order: np.ndarray,
    patch_count: int,
    mode: str,
) -> torch.Tensor:
    if original.shape != (3, 224, 224) or baseline.shape != original.shape:
        raise ValueError("original and baseline images must have shape (3, 224, 224)")
    order = np.asarray(order)
    if (
        order.shape != (196,)
        or not np.issubdtype(order.dtype, np.integer)
        or not np.array_equal(np.sort(order), np.arange(196))
    ):
        raise ValueError("patch order must be a permutation of all 196 patches")
    if not 0 <= patch_count <= 196:
        raise ValueError("patch_count must be between zero and 196")
    if mode not in {"deletion", "insertion"}:
        raise ValueError(f"unknown perturbation mode: {mode}")

    image = original.clone() if mode == "deletion" else baseline.clone()
    source = baseline if mode == "deletion" else original
    for patch_index in order[:patch_count]:
        top, left = divmod(int(patch_index), 14)
        top *= 16
        left *= 16
        image[:, top:top + 16, left:left + 16] = source[
            :, top:top + 16, left:left + 16
        ]
    return image


def cwf_probabilities(
    xgb_probability: np.ndarray,
    vit_probability: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    xgb_probability = np.asarray(xgb_probability, dtype=np.float64)
    vit_probability = np.asarray(vit_probability, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if xgb_probability.shape != vit_probability.shape or xgb_probability.ndim != 2:
        raise ValueError("branch probabilities must have the same two-dimensional shape")
    if weights.shape != (xgb_probability.shape[1],):
        raise ValueError("CWF weights must have one value per class")
    if not (
        np.isfinite(xgb_probability).all()
        and np.isfinite(vit_probability).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("CWF inputs must be finite")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("CWF weights must lie in [0, 1]")
    fused = weights[None, :] * xgb_probability + (1.0 - weights[None, :]) * vit_probability
    totals = fused.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("CWF probability row has no positive mass")
    return fused / totals


def score_fixed_target(probabilities: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    target_classes = np.asarray(target_classes, dtype=np.int64)
    if probabilities.ndim != 2 or target_classes.shape != (probabilities.shape[0],):
        raise ValueError("target classes must align with probability rows")
    if np.any((target_classes < 0) | (target_classes >= probabilities.shape[1])):
        raise ValueError("target class is outside the probability columns")
    return probabilities[np.arange(len(target_classes)), target_classes]


def center_gaussian_map(size: int = 14, sigma: float = 3.5) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    center = (size - 1) / 2.0
    result = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    return (result / result.max()).astype(np.float32)


def build_order_specs(
    image_arrays: dict[str, np.ndarray],
    seed: int,
    random_order_seeds: list[int],
    center_sigma: float = 3.5,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for map_key in MAP_KEYS:
        specs.append({
            "name": map_key,
            "target_source": "fused" if map_key.endswith("_fused") else "vit_top",
            "kind": "map",
            "map_key": map_key,
        })
    n_rows = len(image_arrays["test_row_keys"])
    random_maps = np.random.default_rng(seed).random((n_rows, 14, 14), dtype=np.float32)
    for target_source in ("fused", "vit_top"):
        specs.append({
            "name": f"random_reference_{target_source}",
            "target_source": target_source,
            "kind": "fixed_map",
            "maps": random_maps,
        })
        specs.append({
            "name": f"center_reference_{target_source}",
            "target_source": target_source,
            "kind": "center_map",
            "maps": center_gaussian_map(sigma=center_sigma),
        })
    for target_source in ("fused", "vit_top"):
        for order_seed in random_order_seeds:
            order = np.random.default_rng(order_seed).permutation(196).astype(np.int64)
            specs.append({
                "name": f"random_order_{target_source}_{order_seed}",
                "target_source": target_source,
                "kind": "fixed_order",
                "order": order,
            })
    return specs


def _blur_image(image: torch.Tensor, radius: float) -> torch.Tensor:
    is_uint8 = image.dtype == torch.uint8
    raw_image = image
    if not is_uint8:
        if (
            not image.is_floating_point()
            or not torch.isfinite(image).all()
            or torch.any((image < 0) | (image > 1))
        ):
            raise ValueError("blur input must be uint8 or finite raw pixels in [0, 1]")
        raw_image = (image * 255.0).round().to(torch.uint8)
    pixels = raw_image.permute(1, 2, 0).contiguous().numpy()
    blurred = Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius=radius))
    array = np.asarray(blurred, dtype=np.uint8).copy()
    result = torch.from_numpy(array).permute(2, 0, 1)
    return result if is_uint8 else result.to(torch.float32).div(255.0)


def _load_model(checkpoint: Path, n_classes: int, device: torch.device) -> nn.Module:
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _predict_vit(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    results = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = _normalized_model_inputs(
                images[start:start + batch_size],
                device,
            )
            results.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(results, axis=0)


def _normalized_model_inputs(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device)
    if images.dtype == torch.uint8:
        raw = images.float().div_(255.0)
    elif images.is_floating_point():
        raw = images.float()
        if not torch.isfinite(raw).all() or torch.any((raw < 0) | (raw > 1)):
            raise ValueError("floating-point images must be finite and in [0, 1]")
    else:
        raise TypeError("images must be uint8 or floating-point raw pixels")
    return (raw - MEAN.to(device)) / STD.to(device)


def _normalize_maps(raw_maps: torch.Tensor) -> torch.Tensor:
    maps = torch.nan_to_num(raw_maps, nan=0.0, posinf=0.0, neginf=0.0).relu()
    flat = maps.flatten(1)
    low = flat.min(dim=1).values[:, None, None]
    high = flat.max(dim=1).values[:, None, None]
    return ((maps - low) / (high - low).clamp_min(1e-8)).clamp_(0, 1)


@contextmanager
def _capture_attentions(model: nn.Module):
    captured = []
    originals = []
    for block in model.encoder.layers:
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
            ):
                raise RuntimeError("Unexpected ViT self-attention arguments.")
            batch, sequence, embed = query.shape
            heads = _attention.num_heads
            head_dim = embed // heads
            q, k, v = F.linear(
                query,
                _attention.in_proj_weight,
                _attention.in_proj_bias,
            ).chunk(3, dim=-1)
            q = q.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            q = q * (head_dim ** -0.5)
            k = k.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            v = v.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
            weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
            weights = F.dropout(
                weights,
                p=_attention.dropout,
                training=_attention.training,
            )
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


@contextmanager
def capture_vit_attentions(model: nn.Module):
    with _capture_attentions(model) as captured:
        yield captured


def _gradcam_maps(
    model: nn.Module,
    normalized_images: torch.Tensor,
    target_classes: np.ndarray,
) -> np.ndarray:
    activations = []
    handle = model.encoder.layers[-1].ln_1.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    targets = torch.as_tensor(target_classes, device=normalized_images.device)
    try:
        logits = model(normalized_images)
        score = logits.gather(1, targets[:, None]).sum()
        gradients = torch.autograd.grad(score, activations[0])[0]
    finally:
        handle.remove()
    patch_activations = activations[0][:, 1:, :]
    patch_gradients = gradients[:, 1:, :]
    channel_weights = patch_gradients.mean(dim=1, keepdim=True)
    raw = (patch_activations * channel_weights).sum(dim=-1).reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _attention_rollout_map(model: nn.Module, normalized_images: torch.Tensor) -> np.ndarray:
    with capture_vit_attentions(model) as attentions:
        with torch.no_grad():
            model(normalized_images)
    if len(attentions) != len(model.encoder.layers) or any(
        value is None for value in attentions
    ):
        raise RuntimeError("Failed to capture attention weights from every ViT block.")
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=normalized_images.device).unsqueeze(0)
    joint = identity.expand(normalized_images.shape[0], -1, -1).clone()
    for attention in attentions:
        matrix = attention.mean(dim=1) + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = matrix @ joint
    raw = joint[:, 0, 1:].reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _chefer_maps(
    model: nn.Module,
    normalized_images: torch.Tensor,
    target_classes: np.ndarray,
) -> np.ndarray:
    targets = torch.as_tensor(target_classes, device=normalized_images.device)
    with capture_vit_attentions(model) as attentions:
        logits = model(normalized_images)
        score = logits.gather(1, targets[:, None]).sum()
        gradients = torch.autograd.grad(score, tuple(attentions))
    if len(attentions) != len(model.encoder.layers):
        raise RuntimeError("Failed to capture attention weights from every ViT block.")
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=normalized_images.device).unsqueeze(0)
    joint = identity.expand(normalized_images.shape[0], -1, -1).clone()
    for attention, gradient in zip(attentions, gradients):
        relevance = (attention * gradient).clamp_min(0).mean(dim=1) + identity
        relevance = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = relevance @ joint
    raw = joint[:, 0, 1:].reshape(-1, 14, 14)
    return _normalize_maps(raw).detach().cpu().numpy().astype(np.float32)


def _occlusion_maps(
    model: nn.Module,
    images: torch.Tensor,
    probabilities: np.ndarray,
    target_fused: np.ndarray,
    target_vit: np.ndarray,
    device: torch.device,
    blur_radius: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    patches = [
        (top, left)
        for top in range(0, 224 - 16 + 1, 16)
        for left in range(0, 224 - 16 + 1, 16)
    ]
    if len(patches) != 196:
        raise RuntimeError("occlusion grid must cover exactly 196 ViT patches")
    fused_maps = np.empty((len(images), 14, 14), dtype=np.float32)
    vit_maps = np.empty_like(fused_maps)
    for row, image in enumerate(images):
        baseline = _blur_image(image, blur_radius)
        variants = image.unsqueeze(0).repeat(len(patches), 1, 1, 1)
        for patch_index, (top, left) in enumerate(patches):
            variants[patch_index, :, top:top + 16, left:left + 16] = baseline[
                :, top:top + 16, left:left + 16
            ]
        variant_probabilities = _predict_vit(model, variants, device, batch_size)
        fused_class = int(target_fused[row])
        vit_class = int(target_vit[row])
        fused_drop = probabilities[row, fused_class] - variant_probabilities[:, fused_class]
        vit_drop = probabilities[row, vit_class] - variant_probabilities[:, vit_class]
        fused_maps[row] = _normalize_numpy_map(fused_drop.reshape(14, 14))
        vit_maps[row] = _normalize_numpy_map(vit_drop.reshape(14, 14))
    return fused_maps, vit_maps


def _normalize_numpy_map(raw_map: np.ndarray) -> np.ndarray:
    values = np.maximum(np.nan_to_num(raw_map, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def _compute_explanation_maps(
    model: nn.Module,
    images: torch.Tensor,
    probabilities: np.ndarray,
    target_fused: np.ndarray,
    target_vit: np.ndarray,
    device: torch.device,
    blur_radius: float,
    occlusion_batch_size: int,
) -> dict[str, np.ndarray]:
    normalized_images = _normalized_model_inputs(images, device)
    maps = {
        "gradcam_fused": _gradcam_maps(model, normalized_images, target_fused),
        "gradcam_vit_top": _gradcam_maps(model, normalized_images, target_vit),
    }
    rollout = _attention_rollout_map(model, normalized_images)
    maps["attention_rollout_fused"] = rollout
    maps["attention_rollout_vit_top"] = rollout
    maps["chefer_fused"] = _chefer_maps(model, normalized_images, target_fused)
    maps["chefer_vit_top"] = _chefer_maps(model, normalized_images, target_vit)
    occlusion_fused, occlusion_vit = _occlusion_maps(
        model,
        images,
        probabilities,
        target_fused,
        target_vit,
        device,
        blur_radius,
        occlusion_batch_size,
    )
    maps["occlusion_fused"] = occlusion_fused
    maps["occlusion_vit_top"] = occlusion_vit
    return maps


def _target_classes(images: dict[str, np.ndarray], target_source: str) -> np.ndarray:
    key = "target_fused_class" if target_source == "fused" else "target_vit_top_class"
    return images[key].astype(np.int64, copy=False)


def _patch_order_for_row(
    spec: dict[str, object],
    row: int,
    image_arrays: dict[str, np.ndarray],
) -> np.ndarray:
    kind = spec["kind"]
    if kind == "map":
        return map_to_patch_order(image_arrays[str(spec["map_key"])][row])
    if kind in {"fixed_map", "center_map"}:
        maps = np.asarray(spec["maps"])
        return map_to_patch_order(maps[row] if maps.ndim == 3 else maps)
    return np.asarray(spec["order"], dtype=np.int64)


def _compute_curves(
    model: nn.Module,
    original_images: torch.Tensor,
    xgb_probabilities: np.ndarray,
    weights: np.ndarray,
    image_arrays: dict[str, np.ndarray],
    specs: list[dict[str, object]],
    row_indices: np.ndarray,
    device: torch.device,
    blur_radius: float,
) -> np.ndarray:
    n_steps = len(make_patch_schedule())
    curves = np.empty((len(row_indices), len(specs), 2, 2, n_steps), dtype=np.float32)
    schedule = make_patch_schedule()
    n_classes = len(weights)

    for local_index, row in enumerate(row_indices):
        original = original_images[row]
        baseline = _blur_image(original, blur_radius)
        original_vit = _predict_vit(model, original.unsqueeze(0), device)[0]
        baseline_vit = _predict_vit(model, baseline.unsqueeze(0), device)[0]
        xgb_row = xgb_probabilities[row]
        original_cwf = cwf_probabilities(
            xgb_row[None, :], original_vit[None, :], weights
        )[0]
        baseline_cwf = cwf_probabilities(
            xgb_row[None, :], baseline_vit[None, :], weights
        )[0]
        order_cache: dict[bytes, np.ndarray] = {}

        for order_index, spec in enumerate(specs):
            target_source = str(spec["target_source"])
            target_class = int(_target_classes(image_arrays, target_source)[row])
            order = _patch_order_for_row(spec, row, image_arrays)
            cache_key = order.tobytes()
            if cache_key not in order_cache:
                class_curves = np.empty((2, 2, n_steps, n_classes), dtype=np.float32)
                class_curves[0, 0, 0] = original_vit
                class_curves[1, 0, 0] = original_cwf
                class_curves[0, 1, 0] = baseline_vit
                class_curves[1, 1, 0] = baseline_cwf
                class_curves[0, 0, -1] = baseline_vit
                class_curves[1, 0, -1] = baseline_cwf
                class_curves[0, 1, -1] = original_vit
                class_curves[1, 1, -1] = original_cwf

                views = []
                view_steps = []
                for step_index, patch_count in enumerate(schedule[1:-1], start=1):
                    views.append(apply_ranked_patches(
                        original, baseline, order, int(patch_count), "deletion"
                    ))
                    view_steps.append((step_index, 0))
                    views.append(apply_ranked_patches(
                        original, baseline, order, int(patch_count), "insertion"
                    ))
                    view_steps.append((step_index, 1))

                if views:
                    vit_probabilities = _predict_vit(model, torch.stack(views), device)
                    xgb_rows = np.broadcast_to(xgb_row, vit_probabilities.shape)
                    cwf = cwf_probabilities(xgb_rows, vit_probabilities, weights)
                    for view_index, (step_index, direction) in enumerate(view_steps):
                        class_curves[0, direction, step_index] = vit_probabilities[view_index]
                        class_curves[1, direction, step_index] = cwf[view_index]
                order_cache[cache_key] = class_curves

            curves[local_index, order_index] = order_cache[cache_key][:, :, :, target_class]

    return curves


def _save_chunk(
    path: Path,
    row_keys: np.ndarray,
    order_names: list[str],
    target_sources: list[str],
    target_classes: np.ndarray,
    curves: np.ndarray,
) -> None:
    auc = np.trapezoid(curves, x=np.linspace(0.0, 1.0, curves.shape[-1]), axis=-1)
    temp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temp_path,
        test_row_keys=row_keys,
        order_names=np.asarray(order_names),
        target_sources=np.asarray(target_sources),
        target_classes=target_classes,
        score_spaces=np.asarray(["vit", "cwf"]),
        directions=np.asarray(["deletion", "insertion"]),
        patch_counts=make_patch_schedule(),
        curves=curves,
        auc=auc.astype(np.float32),
    )
    os.replace(temp_path, path)


def _load_test_images(cache_path: Path) -> torch.Tensor:
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    if "te" not in cached:
        raise ValueError("frozen image cache has no 'te' test-image tensor")
    images = cached["te"]
    if images.shape != (461, 3, 224, 224) or images.dtype != torch.uint8:
        raise ValueError(f"unexpected cached test image tensor: {images.shape}, {images.dtype}")
    return images.contiguous()


def _make_model_and_data(root: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase F image evaluation requires the verified CUDA environment")
    device = torch.device("cuda:0")
    torch.ones(1, device=device).add_(1)
    torch.cuda.synchronize()

    frozen = load_frozen_arrays(root)
    model = _load_model(root / "outputs" / "vit_final_v1.pt", 6, device)
    images = _load_test_images(root / "cache" / "images_uint8_224.pt")
    return frozen, model, images, device


def _initialize_run(frozen: dict[str, object], root: Path, resume: bool) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256_file(CONFIG_PATH)
    prepare_run_dir(
        root / config["output_directory"],
        config_hash,
        frozen["input_hashes"],
        resume=resume,
    )
    return config


def run_preflight(
    frozen: dict[str, object],
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    config: dict[str, object],
    output_dir: Path,
    n_rows: int,
    n_orders: int,
) -> None:
    image_arrays = frozen["images"]
    all_specs = build_order_specs(
        image_arrays,
        int(config["seed"]),
        config["image"]["random_order_seeds"],
        float(config["image"]["center_gaussian_sigma"]),
    )
    unique_specs = []
    seen_orders = set()
    for spec in all_specs:
        order_key = _patch_order_for_row(spec, 0, image_arrays).tobytes()
        if order_key not in seen_orders:
            unique_specs.append(spec)
            seen_orders.add(order_key)
    specs = unique_specs[:min(n_orders, len(unique_specs))]
    if len(specs) < 2:
        raise ValueError("preflight requires at least two unique patch orders")
    row_indices = np.arange(min(n_rows, len(images)), dtype=np.int64)
    unique_full_orders = len(seen_orders)
    single_start = time.perf_counter()
    _compute_curves(
        model,
        images,
        frozen["base"]["test_xgb"],
        frozen["fusion"]["w_xgb"],
        image_arrays,
        specs[:1],
        row_indices,
        device,
        float(config["image"]["blur_radius"]),
    )
    torch.cuda.synchronize()
    single_order_seconds = time.perf_counter() - single_start
    multiple_start = time.perf_counter()
    _compute_curves(
        model,
        images,
        frozen["base"]["test_xgb"],
        frozen["fusion"]["w_xgb"],
        image_arrays,
        specs,
        row_indices,
        device,
        float(config["image"]["blur_radius"]),
    )
    torch.cuda.synchronize()
    multiple_order_seconds = time.perf_counter() - multiple_start
    per_row_first_order = single_order_seconds / len(row_indices)
    per_row_additional_order = (
        (multiple_order_seconds - single_order_seconds)
        / (len(row_indices) * (len(specs) - 1))
    )
    estimated_minutes = (
        len(images)
        * (per_row_first_order + per_row_additional_order * (unique_full_orders - 1))
        / 60
    )
    report = {
        "rows_timed": int(len(row_indices)),
        "orders_timed": int(len(specs)),
        "unique_orders_full": unique_full_orders,
        "single_order_seconds": single_order_seconds,
        "multiple_order_seconds": multiple_order_seconds,
        "estimated_full_run_minutes": estimated_minutes,
        "metrics_saved": False,
        "purpose": "timing only; no test outcomes inspected or used for protocol changes",
    }
    report_path = output_dir / "image_preflight_timing.json"
    suffix = 2
    while report_path.exists():
        report_path = output_dir / f"image_preflight_timing_{suffix:03d}.json"
        suffix += 1
    report["timing_file"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def run_full(
    frozen: dict[str, object],
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    config: dict[str, object],
    output_dir: Path,
    chunk_size: int,
    resume: bool,
) -> None:
    image_arrays = frozen["images"]
    specs = build_order_specs(
        image_arrays,
        int(config["seed"]),
        config["image"]["random_order_seeds"],
        float(config["image"]["center_gaussian_sigma"]),
    )
    order_names = [str(spec["name"]) for spec in specs]
    target_sources = [str(spec["target_source"]) for spec in specs]
    all_targets = np.column_stack([
        _target_classes(image_arrays, source) for source in target_sources
    ])
    n_rows = len(images)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        path = output_dir / f"image_curves_{start:03d}_{end:03d}.npz"
        if path.exists():
            if not resume:
                raise FileExistsError(f"result chunk exists; rerun with --resume: {path}")
            with np.load(path, allow_pickle=False) as saved:
                if saved["test_row_keys"].tolist() != frozen["split"]["test_img_ids"][start:end]:
                    raise ValueError(f"saved row keys do not match frozen split: {path}")
                if saved["curves"].shape != (end - start, len(specs), 2, 2, 21):
                    raise ValueError(f"saved curve shape is invalid: {path}")
                if not np.isfinite(saved["curves"]).all():
                    raise ValueError(f"saved curves contain non-finite values: {path}")
            print(f"Verified existing rows {start}:{end}")
            continue

        row_indices = np.arange(start, end, dtype=np.int64)
        curves = _compute_curves(
            model,
            images,
            frozen["base"]["test_xgb"],
            frozen["fusion"]["w_xgb"],
            image_arrays,
            specs,
            row_indices,
            device,
            float(config["image"]["blur_radius"]),
        )
        if curves.shape != (end - start, len(specs), 2, 2, 21):
            raise RuntimeError(f"unexpected image curve output shape: {curves.shape}")
        if not np.isfinite(curves).all():
            raise RuntimeError(f"non-finite image curve values in rows {start}:{end}")
        _save_chunk(
            path,
            np.asarray(frozen["split"]["test_img_ids"][start:end]),
            order_names,
            target_sources,
            all_targets[start:end],
            curves,
        )
        print(f"Saved image curves for rows {start}:{end} to {path}")


def _transform_images(images: torch.Tensor, name: str) -> torch.Tensor:
    if name == "horizontal_flip":
        return torch.flip(images, dims=(-1,))
    if name.startswith("brightness_"):
        return brightness_transform(images, float(name.removeprefix("brightness_")))
    raise ValueError(f"unknown image transformation: {name}")


def _map_similarity_arrays(
    reference_maps: dict[str, np.ndarray],
    candidate_maps: dict[str, np.ndarray],
    fused_mask: np.ndarray | None = None,
    vit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(next(iter(candidate_maps.values())))
    spearman = np.full((n_rows, len(MAP_KEYS)), np.nan, dtype=np.float32)
    ssim = np.full_like(spearman, np.nan)
    for map_index, key in enumerate(MAP_KEYS):
        source_mask = fused_mask if key.endswith("_fused") else vit_mask
        valid_rows = np.arange(n_rows) if source_mask is None else np.flatnonzero(source_mask)
        for row in valid_rows:
            spearman[row, map_index] = spearman_map_similarity(
                reference_maps[key][row],
                candidate_maps[key][row],
            )
            ssim[row, map_index] = ssim_map_similarity(
                reference_maps[key][row],
                candidate_maps[key][row],
            )
    return spearman, ssim


def run_image_stability(
    model: nn.Module,
    images: torch.Tensor,
    frozen: dict[str, object],
    row_indices: np.ndarray,
    transform_name: str,
    device: torch.device,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    originals = images[rows]
    transformed = _transform_images(originals, transform_name)
    original_vit = _predict_vit(
        model,
        originals,
        device,
        int(config["image_batch_size"]),
    )
    transformed_vit = _predict_vit(
        model,
        transformed,
        device,
        int(config["image_batch_size"]),
    )
    xgb = frozen["base"]["test_xgb"][rows]
    weights = frozen["fusion"]["w_xgb"]
    original_cwf = cwf_probabilities(xgb, original_vit, weights)
    transformed_cwf = cwf_probabilities(xgb, transformed_vit, weights)
    targets_fused = _target_classes(frozen["images"], "fused")[rows]
    targets_vit = _target_classes(frozen["images"], "vit_top")[rows]
    fused_unchanged = unchanged_prediction_mask(original_cwf, transformed_cwf)
    vit_unchanged = unchanged_prediction_mask(original_vit, transformed_vit)
    transformed_maps = _compute_explanation_maps(
        model,
        transformed,
        transformed_vit,
        targets_fused,
        targets_vit,
        device,
        float(config["blur_radius"]),
        int(config["occlusion_batch_size"]),
    )
    if transform_name == "horizontal_flip":
        transformed_maps = {
            key: map_back_horizontal_flip(value)
            for key, value in transformed_maps.items()
        }
    reference_maps = {
        key: frozen["images"][key][rows]
        for key in MAP_KEYS
    }
    map_spearman, map_ssim = _map_similarity_arrays(
        reference_maps,
        transformed_maps,
        fused_unchanged,
        vit_unchanged,
    )
    return {
        "test_row_keys": np.asarray(frozen["split"]["test_img_ids"])[rows],
        "transform_name": np.asarray(transform_name),
        "map_keys": np.asarray(MAP_KEYS),
        "target_classes_fused": targets_fused,
        "target_classes_vit_top": targets_vit,
        "original_vit_probabilities": original_vit.astype(np.float32),
        "transformed_vit_probabilities": transformed_vit.astype(np.float32),
        "original_cwf_probabilities": original_cwf.astype(np.float32),
        "transformed_cwf_probabilities": transformed_cwf.astype(np.float32),
        "fused_unchanged_mask": fused_unchanged,
        "vit_unchanged_mask": vit_unchanged,
        "map_spearman": map_spearman,
        "map_ssim": map_ssim,
        "map_spearman_valid": np.isfinite(map_spearman),
        "map_ssim_valid": np.isfinite(map_ssim),
    }


def run_vit_randomization(
    frozen_model: nn.Module,
    randomized_model: nn.Module | None,
    images: torch.Tensor,
    frozen: dict[str, object],
    row_indices: np.ndarray,
    stage: str,
    seed: int,
    device: torch.device,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    if randomized_model is None:
        randomized_model = randomize_vit_parameters(frozen_model, stage, seed)
    randomized_probabilities = _predict_vit(
        randomized_model,
        images[rows],
        device,
        int(config["image_batch_size"]),
    )
    xgb = frozen["base"]["test_xgb"][rows]
    randomized_cwf = cwf_probabilities(
        xgb,
        randomized_probabilities,
        frozen["fusion"]["w_xgb"],
    )
    targets_fused = _target_classes(frozen["images"], "fused")[rows]
    targets_vit = _target_classes(frozen["images"], "vit_top")[rows]
    randomized_maps = _compute_explanation_maps(
        randomized_model,
        images[rows],
        randomized_probabilities,
        targets_fused,
        targets_vit,
        device,
        float(config["blur_radius"]),
        int(config["occlusion_batch_size"]),
    )
    reference_maps = {
        key: frozen["images"][key][rows]
        for key in MAP_KEYS
    }
    map_spearman, map_ssim = _map_similarity_arrays(
        reference_maps,
        randomized_maps,
    )
    original_vit = frozen["base"]["test_vit"][rows]
    original_cwf = frozen["fusion"]["cwf_test"][rows]
    return {
        "test_row_keys": np.asarray(frozen["split"]["test_img_ids"])[rows],
        "randomization_stage": np.asarray(stage),
        "randomization_seed": np.asarray(seed, dtype=np.int64),
        "map_keys": np.asarray(MAP_KEYS),
        "target_classes_fused": targets_fused,
        "target_classes_vit_top": targets_vit,
        "frozen_vit_probabilities": original_vit.astype(np.float32),
        "randomized_vit_probabilities": randomized_probabilities.astype(np.float32),
        "frozen_cwf_probabilities": original_cwf.astype(np.float32),
        "randomized_cwf_probabilities": randomized_cwf.astype(np.float32),
        "vit_argmax_changed": (
            randomized_probabilities.argmax(axis=1) != original_vit.argmax(axis=1)
        ),
        "cwf_argmax_changed": (
            randomized_cwf.argmax(axis=1) != original_cwf.argmax(axis=1)
        ),
        "vit_max_abs_probability_delta": np.max(
            np.abs(randomized_probabilities - original_vit),
            axis=1,
        ).astype(np.float32),
        "cwf_max_abs_probability_delta": np.max(
            np.abs(randomized_cwf - original_cwf),
            axis=1,
        ).astype(np.float32),
        "map_spearman": map_spearman,
        "map_ssim": map_ssim,
        "map_spearman_valid": np.isfinite(map_spearman),
        "map_ssim_valid": np.isfinite(map_ssim),
    }


def _initialize_sanity_run(
    frozen: dict[str, object],
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    config = json.loads(SANITY_CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256_file(SANITY_CONFIG_PATH)
    identity_path = output_dir / "image_sanity_run_identity.json"
    identity = {
        "config_sha256": config_hash,
        "input_sha256": dict(sorted(frozen["input_hashes"].items())),
    }
    snapshot_path = output_dir / "image_sanity_config.json"
    result_chunks = list(output_dir.glob("image_stability_*.npz"))
    result_chunks += list(output_dir.glob("image_randomization_*.npz"))

    if identity_path.exists():
        if not resume:
            raise FileExistsError(
                f"sanity run identity already exists; pass --resume: {identity_path}"
            )
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("sanity resume config or protected input hashes do not match")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume missing sanity run: {identity_path}")
        if result_chunks:
            raise ValueError("sanity result chunks exist without a matching run identity")
        identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    if snapshot_path.exists():
        if sha256_file(snapshot_path) != config_hash:
            raise ValueError("saved sanity configuration differs from the active configuration")
    else:
        snapshot_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if sha256_file(snapshot_path) != config_hash:
            raise RuntimeError("sanity configuration snapshot hash changed during writing")
    return config, identity


def _save_sanity_chunk(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, path)


def _verify_sanity_chunk(
    path: Path,
    expected_ids: list[str],
    is_stability: bool,
) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if saved["test_row_keys"].tolist() != expected_ids:
            raise ValueError(f"saved row keys do not match frozen split: {path}")
        if saved["map_keys"].tolist() != list(MAP_KEYS):
            raise ValueError(f"saved map-key order is invalid: {path}")
        if saved["map_spearman"].shape != (len(expected_ids), len(MAP_KEYS)):
            raise ValueError(f"saved Spearman shape is invalid: {path}")
        if saved["map_ssim"].shape != (len(expected_ids), len(MAP_KEYS)):
            raise ValueError(f"saved SSIM shape is invalid: {path}")
        if not np.array_equal(
            saved["map_spearman_valid"],
            np.isfinite(saved["map_spearman"]),
        ):
            raise ValueError(f"saved Spearman-valid mask is inconsistent: {path}")
        if not np.array_equal(saved["map_ssim_valid"], np.isfinite(saved["map_ssim"])):
            raise ValueError(f"saved SSIM-valid mask is inconsistent: {path}")
        if not np.isfinite(saved["map_ssim"][saved["map_ssim_valid"]]).all():
            raise ValueError(f"saved SSIM values contain non-finite results: {path}")
        if is_stability:
            for key in ("fused_unchanged_mask", "vit_unchanged_mask"):
                if saved[key].shape != (len(expected_ids),):
                    raise ValueError(f"saved unchanged mask is invalid: {path}")
            if not np.isfinite(saved["map_spearman"][saved["map_spearman_valid"]]).all():
                raise ValueError(f"saved Spearman values contain non-finite results: {path}")
        else:
            for key in (
                "randomized_vit_probabilities",
                "randomized_cwf_probabilities",
                "vit_max_abs_probability_delta",
                "cwf_max_abs_probability_delta",
            ):
                if not np.isfinite(saved[key]).all():
                    raise ValueError(f"saved randomized outputs contain non-finite values: {path}")
            if not np.isfinite(saved["map_spearman"][saved["map_spearman_valid"]]).all():
                raise ValueError(f"saved Spearman values contain non-finite results: {path}")


def run_sanity_preflight(
    frozen_model: nn.Module,
    frozen: dict[str, object],
    images: torch.Tensor,
    device: torch.device,
    config: dict[str, object],
    output_dir: Path,
) -> None:
    row_indices = np.arange(min(4, len(images)), dtype=np.int64)
    timings = []
    for transform in config["transformations"]:
        start = time.perf_counter()
        run_image_stability(
            frozen_model,
            images,
            frozen,
            row_indices,
            str(transform),
            device,
            config,
        )
        torch.cuda.synchronize()
        timings.append({
            "condition": f"stability:{transform}",
            "rows": len(row_indices),
            "seconds": time.perf_counter() - start,
        })
    for stage_info in config["randomization_stages"]:
        stage = str(stage_info["name"])
        start = time.perf_counter()
        randomized = randomize_vit_parameters(
            frozen_model,
            stage,
            int(stage_info["seed"]),
        )
        run_vit_randomization(
            frozen_model,
            randomized,
            images,
            frozen,
            row_indices,
            stage,
            int(stage_info["seed"]),
            device,
            config,
        )
        torch.cuda.synchronize()
        timings.append({
            "condition": f"randomization:{stage}",
            "rows": len(row_indices),
            "seconds": time.perf_counter() - start,
        })
        del randomized
        torch.cuda.empty_cache()

    estimated_seconds = sum(
        item["seconds"] * len(images) / item["rows"]
        for item in timings
    )
    report = {
        "rows_timed_per_condition": len(row_indices),
        "conditions": timings,
        "estimated_full_run_minutes": estimated_seconds / 60.0,
        "metrics_saved": False,
        "purpose": "timing only; scores are discarded and not used for protocol changes",
    }
    report_path = output_dir / "image_sanity_preflight_timing.json"
    suffix = 2
    while report_path.exists():
        report_path = output_dir / f"image_sanity_preflight_timing_{suffix:03d}.json"
        suffix += 1
    report["timing_file"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_sanity_full(
    frozen_model: nn.Module,
    frozen: dict[str, object],
    images: torch.Tensor,
    device: torch.device,
    config: dict[str, object],
    output_dir: Path,
    resume: bool,
) -> None:
    n_rows = len(images)
    chunk_size = int(config["chunk_size"])
    starts = list(range(0, n_rows, chunk_size))
    for transform in config["transformations"]:
        for start in starts:
            end = min(start + chunk_size, n_rows)
            path = output_dir / f"image_stability_{transform}_{start:03d}_{end:03d}.npz"
            expected_ids = frozen["split"]["test_img_ids"][start:end]
            if path.exists():
                if not resume:
                    raise FileExistsError(f"sanity chunk exists; rerun with --resume: {path}")
                _verify_sanity_chunk(path, expected_ids, is_stability=True)
                continue
            result = run_image_stability(
                frozen_model,
                images,
                frozen,
                np.arange(start, end, dtype=np.int64),
                str(transform),
                device,
                config,
            )
            _save_sanity_chunk(path, result)
            print(f"Saved stability {transform} rows {start}:{end}")

    for stage_info in config["randomization_stages"]:
        stage = str(stage_info["name"])
        seed = int(stage_info["seed"])
        paths = [
            output_dir / f"image_randomization_{stage}_{start:03d}_{min(start + chunk_size, n_rows):03d}.npz"
            for start in starts
        ]
        if all(path.exists() for path in paths):
            if not resume:
                raise FileExistsError(
                    f"randomization chunks exist for {stage}; rerun with --resume"
                )
            for path, start in zip(paths, starts):
                end = min(start + chunk_size, n_rows)
                _verify_sanity_chunk(
                    path,
                    frozen["split"]["test_img_ids"][start:end],
                    is_stability=False,
                )
            continue

        randomized_model = randomize_vit_parameters(frozen_model, stage, seed)
        try:
            for start, path in zip(starts, paths):
                end = min(start + chunk_size, n_rows)
                expected_ids = frozen["split"]["test_img_ids"][start:end]
                if path.exists():
                    if not resume:
                        raise FileExistsError(
                            f"randomization chunk exists; rerun with --resume: {path}"
                        )
                    _verify_sanity_chunk(path, expected_ids, is_stability=False)
                    continue
                result = run_vit_randomization(
                    frozen_model,
                    randomized_model,
                    images,
                    frozen,
                    np.arange(start, end, dtype=np.int64),
                    stage,
                    seed,
                    device,
                    config,
                )
                _save_sanity_chunk(path, result)
                print(f"Saved randomization {stage} rows {start}:{end}")
        finally:
            del randomized_model
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--sanity-preflight", action="store_true")
    parser.add_argument("--run-sanity", action="store_true")
    parser.add_argument("--preflight-rows", type=int, default=2)
    parser.add_argument("--preflight-orders", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight, args.sanity_preflight, args.run_sanity)) > 1:
        parser.error("choose only one of --preflight, --sanity-preflight, or --run-sanity")
    if args.preflight_rows <= 0 or args.preflight_orders <= 0 or args.chunk_size <= 0:
        parser.error("row, order, and chunk sizes must be positive")

    frozen, model, images, device = _make_model_and_data(args.root)
    if args.sanity_preflight or args.run_sanity:
        config = _initialize_run(frozen, args.root, resume=True)
    else:
        config = _initialize_run(frozen, args.root, resume=args.resume)
    output_dir = args.root / config["output_directory"]
    print("GPU:", torch.cuda.get_device_name(device))
    print("CUDA runtime:", torch.version.cuda)
    print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("Python:", platform.python_version(), "NumPy:", np.__version__)
    print("Pillow:", PIL.__version__)
    print("scikit-image:", skimage.__version__)
    print("GPU memory GiB:", round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2))

    if args.preflight:
        run_preflight(
            frozen,
            model,
            images,
            device,
            config,
            output_dir,
            args.preflight_rows,
            args.preflight_orders,
        )
    elif args.sanity_preflight or args.run_sanity:
        sanity_config, _ = _initialize_sanity_run(
            frozen,
            output_dir,
            resume=args.resume,
        )
        if args.sanity_preflight:
            run_sanity_preflight(
                model,
                frozen,
                images,
                device,
                sanity_config,
                output_dir,
            )
        else:
            run_sanity_full(
                model,
                frozen,
                images,
                device,
                sanity_config,
                output_dir,
                resume=args.resume,
            )
    else:
        run_full(
            frozen,
            model,
            images,
            device,
            config,
            output_dir,
            args.chunk_size,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
