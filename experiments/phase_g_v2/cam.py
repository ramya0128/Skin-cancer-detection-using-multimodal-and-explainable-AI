from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MapBatch:
    maps: np.ndarray
    valid: np.ndarray
    invalid_reasons: np.ndarray
    diagnostics: dict[str, np.ndarray]


def normalize_cam_maps(raw_maps: np.ndarray, epsilon: float = 1e-8) -> MapBatch:
    raw = np.asarray(raw_maps, dtype=np.float64)
    if raw.ndim != 3 or min(raw.shape[1:]) < 1:
        raise ValueError("raw CAM maps must have shape (N, H, W)")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    maps = np.zeros(raw.shape, dtype=np.float32)
    valid = np.zeros(len(raw), dtype=bool)
    reasons = np.full(len(raw), "", dtype="U32")
    raw_min = np.full(len(raw), np.nan, dtype=np.float64)
    raw_max = np.full(len(raw), np.nan, dtype=np.float64)
    positive_count = np.zeros(len(raw), dtype=np.int64)
    for row, values in enumerate(raw):
        if not np.isfinite(values).all():
            reasons[row] = "non_finite_raw_map"
            continue
        raw_min[row] = float(values.min())
        raw_max[row] = float(values.max())
        positive = np.maximum(values, 0.0)
        positive_count[row] = int(np.count_nonzero(positive > 0.0))
        if positive_count[row] == 0:
            reasons[row] = "no_positive_support"
            continue
        low = float(positive.min())
        high = float(positive.max())
        span = high - low
        if not np.isfinite(span) or span <= epsilon:
            reasons[row] = "constant_positive_map"
            continue
        normalized = (positive - low) / span
        total = float(normalized.sum())
        if not np.isfinite(normalized).all() or not np.isfinite(total):
            reasons[row] = "non_finite_normalized_map"
            continue
        if total <= epsilon:
            reasons[row] = "zero_normalized_map"
            continue
        maps[row] = normalized.astype(np.float32)
        valid[row] = True
    return MapBatch(
        maps=maps,
        valid=valid,
        invalid_reasons=reasons,
        diagnostics={
            "raw_min": raw_min,
            "raw_max": raw_max,
            "positive_patch_count": positive_count,
        },
    )


def _model_logits(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
) -> torch.Tensor:
    if model_kind == "image_only":
        if categories is not None:
            raise ValueError("image-only Grad-CAM cannot consume categories")
        return model(images)
    if model_kind == "joint":
        if categories is None:
            raise ValueError("joint Grad-CAM requires categories")
        return model(images, categories)
    raise ValueError("model_kind must be 'image_only' or 'joint'")


def compute_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    model_kind: str,
    categories: torch.Tensor | None = None,
    *,
    layer_index: int = -1,
    weighting: str = "global",
    method_id: str | None = None,
) -> MapBatch:
    if weighting not in {"global", "token_local"}:
        raise ValueError("weighting must be 'global' or 'token_local'")
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("images must have shape (N, 3, 224, 224)")
    if images.dtype != torch.float32:
        raise TypeError("Grad-CAM inputs must be float32 for stable gradients")
    if targets.shape != (len(images),):
        raise ValueError("targets must align with image rows")
    if torch.any((targets < 0) | (targets >= 6)):
        raise ValueError("targets must lie in [0, 5]")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Grad-CAM model must have parameters") from exc
    if images.device != model_device or targets.device != model_device:
        raise ValueError("model, images, and targets must share an explicit device")
    if categories is not None and categories.device != model_device:
        raise ValueError("categories must be on the model device")

    image_encoder = getattr(model, "image_encoder", None)
    layers = getattr(getattr(image_encoder, "encoder", None), "layers", None)
    if layers is None or len(layers) < abs(layer_index):
        raise ValueError("model must expose the requested ViT encoder layer")
    target_layer = layers[layer_index]
    norm = getattr(target_layer, "ln_1", None)
    if norm is None:
        raise ValueError("selected ViT block must expose ln_1")

    activations: list[torch.Tensor] = []
    handle = norm.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        with torch.enable_grad():
            logits = _model_logits(model, model_kind, images, categories)
            score = logits.gather(1, targets[:, None]).sum()
            gradients = torch.autograd.grad(score, activations[0])[0]
    finally:
        handle.remove()

    patch_activations = activations[0][:, 1:, :]
    patch_gradients = gradients[:, 1:, :]
    if weighting == "global":
        weights = patch_gradients.mean(dim=1, keepdim=True)
        raw = (patch_activations * weights).sum(dim=-1)
    else:
        raw = (patch_activations * patch_gradients).sum(dim=-1)
    raw_maps = raw.reshape(-1, 14, 14).detach().cpu().numpy()
    result = normalize_cam_maps(raw_maps)
    diagnostics: dict[str, np.ndarray] = {
        **result.diagnostics,
        "gradient_l2": patch_gradients.detach().flatten(1).norm(dim=1).cpu().numpy(),
        "activation_l2": patch_activations.detach().flatten(1).norm(dim=1).cpu().numpy(),
        "method_id": np.full(
            len(images),
            method_id or ("gradcam_v1" if weighting == "global" else "gradcam_v2_token_local"),
            dtype="U32",
        ),
    }
    return MapBatch(
        maps=result.maps,
        valid=result.valid,
        invalid_reasons=result.invalid_reasons,
        diagnostics=diagnostics,
    )


def serializable_map_batch(batch: MapBatch, prefix: str = "") -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        f"{prefix}maps": batch.maps,
        f"{prefix}valid": batch.valid,
        f"{prefix}invalid_reasons": batch.invalid_reasons,
    }
    values.update(
        {f"{prefix}diagnostic_{name}": value for name, value in batch.diagnostics.items()}
    )
    return values
