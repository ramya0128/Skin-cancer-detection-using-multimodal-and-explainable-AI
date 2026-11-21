from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.phase_g_v2.cam import MapBatch, compute_gradcam


GRADCAM_PROTOCOL_METHODS = (
    "gradcam_v1_last_global",
    "gradcam_v2_last_token_local",
    "gradcam_v2_penultimate_global",
    "gradcam_v2_penultimate_token_local",
)


def gradcam_method_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "method_id": "gradcam_v1_last_global",
            "layer_index": -1,
            "weighting": "global",
        },
        {
            "method_id": "gradcam_v2_last_token_local",
            "layer_index": -1,
            "weighting": "token_local",
        },
        {
            "method_id": "gradcam_v2_penultimate_global",
            "layer_index": -2,
            "weighting": "global",
        },
        {
            "method_id": "gradcam_v2_penultimate_token_local",
            "layer_index": -2,
            "weighting": "token_local",
        },
    )


def sha256_checkpoint(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_protocol_gradcam_maps(
    model: nn.Module,
    model_kind: str,
    images: torch.Tensor,
    targets: torch.Tensor,
    categories: torch.Tensor | None,
    *,
    device: torch.device,
) -> dict[str, MapBatch]:
    if device.type != "cuda" or device.index != 0:
        raise ValueError("v2.2 protocol Grad-CAM requires cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v2.2 Grad-CAM")
    if images.dtype != torch.float32:
        raise TypeError("Grad-CAM images must be float32")
    if images.device != device or targets.device != device:
        raise ValueError("Grad-CAM model, images and targets must be on cuda:0")
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("Grad-CAM images must have shape (N, 3, 224, 224)")
    if targets.shape != (len(images),):
        raise ValueError("Grad-CAM targets must align with images")
    if torch.any((targets < 0) | (targets >= 6)):
        raise ValueError("Grad-CAM targets must lie in [0, 5]")
    if model_kind == "joint":
        if categories is None or categories.device != device:
            raise ValueError("joint Grad-CAM requires clinical categories on cuda:0")
        if categories.shape != (len(images),):
            raise ValueError("joint Grad-CAM categories must align with images")
        if torch.any((categories < 0) | (categories >= 6)):
            raise ValueError("joint Grad-CAM categories must lie in [0, 5]")
    elif categories is not None:
        raise ValueError("image-only Grad-CAM cannot consume clinical categories")

    model = model.to(device).eval()
    outputs: dict[str, MapBatch] = {}
    for spec in gradcam_method_specs():
        result = compute_gradcam(
            model,
            images,
            targets,
            model_kind,
            categories,
            layer_index=int(spec["layer_index"]),
            weighting=str(spec["weighting"]),
            method_id=str(spec["method_id"]),
        )
        if result.maps.shape != (len(images), 14, 14):
            raise RuntimeError(f"{spec['method_id']} produced an invalid map shape")
        if not np.isfinite(result.maps).all():
            raise RuntimeError(f"{spec['method_id']} produced a non-finite map")
        if np.any(result.maps < 0):
            raise RuntimeError(f"{spec['method_id']} produced negative attribution")
        outputs[str(spec["method_id"])] = result
    return outputs
