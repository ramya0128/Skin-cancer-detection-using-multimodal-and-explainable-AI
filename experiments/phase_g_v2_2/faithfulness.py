from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.phase_g_v2.independent import independent_superpixel_faithfulness


PROTOCOL_BASELINES = (
    "renderer_background",
    "mid_gray_128",
)


def build_protocol_baselines(
    images: np.ndarray,
    background_canvases: np.ndarray,
) -> dict[str, np.ndarray]:
    images = np.asarray(images)
    backgrounds = np.asarray(background_canvases)
    if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise ValueError("faithfulness images must have shape (N, 224, 224, 3)")
    if images.dtype != np.uint8 or backgrounds.shape != images.shape:
        raise ValueError("faithfulness images and renderer baselines must align uint8 arrays")
    if backgrounds.dtype != np.uint8:
        raise ValueError("renderer baselines must use uint8")
    return {
        "renderer_background": backgrounds.copy(),
        "mid_gray_128": np.full_like(images, 128),
    }


def independent_protocol_faithfulness(
    model: nn.Module,
    model_kind: str,
    images: np.ndarray,
    background_canvases: np.ndarray,
    targets: np.ndarray,
    attribution_maps: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    patient_ids: np.ndarray,
    *,
    categories: np.ndarray | None = None,
    temperature: float = 1.0,
    seed: int = 20261144,
    slic_seed: int = 20261144,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or device.index != 0:
        raise ValueError("v2.2 protocol faithfulness requires cuda:0")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("faithfulness model has no parameters") from error
    if model_device != device:
        raise ValueError("faithfulness model must already be on cuda:0")
    if slic_seed < 0 or slic_seed > 2**32 - 1:
        raise ValueError("SLIC seed must lie in the supported uint32 range")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("random-control seed must lie in the supported uint32 range")
    np.random.seed(slic_seed)

    result = independent_superpixel_faithfulness(
        model,
        model_kind,
        np.asarray(images, dtype=np.uint8),
        build_protocol_baselines(images, background_canvases),
        np.asarray(targets, dtype=np.int64),
        attribution_maps,
        valid_masks,
        np.asarray(patient_ids),
        categories=categories,
        n_segments=128,
        compactness=10.0,
        random_orders=20,
        seed=seed,
        steps=21,
        batch_size=64,
        temperature=temperature,
        device=device,
    )
    if tuple(str(value) for value in result["baseline_names"]) != PROTOCOL_BASELINES:
        raise RuntimeError("faithfulness baseline names do not match the protocol")
    if not np.asarray(result["region_counts"]).size:
        raise RuntimeError("faithfulness produced no region counts")
    return result
