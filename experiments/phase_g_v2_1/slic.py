from __future__ import annotations

from typing import Any

import numpy as np
import skimage
from skimage.segmentation import slic


def slic_parameters(seed: int = 20261144) -> dict[str, Any]:
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("SLIC seed must lie in [0, 2^32-1]")
    return {
        "n_segments": 128,
        "compactness": 10.0,
        "sigma": 0,
        "spacing": None,
        "convert2lab": False,
        "channel_axis": -1,
        "enforce_connectivity": True,
        "min_size_factor": 0.5,
        "max_size_factor": 3,
        "slic_zero": False,
        "start_label": 0,
        "seed": int(seed),
        "scikit_image": skimage.__version__,
    }


def slic_segment(
    image_rgb: np.ndarray,
    *,
    seed: int = 20261144,
) -> dict[str, Any]:
    image = np.asarray(image_rgb)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("SLIC requires a 224x224 uint8 RGB image")
    parameters = slic_parameters(seed)
    np.random.seed(parameters["seed"])
    labels = slic(
        image.astype(np.float32) / 255.0,
        n_segments=parameters["n_segments"],
        compactness=parameters["compactness"],
        sigma=parameters["sigma"],
        convert2lab=parameters["convert2lab"],
        enforce_connectivity=parameters["enforce_connectivity"],
        min_size_factor=parameters["min_size_factor"],
        max_size_factor=parameters["max_size_factor"],
        slic_zero=parameters["slic_zero"],
        start_label=parameters["start_label"],
        channel_axis=parameters["channel_axis"],
    )
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise RuntimeError("SLIC labels must be contiguous and start at zero")
    return {
        "labels": labels,
        "parameters": parameters,
    }
