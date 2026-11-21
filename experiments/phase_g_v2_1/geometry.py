from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or not source.any():
        raise ValueError("mask must be a nonempty two-dimensional array")
    height, width = source.shape
    ys, xs = np.nonzero(source)
    if (
        int(ys.min()) + dy < 0
        or int(ys.max()) + dy >= height
        or int(xs.min()) + dx < 0
        or int(xs.max()) + dx >= width
    ):
        raise ValueError("translated mask would leave the image")
    result = np.zeros_like(source)
    result[ys + dy, xs + dx] = True
    return result


def valid_translation_offsets(
    lesion: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    lesion = np.asarray(lesion, dtype=bool)
    occupied = np.asarray(occupied, dtype=bool)
    if lesion.ndim != 2 or lesion.shape != occupied.shape or not lesion.any():
        raise ValueError("lesion and occupied masks must be aligned nonempty 2-D masks")
    height, width = lesion.shape
    ys, xs = np.nonzero(lesion)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    overlaps = fftconvolve(
        occupied.astype(np.float64),
        lesion[::-1, ::-1].astype(np.float64),
        mode="full",
    )
    offsets = []
    for dy in range(-min_y, height - 1 - max_y + 1):
        for dx in range(-min_x, width - 1 - max_x + 1):
            overlap = overlaps[dy + height - 1, dx + width - 1]
            if abs(float(overlap)) < 0.5:
                offsets.append((dy, dx))
    if not offsets:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(offsets, dtype=np.int64)


def shape_preserving_controls(
    lesion_masks: np.ndarray,
    marker_masks: np.ndarray,
    *,
    seed: int,
    repeats: int = 20,
) -> dict[str, np.ndarray]:
    lesions = np.asarray(lesion_masks, dtype=bool)
    markers = np.asarray(marker_masks, dtype=bool)
    if (
        lesions.ndim != 3
        or lesions.shape != markers.shape
        or not np.all(lesions.reshape(len(lesions), -1).any(axis=1))
    ):
        raise ValueError("lesion_masks and marker_masks must be aligned nonempty masks")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if np.any(lesions & markers):
        raise ValueError("lesion and marker masks must not overlap")

    count, height, width = lesions.shape
    rng = np.random.default_rng(seed)
    controls = np.zeros((count, repeats, height, width), dtype=bool)
    offsets = np.full((count, repeats, 2), -1, dtype=np.int64)
    placement_counts = np.zeros((count, repeats), dtype=np.int64)
    available = np.zeros((count, repeats), dtype=bool)
    for row in range(count):
        candidates = valid_translation_offsets(
            lesions[row],
            lesions[row] | markers[row],
        )
        placement_counts[row, :] = len(candidates)
        if not len(candidates):
            continue
        for repeat in range(repeats):
            selected = candidates[int(rng.integers(0, len(candidates)))]
            dy, dx = map(int, selected)
            controls[row, repeat] = translate_mask(lesions[row], dy, dx)
            offsets[row, repeat] = (dy, dx)
            available[row, repeat] = True
    return {
        "control_masks": controls,
        "selected_offsets": offsets,
        "valid_placement_counts": placement_counts,
        "available": available,
        "patient_available": available.any(axis=1),
        "total_requested": np.asarray(count * repeats, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
    }


def area_matched_random_pixel_null(
    lesion: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    lesion = np.asarray(lesion, dtype=bool)
    occupied = np.asarray(occupied, dtype=bool)
    if lesion.ndim != 2 or lesion.shape != occupied.shape or not lesion.any():
        raise ValueError("lesion and occupied masks must be aligned and nonempty")
    available_pixels = np.flatnonzero(~occupied)
    desired = int(lesion.sum())
    if len(available_pixels) < desired:
        raise RuntimeError("not enough unoccupied pixels for the random-pixel null")
    chosen = rng.choice(available_pixels, size=desired, replace=False)
    control = np.zeros_like(lesion)
    control.reshape(-1)[chosen] = True
    return control
