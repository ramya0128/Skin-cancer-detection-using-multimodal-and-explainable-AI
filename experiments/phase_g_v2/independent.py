from __future__ import annotations

from typing import Any

import numpy as np
import torch
from skimage.segmentation import slic
from torch import nn

from experiments.phase_g.generate import _make_glyph


def replace_marker_glyph(
    images: np.ndarray, marker_masks: np.ndarray, cue_values: np.ndarray
) -> np.ndarray:
    source = np.asarray(images)
    masks = np.asarray(marker_masks, dtype=bool)
    values = np.asarray(cue_values, dtype=np.int64)
    single = source.ndim == 3
    if single:
        source = source[None, ...]
        masks = masks[None, ...]
        values = values.reshape(1)
    if (
        source.ndim != 4
        or source.shape[1:] != (224, 224, 3)
        or source.dtype != np.uint8
        or masks.shape != source.shape[:3]
        or values.shape != (len(source),)
        or np.any((values < 0) | (values >= 6))
    ):
        raise ValueError("images, marker masks, or cue values have invalid shapes or values")
    result = source.copy()
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(ys) != 16 * 16:
            raise ValueError("each marker mask must contain exactly a 16x16 glyph")
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        if bottom - top != 16 or right - left != 16:
            raise ValueError("marker mask must be a contiguous 16x16 square")
        expected = np.zeros_like(mask)
        expected[top:bottom, left:right] = True
        if not np.array_equal(mask, expected):
            raise ValueError("marker mask must contain one contiguous 16x16 square")
        result[index, top:bottom, left:right] = _make_glyph(int(values[index]))
    return result[0] if single else result


def segment_superpixels(
    image_rgb: np.ndarray, n_segments: int = 128, compactness: float = 10.0
) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("SLIC requires a 224x224 uint8 RGB image")
    if n_segments < 2 or not np.isfinite(compactness) or compactness <= 0:
        raise ValueError("n_segments and compactness must be positive")
    labels = slic(
        image.astype(np.float32) / 255.0,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
        convert2lab=False,
    )
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise RuntimeError("SLIC labels must be contiguous and start at zero")
    return labels


def _ordered_region_masks(
    labels: np.ndarray, order: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    flat = labels.reshape(-1)
    areas = np.bincount(flat, minlength=int(flat.max()) + 1)
    if order.shape != (len(areas),) or not np.array_equal(
        np.sort(order), np.arange(len(areas))
    ):
        raise ValueError("region order must be a permutation of all superpixels")
    masks = np.zeros((steps, *labels.shape), dtype=bool)
    fractions = np.zeros(steps, dtype=np.float64)
    selected = np.zeros(len(areas), dtype=bool)
    cursor = 0
    selected_area = 0
    pixel_count = flat.size
    for step, target in enumerate(np.linspace(0.0, 1.0, steps)):
        target_area = target * pixel_count
        if step == steps - 1:
            selected[:] = True
            selected_area = pixel_count
            cursor = len(order)
        else:
            while selected_area < target_area and cursor < len(order):
                region = int(order[cursor])
                selected[region] = True
                selected_area += int(areas[region])
                cursor += 1
        masks[step] = selected[flat].reshape(labels.shape)
        fractions[step] = float(selected_area / pixel_count)
    return masks, fractions


def _model_target_probabilities(
    model: nn.Module,
    model_kind: str,
    images: np.ndarray,
    categories: np.ndarray | None,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    results = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            tensor = torch.from_numpy(images[start:stop]).to(
                device=device, dtype=torch.float32
            )
            tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
            category_tensor = (
                None
                if categories is None
                else torch.as_tensor(categories[start:stop], device=device, dtype=torch.long)
            )
            if model_kind == "image_only":
                if category_tensor is not None:
                    raise ValueError("image-only evaluation cannot consume categories")
                logits = model(tensor)
            elif model_kind == "joint":
                if category_tensor is None:
                    raise ValueError("joint evaluation requires fixed clinical categories")
                logits = model(tensor, category_tensor)
            else:
                raise ValueError("model_kind must be 'image_only' or 'joint'")
            probability = torch.softmax(logits.float() / temperature, dim=1)
            target_tensor = torch.as_tensor(targets[start:stop], device=device)
            results.append(
                probability.gather(1, target_tensor[:, None]).squeeze(1).cpu().numpy()
            )
    return np.concatenate(results) if results else np.empty(0, dtype=np.float32)


def _curves_for_regions(
    image: np.ndarray,
    baseline: np.ndarray,
    masks: np.ndarray,
    model: nn.Module,
    model_kind: str,
    category: int | None,
    target: int,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    pixel_masks = masks[..., None]
    deletion = np.where(pixel_masks, baseline[None, ...], image[None, ...])
    insertion = np.where(pixel_masks, image[None, ...], baseline[None, ...])
    variants = np.concatenate((deletion, insertion), axis=0)
    categories = (
        None
        if category is None
        else np.full(len(variants), category, dtype=np.int64)
    )
    targets = np.full(len(variants), target, dtype=np.int64)
    scores = _model_target_probabilities(
        model,
        model_kind,
        variants,
        categories,
        targets,
        device,
        batch_size,
        temperature,
    )
    return scores.reshape(2, len(masks))


def independent_superpixel_faithfulness(
    model: nn.Module,
    model_kind: str,
    images: np.ndarray,
    baselines: dict[str, np.ndarray],
    targets: np.ndarray,
    attribution_maps: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    patient_ids: np.ndarray,
    *,
    categories: np.ndarray | None = None,
    n_segments: int = 128,
    compactness: float = 10.0,
    random_orders: int = 20,
    seed: int = 20261135,
    steps: int = 21,
    batch_size: int = 64,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    images = np.asarray(images)
    targets = np.asarray(targets, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    count = len(images)
    if images.shape != (count, 224, 224, 3) or images.dtype != np.uint8:
        raise ValueError("images must have shape (N, 224, 224, 3) and dtype uint8")
    if targets.shape != (count,) or patient_ids.shape != (count,):
        raise ValueError("targets and patient_ids must align with images")
    if len(np.unique(patient_ids)) != count:
        raise ValueError("patient_ids must be unique")
    if model_kind == "joint":
        if categories is None or np.asarray(categories).shape != (count,):
            raise ValueError("joint evaluation requires one fixed category per patient")
        categories = np.asarray(categories, dtype=np.int64)
        if np.any((categories < 0) | (categories >= 6)):
            raise ValueError("categories must lie in [0, 5]")
    elif categories is not None:
        raise ValueError("image-only evaluation cannot consume categories")
    if not baselines:
        raise ValueError("at least one replacement baseline is required")
    for name, baseline in baselines.items():
        if np.asarray(baseline).shape != images.shape or np.asarray(baseline).dtype != np.uint8:
            raise ValueError(f"baseline {name} must align with uint8 images")
    if set(attribution_maps) != set(valid_masks) or not attribution_maps:
        raise ValueError("each attribution map must have a matching validity mask")
    if random_orders < 1 or steps < 2 or batch_size < 1:
        raise ValueError("random_orders, steps, and batch_size are invalid")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if device is None:
        raise ValueError("an explicit inference device is required")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must have parameters") from exc
    if model_device != device:
        raise ValueError("model must already be placed on the explicit inference device")
    if any(not str(name) for name in baselines):
        raise ValueError("baseline names must be nonempty")

    method_names = tuple(attribution_maps)
    normalized_maps = {}
    method_valid = {}
    for name in method_names:
        values = np.asarray(attribution_maps[name], dtype=np.float64)
        valid = np.asarray(valid_masks[name], dtype=bool)
        if values.shape != (count, 14, 14) or valid.shape != (count,):
            raise ValueError(f"{name} maps or validity mask have invalid shapes")
        if np.any(valid & (~np.isfinite(values).all(axis=(1, 2)))):
            raise ValueError(f"{name} marks non-finite attribution maps valid")
        if np.any(valid & (values < 0).any(axis=(1, 2))):
            raise ValueError(f"{name} marks negative attribution maps valid")
        normalized_maps[name] = values
        method_valid[name] = valid

    baseline_names = tuple(baselines)
    curve_store = {
        name: np.full((count, len(baseline_names), 2, steps), np.nan, dtype=np.float32)
        for name in method_names
    }
    fraction_store = {
        name: np.full((count, len(baseline_names), 2, steps), np.nan, dtype=np.float32)
        for name in method_names
    }
    random_curves = np.empty(
        (count, random_orders, len(baseline_names), 2, steps), dtype=np.float32
    )
    random_fractions = np.empty_like(random_curves)
    center_curves = np.empty((count, len(baseline_names), 2, steps), dtype=np.float32)
    center_fractions = np.empty_like(center_curves)
    region_counts = np.empty(count, dtype=np.int32)
    random_seeds = np.empty((count, random_orders), dtype=np.uint32)
    rng = np.random.default_rng(seed)

    for row in range(count):
        labels = segment_superpixels(images[row], n_segments, compactness)
        segment_count = int(labels.max()) + 1
        region_counts[row] = segment_count
        centroids = np.asarray(
            [
                np.argwhere(labels == region).mean(axis=0)
                for region in range(segment_count)
            ]
        )
        center_order = np.argsort(
            np.sum((centroids - np.asarray([111.5, 111.5])) ** 2, axis=1),
            kind="stable",
        )
        patient_random_orders = []
        for random_index in range(random_orders):
            random_seeds[row, random_index] = rng.integers(0, 2**32, dtype=np.uint32)
            patient_random_orders.append(
                np.random.default_rng(int(random_seeds[row, random_index])).permutation(
                    segment_count
                )
            )
        category = None if categories is None else int(categories[row])
        for baseline_index, baseline_name in enumerate(baseline_names):
            baseline = np.asarray(baselines[baseline_name])[row]
            for name in method_names:
                if not method_valid[name][row]:
                    continue
                patch_map = normalized_maps[name][row]
                pixel_scores = np.repeat(np.repeat(patch_map, 16, axis=0), 16, axis=1)
                segment_scores = np.bincount(
                    labels.reshape(-1),
                    weights=pixel_scores.reshape(-1),
                    minlength=segment_count,
                )
                order = np.argsort(-segment_scores, kind="stable")
                masks, fractions = _ordered_region_masks(labels, order, steps)
                curve_store[name][row, baseline_index] = _curves_for_regions(
                    images[row],
                    baseline,
                    masks,
                    model,
                    model_kind,
                    category,
                    int(targets[row]),
                    device,
                    batch_size,
                    temperature,
                )
                fraction_store[name][row, baseline_index] = np.broadcast_to(
                    fractions[None, :], (2, steps)
                )
            for random_index, order in enumerate(patient_random_orders):
                masks, fractions = _ordered_region_masks(labels, order, steps)
                random_curves[row, random_index, baseline_index] = _curves_for_regions(
                    images[row],
                    baseline,
                    masks,
                    model,
                    model_kind,
                    category,
                    int(targets[row]),
                    device,
                    batch_size,
                    temperature,
                )
                random_fractions[row, random_index, baseline_index] = np.broadcast_to(
                    fractions[None, :], (2, steps)
                )
            masks, fractions = _ordered_region_masks(labels, center_order, steps)
            center_curves[row, baseline_index] = _curves_for_regions(
                images[row],
                baseline,
                masks,
                model,
                model_kind,
                category,
                int(targets[row]),
                device,
                batch_size,
                temperature,
            )
            center_fractions[row, baseline_index] = np.broadcast_to(
                fractions[None, :], (2, steps)
            )

    method_auc = {
        name: np.trapezoid(curve_store[name], x=fraction_store[name], axis=-1)
        for name in method_names
    }
    random_auc = np.trapezoid(random_curves, x=random_fractions, axis=-1)
    center_auc = np.trapezoid(center_curves, x=center_fractions, axis=-1)
    return {
        "patient_ids": patient_ids.copy(),
        "patient_count": count,
        "valid_patient_count": {
            name: int(method_valid[name].sum()) for name in method_names
        },
        "valid_masks": method_valid,
        "method_names": np.asarray(method_names),
        "baseline_names": np.asarray(baseline_names),
        "directions": np.asarray(["deletion", "insertion"]),
        "region_counts": region_counts,
        "curves": curve_store,
        "affected_area_fractions": fraction_store,
        "auc": method_auc,
        "random_curves": random_curves,
        "random_affected_area_fractions": random_fractions,
        "random_auc": random_auc,
        "center_curves": center_curves,
        "center_affected_area_fractions": center_fractions,
        "center_auc": center_auc,
        "random_order_seeds": random_seeds,
        "seed": np.asarray(seed, dtype=np.int64),
        "intervention_family": np.asarray("connected_slic_superpixels"),
        "replacement_description": np.asarray(
            "replace or restore whole connected superpixels with each named baseline"
        ),
        "model_kind": np.asarray(model_kind),
        "device": np.asarray(str(device)),
    }


def marker_counterfactual_sensitivity(
    model: nn.Module,
    model_kind: str,
    images: np.ndarray,
    background_canvases: np.ndarray,
    marker_masks: np.ndarray,
    original_cues: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
    *,
    categories: np.ndarray | None = None,
    seed: int = 20261136,
    batch_size: int = 64,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    images = np.asarray(images)
    backgrounds = np.asarray(background_canvases)
    masks = np.asarray(marker_masks, dtype=bool)
    cues = np.asarray(original_cues, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    count = len(images)
    if (
        images.shape != (count, 224, 224, 3)
        or backgrounds.shape != images.shape
        or masks.shape != images.shape[:3]
        or cues.shape != (count,)
        or targets.shape != (count,)
        or patient_ids.shape != (count,)
    ):
        raise ValueError("marker-intervention inputs must align by patient")
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, 6, size=count)
    replacement_cues = (cues + offsets) % 6
    swapped = replace_marker_glyph(images, masks, replacement_cues)
    removed = images.copy()
    removed[masks] = backgrounds[masks]
    image_sets = (images, swapped, removed)
    set_names = ("original", "cue_swapped", "marker_removed")
    if device is None:
        raise ValueError("an explicit inference device is required")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must have parameters") from exc
    if model_device != device:
        raise ValueError("model must already be placed on the explicit inference device")
    if model_kind == "joint":
        if categories is None or np.asarray(categories).shape != (count,):
            raise ValueError("joint intervention requires fixed patient categories")
        categories = np.asarray(categories, dtype=np.int64)
    elif categories is not None:
        raise ValueError("image-only intervention cannot consume categories")

    scores = {}
    for name, batch in zip(set_names, image_sets, strict=True):
        batch_categories = None
        if categories is not None:
            batch_categories = np.tile(categories, 1)
        scores[name] = _model_target_probabilities(
            model,
            model_kind,
            batch,
            batch_categories,
            targets,
            device,
            batch_size,
            temperature,
        )
    return {
        "patient_ids": patient_ids.copy(),
        "patient_count": count,
        "original_cues": cues,
        "replacement_cues": replacement_cues,
        "target_classes": targets,
        "target_probability_original": scores["original"],
        "target_probability_cue_swapped": scores["cue_swapped"],
        "target_probability_marker_removed": scores["marker_removed"],
        "cue_swap_probability_change": scores["cue_swapped"] - scores["original"],
        "marker_removal_probability_change": scores["marker_removed"] - scores["original"],
        "seed": np.asarray(seed, dtype=np.int64),
        "intervention_family": np.asarray("renderer_glyph_counterfactual"),
        "clinical_category_fixed": categories,
        "replacement_description": np.asarray(
            "replace glyph with another renderer code or restore its exact background canvas"
        ),
    }
