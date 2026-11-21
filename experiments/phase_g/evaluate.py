from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

from experiments.phase_g.explain import (
    _device,
    _image_tensor,
    _logits,
    _validate_temperature,
    compute_image_maps,
)
from experiments.phase_g.generate import SyntheticPatient
from experiments.phase_g.model import fit_temperature


MODEL_NAMES = ("image_only", "clinical_only", "joint")
CONTRAST_NAMES = ("image_only_minus_joint", "clinical_only_minus_joint")
CLASS_COUNT = 6
ACCURACY_FLOOR = 0.80
HOLM_ALPHA = 0.05


def oracle_metrics(reliability: float = 0.85) -> dict[str, float]:
    if not 0.0 < reliability < 1.0:
        raise ValueError("reliability must be strictly between zero and one")
    error_probability = (1.0 - reliability) / (CLASS_COUNT - 1)
    channel = np.full((CLASS_COUNT, CLASS_COUNT), error_probability, dtype=np.float64)
    np.fill_diagonal(channel, reliability)
    prior = 1.0 / CLASS_COUNT

    unimodal_nll = 0.0
    for label in range(CLASS_COUNT):
        for observed in range(CLASS_COUNT):
            probability = prior * channel[label, observed]
            posterior = channel[label, observed]
            unimodal_nll -= probability * np.log(posterior)

    joint_nll = 0.0
    joint_accuracy = 0.0
    for image_value in range(CLASS_COUNT):
        for clinical_value in range(CLASS_COUNT):
            joint_by_label = (
                prior
                * channel[:, image_value]
                * channel[:, clinical_value]
            )
            marginal = float(joint_by_label.sum())
            posterior = joint_by_label / marginal
            joint_nll -= float(np.sum(joint_by_label * np.log(posterior)))
            joint_accuracy += float(joint_by_label.max())

    return {
        "unimodal_accuracy": reliability,
        "joint_accuracy": joint_accuracy,
        "unimodal_nll": float(unimodal_nll),
        "joint_nll": float(joint_nll),
        "ideal_nll_gain": float(unimodal_nll - joint_nll),
    }


def _validate_probability_inputs(
    probabilities: dict[str, np.ndarray], labels: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if set(probabilities) != set(MODEL_NAMES):
        raise ValueError(f"probabilities must contain exactly {MODEL_NAMES}")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise ValueError("labels must be a one-dimensional array of six-class indices")
    normalized: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        values = np.asarray(probabilities[name], dtype=np.float64)
        if values.shape != (len(labels), CLASS_COUNT):
            raise ValueError(f"{name} probabilities must have shape (N, 6)")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} probabilities contain invalid values")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{name} probabilities must sum to one per patient")
        normalized[name] = values
    return normalized, labels


def _per_patient_nll(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    selected = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0))


def paired_nll_gains(
    probabilities: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, np.ndarray]:
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    joint_nll = _per_patient_nll(probabilities["joint"], labels)
    return {
        "image_only_minus_joint": _per_patient_nll(
            probabilities["image_only"], labels
        )
        - joint_nll,
        "clinical_only_minus_joint": _per_patient_nll(
            probabilities["clinical_only"], labels
        )
        - joint_nll,
    }


def centered_bootstrap_p_value(
    bootstrap_estimates: np.ndarray,
    estimate: float,
    threshold: float,
) -> float:
    bootstrap_estimates = np.asarray(bootstrap_estimates, dtype=np.float64)
    if bootstrap_estimates.ndim != 1 or bootstrap_estimates.size == 0:
        raise ValueError("bootstrap estimates must be a nonempty one-dimensional array")
    if not np.isfinite(bootstrap_estimates).all() or not np.isfinite(estimate + threshold):
        raise ValueError("bootstrap p-value inputs must be finite")
    centered_errors = bootstrap_estimates - estimate
    tail_count = int(np.count_nonzero(centered_errors >= estimate - threshold))
    return (1 + tail_count) / (len(bootstrap_estimates) + 1)


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or p_values.size == 0:
        raise ValueError("p-values must be a nonempty one-dimensional array")
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(p_values, kind="stable")
    sorted_values = p_values[order]
    family_size = len(sorted_values)
    adjusted_sorted = np.empty_like(sorted_values)
    running_max = 0.0
    for rank, value in enumerate(sorted_values):
        adjusted = min(1.0, (family_size - rank) * float(value))
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p_values)
    adjusted[order] = adjusted_sorted
    return adjusted


def _softmax_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _stratified_resample_indices(
    classes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    classes = np.asarray(classes, dtype=np.int64)
    if classes.ndim != 1 or set(np.unique(classes)) != set(range(CLASS_COUNT)):
        raise ValueError("stratified resampling requires all six class indices")
    samples = []
    for class_index in range(CLASS_COUNT):
        class_indices = np.flatnonzero(classes == class_index)
        samples.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    return np.concatenate(samples)


def _threshold_p_values(
    bootstrap_estimates: dict[str, np.ndarray],
    estimates: dict[str, float],
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    raw = {
        name: centered_bootstrap_p_value(
            bootstrap_estimates[name], estimates[name], threshold
        )
        for name in CONTRAST_NAMES
    }
    adjusted_array = holm_adjusted_p_values(
        np.asarray([raw[name] for name in CONTRAST_NAMES], dtype=np.float64)
    )
    adjusted = {
        name: float(adjusted_array[index])
        for index, name in enumerate(CONTRAST_NAMES)
    }
    return raw, adjusted


def holm_lower_bounds(
    bootstrap_estimates: dict[str, np.ndarray],
    estimates: dict[str, float],
    alpha: float = HOLM_ALPHA,
) -> dict[str, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if set(bootstrap_estimates) != set(CONTRAST_NAMES) or set(estimates) != set(
        CONTRAST_NAMES
    ):
        raise ValueError(f"bootstrap estimates and point estimates must contain {CONTRAST_NAMES}")
    breakpoints = np.concatenate(
        [
            estimates[name] - (bootstrap_estimates[name] - estimates[name])
            for name in CONTRAST_NAMES
        ]
    )
    lower = float(np.min(breakpoints) - 1.0)
    upper = float(np.max(breakpoints) + 1.0)
    _, adjusted_at_lower = _threshold_p_values(
        bootstrap_estimates, estimates, lower
    )
    bounds = {}
    for name in CONTRAST_NAMES:
        if adjusted_at_lower[name] > alpha:
            bounds[name] = float("-inf")
            continue
        lo = lower
        hi = upper
        for _ in range(60):
            middle = (lo + hi) / 2.0
            _, adjusted = _threshold_p_values(
                bootstrap_estimates, estimates, middle
            )
            if adjusted[name] <= alpha:
                lo = middle
            else:
                hi = middle
        bounds[name] = float(lo)
    return bounds


def _validate_logits(
    logits: dict[str, np.ndarray],
    labels: np.ndarray,
    input_name: str,
) -> dict[str, np.ndarray]:
    if set(logits) != set(MODEL_NAMES):
        raise ValueError(f"{input_name} must contain exactly {MODEL_NAMES}")
    normalized = {}
    for name in MODEL_NAMES:
        values = np.asarray(logits[name], dtype=np.float64)
        if values.shape != (len(labels), CLASS_COUNT):
            raise ValueError(f"{input_name}[{name}] must have shape (N, 6)")
        if not np.isfinite(values).all():
            raise ValueError(f"{input_name}[{name}] contains non-finite logits")
        normalized[name] = values
    return normalized


def primary_nll_inference(
    probabilities: dict[str, np.ndarray],
    test_logits: dict[str, np.ndarray],
    test_labels: np.ndarray,
    calibration_logits: dict[str, np.ndarray],
    calibration_labels: np.ndarray,
    test_classes: np.ndarray,
    replicates: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    probabilities, test_labels = _validate_probability_inputs(probabilities, test_labels)
    test_logits = _validate_logits(test_logits, test_labels, "test_logits")
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    calibration_logits = _validate_logits(
        calibration_logits, calibration_labels, "calibration_logits"
    )
    test_classes = np.asarray(test_classes, dtype=np.int64)
    if test_classes.shape != test_labels.shape or not np.array_equal(
        test_classes, test_labels
    ):
        raise ValueError("test_classes must exactly match true test labels")
    if set(np.unique(calibration_labels)) != set(range(CLASS_COUNT)):
        raise ValueError("calibration data must contain all six classes")
    if set(np.unique(test_classes)) != set(range(CLASS_COUNT)):
        raise ValueError("test data must contain all six classes")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    estimates = {
        name: float(values.mean())
        for name, values in paired_nll_gains(probabilities, test_labels).items()
    }
    bootstrap_estimates = {
        name: np.empty(replicates, dtype=np.float64)
        for name in CONTRAST_NAMES
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        calibration_indices = _stratified_resample_indices(calibration_labels, rng)
        test_indices = _stratified_resample_indices(test_classes, rng)
        resampled_labels = test_labels[test_indices]
        resampled_probabilities = {}
        for name in MODEL_NAMES:
            temperature = fit_temperature(
                calibration_logits[name][calibration_indices],
                calibration_labels[calibration_indices],
            )
            resampled_probabilities[name] = _softmax_logits(
                test_logits[name][test_indices], temperature
            )
        gains = paired_nll_gains(resampled_probabilities, resampled_labels)
        for name in CONTRAST_NAMES:
            bootstrap_estimates[name][replicate] = float(gains[name].mean())

    raw_p_values, adjusted_p_values = _threshold_p_values(
        bootstrap_estimates, estimates, threshold
    )
    lower_bounds = holm_lower_bounds(
        bootstrap_estimates, estimates, HOLM_ALPHA
    )
    uncalibrated_nll = {
        name: float(_per_patient_nll(_softmax_logits(test_logits[name], 1.0), test_labels).mean())
        for name in MODEL_NAMES
    }
    calibrated_nll = {
        name: float(_per_patient_nll(probabilities[name], test_labels).mean())
        for name in MODEL_NAMES
    }
    joint_accuracy = float(np.mean(np.argmax(probabilities["joint"], axis=1) == test_labels))
    return {
        "estimates": estimates,
        "bootstrap_estimates": bootstrap_estimates,
        "raw_p_values": raw_p_values,
        "holm_adjusted_p_values": adjusted_p_values,
        "lower_bounds": lower_bounds,
        "threshold": float(threshold),
        "calibrated_nll": calibrated_nll,
        "uncalibrated_nll": uncalibrated_nll,
        "joint_accuracy": joint_accuracy,
        "passes": bool(
            joint_accuracy >= ACCURACY_FLOOR
            and all(bound > threshold for bound in lower_bounds.values())
        ),
    }


def _fractional_patch_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if masks.shape[1:] == (224, 224):
        return masks.astype(np.float32).reshape(-1, 14, 16, 14, 16).mean(axis=(2, 4))
    if masks.shape[1:] == (14, 14):
        values = masks.astype(np.float32)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("fractional patch masks must lie in [0, 1]")
        return values
    raise ValueError("masks must be 224x224 pixels or 14x14 fractional patches")


def _random_patch_mask(
    occupied: np.ndarray, rng: np.random.Generator, side: int = 16
) -> np.ndarray:
    integral = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for _ in range(10000):
        top = int(rng.integers(0, occupied.shape[0] - side + 1))
        left = int(rng.integers(0, occupied.shape[1] - side + 1))
        bottom, right = top + side, left + side
        area = (
            integral[bottom, right]
            - integral[top, right]
            - integral[bottom, left]
            + integral[top, left]
        )
        if area == 0:
            mask = np.zeros_like(occupied, dtype=bool)
            mask[top:bottom, left:right] = True
            return mask
    raise RuntimeError("could not sample an area-matched marker control")


def _random_lesion_translation(
    lesion: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if lesion.shape != occupied.shape:
        raise ValueError("lesion and occupied masks must have matching shapes")
    desired_area = int(np.count_nonzero(lesion))
    if desired_area == 0:
        raise ValueError("lesion masks cannot be empty")
    available = np.flatnonzero(~occupied.reshape(-1))
    if available.size < desired_area:
        raise RuntimeError(
            "could not sample an area-matched lesion control: "
            f"available pixels={available.size}, desired={desired_area}"
        )
    chosen = rng.choice(available, size=desired_area, replace=False)
    translated = np.zeros_like(lesion, dtype=bool)
    translated.reshape(-1)[chosen] = True
    return translated


def _center_gaussian(size: int = 14, sigma: float = 3.5) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    center = (size - 1) / 2.0
    values = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
    return (values / values.max()).astype(np.float32)


def build_localization_controls(
    masks: dict[str, np.ndarray], seed: int, repeats: int = 20
) -> dict[str, np.ndarray]:
    if set(masks) != {"lesion", "marker"}:
        raise ValueError("masks must contain exactly lesion and marker")
    lesion = np.asarray(masks["lesion"], dtype=bool)
    marker = np.asarray(masks["marker"], dtype=bool)
    if lesion.ndim != 3 or lesion.shape != marker.shape or lesion.shape[1:] != (224, 224):
        raise ValueError("lesion and marker masks must align as (N, 224, 224)")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if np.any(lesion & marker):
        raise ValueError("lesion and marker masks must be disjoint")
    rng = np.random.default_rng(seed)
    marker_controls = np.empty((len(lesion), repeats, 14, 14), dtype=np.float32)
    lesion_controls = np.empty_like(marker_controls)
    random_maps = rng.random((len(lesion), repeats, 14, 14), dtype=np.float32)
    for row in range(len(lesion)):
        occupied = lesion[row] | marker[row]
        for repeat in range(repeats):
            marker_controls[row, repeat] = _fractional_patch_masks(
                _random_patch_mask(occupied, rng)[None, ...]
            )[0]
            lesion_controls[row, repeat] = _fractional_patch_masks(
                _random_lesion_translation(
                    lesion[row], occupied, rng
                )[None, ...]
            )[0]
    return {
        "marker_random_masks": marker_controls,
        "lesion_random_masks": lesion_controls,
        "random_maps": random_maps,
        "center_gaussian": _center_gaussian(),
    }


def _region_mass(attribution: np.ndarray, region: np.ndarray) -> np.ndarray:
    totals = attribution.sum(axis=(1, 2))
    numerator = (attribution * region).sum(axis=(1, 2))
    return np.divide(numerator, totals, out=np.zeros_like(numerator), where=totals > 0)


def localization_metrics(
    maps: np.ndarray,
    masks: dict[str, np.ndarray],
    controls: dict[str, np.ndarray],
    patient_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    attribution = np.asarray(maps, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    if attribution.ndim != 3 or attribution.shape[1:] != (14, 14):
        raise ValueError("maps must have shape (N, 14, 14)")
    if not np.isfinite(attribution).all() or np.any(attribution < 0):
        raise ValueError("maps must be finite and nonnegative")
    if patient_ids.shape != (len(attribution),):
        raise ValueError("patient_ids must align with maps")
    lesion = _fractional_patch_masks(masks["lesion"])
    marker = _fractional_patch_masks(masks["marker"])
    if lesion.shape != attribution.shape or marker.shape != attribution.shape:
        raise ValueError("masks and maps must have the same patient rows")
    if np.any(lesion + marker > 1.0 + 1e-6):
        raise ValueError("lesion and marker masks overlap")
    background = np.clip(1.0 - lesion - marker, 0.0, 1.0)
    totals = attribution.sum(axis=(1, 2), keepdims=True)
    normalized = np.divide(
        attribution,
        totals,
        out=np.zeros_like(attribution),
        where=totals > 0,
    )
    marker_mass = _region_mass(normalized, marker)
    lesion_mass = _region_mass(normalized, lesion)
    background_mass = _region_mass(normalized, background)
    marker_area = marker.mean(axis=(1, 2))
    lesion_area = lesion.mean(axis=(1, 2))
    peak = np.argmax(normalized.reshape(len(normalized), -1), axis=1)
    rows = np.arange(len(normalized))
    marker_random = np.asarray(controls["marker_random_masks"], dtype=np.float64)
    lesion_random = np.asarray(controls["lesion_random_masks"], dtype=np.float64)
    random_maps = np.asarray(controls["random_maps"], dtype=np.float64)
    if (
        marker_random.ndim != 4
        or lesion_random.shape != marker_random.shape
        or marker_random.shape[0] != len(normalized)
        or marker_random.shape[2:] != (14, 14)
    ):
        raise ValueError("area-matched controls must have shape (N, repeats, 14, 14)")
    if random_maps.shape != marker_random.shape:
        raise ValueError("random maps must match the area-matched control shape")
    marker_controls = np.einsum("nhw,nrhw->nr", normalized, marker_random)
    lesion_controls = np.einsum("nhw,nrhw->nr", normalized, lesion_random)
    random_normalized = np.divide(
        random_maps,
        random_maps.sum(axis=(2, 3), keepdims=True),
        out=np.zeros_like(random_maps),
        where=random_maps.sum(axis=(2, 3), keepdims=True) > 0,
    )
    random_marker_mass = np.einsum("nrhw,nhw->nr", random_normalized, marker)
    random_lesion_mass = np.einsum("nrhw,nhw->nr", random_normalized, lesion)
    center = np.asarray(controls["center_gaussian"], dtype=np.float64)
    if center.shape != (14, 14) or not np.isfinite(center).all() or np.any(center < 0):
        raise ValueError("center_gaussian must be a finite nonnegative 14x14 map")
    center /= max(float(center.sum()), np.finfo(np.float64).tiny)
    center_marker_mass = np.einsum("hw,nhw->n", center, marker)
    center_lesion_mass = np.einsum("hw,nhw->n", center, lesion)
    return {
        "patient_ids": patient_ids.copy(),
        "marker_mass": marker_mass.astype(np.float32),
        "lesion_mass": lesion_mass.astype(np.float32),
        "background_mass": background_mass.astype(np.float32),
        "marker_area_fraction": marker_area.astype(np.float32),
        "lesion_area_fraction": lesion_area.astype(np.float32),
        "marker_area_enrichment": np.divide(
            marker_mass, marker_area, out=np.zeros_like(marker_mass), where=marker_area > 0
        ).astype(np.float32),
        "lesion_area_enrichment": np.divide(
            lesion_mass, lesion_area, out=np.zeros_like(lesion_mass), where=lesion_area > 0
        ).astype(np.float32),
        "marker_pointing_game": (marker.reshape(len(marker), -1)[rows, peak] > 0).astype(np.float32),
        "lesion_pointing_game": (lesion.reshape(len(lesion), -1)[rows, peak] > 0).astype(np.float32),
        "marker_random_mass": marker_controls.astype(np.float32),
        "lesion_random_mass": lesion_controls.astype(np.float32),
        "random_map_marker_mass": random_marker_mass.astype(np.float32),
        "random_map_lesion_mass": random_lesion_mass.astype(np.float32),
        "center_gaussian_marker_mass": center_marker_mass.astype(np.float32),
        "center_gaussian_lesion_mass": center_lesion_mass.astype(np.float32),
        "outcome_type": np.asarray("localization"),
    }


def _as_uint_or_float(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.float().div(255.0)
    return image.float()


def _gaussian_blur_batch(images: torch.Tensor, radius: float) -> torch.Tensor:
    blurred = []
    for image in images:
        pixels = (_as_uint_or_float(image).cpu().clamp(0, 1) * 255).round().byte()
        array = pixels.permute(1, 2, 0).contiguous().numpy()
        result = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=radius))
        blurred.append(
            torch.from_numpy(np.asarray(result).copy()).permute(2, 0, 1).float().div_(255.0)
        )
    return torch.stack(blurred)


def _predict_target_scores(
    model: torch.nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: torch.Tensor | None,
    targets: torch.Tensor,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    device = _device(model)
    scores = []
    inference_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), inference_context:
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch = _image_tensor(images[start:end], device)
            batch_categories = None if categories is None else categories[start:end].to(device)
            logits = _logits(model, model_kind, batch, batch_categories)
            probabilities = torch.softmax(logits / temperature, dim=1)
            scores.extend(
                probabilities.gather(1, targets[start:end].to(device)[:, None])
                .squeeze(1)
                .cpu()
                .numpy()
                .tolist()
            )
    return np.asarray(scores, dtype=np.float64)


def _ranked_variants(
    original: torch.Tensor,
    baseline: torch.Tensor,
    order: np.ndarray,
    patch_counts: np.ndarray,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    deletion = []
    insertion = []
    for count in patch_counts:
        deleted = original.clone()
        inserted = baseline.clone()
        for patch in order[:int(count)]:
            top, left = divmod(int(patch), 14)
            top *= 16
            left *= 16
            deleted[:, top:top + 16, left:left + 16] = baseline[
                :, top:top + 16, left:left + 16
            ]
            inserted[:, top:top + 16, left:left + 16] = original[
                :, top:top + 16, left:left + 16
            ]
        deletion.append(deleted)
        insertion.append(inserted)
    return deletion, insertion


def _ranked_variants_batched(
    originals: torch.Tensor,
    baselines: torch.Tensor,
    orders: np.ndarray | torch.Tensor,
    patch_counts: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if originals.ndim != 4 or originals.shape != baselines.shape:
        raise ValueError("originals and baselines must align as (N, 3, 224, 224)")
    count = len(originals)
    if tuple(originals.shape[1:]) != (3, 224, 224):
        raise ValueError("originals and baselines must be 224x224 RGB images")
    order_tensor = torch.as_tensor(orders, dtype=torch.long, device=device)
    if order_tensor.ndim == 1:
        order_tensor = order_tensor.unsqueeze(0).expand(count, -1)
    if order_tensor.shape != (count, 196):
        raise ValueError("orders must contain one 196-patch permutation per image")
    if torch.any(torch.sort(order_tensor, dim=1).values != torch.arange(196, device=device)):
        raise ValueError("each order must be a permutation of the 196 image patches")

    original_tensor = originals.to(device=device, dtype=torch.float32)
    baseline_tensor = baselines.to(device=device, dtype=torch.float32)
    counts = torch.as_tensor(patch_counts, dtype=torch.long, device=device)
    if counts.ndim != 1 or torch.any((counts < 0) | (counts > 196)):
        raise ValueError("patch_counts must be a vector of values in [0, 196]")
    patch_rank = torch.empty_like(order_tensor)
    patch_rank.scatter_(
        1,
        order_tensor,
        torch.arange(196, device=device).expand(count, -1),
    )
    selected = patch_rank[:, None, :] < counts[None, :, None]
    pixel_mask = (
        selected.reshape(count, len(counts), 14, 14)
        .repeat_interleave(16, dim=-2)
        .repeat_interleave(16, dim=-1)
        .unsqueeze(2)
    )
    originals_expanded = original_tensor[:, None]
    baselines_expanded = baseline_tensor[:, None]
    deletion = torch.where(pixel_mask, baselines_expanded, originals_expanded)
    insertion = torch.where(pixel_mask, originals_expanded, baselines_expanded)
    return torch.stack((deletion, insertion), dim=1)


def _curves_for_order(
    model: torch.nn.Module,
    model_kind: str,
    original: torch.Tensor,
    baseline: torch.Tensor,
    category: torch.Tensor | None,
    target: int,
    order: np.ndarray,
    patch_counts: np.ndarray,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    deletion, insertion = _ranked_variants(original, baseline, order, patch_counts)
    variants = torch.stack(deletion + insertion)
    categories = None if category is None else category.expand(len(variants))
    targets = torch.full((len(variants),), int(target), dtype=torch.long)
    scores = _predict_target_scores(
        model, model_kind, variants, categories, targets, batch_size, temperature
    )
    return scores.reshape(2, len(patch_counts)).astype(np.float32)


def deletion_insertion_metrics(
    model: torch.nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: np.ndarray | None,
    background_canvases: torch.Tensor,
    maps: dict[str, np.ndarray],
    targets: np.ndarray,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    model.eval()
    if model_kind not in {"image_only", "joint"}:
        raise ValueError("model_kind must be 'image_only' or 'joint'")
    if images.shape != background_canvases.shape or images.ndim != 4:
        raise ValueError("images and background canvases must be aligned batches")
    n = len(images)
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (n,) or np.any((targets < 0) | (targets >= CLASS_COUNT)):
        raise ValueError("targets must align with images and lie in [0, 5]")
    if not maps:
        raise ValueError("at least one map is required")
    normalized_maps = {}
    for name, value in maps.items():
        scores = np.asarray(value, dtype=np.float64)
        if scores.shape != (n, 14, 14) or not np.isfinite(scores).all():
            raise ValueError(f"{name} must be a finite (N, 14, 14) map")
        normalized_maps[name] = scores
    category_tensor = None
    if model_kind == "joint":
        if categories is None:
            raise ValueError("joint faithfulness requires fixed clinical categories")
        category_tensor = torch.as_tensor(categories, dtype=torch.long)
        if category_tensor.shape != (n,) or torch.any(
            (category_tensor < 0) | (category_tensor >= CLASS_COUNT)
        ):
            raise ValueError("joint clinical categories must align and lie in [0, 5]")
    elif categories is not None:
        raise ValueError("image-only faithfulness cannot consume clinical categories")

    seed = int(config.get("seed", 20261119))
    random_count = int(config.get("random_order_count", 20))
    batch_size = int(config.get("curve_batch_size", 64))
    blur_radius = float(config.get("gaussian_blur_radius", 6.0))
    if random_count < 1 or batch_size < 1 or not np.isfinite(blur_radius) or blur_radius <= 0:
        raise ValueError("random count, batch size, and blur radius must be positive")
    temperature = _validate_temperature(float(config.get("temperature", 1.0)))
    patch_counts = np.round(np.linspace(0, 196, 21)).astype(np.int64)
    baseline_names = ("background_canvas", "gaussian_blur")
    method_names = tuple(normalized_maps)
    curves = np.empty((n, len(method_names), 2, 2, len(patch_counts)), dtype=np.float32)
    random_curves = np.empty((n, random_count, 2, 2, len(patch_counts)), dtype=np.float32)
    center_curves = np.empty((n, 2, 2, len(patch_counts)), dtype=np.float32)
    center_map = _center_gaussian()
    random_seeds = np.random.SeedSequence(seed).generate_state(random_count)
    device = _device(model)
    patient_chunk_size = 8
    inference_batch_size = max(batch_size, 128) if device.type == "cuda" else batch_size
    random_orders = [
        np.random.default_rng(int(order_seed)).permutation(196).astype(np.int64)
        for order_seed in random_seeds
    ]
    center_order = np.argsort(-center_map.reshape(-1), kind="stable")

    for chunk_start in range(0, n, patient_chunk_size):
        chunk_stop = min(chunk_start + patient_chunk_size, n)
        chunk_rows = np.arange(chunk_start, chunk_stop)
        originals = [
            _image_tensor(images[row:row + 1], torch.device("cpu"))[0]
            for row in chunk_rows
        ]
        backgrounds = [
            _image_tensor(background_canvases[row:row + 1], torch.device("cpu"))[0]
            for row in chunk_rows
        ]
        blurred = [
            _gaussian_blur_batch(original.unsqueeze(0), blur_radius)[0]
            for original in originals
        ]

        def batched_patient_curves(
            method_order: np.ndarray | list[np.ndarray],
            baseline_images: list[torch.Tensor],
            target_chunk: np.ndarray,
        ) -> np.ndarray:
            order_array = (
                np.stack(method_order)
                if isinstance(method_order, list)
                else method_order
            )
            variants = _ranked_variants_batched(
                torch.stack(originals),
                torch.stack(baseline_images),
                order_array,
                patch_counts,
                device,
            )
            patient_count = len(chunk_rows)
            variants = variants.reshape(
                patient_count * 2 * len(patch_counts), 3, 224, 224
            )
            categories = (
                category_tensor[chunk_rows]
                .to(device)
                .repeat_interleave(2 * len(patch_counts))
                if category_tensor is not None
                else None
            )
            target_tensor = torch.as_tensor(
                target_chunk, dtype=torch.long, device=device
            ).repeat_interleave(2 * len(patch_counts))
            scores = _predict_target_scores(
                model,
                model_kind,
                variants,
                categories,
                target_tensor,
                inference_batch_size,
                temperature,
            )
            return scores.reshape(patient_count, 2, len(patch_counts)).astype(
                np.float32
            )

        for method_index, (name, value) in enumerate(normalized_maps.items()):
            method_orders = [
                np.argsort(-value[row].reshape(-1), kind="stable")
                for row in chunk_rows
            ]
            for baseline_index, baseline_images in enumerate(
                (backgrounds, blurred)
            ):
                chunk_curves = batched_patient_curves(
                    method_orders,
                    baseline_images,
                    targets[chunk_start:chunk_stop],
                )
                curves[chunk_start:chunk_stop, method_index, baseline_index] = (
                    chunk_curves
                )

        for random_index, order in enumerate(random_orders):
            for baseline_index, baseline_images in enumerate(
                (backgrounds, blurred)
            ):
                chunk_curves = batched_patient_curves(
                    order,
                    baseline_images,
                    targets[chunk_start:chunk_stop],
                )
                random_curves[
                    chunk_start:chunk_stop, random_index, baseline_index
                ] = chunk_curves

        for baseline_index, baseline_images in enumerate((backgrounds, blurred)):
            chunk_curves = batched_patient_curves(
                center_order,
                baseline_images,
                targets[chunk_start:chunk_stop],
            )
            center_curves[chunk_start:chunk_stop, baseline_index] = chunk_curves
    auc = np.trapezoid(curves, x=np.linspace(0, 1, len(patch_counts)), axis=-1)
    random_auc = np.trapezoid(
        random_curves, x=np.linspace(0, 1, len(patch_counts)), axis=-1
    )
    center_auc = np.trapezoid(
        center_curves, x=np.linspace(0, 1, len(patch_counts)), axis=-1
    )
    return {
        "patient_count": np.asarray(n, dtype=np.int64),
        "target_classes": targets,
        "model_kind": np.asarray(model_kind),
        "method_names": np.asarray(method_names),
        "baseline_names": np.asarray(baseline_names),
        "directions": np.asarray(["deletion", "insertion"]),
        "patch_counts": patch_counts,
        "curves": curves,
        "auc": auc.astype(np.float32),
        "random_order_curves": random_curves,
        "random_order_auc": random_auc.astype(np.float32),
        "center_gaussian_curves": center_curves,
        "center_gaussian_auc": center_auc.astype(np.float32),
        "random_order_seeds": random_seeds.astype(np.uint32),
        "temperature": np.asarray(temperature, dtype=np.float64),
        "execution_device": np.asarray(str(device)),
        "inference_batch_size": np.asarray(inference_batch_size, dtype=np.int64),
        "outcome_type": np.asarray("faithfulness"),
        "attention_rollout_role": np.asarray("class_independent_control"),
    }


def _brightness(images: torch.Tensor, factor: float) -> torch.Tensor:
    if images.dtype == torch.uint8:
        raw = images.float().div(255.0)
    else:
        raw = images.float()
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("brightness factors must be finite and positive")
    return (raw * factor).clamp(0.0, 1.0)


def _spearman_map(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = rankdata(first.reshape(-1), method="average")
    second_rank = rankdata(second.reshape(-1), method="average")
    if first_rank.std() == 0 or second_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def stability_metrics(
    model: torch.nn.Module,
    model_kind: str,
    images: torch.Tensor,
    categories: np.ndarray | None,
    maps: dict[str, np.ndarray],
    targets: np.ndarray,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    model.eval()
    if model_kind not in {"image_only", "joint"}:
        raise ValueError("model_kind must be 'image_only' or 'joint'")
    targets = np.asarray(targets, dtype=np.int64)
    n = len(images)
    if targets.shape != (n,):
        raise ValueError("targets must align with images")
    map_names = tuple(maps)
    if not map_names:
        raise ValueError("at least one reference map is required")
    for name, value in maps.items():
        if np.asarray(value).shape != (n, 14, 14):
            raise ValueError(f"{name} maps must have shape (N, 14, 14)")
    if model_kind == "joint":
        if categories is None or np.asarray(categories).shape != (n,):
            raise ValueError("joint stability requires aligned fixed categories")
        category_tensor = torch.as_tensor(categories, dtype=torch.long)
    else:
        if categories is not None:
            raise ValueError("image-only stability cannot consume clinical categories")
        category_tensor = None
    factors = config.get("brightness_factors", [0.9, 1.1])
    if list(factors) != [0.9, 1.1]:
        raise ValueError("approved stability brightness factors are exactly 0.9 and 1.1")
    transforms: list[tuple[str, torch.Tensor, bool]] = [
        ("horizontal_flip", torch.flip(images, dims=(-1,)), True)
    ]
    transforms.extend(
        (f"brightness_{factor:.1f}", _brightness(images, float(factor)), False)
        for factor in factors
    )
    device = _device(model)
    temperature = _validate_temperature(float(config.get("temperature", 1.0)))
    batch_size = int(config.get("image_batch_size", 16))
    if batch_size < 1:
        raise ValueError("image_batch_size must be positive")
    original_probabilities = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = _image_tensor(images[start:end], device)
            batch_categories = None if category_tensor is None else category_tensor[start:end].to(device)
            original_probabilities.append(
                torch.softmax(
                    _logits(model, model_kind, batch, batch_categories) / temperature,
                    dim=1,
                )
                .cpu().numpy()
            )
    original_probabilities_np = np.concatenate(original_probabilities, axis=0)
    n_transforms = len(transforms)
    prediction_changed = np.zeros((n_transforms, n), dtype=bool)
    valid_subset = np.zeros_like(prediction_changed)
    spearman = np.full((n_transforms, n, len(map_names)), np.nan, dtype=np.float32)
    ssim = np.full_like(spearman, np.nan)
    transformed_probabilities = np.empty(
        (n_transforms, n, CLASS_COUNT), dtype=np.float32
    )
    for transform_index, (name, transformed, flipped) in enumerate(transforms):
        probabilities = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = _image_tensor(transformed[start:end], device)
                batch_categories = (
                    None if category_tensor is None else category_tensor[start:end].to(device)
                )
                probabilities.append(
                    torch.softmax(
                        _logits(model, model_kind, batch, batch_categories) / temperature,
                        dim=1,
                    ).cpu().numpy()
                )
        transformed_probabilities[transform_index] = np.concatenate(probabilities, axis=0)
        prediction_changed[transform_index] = (
            original_probabilities_np.argmax(axis=1)
            != transformed_probabilities[transform_index].argmax(axis=1)
        )
        valid = ~prediction_changed[transform_index]
        valid_subset[transform_index] = valid
        transformed_maps = compute_image_maps(
            model,
            model_kind,
            transformed,
            targets,
            categories,
            map_names,
            batch_size,
            temperature,
        )
        if flipped:
            transformed_maps = {
                key: value[:, :, ::-1].copy() for key, value in transformed_maps.items()
            }
        for row in np.flatnonzero(valid):
            for map_index, key in enumerate(map_names):
                reference = np.asarray(maps[key][row], dtype=np.float64)
                candidate = transformed_maps[key][row].astype(np.float64)
                spearman[transform_index, row, map_index] = _spearman_map(
                    reference, candidate
                )
                ssim[transform_index, row, map_index] = structural_similarity(
                    reference,
                    candidate,
                    data_range=1.0,
                    win_size=7,
                    gaussian_weights=False,
                )
    return {
        "transform_names": np.asarray([item[0] for item in transforms]),
        "map_names": np.asarray(map_names),
        "target_classes": targets,
        "temperature": np.asarray(temperature, dtype=np.float64),
        "original_probabilities": original_probabilities_np.astype(np.float32),
        "transformed_probabilities": transformed_probabilities,
        "prediction_changed": prediction_changed,
        "valid_subset": valid_subset,
        "spearman": spearman,
        "ssim": ssim,
        "outcome_type": np.asarray("stability"),
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    replicates: int = 2000,
    seed: int = 20261119,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite patient-level vector")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    chunk = max(1, min(256, replicates))
    for start in range(0, replicates, chunk):
        end = min(start + chunk, replicates)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        estimates[start:end] = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def paired_signflip_p(
    values_a: np.ndarray,
    values_b: np.ndarray,
    draws: int = 10000,
    seed: int = 20261119,
) -> float:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if (
        values_a.ndim != 1
        or values_a.shape != values_b.shape
        or values_a.size == 0
        or not np.isfinite(values_a).all()
        or not np.isfinite(values_b).all()
    ):
        raise ValueError("paired inputs must be aligned finite patient-level vectors")
    if draws < 1:
        raise ValueError("draws must be positive")
    differences = values_a - values_b
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    chunk = max(1, min(256, draws))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(differences)))
        null = np.abs((signs * differences).mean(axis=1))
        exceedances += int(np.count_nonzero(null >= observed))
    return float((exceedances + 1) / (draws + 1))
