from __future__ import annotations

from typing import Any

import numpy as np

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
