from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import norm

from experiments.phase_g.model import fit_temperature


PRIMARY_CONTRAST_NAMES = (
    "image_dominant.image_minus_joint",
    "image_dominant.clinical_minus_joint",
    "clinical_dominant.image_minus_joint",
    "clinical_dominant.clinical_minus_joint",
    "redundant.image_minus_joint",
    "redundant.clinical_minus_joint",
    "conflicting.image_minus_joint",
    "conflicting.clinical_minus_joint",
    "synergistic.image_minus_joint",
    "synergistic.clinical_minus_joint",
)

# Phase G has five conditions and two contrasts. The existing first pair is
# diagnostic, not the protocol's primary family.


def holm_adjusted_p_values(p_values: np.ndarray | Iterable[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("p-values must be a nonempty one-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    family_size = len(sorted_values)
    adjusted_sorted = np.empty_like(sorted_values)
    running_max = 0.0
    for rank, value in enumerate(sorted_values):
        adjusted = min(1.0, (family_size - rank) * float(value))
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def _softmax_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _patient_nll(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise ValueError("probabilities and labels must align")
    if probabilities.shape[1] != 6:
        raise ValueError("patient NLL probabilities must contain six classes")
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= 6)):
        raise ValueError("patient NLL labels must be six-class indices")
    selected = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0))


def paired_stratified_bootstrap(
    test_logits: Mapping[str, np.ndarray] | Mapping[str, Mapping[str, np.ndarray]],
    test_labels: np.ndarray | Mapping[str, np.ndarray],
    calibration_logits: Mapping[str, np.ndarray],
    calibration_labels: np.ndarray,
    *,
    contrasts: tuple[str, ...],
    replicates: int = 10000,
    test_seed: int = 20261123,
    calibration_seed: int = 20261129,
    test_patient_ids: np.ndarray | Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    model_names = ("image_only", "clinical_only", "joint")
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    if set(calibration_logits) != set(model_names):
        raise ValueError("bootstrap requires exactly image, clinical and joint calibration logits")
    if set(np.unique(calibration_labels)) != set(range(6)):
        raise ValueError("calibration labels must contain all six classes")

    condition_test_logits: dict[str, dict[str, np.ndarray]] = {}
    condition_test_labels: dict[str, np.ndarray] = {}
    if set(test_logits) == set(model_names):
        if not isinstance(test_labels, np.ndarray) and not isinstance(test_labels, list):
            raise ValueError("single-condition test labels must be an array")
        labels = np.asarray(test_labels, dtype=np.int64)
        if set(np.unique(labels)) != set(range(6)):
            raise ValueError("test labels must contain all six classes")
        condition_test_labels["single"] = labels
        condition_test_logits["single"] = {}
        for name in model_names:
            values = np.asarray(test_logits[name], dtype=np.float64)
            if values.shape != (len(labels), 6) or not np.isfinite(values).all():
                raise ValueError(f"invalid test logits for {name}")
            condition_test_logits["single"][name] = values
    else:
        if not isinstance(test_labels, Mapping):
            raise ValueError("multi-condition test labels must be a mapping")
        if set(test_logits) != set(test_labels):
            raise ValueError("test logits and labels must identify the same conditions")
        if test_patient_ids is not None:
            if not isinstance(test_patient_ids, Mapping):
                raise ValueError("multi-condition test_patient_ids must be a mapping")
            if set(test_patient_ids) != set(test_labels):
                raise ValueError("test patient IDs and labels must identify the same conditions")
            patient_id_arrays = {
                condition: np.asarray(ids)
                for condition, ids in test_patient_ids.items()
            }
            first_condition = next(iter(patient_id_arrays))
            first_ids = patient_id_arrays[first_condition]
            first_labels = np.asarray(test_labels[first_condition], dtype=np.int64)
            for condition, ids in patient_id_arrays.items():
                if len(ids) != len(test_labels[condition]):
                    raise ValueError(f"patient IDs and test labels must align for {condition}")
                if not np.array_equal(ids, first_ids):
                    raise ValueError("multi-condition test patient IDs must be identical")
                if not np.array_equal(
                    np.asarray(test_labels[condition], dtype=np.int64),
                    first_labels,
                ):
                    raise ValueError("multi-condition test labels must match shared patient IDs")
        for condition, model_logits in test_logits.items():
            if set(model_logits) != set(model_names):
                raise ValueError(f"{condition} requires exactly three model logits")
            labels = np.asarray(test_labels[condition], dtype=np.int64)
            if set(np.unique(labels)) != set(range(6)):
                raise ValueError(f"test labels for {condition} must contain all six classes")
            condition_test_labels[condition] = labels
            condition_test_logits[condition] = {}
            for name in model_names:
                values = np.asarray(model_logits[name], dtype=np.float64)
                if values.shape != (len(labels), 6) or not np.isfinite(values).all():
                    raise ValueError(f"invalid test logits for {condition}/{name}")
                condition_test_logits[condition][name] = values
    calibration_arrays = {}
    for name in model_names:
        values = np.asarray(calibration_logits[name], dtype=np.float64)
        if values.shape != (len(calibration_labels), 6) or not np.isfinite(values).all():
            raise ValueError(f"invalid calibration logits for {name}")
        calibration_arrays[name] = values

    test_rng = np.random.default_rng(test_seed)
    calibration_rng = np.random.default_rng(calibration_seed)
    condition_label_arrays = list(condition_test_labels.values())
    shared_condition_resampling = (
        len(condition_label_arrays) > 1
        and all(
            np.array_equal(condition_label_arrays[0], labels)
            for labels in condition_label_arrays[1:]
        )
    )
    bootstrap = {
        contrast: np.empty(replicates, dtype=np.float64)
        for contrast in contrasts
    }

    for replicate in range(replicates):
        calibration_indices = []
        for label in range(6):
            indices = np.flatnonzero(calibration_labels == label)
            calibration_indices.append(
                calibration_rng.choice(indices, size=len(indices), replace=True)
            )
        calibration_resample = np.concatenate(calibration_indices)

        temperatures = {}
        for name in model_names:
            temperatures[name] = fit_temperature(
                calibration_arrays[name][calibration_resample],
                calibration_labels[calibration_resample],
            )

        shared_test_resample = None
        condition_label_arrays = list(condition_test_labels.values())
        if condition_label_arrays and all(
            np.array_equal(condition_label_arrays[0], labels)
            for labels in condition_label_arrays[1:]
        ):
            labels = condition_label_arrays[0]
            test_indices = []
            for label in range(6):
                indices = np.flatnonzero(labels == label)
                test_indices.append(test_rng.choice(indices, size=len(indices), replace=True))
            shared_test_resample = np.concatenate(test_indices)

        gains_by_condition = {}
        for condition, labels in condition_test_labels.items():
            if shared_test_resample is not None:
                test_resample = shared_test_resample
            else:
                test_indices = []
                for label in range(6):
                    indices = np.flatnonzero(labels == label)
                    test_indices.append(test_rng.choice(indices, size=len(indices), replace=True))
                test_resample = np.concatenate(test_indices)
            resampled_labels = labels[test_resample]
            probabilities = {
                name: _softmax_logits(
                    condition_test_logits[condition][name][test_resample],
                    temperatures[name],
                )
                for name in model_names
            }
            joint_nll = _patient_nll(probabilities["joint"], resampled_labels)
            gains_by_condition[condition] = {
                "image_minus_joint": float(
                    (
                        _patient_nll(probabilities["image_only"], resampled_labels)
                        - joint_nll
                    ).mean()
                ),
                "clinical_minus_joint": float(
                    (
                        _patient_nll(probabilities["clinical_only"], resampled_labels)
                        - joint_nll
                    ).mean()
                ),
            }

        for contrast in contrasts:
            condition = contrast.split(".", 1)[0]
            gain_name = contrast.rsplit(".", 1)[1]
            if condition == "single" and "single" in gains_by_condition:
                bootstrap[contrast][replicate] = gains_by_condition["single"][gain_name]
            elif "single" in gains_by_condition:
                bootstrap[contrast][replicate] = gains_by_condition["single"][gain_name]
            elif condition in gains_by_condition:
                bootstrap[contrast][replicate] = gains_by_condition[condition][gain_name]
            else:
                raise ValueError(f"unsupported contrast or missing condition: {contrast}")

    return {
        "bootstrap": bootstrap,
        "temperature_calibration_resampled": True,
        "test_patient_resampled": True,
        "shared_test_resample_across_conditions": shared_condition_resampling,
        "replicates": replicates,
        "test_seed": test_seed,
        "calibration_seed": calibration_seed,
    }


def bootstrap_centered_p_value(
    bootstrap_draws: np.ndarray,
    estimate: float,
    threshold: float = 0.0,
) -> float:
    draws = np.asarray(bootstrap_draws, dtype=np.float64)
    if draws.ndim != 1 or not len(draws):
        raise ValueError("bootstrap draws must be a nonempty vector")
    if not np.isfinite(draws).all() or not np.isfinite(estimate) or not np.isfinite(threshold):
        raise ValueError("bootstrap p-value inputs must be finite")
    # For H0: Delta <= threshold, the bootstrap test tail is the fraction
    # of bootstrap estimates less than or equal to the threshold.
    tail = np.count_nonzero(draws <= threshold)
    return float((1 + tail) / (len(draws) + 1))


def summarize_bootstrap(
    estimates: Mapping[str, float],
    bootstrap: Mapping[str, np.ndarray],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    contrast_names = tuple(estimates)
    if set(bootstrap) != set(contrast_names):
        raise ValueError("bootstrap draws must align with point estimates")
    raw_p_values = {}
    marginal_intervals = {}
    for contrast in contrast_names:
        draws = np.asarray(bootstrap[contrast], dtype=np.float64)
        estimate = float(estimates[contrast])
        raw_p_values[contrast] = bootstrap_centered_p_value(draws, estimate, 0.0)
        marginal_intervals[contrast] = [
            float(value)
            for value in np.quantile(draws, [0.025, 0.975])
        ]
    adjusted = holm_adjusted_p_values(
        [raw_p_values[contrast] for contrast in contrast_names]
    )
    adjusted_p_values = {
        contrast: float(adjusted[index])
        for index, contrast in enumerate(contrast_names)
    }
    adjusted_lower_bounds = {}
    for contrast in contrast_names:
        draws = np.asarray(bootstrap[contrast], dtype=np.float64)
        estimate = float(estimates[contrast])
        lower = float(np.min(draws))
        upper = float(np.max(draws))
        for _ in range(60):
            middle = (lower + upper) / 2
            family = {}
            for candidate in contrast_names:
                candidate_draws = np.asarray(bootstrap[candidate], dtype=np.float64)
                candidate_estimate = float(estimates[candidate])
                family[candidate] = bootstrap_centered_p_value(
                    candidate_draws, candidate_estimate, middle
                )
            candidate_adjusted = holm_adjusted_p_values(
                [family[candidate] for candidate in contrast_names]
            )
            candidate_index = contrast_names.index(contrast)
            if candidate_adjusted[candidate_index] <= alpha:
                lower = middle
            else:
                upper = middle
        adjusted_lower_bounds[contrast] = lower
    return {
        "marginal_percentile_95": marginal_intervals,
        "raw_one_sided_p_values": raw_p_values,
        "holm_adjusted_p_values": adjusted_p_values,
        "holm_adjusted_one_sided_lower_bounds": adjusted_lower_bounds,
        "simultaneous_interpretation": False,
    }


def contrast_success(adjusted_lower_bounds: Mapping[str, float], contrast: str) -> bool:
    if contrast not in adjusted_lower_bounds:
        raise KeyError(f"unknown contrast: {contrast}")
    return math.isfinite(float(adjusted_lower_bounds[contrast])) and float(
        adjusted_lower_bounds[contrast]
    ) > 0


def condition_success(
    adjusted_lower_bounds: Mapping[str, float],
    condition: str,
) -> bool:
    image = f"{condition}.image_minus_joint"
    clinical = f"{condition}.clinical_minus_joint"
    if image not in adjusted_lower_bounds or clinical not in adjusted_lower_bounds:
        raise KeyError(f"missing primary contrasts for condition: {condition}")
    return contrast_success(adjusted_lower_bounds, image) and contrast_success(
        adjusted_lower_bounds, clinical
    )


def experiment_success(
    adjusted_lower_bounds: Mapping[str, float],
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    condition_results = {
        condition: condition_success(adjusted_lower_bounds, condition)
        for condition in conditions
    }
    succeeded = [condition for condition, passed in condition_results.items() if passed]
    return {
        "condition_results": condition_results,
        "succeeded_conditions": succeeded,
        "all_conditions": len(succeeded) == len(conditions),
        "some_conditions": 0 < len(succeeded) < len(conditions),
        "no_conditions": len(succeeded) == 0,
    }


def normal_contrast_power(
    effect: float,
    sd: float,
    per_class_n: int,
    *,
    alpha: float = 0.05,
    family_size: int = 10,
    threshold: float = 0.0,
) -> dict[str, float]:
    if not np.isfinite(effect) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("effect and SD must be finite and SD positive")
    if per_class_n < 1:
        raise ValueError("per-class n must be positive")
    if not 0 < alpha < 1 or family_size < 1:
        raise ValueError("alpha and family size are invalid")
    z = norm.isf(alpha / family_size)
    total_n = per_class_n * 6
    power = float(
        norm.cdf(
            (effect - threshold) * math.sqrt(total_n) / sd - z
        )
    )
    return {
        "per_class_n": float(per_class_n),
        "total_n": float(total_n),
        "effect": float(effect),
        "threshold": float(threshold),
        "margin": float(effect - threshold),
        "sd": float(sd),
        "alpha": float(alpha),
        "family_size": float(family_size),
        "contrast_power": power,
    }


def monte_carlo_contrast_power(
    effect: float,
    sd: float,
    per_class_n: int,
    *,
    family_size: int = 10,
    replicates: int = 20000,
    seed: int = 20261123,
    alpha: float = 0.05,
    threshold: float = 0.0,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("power simulation replicates must be at least 100")
    if family_size < 1:
        raise ValueError("family size must be positive")
    rng = np.random.default_rng(seed)
    total_n = per_class_n * 6
    if not np.isfinite(effect) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("effect and SD must be finite and SD positive")
    z_effect = (effect - threshold) * math.sqrt(total_n) / sd
    z_matrix = rng.normal(z_effect, 1.0, size=(replicates, family_size))
    p_matrix = norm.sf(z_matrix)
    holm_success = np.empty(replicates, dtype=bool)
    for replicate in range(replicates):
        adjusted = holm_adjusted_p_values(p_matrix[replicate])
        holm_success[replicate] = np.min(adjusted) <= alpha
    power = float(holm_success.mean())
    standard_error = float(
        math.sqrt(power * (1 - power) / replicates)
    )
    interval = [
        float(max(0.0, power - 1.96 * standard_error)),
        float(min(1.0, power + 1.96 * standard_error)),
    ]
    return {
        "contrast_power": power,
        "monte_carlo_standard_error": standard_error,
        "approximate_95_interval": interval,
        "replicates": replicates,
        "seed": seed,
        "per_class_n": per_class_n,
        "total_n": total_n,
        "effect": effect,
        "threshold": threshold,
        "sd": sd,
        "family_size": family_size,
    }
