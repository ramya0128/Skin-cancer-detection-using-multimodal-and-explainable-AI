from __future__ import annotations

from typing import Any

import numpy as np


def translated_control_summary(
    translated_control_mass: np.ndarray,
    translated_control_available: np.ndarray,
) -> dict[str, Any]:
    """Summarize translated-control attribution without counting unavailable slots."""
    mass = np.asarray(translated_control_mass, dtype=np.float64)
    availability = np.asarray(translated_control_available, dtype=bool)
    if availability.shape != mass.shape:
        raise ValueError(
            "translated_control_available must align exactly with translated_control_mass"
        )
    if mass.ndim not in (1, 2):
        raise ValueError("translated control mass must be one- or two-dimensional")
    if not np.all(np.isfinite(mass[availability]) | np.isnan(mass[availability])):
        raise ValueError("translated control mass contains invalid non-finite values")

    mass_1d = mass.reshape(-1) if mass.ndim == 1 else mass
    available_1d = availability.reshape(-1) if availability.ndim == 1 else availability
    mass_1d = mass.reshape(-1) if mass.ndim == 1 else mass

    if mass.ndim == 1:
        numeric_slots = availability & np.isfinite(mass)
        values = mass[numeric_slots]
        patient_means = np.asarray([float(values.mean())]) if len(values) else np.asarray([np.nan])
        patient_available = bool(np.any(availability))
        patients_with_valid = int(patient_available)
        patients_without_valid = int(not patient_available)
    else:
        numeric_slots = availability & np.isfinite(mass)
        patient_means = np.full(mass.shape[0], np.nan, dtype=np.float64)
        numeric_counts = numeric_slots.sum(axis=1)
        geometry_counts = availability.sum(axis=1)
        for index in np.flatnonzero(numeric_counts):
            patient_means[index] = float(mass[index, numeric_slots[index]].mean())
        patient_available = bool(geometry_counts.sum() > 0)
        patients_with_valid = int(np.count_nonzero(geometry_counts))
        patients_without_valid = int(mass.shape[0] - patients_with_valid)
        values = mass[numeric_slots]

    total_requested = int(mass.size)
    available_translations = int(np.count_nonzero(availability))
    unavailable_translations = int(total_requested - available_translations)
    mean = float(values.mean()) if len(values) else np.nan

    return {
        "patient_available": patient_available,
        "mean": mean,
        "patient_means": patient_means,
        "total_requested": total_requested,
        "available_translations": available_translations,
        "unavailable_translations": unavailable_translations,
        "patients_with_valid_translation": patients_with_valid,
        "patients_with_no_valid_translation": patients_without_valid,
        "actual_denominator": int(len(values)),
    }
