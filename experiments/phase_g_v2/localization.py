from __future__ import annotations

from typing import Any

import numpy as np


def _to_patch_masks(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 3:
        raise ValueError("region masks must have shape (N, H, W)")
    if values.shape[1:] == (224, 224):
        return values.astype(np.float64).reshape(-1, 14, 16, 14, 16).mean(axis=(2, 4))
    if values.shape[1:] == (14, 14):
        patch_masks = values.astype(np.float64)
        if not np.isfinite(patch_masks).all() or np.any(
            (patch_masks < 0.0) | (patch_masks > 1.0)
        ):
            raise ValueError("fractional patch masks must lie in [0, 1]")
        return patch_masks
    raise ValueError("region masks must be 224x224 or 14x14")


def localization_metrics(
    maps: np.ndarray,
    patient_ids: np.ndarray,
    lesion_masks: np.ndarray,
    marker_masks: np.ndarray,
    *,
    upstream_valid: np.ndarray | None = None,
    upstream_reasons: np.ndarray | None = None,
    translated_control_masks: np.ndarray | None = None,
) -> dict[str, Any]:
    attribution = np.asarray(maps, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    if attribution.ndim != 3 or attribution.shape[1:] != (14, 14):
        raise ValueError("maps must have shape (N, 14, 14)")
    count = len(attribution)
    if patient_ids.shape != (count,) or len(np.unique(patient_ids)) != count:
        raise ValueError("patient_ids must be unique and align with maps")
    lesion = _to_patch_masks(lesion_masks)
    marker = _to_patch_masks(marker_masks)
    if lesion.shape != attribution.shape or marker.shape != attribution.shape:
        raise ValueError("maps and region masks must align")
    if np.any(lesion + marker > 1.0 + 1e-8):
        raise ValueError("lesion and marker masks must not overlap")
    if upstream_valid is None:
        valid = np.ones(count, dtype=bool)
    else:
        valid = np.asarray(upstream_valid, dtype=bool).copy()
        if valid.shape != (count,):
            raise ValueError("upstream_valid must align with maps")
    reasons = np.full(count, "", dtype="U48")
    if upstream_reasons is not None:
        given = np.asarray(upstream_reasons, dtype="U48")
        if given.shape != (count,):
            raise ValueError("upstream_reasons must align with maps")
        reasons[:] = given

    for index, row in enumerate(attribution):
        if not np.isfinite(row).all():
            valid[index] = False
            reasons[index] = "non_finite_map"
        elif np.any(row < 0.0):
            valid[index] = False
            reasons[index] = "negative_attribution"
        elif float(row.sum()) <= np.finfo(np.float64).eps:
            valid[index] = False
            reasons[index] = "zero_map"
        elif not valid[index] and not reasons[index]:
            reasons[index] = "invalid_upstream"

    background = np.clip(1.0 - lesion - marker, 0.0, 1.0)
    shares = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in ("marker_share", "lesion_share", "background_share")
    }
    enrichment = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in ("marker_area_enrichment", "lesion_area_enrichment")
    }
    pointing = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in ("marker_pointing_game", "lesion_pointing_game")
    }
    translated_control_mass = None
    if translated_control_masks is not None:
        controls = np.asarray(translated_control_masks)
        if controls.ndim != 4 or controls.shape[0] != count:
            raise ValueError(
                "translated_control_masks must have shape (N, R, H, W)"
            )
        repeat_count = controls.shape[1]
        control_patches = _to_patch_masks(
            controls.reshape(count * repeat_count, *controls.shape[2:])
        ).reshape(count, repeat_count, 14, 14)
        translated_control_mass = np.full(
            (count, repeat_count), np.nan, dtype=np.float64
        )
    for index in np.flatnonzero(valid):
        normalized = attribution[index] / attribution[index].sum()
        marker_share = float(np.sum(normalized * marker[index]))
        lesion_share = float(np.sum(normalized * lesion[index]))
        background_share = float(np.sum(normalized * background[index]))
        if not np.isclose(marker_share + lesion_share + background_share, 1.0, atol=1e-6):
            valid[index] = False
            reasons[index] = "region_partition_not_unit_mass"
            continue
        shares["marker_share"][index] = marker_share
        shares["lesion_share"][index] = lesion_share
        shares["background_share"][index] = background_share
        marker_area = float(marker[index].mean())
        lesion_area = float(lesion[index].mean())
        enrichment["marker_area_enrichment"][index] = (
            marker_share / marker_area if marker_area > 0 else np.nan
        )
        enrichment["lesion_area_enrichment"][index] = (
            lesion_share / lesion_area if lesion_area > 0 else np.nan
        )
        peak = int(np.argmax(normalized))
        pointing["marker_pointing_game"][index] = float(marker[index].reshape(-1)[peak] > 0)
        pointing["lesion_pointing_game"][index] = float(lesion[index].reshape(-1)[peak] > 0)
        if translated_control_mass is not None:
            translated_control_mass[index] = np.sum(
                control_patches[index] * normalized[None, :, :], axis=(1, 2)
            )

    return {
        "patient_ids": patient_ids.copy(),
        "valid": valid,
        "invalid_reasons": reasons,
        "patient_count": count,
        "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if count else 0.0,
        "invalid_fraction": float((~valid).mean()) if count else 0.0,
        **shares,
        **enrichment,
        **pointing,
        "translated_control_mass": translated_control_mass,
    }


def summarize_localization(
    metrics: dict[str, Any], replicates: int = 2000, seed: int = 20261126
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    valid = np.asarray(metrics["valid"], dtype=bool)
    if valid.ndim != 1:
        raise ValueError("validity must be a patient-level vector")
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {
        "patient_count": int(len(valid)),
        "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "invalid_fraction": float((~valid).mean()) if len(valid) else 0.0,
        "valid_patient_ids": np.asarray(metrics["patient_ids"])[valid].tolist(),
        "metrics": {},
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }
    for key in (
        "marker_share",
        "lesion_share",
        "background_share",
        "marker_area_enrichment",
        "lesion_area_enrichment",
        "marker_pointing_game",
        "lesion_pointing_game",
    ):
        values = np.asarray(metrics[key], dtype=np.float64)[valid]
        if not len(values):
            summary["metrics"][key] = None
            continue
        draws = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 128):
            stop = min(start + 128, replicates)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] = values[indices].mean(axis=1)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "n": int(len(values)),
        }
    control_mass = metrics.get("translated_control_mass")
    if control_mass is not None:
        control_values = np.asarray(control_mass, dtype=np.float64)
        control_counts = np.isfinite(control_values).sum(axis=1)
        per_patient = np.divide(
            np.nansum(control_values, axis=1),
            control_counts,
            out=np.full(control_values.shape[0], np.nan, dtype=np.float64),
            where=control_counts > 0,
        )
        values = per_patient[valid & np.isfinite(per_patient)]
        if len(values):
            draws = np.empty(replicates, dtype=np.float64)
            for start in range(0, replicates, 128):
                stop = min(start + 128, replicates)
                indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                draws[start:stop] = values[indices].mean(axis=1)
            summary["metrics"]["translated_control_mass"] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
                "n": int(len(values)),
            }
        else:
            summary["metrics"]["translated_control_mass"] = None
    return summary
