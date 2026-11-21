from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import scipy
import skimage
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_g.generate import SyntheticPatient, generate_split
from experiments.phase_g.model import build_model
from experiments.phase_g.run import atomic_save_npz, atomic_write_json
from experiments.phase_g_v2.cam import MapBatch, compute_gradcam, serializable_map_batch
from experiments.phase_g_v2.geometry import shape_preserving_controls
from experiments.phase_g_v2.independent import (
    independent_superpixel_faithfulness,
    marker_counterfactual_sensitivity,
)
from experiments.phase_g_v2.localization import localization_metrics, summarize_localization


V1_ROOT = Path("outputs") / "synthetic_phase_g_v1" / "pilot_balanced_085_085_v1"
BASELINE_MANIFEST = (
    Path("outputs") / "synthetic_phase_g_v2" / "v1_preservation_sha256_before.json"
)
OUTPUT_DIRECTORY = (
    Path("outputs") / "synthetic_phase_g_v2" / "validation_pilot_balanced_v2"
)
V2_SEED = 20261142
GEOMETRY_SEED = 20261143
INDEPENDENT_SEED = 20261144
INTERVENTION_SEED = 20261145
INDEPENDENT_PATIENTS_PER_CLASS = 12
GEOMETRY_REPEATS = 20
MAP_METHODS = (
    "gradcam_v1_last_global",
    "gradcam_v2_last_token_local",
    "gradcam_v2_penultimate_global",
    "gradcam_v2_penultimate_token_local",
    "chefer",
    "attention_rollout",
    "occlusion",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_npz_arrays(
    values: dict[str, Any], prefix: str = ""
) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            flattened.update(flatten_npz_arrays(value, f"{name}__"))
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"cannot save object array at {name}")
        flattened[name] = array
    return flattened


def _verify_v1_baseline(root: Path) -> dict[str, str]:
    manifest_path = root / BASELINE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    if not isinstance(expected, dict) or len(expected) != int(manifest["file_count"]):
        raise ValueError("v1 preservation baseline is incomplete")
    observed = {}
    for relative, expected_hash in expected.items():
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"protected v1 artifact is missing: {relative}")
        observed[relative] = _sha256_file(path)
        if observed[relative] != expected_hash:
            raise RuntimeError(f"protected v1 artifact changed: {relative}")
    return observed


def select_validation_patient_ids(
    split_record: dict[str, Any], per_class: int | None = None
) -> list[str]:
    partitions = split_record.get("partition_patient_ids", {})
    validation_ids = partitions.get("validation_model_selection")
    test_ids = partitions.get("test")
    labels = split_record.get("development_labels")
    if not isinstance(validation_ids, list) or not validation_ids:
        raise ValueError("validation_model_selection patient IDs are missing")
    if not isinstance(test_ids, list) or not isinstance(labels, dict):
        raise ValueError("saved split record is incomplete")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("validation_model_selection patient IDs are not unique")
    if set(validation_ids).intersection(test_ids):
        raise ValueError("validation_model_selection IDs overlap the test partition")
    if any(patient_id not in labels for patient_id in validation_ids):
        raise ValueError("validation labels are missing for selected patient IDs")
    if per_class is None:
        return list(validation_ids)
    if per_class < 1:
        raise ValueError("per_class must be positive")
    class_ids = sorted({int(labels[patient_id]) for patient_id in validation_ids})
    result = []
    for class_id in class_ids:
        class_ids = [
            patient_id
            for patient_id in validation_ids
            if int(labels[patient_id]) == class_id
        ]
        if len(class_ids) < per_class:
            raise ValueError(
                f"class {class_id} has fewer than {per_class} validation patients"
            )
        result.extend(class_ids[:per_class])
    return result


def summarize_faithfulness(
    result: dict[str, Any], replicates: int = 2000, seed: int = INDEPENDENT_SEED
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    patient_ids = np.asarray(result["patient_ids"])
    baseline_names = [str(value) for value in result["baseline_names"]]
    directions = [str(value) for value in result["directions"]]
    random_auc = np.asarray(result["random_auc"], dtype=np.float64)
    center_auc = np.asarray(result["center_auc"], dtype=np.float64)
    if random_auc.ndim != 4 or center_auc.ndim != 3:
        raise ValueError("faithfulness control AUC arrays have invalid dimensions")
    if random_auc.shape[0] != len(patient_ids) or center_auc.shape[0] != len(patient_ids):
        raise ValueError("faithfulness controls must align with patients")
    if random_auc.shape[2:] != center_auc.shape[1:]:
        raise ValueError("random and center controls must share baseline/direction axes")
    rng = np.random.default_rng(seed)

    def interval(values: np.ndarray) -> list[float]:
        draws = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 128):
            stop = min(start + 128, replicates)
            sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] = values[sampled].mean(axis=1)
        return [float(value) for value in np.quantile(draws, [0.025, 0.975])]

    summary: dict[str, Any] = {
        "patient_count": int(len(patient_ids)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "methods": {},
    }
    for method, method_auc in result["auc"].items():
        auc = np.asarray(method_auc, dtype=np.float64)
        valid = np.asarray(result["valid_masks"][method], dtype=bool)
        if auc.shape != center_auc.shape or valid.shape != (len(patient_ids),):
            raise ValueError(f"{method} AUC or validity arrays are misaligned")
        summary["methods"][method] = {}
        for baseline_index, baseline_name in enumerate(baseline_names):
            summary["methods"][method][baseline_name] = {}
            for direction_index, direction in enumerate(directions):
                method_values = auc[:, baseline_index, direction_index]
                random_values = random_auc[:, :, baseline_index, direction_index].mean(axis=1)
                center_values = center_auc[:, baseline_index, direction_index]
                paired = (
                    valid
                    & np.isfinite(method_values)
                    & np.isfinite(random_values)
                    & np.isfinite(center_values)
                )
                method_values = method_values[paired]
                random_values = random_values[paired]
                center_values = center_values[paired]
                if not len(method_values):
                    summary["methods"][method][baseline_name][direction] = {
                        "n": 0,
                        "method_mean_auc": None,
                        "random_mean_auc": None,
                        "center_mean_auc": None,
                        "method_minus_random_mean": None,
                        "method_minus_center_mean": None,
                    }
                    continue
                difference_random = method_values - random_values
                difference_center = method_values - center_values
                summary["methods"][method][baseline_name][direction] = {
                    "n": int(len(method_values)),
                    "method_mean_auc": float(method_values.mean()),
                    "method_mean_auc_ci_95": interval(method_values),
                    "random_mean_auc": float(random_values.mean()),
                    "random_mean_auc_ci_95": interval(random_values),
                    "center_mean_auc": float(center_values.mean()),
                    "center_mean_auc_ci_95": interval(center_values),
                    "method_minus_random_mean": float(difference_random.mean()),
                    "method_minus_random_ci_95": interval(difference_random),
                    "method_minus_center_mean": float(difference_center.mean()),
                    "method_minus_center_ci_95": interval(difference_center),
                }
    return summary


def _format_interval(value: dict[str, Any] | None) -> str:
    if value is None or value.get("mean") is None:
        return "n/a"
    interval = value.get("ci_95")
    if not interval:
        return f"{float(value['mean']):.4f}"
    return f"{float(value['mean']):.4f} [{float(interval[0]):.4f}, {float(interval[1]):.4f}]"


def _write_validation_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase G v2 validation-only report",
        "",
        "This is a validation-only methodological re-evaluation. Frozen v1 models "
        "were used without retraining; test rows, labels, and predictions were not "
        "loaded or analyzed. Calibration temperatures came from the frozen "
        "`validation_calibration` partition.",
        "",
        f"- Evaluated partition: `{summary['partition']}` ({summary['patient_count']} patients)",
        f"- Independent perturbation subset: {summary['independent_evaluation_patient_count']} "
        f"patients ({INDEPENDENT_PATIENTS_PER_CLASS} per synthetic class)",
        f"- Device: `{summary['device']}`",
        f"- Test partition evaluated: `{summary['test_partition_evaluated']}`",
        f"- Models retrained: `{summary['models_retrained']}`",
        "",
        "## Validity and localization",
        "",
        "Attribution masses are calculated only for valid maps. Bootstrap intervals "
        "resample patients; invalid maps are excluded and counted.",
        "",
        "| Model | Method | Valid / total | Lesion mass | Marker mass | Background mass | Translated-control mass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_kind, method_summaries in summary["map_validity_and_localization"].items():
        for method, result in method_summaries.items():
            metrics = result["metrics"]
            lines.append(
                "| {model} | {method} | {valid} / {total} | {lesion} | {marker} | "
                "{background} | {control} |".format(
                    model=model_kind,
                    method=method,
                    valid=result["valid_count"],
                    total=result["patient_count"],
                    lesion=_format_interval(metrics.get("lesion_share")),
                    marker=_format_interval(metrics.get("marker_share")),
                    background=_format_interval(metrics.get("background_share")),
                    control=_format_interval(metrics.get("translated_control_mass")),
                )
            )
    lines.extend(
        [
            "",
            "The 95% intervals are descriptive patient-bootstrap intervals. "
            "Translated-control placement and mass are reported separately from "
            "the lesion and marker regions.",
            "",
            "## Shape-preserving geometry controls",
            "",
            "| Model | Patients | Available translated masks | Patients with no valid placement | Valid-placement count range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_kind, values in summary["geometry_controls"].items():
        limits = values["valid_placement_count_range"]
        lines.append(
            f"| {model_kind} | {values['patient_count']} | "
            f"{values['available_control_count']} / "
            f"{values['patient_count'] * values['repeats_per_patient']} | "
            f"{values['unavailable_patient_count']} | {limits[0]}-{limits[1]} |"
        )
    lines.extend(
        [
            "",
            "## Independent superpixel faithfulness",
            "",
            "AUCs are shown as mean [95% patient-bootstrap interval]. Paired "
            "method-minus-control differences are calculated within patient; "
            "random controls are averaged over their 20 orders within patient "
            "before the bootstrap. Lower deletion AUC and higher insertion AUC "
            "indicate stronger target-probability change/recovery under the "
            "specified replacement baseline, but results remain descriptive.",
            "",
            "| Model | Method | Baseline | Direction | Method AUC | Random AUC | Center AUC | Method - random | Method - center | n |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_kind, result in summary["independent_superpixel_faithfulness"].items():
        for method, baseline_summaries in result["auc_summary"]["methods"].items():
            for baseline_name, directions in baseline_summaries.items():
                for direction, values in directions.items():
                    if values["n"] == 0:
                        method_auc = random_auc = center_auc = diff_random = diff_center = "n/a"
                    else:
                        method_auc = _format_interval(
                            {
                                "mean": values["method_mean_auc"],
                                "ci_95": values["method_mean_auc_ci_95"],
                            }
                        )
                        random_auc = _format_interval(
                            {
                                "mean": values["random_mean_auc"],
                                "ci_95": values["random_mean_auc_ci_95"],
                            }
                        )
                        center_auc = _format_interval(
                            {
                                "mean": values["center_mean_auc"],
                                "ci_95": values["center_mean_auc_ci_95"],
                            }
                        )
                        diff_random = _format_interval(
                            {
                                "mean": values["method_minus_random_mean"],
                                "ci_95": values["method_minus_random_ci_95"],
                            }
                        )
                        diff_center = _format_interval(
                            {
                                "mean": values["method_minus_center_mean"],
                                "ci_95": values["method_minus_center_ci_95"],
                            }
                        )
                    lines.append(
                        f"| {model_kind} | {method} | {baseline_name} | {direction} | "
                        f"{method_auc} | {random_auc} | {center_auc} | {diff_random} | "
                        f"{diff_center} | {values['n']} |"
                    )
    lines.extend(
        [
            "",
            "## Separate marker intervention sensitivity",
            "",
            "| Model | Patients | Mean target-probability change after glyph swap | "
            "Mean change after marker removal |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_kind, values in summary["marker_counterfactual_sensitivity"].items():
        lines.append(
            f"| {model_kind} | {values['patient_count']} | "
            f"{values['mean_cue_swap_probability_change']:.4f} | "
            f"{values['mean_marker_removal_probability_change']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These renderer-based cue interventions are sensitivity analyses, not "
            "explanation-faithfulness scores.",
            "",
            "## Geometry controls and limitations",
            "",
            "Shape-preserving controls are integer translations with exact lesion "
            "shape and pixel area, outside both the original lesion and marker. "
            "Unavailable controls are not replaced with random-pixel masks. "
            "Superpixel results use connected SLIC regions, renderer-background "
            "and mid-gray replacements, 20 random orders, and a center-first "
            "control. The validation data are not an independent test set; these "
            "findings do not establish clinical or external validity.",
            "",
        ]
    )
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def _environment(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase G v2 validation requires an available CUDA device")
    if device.index is None or device.index != 0:
        raise ValueError("Phase G v2 validation is fixed to cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "scikit_image": skimage.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _load_model(
    kind: str, config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen {kind} checkpoint is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != kind or "state_dict" not in checkpoint:
        raise ValueError(f"frozen {kind} checkpoint has an invalid structure")
    model = build_model(kind, config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _method_batch(values: np.ndarray, name: str) -> MapBatch:
    maps = np.asarray(values, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[1:] != (14, 14):
        raise ValueError(f"{name} maps must have shape (N, 14, 14)")
    finite = np.isfinite(maps).all(axis=(1, 2))
    nonnegative = (maps >= 0).all(axis=(1, 2))
    positive = maps.sum(axis=(1, 2)) > np.finfo(np.float32).eps
    valid = finite & nonnegative & positive
    reasons = np.full(len(maps), "", dtype="U48")
    reasons[~finite] = "non_finite_legacy_map"
    reasons[finite & ~nonnegative] = "negative_legacy_attribution"
    reasons[finite & nonnegative & ~positive] = "zero_map_after_legacy_normalization"
    diagnostics = {"method_id": np.full(len(maps), name, dtype="U48")}
    return MapBatch(
        maps=np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0),
        valid=valid,
        invalid_reasons=reasons,
        diagnostics=diagnostics,
    )


def _map_batches(
    model: torch.nn.Module,
    model_kind: str,
    patients: list[SyntheticPatient],
    device: torch.device,
    batch_size: int = 16,
) -> tuple[dict[str, MapBatch], dict[str, dict[str, np.ndarray]]]:
    count = len(patients)
    labels = np.asarray([patient.label for patient in patients], dtype=np.int64)
    categories = np.asarray(
        [patient.clinical_value for patient in patients], dtype=np.int64
    )
    images = np.stack([patient.image for patient in patients])
    outputs = {
        name: np.zeros((count, 14, 14), dtype=np.float32) for name in MAP_METHODS
    }
    valid = {name: np.zeros(count, dtype=bool) for name in MAP_METHODS}
    reasons = {name: np.full(count, "", dtype="U48") for name in MAP_METHODS}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    gradcam_specs = (
        ("gradcam_v1_last_global", -1, "global"),
        ("gradcam_v2_last_token_local", -1, "token_local"),
        ("gradcam_v2_penultimate_global", -2, "global"),
        ("gradcam_v2_penultimate_token_local", -2, "token_local"),
    )
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        image_batch = torch.from_numpy(
            images[start:stop].transpose(0, 3, 1, 2).copy()
        ).to(device=device, dtype=torch.float32).div_(255.0)
        target_batch = torch.as_tensor(labels[start:stop], device=device)
        category_batch = (
            None
            if model_kind == "image_only"
            else torch.as_tensor(categories[start:stop], device=device)
        )
        for name, layer_index, weighting in gradcam_specs:
            result = compute_gradcam(
                model,
                image_batch,
                target_batch,
                model_kind,
                category_batch,
                layer_index=layer_index,
                weighting=weighting,
                method_id=name,
            )
            outputs[name][start:stop] = result.maps
            valid[name][start:stop] = result.valid
            reasons[name][start:stop] = result.invalid_reasons
            diagnostics.setdefault(name, {})
            for key, value in result.diagnostics.items():
                diagnostics[name].setdefault(
                    key, np.empty((count, *value.shape[1:]), dtype=value.dtype)
                )[start:stop] = value

    from experiments.phase_g.explain import compute_image_maps

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        image_batch = torch.from_numpy(
            images[start:stop].transpose(0, 3, 1, 2).copy()
        )
        category_batch = (
            None if model_kind == "image_only" else categories[start:stop]
        )
        legacy = compute_image_maps(
            model,
            model_kind,
            image_batch,
            labels[start:stop],
            category_batch,
            ("chefer", "attention_rollout", "occlusion"),
            batch_size=batch_size,
        )
        for name, values in legacy.items():
            result = _method_batch(values, name)
            outputs[name][start:stop] = result.maps
            valid[name][start:stop] = result.valid
            reasons[name][start:stop] = result.invalid_reasons

    batches = {
        name: MapBatch(outputs[name], valid[name], reasons[name], diagnostics.get(name, {}))
        for name in MAP_METHODS
    }
    return batches, diagnostics


def _localization_outputs(
    patients: list[SyntheticPatient],
    maps: dict[str, MapBatch],
    output_dir: Path,
    chunk_size: int = 32,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    patient_ids = np.asarray([patient.patient_id for patient in patients])
    lesions = np.stack([patient.lesion_mask for patient in patients])
    markers = np.stack([patient.marker_mask for patient in patients])
    geometry: dict[str, list[np.ndarray]] = {
        "selected_offsets": [],
        "valid_placement_counts": [],
        "available": [],
    }
    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in maps
    }
    for start in range(0, len(patients), chunk_size):
        stop = min(start + chunk_size, len(patients))
        controls = shape_preserving_controls(
            lesions[start:stop],
            markers[start:stop],
            seed=GEOMETRY_SEED + start,
            repeats=GEOMETRY_REPEATS,
        )
        for key in geometry:
            geometry[key].append(controls[key])
        for name, batch in maps.items():
            result = localization_metrics(
                batch.maps[start:stop],
                patient_ids[start:stop],
                lesions[start:stop],
                markers[start:stop],
                upstream_valid=batch.valid[start:stop],
                upstream_reasons=batch.invalid_reasons[start:stop],
                translated_control_masks=controls["control_masks"],
            )
            for key, value in result.items():
                if key == "patient_ids" or value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 0 or array.shape[0] != stop - start:
                    continue
                accumulated[name].setdefault(key, []).append(array)

    geometry_arrays = {key: np.concatenate(parts, axis=0) for key, parts in geometry.items()}
    map_metrics = {}
    saved_arrays: dict[str, np.ndarray] = {
        "patient_ids": patient_ids,
        "selected_offsets": geometry_arrays["selected_offsets"],
        "valid_placement_counts": geometry_arrays["valid_placement_counts"],
        "available": geometry_arrays["available"],
    }
    for name, fields in accumulated.items():
        combined = {key: np.concatenate(parts, axis=0) for key, parts in fields.items()}
        combined["patient_ids"] = patient_ids.copy()
        map_metrics[name] = summarize_localization(combined)
        for key, value in combined.items():
            saved_arrays[f"{name}__{key}"] = np.asarray(value)
    atomic_save_npz(output_dir / "localization_and_geometry.npz", saved_arrays)
    return map_metrics, {
        "selected_offsets": geometry_arrays["selected_offsets"],
        "valid_placement_counts": geometry_arrays["valid_placement_counts"],
        "available": geometry_arrays["available"],
    }


def _patient_arrays(patients: list[SyntheticPatient]) -> dict[str, np.ndarray]:
    return {
        "patient_ids": np.asarray([patient.patient_id for patient in patients]),
        "labels": np.asarray([patient.label for patient in patients], dtype=np.int64),
        "categories": np.asarray(
            [patient.clinical_value for patient in patients], dtype=np.int64
        ),
        "image_cues": np.asarray([patient.image_cue for patient in patients], dtype=np.int64),
        "images": np.stack([patient.image for patient in patients]),
        "backgrounds": np.stack([patient.background_canvas for patient in patients]),
        "lesion_masks": np.stack([patient.lesion_mask for patient in patients]),
        "marker_masks": np.stack([patient.marker_mask for patient in patients]),
    }


def _run_independent_evaluation(
    model: torch.nn.Module,
    model_kind: str,
    patients: list[SyntheticPatient],
    map_batches: dict[str, MapBatch],
    selected_indices: np.ndarray,
    temperature: float,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [patients[int(index)] for index in selected_indices]
    arrays = _patient_arrays(selected)
    baseline_images = {
        "renderer_background": arrays["backgrounds"],
        "midgray_128": np.full_like(arrays["images"], 128),
    }
    selected_maps = {
        name: batch.maps[selected_indices] for name, batch in map_batches.items()
    }
    selected_validity = {
        name: batch.valid[selected_indices] for name, batch in map_batches.items()
    }
    torch.cuda.reset_peak_memory_stats(device)
    benchmark_start = time.perf_counter()
    benchmark = independent_superpixel_faithfulness(
        model,
        model_kind,
        arrays["images"][:1],
        {name: values[:1] for name, values in baseline_images.items()},
        arrays["labels"][:1],
        {"attention_rollout": selected_maps["attention_rollout"][:1]},
        {"attention_rollout": selected_validity["attention_rollout"][:1]},
        arrays["patient_ids"][:1],
        categories=arrays["categories"][:1] if model_kind == "joint" else None,
        n_segments=128,
        random_orders=2,
        seed=INDEPENDENT_SEED,
        steps=21,
        batch_size=64,
        temperature=temperature,
        device=device,
    )
    torch.cuda.synchronize(device)
    benchmark_record = {
        "patient_count": 1,
        "method": "attention_rollout",
        "n_segments": 128,
        "random_orders": 2,
        "steps": 21,
        "elapsed_seconds": time.perf_counter() - benchmark_start,
        "device": str(device),
        "valid_patient_count": benchmark["valid_patient_count"],
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = independent_superpixel_faithfulness(
        model,
        model_kind,
        arrays["images"],
        baseline_images,
        arrays["labels"],
        selected_maps,
        selected_validity,
        arrays["patient_ids"],
        categories=arrays["categories"] if model_kind == "joint" else None,
        n_segments=128,
        random_orders=20,
        seed=INDEPENDENT_SEED,
        steps=21,
        batch_size=64,
        temperature=temperature,
        device=device,
    )
    torch.cuda.synchronize(device)
    result["elapsed_seconds"] = np.asarray(time.perf_counter() - start)
    result["peak_memory_bytes"] = np.asarray(torch.cuda.max_memory_allocated(device))
    result["temperature"] = np.asarray(temperature, dtype=np.float64)
    return result, benchmark_record


def _run_marker_interventions(
    model: torch.nn.Module,
    model_kind: str,
    patients: list[SyntheticPatient],
    selected_indices: np.ndarray,
    temperature: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    arrays = _patient_arrays([patients[int(index)] for index in selected_indices])
    return marker_counterfactual_sensitivity(
        model,
        model_kind,
        arrays["images"],
        arrays["backgrounds"],
        arrays["marker_masks"],
        arrays["image_cues"],
        arrays["labels"],
        arrays["patient_ids"],
        categories=arrays["categories"] if model_kind == "joint" else None,
        seed=INTERVENTION_SEED,
        batch_size=64,
        temperature=temperature,
        device=device,
    )


def _save_map_outputs(
    output_dir: Path,
    all_maps: dict[str, dict[str, MapBatch]],
    all_diagnostics: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    archive: dict[str, np.ndarray] = {}
    for model_kind, method_batches in all_maps.items():
        for name, batch in method_batches.items():
            archive.update(serializable_map_batch(batch, f"{model_kind}__{name}__"))
            for key, value in all_diagnostics[model_kind].get(name, {}).items():
                archive[f"{model_kind}__{name}__diagnostic_{key}"] = value
    atomic_save_npz(output_dir / "validation_maps.npz", archive)


def run_validation(root: Path = ROOT) -> Path:
    root = Path(root).resolve()
    run_directory = Path(
        time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
    )
    output_dir = root / OUTPUT_DIRECTORY / run_directory
    if output_dir.exists():
        raise FileExistsError(f"v2 output directory already exists: {output_dir}")
    relative_output_dir = OUTPUT_DIRECTORY / run_directory
    print("Checking protected v1 hashes and CUDA preflight.", flush=True)
    baseline_before = _verify_v1_baseline(root)
    device = torch.device("cuda:0")
    environment = _environment(device)
    source_config = json.loads(
        (root / V1_ROOT / "run_config.json").read_text(encoding="utf-8")
    )
    split_record = json.loads(
        (root / V1_ROOT / "synthetic_patient_splits.json").read_text(encoding="utf-8")
    )
    patient_ids = select_validation_patient_ids(split_record)
    labels = {
        str(patient_id): int(label)
        for patient_id, label in split_record["development_labels"].items()
    }
    test_ids = set(split_record["partition_patient_ids"]["test"])
    if set(patient_ids).intersection(test_ids):
        raise RuntimeError("validation patient selection touched test IDs")
    patients = generate_split(
        patient_ids, labels, "validation_model_selection", source_config
    )
    if len(patients) != 900 or any(
        patient.partition != "validation_model_selection" for patient in patients
    ):
        raise RuntimeError("the validation runner generated an unexpected partition")
    ids_by_class: dict[int, list[int]] = {class_id: [] for class_id in range(6)}
    for index, patient in enumerate(patients):
        ids_by_class[patient.label].append(index)
    if any(len(indices) != 150 for indices in ids_by_class.values()):
        raise RuntimeError("validation patients are not class-balanced")
    subset_rng = np.random.default_rng(V2_SEED)
    subset_indices = np.asarray(
        [
            int(index)
            for class_id in range(6)
            for index in subset_rng.permutation(ids_by_class[class_id])[
                :INDEPENDENT_PATIENTS_PER_CLASS
            ]
        ],
        dtype=np.int64,
    )
    temperatures_record = json.loads(
        (root / V1_ROOT / "calibration_temperatures.json").read_text(encoding="utf-8")
    )
    if (
        temperatures_record.get("fit_partition") != "validation_calibration"
        or temperatures_record.get("test_labels_used") is not False
    ):
        raise ValueError("frozen calibration metadata does not meet the v2 protocol")
    temperatures = temperatures_record["temperatures"]

    output_dir.mkdir(parents=True)
    print(
        f"Generated {len(patients)} validation_model_selection patients; no test rows loaded.",
        flush=True,
    )
    atomic_write_json(output_dir / "environment.json", environment)
    atomic_write_json(
        output_dir / "validation_selection.json",
        {
            "evaluated_partition": "validation_model_selection",
            "patient_count": len(patients),
            "independent_evaluation_patient_count": len(subset_indices),
            "independent_patients_per_class": INDEPENDENT_PATIENTS_PER_CLASS,
            "class_counts": {str(key): len(value) for key, value in ids_by_class.items()},
            "test_rows_or_labels_loaded": False,
            "test_ids_used_only_for_partition_disjointness_check": True,
            "validation_patient_ids": patient_ids,
            "independent_patient_ids": [
                patients[int(index)].patient_id for index in subset_indices
            ],
            "seed": V2_SEED,
            "prespecified_methods": list(MAP_METHODS),
        },
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    map_batches_by_model: dict[str, dict[str, MapBatch]] = {}
    diagnostics_by_model: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    localization_summaries = {}
    geometry_summary = {}
    faithfulness = {}
    marker_interventions = {}
    benchmark_records = {}
    timings = {}
    for model_kind in ("image_only", "joint"):
        print(f"Computing {model_kind} explanation maps on {device}.", flush=True)
        checkpoint_path = root / V1_ROOT / "checkpoints" / f"{model_kind}.pt"
        model = _load_model(model_kind, source_config, checkpoint_path, device)
        start = time.perf_counter()
        map_batches, diagnostics = _map_batches(model, model_kind, patients, device)
        torch.cuda.synchronize(device)
        timings[f"{model_kind}_map_computation_seconds"] = time.perf_counter() - start
        map_batches_by_model[model_kind] = map_batches
        diagnostics_by_model[model_kind] = diagnostics

        (output_dir / model_kind).mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        localization_summaries[model_kind], geometry_summary[model_kind] = (
            _localization_outputs(
                patients,
                map_batches,
                output_dir / model_kind,
            )
        )
        timings[f"{model_kind}_localization_and_geometry_seconds"] = (
            time.perf_counter() - start
        )

        print(
            f"Running small-batch benchmark and independent superpixel evaluation for {model_kind}.",
            flush=True,
        )
        result, benchmark = _run_independent_evaluation(
            model,
            model_kind,
            patients,
            map_batches,
            subset_indices,
            float(temperatures[model_kind]),
            device,
            output_dir,
        )
        faithfulness[model_kind] = result
        benchmark_records[model_kind] = benchmark
        atomic_save_npz(
            output_dir / f"faithfulness_{model_kind}.npz",
            flatten_npz_arrays(result, prefix=f"{model_kind}__"),
        )

        marker_interventions[model_kind] = _run_marker_interventions(
            model,
            model_kind,
            patients,
            subset_indices,
            float(temperatures[model_kind]),
            device,
        )
        atomic_save_npz(
            output_dir / f"marker_interventions_{model_kind}.npz",
            flatten_npz_arrays(
                marker_interventions[model_kind], prefix=f"{model_kind}__"
            ),
        )
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    _save_map_outputs(output_dir, map_batches_by_model, diagnostics_by_model)
    atomic_write_json(
        output_dir / "small_batch_benchmarks.json", benchmark_records
    )
    summary = {
        "status": "completed_validation_only",
        "condition": "balanced",
        "partition": "validation_model_selection",
        "device": str(device),
        "models_retrained": False,
        "test_partition_evaluated": False,
        "calibration_partition": "validation_calibration",
        "calibration_temperatures": temperatures,
        "patient_count": len(patients),
        "independent_evaluation_patient_count": len(subset_indices),
        "map_validity_and_localization": localization_summaries,
        "geometry_controls": {
            model_kind: {
                "patient_count": int(len(value["available"])),
                "available_control_count": int(value["available"].sum()),
                "unavailable_patient_count": int(
                    (~value["available"].any(axis=1)).sum()
                ),
                "repeats_per_patient": GEOMETRY_REPEATS,
                "valid_placement_count_range": [
                    int(value["valid_placement_counts"].min()),
                    int(value["valid_placement_counts"].max()),
                ],
                "seed": GEOMETRY_SEED,
            }
            for model_kind, value in geometry_summary.items()
        },
        "independent_superpixel_faithfulness": {
            model_kind: {
                "patient_count": int(result["patient_count"]),
                "valid_patient_count": result["valid_patient_count"],
                "auc_summary": summarize_faithfulness(
                    result, seed=INDEPENDENT_SEED
                ),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "peak_memory_bytes": int(result["peak_memory_bytes"]),
                "device": str(device),
                "baseline_names": result["baseline_names"].tolist(),
                "random_control_order_count": 20,
                "center_first_control": True,
                "intervention_family": "connected_slic_superpixels",
            }
            for model_kind, result in faithfulness.items()
        },
        "small_batch_benchmarks": benchmark_records,
        "marker_counterfactual_sensitivity": {
            model_kind: {
                "patient_count": int(len(value["patient_ids"])),
                "mean_cue_swap_probability_change": float(
                    value["cue_swap_probability_change"].mean()
                ),
                "mean_marker_removal_probability_change": float(
                    value["marker_removal_probability_change"].mean()
                ),
                "intervention_family": "renderer_glyph_counterfactual",
            }
            for model_kind, value in marker_interventions.items()
        },
        "timings_seconds": timings,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "v1_protection_file_count": len(baseline_before),
    }
    atomic_write_json(output_dir / "validation_summary.json", summary)
    _write_validation_report(output_dir / "validation_report.md", summary)
    baseline_after = _verify_v1_baseline(root)
    if baseline_before != baseline_after:
        raise RuntimeError("v1 preservation hashes changed during validation")
    atomic_write_json(
        output_dir / "v1_preservation_verification.json",
        {
            "baseline_manifest": str(BASELINE_MANIFEST),
            "file_count": len(baseline_after),
            "unchanged": True,
            "sha256": baseline_after,
        },
    )
    generated_files = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    source_files = sorted(
        path
        for path in (root / "experiments" / "phase_g_v2").rglob("*")
        if path.is_file() and path.suffix in {".py", ".md"}
    )
    input_files = {
        "v1_run_config": root / V1_ROOT / "run_config.json",
        "v1_patient_splits": root / V1_ROOT / "synthetic_patient_splits.json",
        "v1_calibration_temperatures": root / V1_ROOT / "calibration_temperatures.json",
        "v1_image_only_checkpoint": root / V1_ROOT / "checkpoints" / "image_only.pt",
        "v1_joint_checkpoint": root / V1_ROOT / "checkpoints" / "joint.pt",
    }
    manifest = {
        "status": "completed",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_directory": str(relative_output_dir),
        "condition": "balanced",
        "partition": "validation_model_selection",
        "test_partition_evaluated": False,
        "models_retrained": False,
        "device": str(device),
        "v1_preservation_manifest": str(BASELINE_MANIFEST),
        "v1_preservation_manifest_sha256": _sha256_file(
            root / BASELINE_MANIFEST
        ),
        "source_sha256": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in source_files
        },
        "input_sha256": {
            name: _sha256_file(path) for name, path in input_files.items()
        },
        "execution_specification": {
            "map_methods": list(MAP_METHODS),
            "geometry_repeats": GEOMETRY_REPEATS,
            "independent_patients_per_class": INDEPENDENT_PATIENTS_PER_CLASS,
            "superpixels_per_image": 128,
            "faithfulness_steps": 21,
            "random_control_orders": 20,
            "bootstrap_replicates": 2000,
            "seeds": {
                "validation_subset": V2_SEED,
                "geometry": GEOMETRY_SEED,
                "independent_evaluation": INDEPENDENT_SEED,
                "marker_interventions": INTERVENTION_SEED,
            },
        },
        "artifacts": {
            relative: _sha256_file(output_dir / Path(relative))
            for relative in generated_files
        },
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return output_dir


if __name__ == "__main__":
    print(run_validation())
