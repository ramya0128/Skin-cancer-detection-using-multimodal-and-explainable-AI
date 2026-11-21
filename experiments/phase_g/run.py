from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import PIL
import scipy
import skimage
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torchvision
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_g.evaluate import (
    bootstrap_mean_interval,
    build_localization_controls,
    deletion_insertion_metrics,
    holm_adjusted_p_values,
    localization_metrics,
    paired_nll_gains,
    paired_signflip_p,
    primary_nll_inference,
    stability_metrics,
)
from experiments.phase_g.explain import (
    compute_clinical_attributions,
    compute_image_maps,
    compute_joint_shapley,
    cue_mask_sensitivity,
)
from experiments.phase_g.generate import (
    CLASS_COUNT,
    SyntheticPatient,
    generate_split,
    make_patient_splits,
)
from experiments.phase_g.io import hash_protected_inputs, prepare_run_dir, validate_config
from experiments.phase_g.model import (
    MODEL_NAMES,
    build_model,
    fit_temperature,
    predict_logits,
    train_model,
)


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
SOURCE_FILES = (
    Path("experiments/phase_g/config.json"),
    Path("experiments/phase_g/generate.py"),
    Path("experiments/phase_g/io.py"),
    Path("experiments/phase_g/model.py"),
    Path("experiments/phase_g/evaluate.py"),
    Path("experiments/phase_g/explain.py"),
    Path("experiments/phase_g/run.py"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    if not arrays:
        raise ValueError("cannot save an empty array archive")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise TypeError("atomic NumPy archives cannot contain object arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def atomic_save_torch(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def assert_test_ready(
    checkpoint_paths: dict[str, Path],
    temperatures: dict[str, float],
) -> None:
    if set(checkpoint_paths) != set(MODEL_NAMES):
        raise RuntimeError("test evaluation requires all three frozen model checkpoints")
    if set(temperatures) != set(MODEL_NAMES):
        raise RuntimeError("test evaluation requires all three frozen calibrators")
    missing = [str(checkpoint_paths[name]) for name in MODEL_NAMES if not Path(checkpoint_paths[name]).is_file()]
    if missing:
        raise RuntimeError("test evaluation checkpoints are missing: " + ", ".join(missing))
    invalid = [
        name
        for name in MODEL_NAMES
        if not np.isfinite(float(temperatures[name])) or float(temperatures[name]) <= 0
    ]
    if invalid:
        raise RuntimeError("test evaluation has invalid frozen temperatures: " + ", ".join(invalid))


def _environment_record(device: torch.device) -> dict[str, Any]:
    driver_version = None
    gpu = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
        gpu_properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": gpu_properties.name,
            "total_memory_bytes": int(gpu_properties.total_memory),
            "compute_capability": [
                int(gpu_properties.major),
                int(gpu_properties.minor),
            ],
        }
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        driver_version = result.stdout.strip().splitlines()[0]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "scikit_image": skimage.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": gpu,
        "nvidia_driver": driver_version,
        "selected_device": str(device),
    }


def run_preflight(
    config: dict[str, Any],
    root: Path,
    device_override: torch.device | None = None,
    check_output_unused: bool = True,
) -> dict[str, Any]:
    validate_config(config)
    root = Path(root)
    output_dir = root / str(config["output_directory"])
    if check_output_unused and output_dir.exists():
        raise FileExistsError(f"preflight output path is already in use: {output_dir}")
    if device_override is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase G pilot requires the selected CUDA environment")
        device = torch.device("cuda:0")
    else:
        device = torch.device(device_override)
    environment = _environment_record(device)
    model = build_model("joint", config).to(device).train()
    batch_size = (
        2
        if device.type == "cpu"
        else int(config["model"]["training"]["batch_size"])
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["master_seed"]))
    images = torch.rand(
        (batch_size, 3, 224, 224), generator=generator, device=device
    )
    categories = torch.arange(batch_size, dtype=torch.long, device=device) % 6
    labels = categories.clone()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(images, categories)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError("synthetic model preflight produced non-finite values")
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
    result = {
        "environment": environment,
        "device": str(device),
        "batch_size": batch_size,
        "forward_backward_seconds": elapsed,
        "loss": float(loss.detach().cpu()),
        "peak_memory_bytes": peak_memory,
        "available_storage_bytes": int(shutil.disk_usage(root).free),
        "output_path": str(output_dir),
        "output_path_unused": not output_dir.exists(),
    }
    return result


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _source_hashes(root: Path) -> dict[str, str]:
    hashes = {
        str(relative): _sha256_file(root / relative)
        for relative in SOURCE_FILES
    }
    hashes.update(
        {
            "protected:" + relative: digest
            for relative, digest in hash_protected_inputs(root).items()
        }
    )
    return dict(sorted(hashes.items()))


def _log(output_dir: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()


def _write_final_manifest(
    output_dir: Path,
    metadata: dict[str, Any],
    completion_message: str,
) -> None:
    _log(output_dir, completion_message)
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".tmp"):
            artifacts[str(path.relative_to(output_dir))] = _sha256_file(path)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {**metadata, "artifact_sha256": artifacts},
    )


def _stage_path(output_dir: Path, name: str) -> Path:
    return output_dir / "stages" / f"{name}.json"


def _stage_done(output_dir: Path, name: str) -> bool:
    return _stage_path(output_dir, name).is_file()


def _verify_stage(output_dir: Path, name: str) -> bool:
    stage_path = _stage_path(output_dir, name)
    if not stage_path.is_file():
        return False
    try:
        record = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"stage record is unreadable: {stage_path}") from error
    if record.get("stage") != name or not isinstance(record.get("outputs_sha256"), dict):
        raise RuntimeError(f"stage record is malformed: {stage_path}")
    for relative, expected_hash in record["outputs_sha256"].items():
        artifact = output_dir / relative
        if not artifact.is_file() or _sha256_file(artifact) != expected_hash:
            raise RuntimeError(
                f"stage artifact changed after stage completion: {artifact}"
            )
    return True


def _mark_stage(output_dir: Path, name: str, outputs: list[Path]) -> None:
    hashes = {str(path.relative_to(output_dir)): _sha256_file(path) for path in outputs}
    atomic_write_json(
        _stage_path(output_dir, name),
        {
            "stage": name,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs_sha256": hashes,
        },
    )


def _validate_splits(
    splits: dict[str, list[str]],
    labels: dict[str, int],
    config: dict[str, Any],
) -> None:
    expected = {
        "train": int(config["dataset"]["class_counts"]["train_per_class"]) * CLASS_COUNT,
        "validation_model_selection": int(
            config["dataset"]["class_counts"]["validation_model_selection_per_class"]
        ) * CLASS_COUNT,
        "validation_calibration": int(
            config["dataset"]["class_counts"]["validation_calibration_per_class"]
        ) * CLASS_COUNT,
        "test": int(config["dataset"]["class_counts"]["test_per_class"]) * CLASS_COUNT,
    }
    all_ids: list[str] = []
    for partition, count in expected.items():
        ids = splits[partition]
        if len(ids) != count or len(set(ids)) != count:
            raise RuntimeError(f"{partition} split has invalid size or duplicate patient IDs")
        observed = np.bincount([labels[patient_id] for patient_id in ids], minlength=CLASS_COUNT)
        per_class = count // CLASS_COUNT
        if not np.array_equal(observed, np.full(CLASS_COUNT, per_class)):
            raise RuntimeError(f"{partition} split is not class balanced")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(labels):
        raise RuntimeError("generated patient partitions are not disjoint and exhaustive")


def _validate_training_cues(patients: Sequence[SyntheticPatient]) -> dict[str, float]:
    if not patients or any(patient.partition != "train" for patient in patients):
        raise ValueError("cue reliability must be checked on synthetic training patients only")
    labels = np.asarray([patient.label for patient in patients], dtype=np.int64)
    image = np.asarray([patient.image_cue for patient in patients], dtype=np.int64)
    clinical = np.asarray([patient.clinical_value for patient in patients], dtype=np.int64)
    expected = 0.85
    tolerance = 5.0 * np.sqrt(len(labels) * expected * (1.0 - expected))
    for name, cue in (("image", image), ("clinical", clinical)):
        correct = int(np.count_nonzero(cue == labels))
        if abs(correct - expected * len(labels)) > tolerance:
            raise RuntimeError(f"training {name} cue reliability exceeds the prespecified 5-SD check")
    return {
        "image_reliability": float(np.mean(image == labels)),
        "clinical_reliability": float(np.mean(clinical == labels)),
    }


def _select_stratified_donors(
    train_data: Sequence[SyntheticPatient], count: int, seed: int
) -> tuple[list[SyntheticPatient], np.ndarray]:
    if count < CLASS_COUNT:
        raise ValueError("Shapley donor count must include every class")
    by_class = {
        label: np.asarray(
            [index for index, patient in enumerate(train_data) if patient.label == label],
            dtype=np.int64,
        )
        for label in range(CLASS_COUNT)
    }
    if any(len(indices) == 0 for indices in by_class.values()):
        raise ValueError("training donors must include all six classes")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    base_count, remainder = divmod(count, CLASS_COUNT)
    for label in range(CLASS_COUNT):
        take = base_count + int(label < remainder)
        if len(by_class[label]) < take:
            raise ValueError(f"insufficient training donors for class {label}")
        selected.extend(
            rng.choice(by_class[label], size=take, replace=False).astype(int).tolist()
        )
    selected_array = np.asarray(selected, dtype=np.int64)
    return [train_data[index] for index in selected], selected_array


def _checkpoint_path(output_dir: Path, kind: str) -> Path:
    return output_dir / "checkpoints" / f"{kind}.pt"


def _load_checkpoint(path: Path, kind: str, config: dict[str, Any]) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != kind or "state_dict" not in payload:
        raise ValueError(f"checkpoint metadata does not match model kind {kind}: {path}")
    model = build_model(kind, config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _softmax_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _test_input_arrays(patients: Sequence[SyntheticPatient]) -> dict[str, np.ndarray]:
    return {
        "patient_ids": np.asarray([patient.patient_id for patient in patients], dtype="U25"),
        "labels": np.asarray([patient.label for patient in patients], dtype=np.int64),
        "image_cues": np.asarray([patient.image_cue for patient in patients], dtype=np.int64),
        "clinical_values": np.asarray(
            [patient.clinical_value for patient in patients], dtype=np.int64
        ),
        "images": np.stack([patient.image for patient in patients]),
        "background_canvases": np.stack([patient.background_canvas for patient in patients]),
        "lesion_masks": np.stack([patient.lesion_mask for patient in patients]),
        "marker_masks": np.stack([patient.marker_mask for patient in patients]),
        "marker_top_left": np.asarray(
            [patient.render_metadata["marker_top_left"] for patient in patients],
            dtype=np.int16,
        ),
        "replacement_image_cues": np.asarray(
            [patient.render_metadata["replacement_image_cues"] for patient in patients],
            dtype=np.int64,
        ),
        "replacement_clinical_cues": np.asarray(
            [patient.render_metadata["replacement_clinical_cues"] for patient in patients],
            dtype=np.int64,
        ),
    }


def _patients_from_archive(arrays: dict[str, np.ndarray]) -> list[SyntheticPatient]:
    n = len(arrays["patient_ids"])
    expected = {
        "labels": (n,),
        "image_cues": (n,),
        "clinical_values": (n,),
        "images": (n, 224, 224, 3),
        "background_canvases": (n, 224, 224, 3),
        "lesion_masks": (n, 224, 224),
        "marker_masks": (n, 224, 224),
        "marker_top_left": (n, 2),
        "replacement_image_cues": (n, 20),
        "replacement_clinical_cues": (n, 20),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"saved test input {name} has unexpected shape {arrays[name].shape}")
    return [
        SyntheticPatient(
            patient_id=str(arrays["patient_ids"][row]),
            label=int(arrays["labels"][row]),
            partition="test",
            image=arrays["images"][row],
            image_cue=int(arrays["image_cues"][row]),
            clinical_value=int(arrays["clinical_values"][row]),
            background_canvas=arrays["background_canvases"][row],
            lesion_mask=arrays["lesion_masks"][row].astype(bool, copy=False),
            marker_mask=arrays["marker_masks"][row].astype(bool, copy=False),
            render_metadata={
                "marker_top_left": arrays["marker_top_left"][row].astype(int).tolist(),
                "replacement_image_cues": arrays["replacement_image_cues"][row].astype(int).tolist(),
                "replacement_clinical_cues": arrays["replacement_clinical_cues"][row].astype(int).tolist(),
            },
        )
        for row in range(n)
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _patient_bootstrap_record(
    values: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    lower, upper = bootstrap_mean_interval(
        values,
        replicates=int(config["descriptive_bootstrap_replicates"]),
        seed=int(config["descriptive_bootstrap_seed"]),
    )
    return {
        "mean": float(values.mean()),
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "patient_count": int(values.size),
    }


def _paired_comparison(
    values: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    summary = _patient_bootstrap_record(values, config)
    summary["p_value"] = paired_signflip_p(
        np.asarray(values, dtype=np.float64),
        np.zeros(len(values), dtype=np.float64),
        draws=int(config["paired_signflip_draws"]),
        seed=int(config["paired_signflip_seed"]),
    )
    return summary


def _train_models(
    output_dir: Path,
    config: dict[str, Any],
    train_data: list[SyntheticPatient],
    selection_data: list[SyntheticPatient],
) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    stage_outputs = []
    for kind in MODEL_NAMES:
        checkpoint = _checkpoint_path(output_dir, kind)
        history_path = output_dir / "training" / f"{kind}_history.json"
        if checkpoint.is_file() and history_path.is_file():
            model = _load_checkpoint(checkpoint, kind, config)
            _log(output_dir, f"Loaded frozen {kind} checkpoint for resume")
        else:
            _log(output_dir, f"Training {kind} on synthetic training patients")
            result = train_model(kind, train_data, selection_data, config)
            model = result.model.eval()
            atomic_write_json(
                history_path,
                {
                    "kind": kind,
                    "seed": int(config["model"]["training"]["model_initialization_seeds"][kind]),
                    "selected_epoch": result.selected_epoch,
                    "history": result.history,
                },
            )
        models[kind] = model.eval()
        stage_outputs.extend((checkpoint, history_path))
    _mark_stage(output_dir, "models_trained", stage_outputs)
    return models


def _calibrate_models(
    output_dir: Path,
    models: dict[str, nn.Module],
    calibration_data: list[SyntheticPatient],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
    logits_path = output_dir / "calibration_predictions.npz"
    temperatures_path = output_dir / "calibration_temperatures.json"
    labels = np.asarray([patient.label for patient in calibration_data], dtype=np.int64)
    if logits_path.is_file() and temperatures_path.is_file():
        arrays = _load_npz(logits_path)
        if not np.array_equal(arrays["labels"], labels):
            raise ValueError("saved calibration labels differ from the regenerated partition")
        calibration_logits = {
            name: arrays[f"logits_{name}"] for name in MODEL_NAMES
        }
        temperatures = json.loads(temperatures_path.read_text(encoding="utf-8"))[
            "temperatures"
        ]
    else:
        batch_size = int(config["model"]["training"]["batch_size"])
        calibration_logits = {
            name: predict_logits(
                models[name],
                calibration_data,
                name,
                batch_size=batch_size,
                device=torch.device("cuda:0"),
            )
            for name in MODEL_NAMES
        }
        temperatures = {
            name: fit_temperature(calibration_logits[name], labels)
            for name in MODEL_NAMES
        }
        atomic_save_npz(
            logits_path,
            {
                "labels": labels,
                **{f"logits_{name}": calibration_logits[name] for name in MODEL_NAMES},
            },
        )
        atomic_write_json(
            temperatures_path,
            {
                "fit_partition": "validation_calibration",
                "test_labels_used": False,
                "temperatures": temperatures,
            },
        )
    temperatures = {name: float(temperatures[name]) for name in MODEL_NAMES}
    outputs = [logits_path, temperatures_path]
    _mark_stage(output_dir, "calibrators_frozen", outputs)
    return temperatures, calibration_logits, labels


def _load_or_create_test_data(
    output_dir: Path,
    test_ids: list[str],
    labels: dict[str, int],
    config: dict[str, Any],
) -> list[SyntheticPatient]:
    archive_path = output_dir / "test_inputs.npz"
    if archive_path.is_file():
        test_data = _patients_from_archive(_load_npz(archive_path))
        if [patient.patient_id for patient in test_data] != test_ids:
            raise ValueError("saved synthetic test IDs do not match the frozen split")
        return test_data
    test_data = generate_split(test_ids, labels, "test", config)
    arrays = _test_input_arrays(test_data)
    atomic_save_npz(archive_path, arrays)
    _mark_stage(output_dir, "test_inputs_generated", [archive_path])
    return test_data


def _test_predictions(
    output_dir: Path,
    models: dict[str, nn.Module],
    test_data: list[SyntheticPatient],
    temperatures: dict[str, float],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    prediction_path = output_dir / "test_predictions.npz"
    patient_ids = np.asarray([patient.patient_id for patient in test_data], dtype="U25")
    labels = np.asarray([patient.label for patient in test_data], dtype=np.int64)
    if prediction_path.is_file():
        arrays = _load_npz(prediction_path)
        if not np.array_equal(arrays["patient_ids"], patient_ids):
            raise ValueError("saved test prediction IDs differ from the frozen synthetic split")
        if not np.array_equal(arrays["labels"], labels):
            raise ValueError("saved test labels differ from regenerated synthetic patients")
        return arrays
    logits = {
        name: predict_logits(
            models[name],
            test_data,
            name,
            batch_size=int(config["model"]["training"]["batch_size"]),
            device=torch.device("cuda:0"),
        )
        for name in MODEL_NAMES
    }
    probabilities = {
        name: _softmax_logits(logits[name], temperatures[name])
        for name in MODEL_NAMES
    }
    arrays = {
        "patient_ids": patient_ids,
        "labels": labels,
        **{f"logits_{name}": logits[name] for name in MODEL_NAMES},
        **{f"probabilities_{name}": probabilities[name] for name in MODEL_NAMES},
    }
    atomic_save_npz(prediction_path, arrays)
    _mark_stage(output_dir, "test_predictions_frozen", [prediction_path])
    return arrays


def _primary_results(
    output_dir: Path,
    predictions: dict[str, np.ndarray],
    calibration_logits: dict[str, np.ndarray],
    calibration_labels: np.ndarray,
    temperatures: dict[str, float],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary_path = output_dir / "primary_statistics.json"
    gains_path = output_dir / "primary_patient_metrics.npz"
    if summary_path.is_file() and gains_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary, _load_npz(gains_path)
    labels = predictions["labels"]
    probabilities = {
        name: predictions[f"probabilities_{name}"] for name in MODEL_NAMES
    }
    test_logits = {name: predictions[f"logits_{name}"] for name in MODEL_NAMES}
    inference = primary_nll_inference(
        probabilities=probabilities,
        test_logits=test_logits,
        test_labels=labels,
        calibration_logits=calibration_logits,
        calibration_labels=calibration_labels,
        test_classes=labels.copy(),
        replicates=int(config["primary_incremental_prediction_criterion"]["bootstrap_replicates"]),
        seed=int(config["primary_incremental_prediction_criterion"]["bootstrap_seed"]),
        threshold=float(config["primary_incremental_prediction_criterion"]["nll_effect_threshold"]),
    )
    accuracy = {
        name: float(np.mean(probabilities[name].argmax(axis=1) == labels))
        for name in MODEL_NAMES
    }
    gains = paired_nll_gains(probabilities, labels)
    summary = {
        "n_test_patients": int(len(labels)),
        "accuracy": accuracy,
        "joint_accuracy_floor": float(
            config["primary_incremental_prediction_criterion"]["full_accuracy_floor"]
        ),
        "joint_accuracy_passes": bool(
            accuracy["joint"]
            >= config["primary_incremental_prediction_criterion"]["full_accuracy_floor"]
        ),
        "nll_effects": inference["estimates"],
        "holm_adjusted_lower_bounds": inference["lower_bounds"],
        "raw_p_values_at_0_10": inference["raw_p_values"],
        "holm_adjusted_p_values_at_0_10": inference["holm_adjusted_p_values"],
        "nll_effect_threshold": inference["threshold"],
        "calibrated_nll": inference["calibrated_nll"],
        "uncalibrated_nll": inference["uncalibrated_nll"],
        "passes_primary_gate": inference["passes"],
        "calibration_temperatures": temperatures,
        "bootstrap_replicates": int(
            config["primary_incremental_prediction_criterion"]["bootstrap_replicates"]
        ),
        "bootstrap_seed": int(
            config["primary_incremental_prediction_criterion"]["bootstrap_seed"]
        ),
    }
    bootstrap_arrays = {
        f"bootstrap_{name}": inference["bootstrap_estimates"][name]
        for name in inference["bootstrap_estimates"]
    }
    atomic_write_json(summary_path, summary)
    atomic_save_npz(
        gains_path,
        {
            "labels": labels,
            "nll_gain_image_only_minus_joint": gains["image_only_minus_joint"],
            "nll_gain_clinical_only_minus_joint": gains["clinical_only_minus_joint"],
            **bootstrap_arrays,
        },
    )
    _mark_stage(output_dir, "primary_evaluation", [summary_path, gains_path])
    return summary, _load_npz(gains_path)


def _compute_explanation_outputs(
    output_dir: Path,
    test_data: list[SyntheticPatient],
    train_data: list[SyntheticPatient],
    models: dict[str, nn.Module],
    temperatures: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    patient_ids = np.asarray([patient.patient_id for patient in test_data], dtype="U25")
    labels = np.asarray([patient.label for patient in test_data], dtype=np.int64)
    categories = np.asarray([patient.clinical_value for patient in test_data], dtype=np.int64)
    images = torch.from_numpy(np.stack([patient.image.transpose(2, 0, 1) for patient in test_data]))
    backgrounds = torch.from_numpy(
        np.stack([patient.background_canvas.transpose(2, 0, 1) for patient in test_data])
    )
    masks = {
        "lesion": np.stack([patient.lesion_mask for patient in test_data]),
        "marker": np.stack([patient.marker_mask for patient in test_data]),
    }
    evaluation = config["explanation_evaluation"]
    methods = tuple(evaluation["image_methods"])
    maps_path = output_dir / "image_maps.npz"
    if maps_path.is_file():
        map_arrays = _load_npz(maps_path)
        maps_by_model = {
            kind: {
                method: map_arrays[f"{kind}_{method}"]
                for method in methods
            }
            for kind in ("image_only", "joint")
        }
    else:
        maps_by_model: dict[str, dict[str, np.ndarray]] = {}
        for kind in ("image_only", "joint"):
            categories_for_model = categories if kind == "joint" else None
            models[kind].to(torch.device("cuda:0")).eval()
            maps_by_model[kind] = compute_image_maps(
                models[kind],
                kind,
                images,
                labels,
                categories_for_model,
                methods,
                batch_size=int(evaluation["image_batch_size"]),
                temperature=temperatures[kind],
            )
            models[kind].to("cpu")
        atomic_save_npz(
            maps_path,
            {
                "patient_ids": patient_ids,
                "target_classes": labels,
                **{
                    f"{kind}_{method}": maps_by_model[kind][method]
                    for kind in maps_by_model
                    for method in methods
                },
            },
        )
        _mark_stage(output_dir, "image_maps", [maps_path])

    clinical_path = output_dir / "clinical_attributions.npz"
    if clinical_path.is_file():
        clinical_arrays = _load_npz(clinical_path)
    else:
        uniform_reference = np.full(CLASS_COUNT, 1.0 / CLASS_COUNT, dtype=np.float64)
        models["clinical_only"].to(torch.device("cuda:0")).eval()
        clinical = compute_clinical_attributions(
            models["clinical_only"],
            categories,
            labels,
            uniform_reference,
            temperature=temperatures["clinical_only"],
        )
        models["clinical_only"].to("cpu")
        clinical_arrays = {"patient_ids": patient_ids, **clinical}
        atomic_save_npz(clinical_path, clinical_arrays)
        _mark_stage(output_dir, "clinical_attributions", [clinical_path])

    donor_count = int(config["shapley"]["background_patients"])
    donors, donor_indices = _select_stratified_donors(
        train_data, donor_count, int(config["shapley"]["seed"])
    )
    shapley_path = output_dir / "fusion_shapley.npz"
    if shapley_path.is_file():
        shapley_arrays = _load_npz(shapley_path)
    else:
        models["joint"].to(torch.device("cuda:0")).eval()
        modality = compute_joint_shapley(
            models["joint"],
            test_data,
            labels,
            donors,
            "modality",
            temperature=temperatures["joint"],
        )
        cue = compute_joint_shapley(
            models["joint"],
            test_data,
            labels,
            donors,
            "cue",
            temperature=temperatures["joint"],
        )
        models["joint"].to("cpu")
        tolerance = float(config["shapley"]["efficiency_tolerance"])
        if (
            np.max(np.abs(modality["efficiency_residual"])) > tolerance
            or np.max(np.abs(cue["efficiency_residual"])) > tolerance
        ):
            raise RuntimeError("joint Shapley values failed the configured efficiency tolerance")
        shapley_arrays = {
            "patient_ids": patient_ids,
            "targets": labels,
            "donor_indices_in_training_partition": donor_indices,
            "donor_patient_ids": np.asarray(
                [patient.patient_id for patient in donors], dtype="U25"
            ),
            **{
                f"modality_{name}": value
                for name, value in modality.items()
                if isinstance(value, np.ndarray)
            },
            **{
                f"cue_{name}": value
                for name, value in cue.items()
                if isinstance(value, np.ndarray)
            },
        }
        atomic_save_npz(shapley_path, shapley_arrays)
        _mark_stage(output_dir, "fusion_shapley", [shapley_path])

    cue_path = output_dir / "cue_intervention_sensitivity.npz"
    if cue_path.is_file():
        cue_arrays = _load_npz(cue_path)
    else:
        models["joint"].to(torch.device("cuda:0")).eval()
        cue_results = cue_mask_sensitivity(
            models["joint"],
            test_data,
            {
                "draws_per_test_patient": int(config["cue_replacements"]["draws_per_test_patient"]),
                "temperature": temperatures["joint"],
            },
        )
        models["joint"].to("cpu")
        cue_arrays = {"patient_ids": patient_ids, "labels": labels, **cue_results}
        atomic_save_npz(cue_path, cue_arrays)
        _mark_stage(output_dir, "cue_intervention_sensitivity", [cue_path])

    localization_controls_path = output_dir / "localization_controls.npz"
    if localization_controls_path.is_file():
        localization_controls = _load_npz(localization_controls_path)
    else:
        localization_config = evaluation["localization"]
        localization_controls = build_localization_controls(
            masks,
            seed=int(localization_config["seed"]),
            repeats=int(localization_config["area_matched_repeats"]),
        )
        atomic_save_npz(localization_controls_path, localization_controls)
        _mark_stage(output_dir, "localization_controls", [localization_controls_path])
    localization_results = {}
    for kind in ("image_only", "joint"):
        localization_path = output_dir / f"localization_{kind}.npz"
        if localization_path.is_file():
            localization_results[kind] = _load_npz(localization_path)
            continue
        arrays: dict[str, np.ndarray] = {"patient_ids": patient_ids}
        for method in methods:
            method_metrics = localization_metrics(
                maps_by_model[kind][method],
                masks,
                localization_controls,
                patient_ids,
            )
            arrays.update({f"{method}_{name}": value for name, value in method_metrics.items()})
        atomic_save_npz(localization_path, arrays)
        _mark_stage(output_dir, f"localization_{kind}", [localization_path])
        localization_results[kind] = arrays

    faithfulness_results = {}
    stability_results = {}
    for kind in ("image_only", "joint"):
        faithfulness_path = output_dir / f"faithfulness_{kind}.npz"
        stability_path = output_dir / f"stability_{kind}.npz"
        if faithfulness_path.is_file():
            faithfulness_results[kind] = _load_npz(faithfulness_path)
        else:
            _log(output_dir, f"Starting batched CUDA faithfulness for {kind}")
            models[kind].to(torch.device("cuda:0")).eval()
            faithfulness_config = {
                **evaluation["faithfulness"],
                "temperature": temperatures[kind],
            }
            faithfulness_results[kind] = deletion_insertion_metrics(
                models[kind],
                kind,
                images,
                categories if kind == "joint" else None,
                backgrounds,
                maps_by_model[kind],
                labels,
                faithfulness_config,
            )
            models[kind].to("cpu")
            faithfulness_results[kind]["patient_ids"] = patient_ids
            atomic_save_npz(faithfulness_path, faithfulness_results[kind])
            _mark_stage(output_dir, f"faithfulness_{kind}", [faithfulness_path])
            _log(
                output_dir,
                f"Completed faithfulness for {kind} on "
                f"{faithfulness_results[kind]['execution_device'].item()}",
            )
        if stability_path.is_file():
            stability_results[kind] = _load_npz(stability_path)
        else:
            _log(output_dir, f"Starting batched stability and occlusion maps for {kind}")
            models[kind].to(torch.device("cuda:0")).eval()
            stability_config = {
                **evaluation["stability"],
                "temperature": temperatures[kind],
            }
            stability_results[kind] = stability_metrics(
                models[kind],
                kind,
                images,
                categories if kind == "joint" else None,
                maps_by_model[kind],
                labels,
                stability_config,
            )
            models[kind].to("cpu")
            stability_results[kind]["patient_ids"] = patient_ids
            atomic_save_npz(stability_path, stability_results[kind])
            _mark_stage(output_dir, f"stability_{kind}", [stability_path])
            _log(output_dir, f"Completed stability for {kind}")

    return {
        "maps_by_model": maps_by_model,
        "clinical": clinical_arrays,
        "shapley": shapley_arrays,
        "cue_sensitivity": cue_arrays,
        "localization": localization_results,
        "faithfulness": faithfulness_results,
        "stability": stability_results,
    }


def _secondary_statistics(
    output_dir: Path,
    primary: dict[str, Any],
    predictions: dict[str, np.ndarray],
    explanation_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["explanation_evaluation"]["statistics"]
    cue = explanation_results["cue_sensitivity"]
    cue_family = {}
    cue_names = ("image_replacement", "clinical_replacement")
    labels = np.asarray(predictions["labels"], dtype=np.int64)
    per_class_cue = {}
    for name in cue_names:
        drop = -np.asarray(cue[f"{name}_accuracy_change"], dtype=np.float64)
        cue_family[name] = _paired_comparison(drop, settings)
        cue_family[name]["accuracy_drop_ge_0_05"] = bool(cue_family[name]["mean"] >= 0.05)
        per_class_cue[name] = {}
        for label in range(CLASS_COUNT):
            selected = labels == label
            per_class_cue[name][str(label)] = {
                "patient_count": int(selected.sum()),
                "accuracy_drop": float(drop[selected].mean()),
                "target_probability_change": float(
                    np.mean(cue[f"{name}_target_probability_change"][selected])
                ),
                "nll_change": float(np.mean(cue[f"{name}_nll_change"][selected])),
            }
    cue_family_p = holm_adjusted_p_values(
        np.asarray([cue_family[name]["p_value"] for name in cue_names])
    )
    for index, name in enumerate(cue_names):
        cue_family[name]["holm_adjusted_p_value"] = float(cue_family_p[index])

    localization_families = {}
    map_methods = ("gradcam", "chefer", "attention_rollout", "occlusion")
    for kind in ("image_only", "joint"):
        metrics = explanation_results["localization"][kind]
        for region in ("marker", "lesion"):
            family_key = f"{kind}_{region}"
            comparisons = {}
            for method in map_methods:
                values = np.asarray(metrics[f"{method}_{region}_mass"], dtype=np.float64)
                controls = np.asarray(metrics[f"{method}_{region}_random_mass"], dtype=np.float64)
                differences = values - controls.mean(axis=1)
                comparisons[method] = _paired_comparison(differences, settings)
            adjusted = holm_adjusted_p_values(
                np.asarray([comparisons[method]["p_value"] for method in map_methods])
            )
            for index, method in enumerate(map_methods):
                comparisons[method]["holm_adjusted_p_value"] = float(adjusted[index])
            localization_families[family_key] = comparisons

    faithfulness_families = {}
    target_methods = ("gradcam", "chefer", "occlusion")
    for kind in ("image_only", "joint"):
        metrics = explanation_results["faithfulness"][kind]
        method_names = [str(value) for value in metrics["method_names"]]
        baseline_index = list(metrics["baseline_names"]).index("background_canvas")
        random_auc = metrics["random_order_auc"][:, :, baseline_index, :]
        center_auc = metrics["center_gaussian_auc"][:, baseline_index, :]
        method_auc = metrics["auc"][:, :, baseline_index, :]
        random_mean = random_auc.mean(axis=1)
        for direction_index, direction in enumerate(("deletion", "insertion")):
            family_key = f"{kind}_{direction}"
            comparisons = {}
            for method in target_methods:
                method_index = method_names.index(method)
                if direction == "deletion":
                    against_random = random_mean[:, 0] - method_auc[:, method_index, 0]
                    against_center = center_auc[:, 0] - method_auc[:, method_index, 0]
                else:
                    against_random = method_auc[:, method_index, 1] - random_mean[:, 1]
                    against_center = method_auc[:, method_index, 1] - center_auc[:, 1]
                comparisons[f"{method}_vs_random"] = _paired_comparison(
                    against_random, settings
                )
                comparisons[f"{method}_vs_center"] = _paired_comparison(
                    against_center, settings
                )
            comparison_names = list(comparisons)
            adjusted = holm_adjusted_p_values(
                np.asarray([comparisons[name]["p_value"] for name in comparison_names])
            )
            for index, name in enumerate(comparison_names):
                comparisons[name]["holm_adjusted_p_value"] = float(adjusted[index])
            faithfulness_families[family_key] = comparisons

    stability_summary = {}
    for kind in ("image_only", "joint"):
        values = explanation_results["stability"][kind]
        stability_summary[kind] = {}
        for transform_index, transform in enumerate(values["transform_names"]):
            name = str(transform)
            stability_summary[kind][name] = {
                "prediction_change_rate": _patient_bootstrap_record(
                    values["prediction_changed"][transform_index].astype(np.float64),
                    settings,
                ),
                "maps": {},
            }
            for map_index, method in enumerate(values["map_names"]):
                valid = values["valid_subset"][transform_index]
                spearman_values = values["spearman"][transform_index, valid, map_index]
                finite_spearman = spearman_values[np.isfinite(spearman_values)]
                ssim_values = values["ssim"][transform_index, valid, map_index]
                finite_ssim = ssim_values[np.isfinite(ssim_values)]
                stability_summary[kind][name]["maps"][str(method)] = {
                    "valid_patient_count": int(valid.sum()),
                    "spearman_valid_count": int(finite_spearman.size),
                    "spearman": _patient_bootstrap_record(
                        finite_spearman.astype(np.float64),
                        settings,
                    )
                    if finite_spearman.size
                    else None,
                    "ssim_valid_count": int(finite_ssim.size),
                    "ssim": _patient_bootstrap_record(
                        finite_ssim.astype(np.float64),
                        settings,
                    )
                    if finite_ssim.size
                    else None,
                }
    summary = {
        "cue_mask_sensitivity": cue_family,
        "cue_mask_per_class": per_class_cue,
        "localization": localization_families,
        "faithfulness": faithfulness_families,
        "stability": stability_summary,
        "family_adjustments": {
            "cue_mask": "Holm within two intervention comparisons",
            "localization": "Holm separately by model target and region",
            "faithfulness": "Holm separately by model target and direction",
            "stability": "descriptive only; no hypothesis tests",
        },
        "patient_count": int(len(predictions["labels"])),
        "primary_gate_passed": bool(primary["passes_primary_gate"]),
    }
    path = output_dir / "secondary_statistics.json"
    atomic_write_json(path, summary)
    _mark_stage(output_dir, "secondary_statistics", [path])
    return summary


def _explanation_resume_readiness(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    required = (
        "run_config.json",
        "calibration_temperatures.json",
        "checkpoints_and_calibrators_frozen.json",
        "synthetic_patient_splits.json",
        "test_inputs.npz",
        "test_predictions.npz",
        "primary_statistics.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "explanation resume requires frozen artifacts: " + ", ".join(missing)
        )
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    validate_config(config)
    frozen = json.loads(
        (output_dir / "checkpoints_and_calibrators_frozen.json").read_text(encoding="utf-8")
    )
    if frozen.get("frozen_before_test_generation") is not True:
        raise RuntimeError("calibrators and checkpoints were not frozen before test generation")
    if frozen.get("test_labels_used") is not False:
        raise RuntimeError("test labels must not have been used before freezing")
    temperatures = json.loads(
        (output_dir / "calibration_temperatures.json").read_text(encoding="utf-8")
    )["temperatures"]
    assert_test_ready(
        {name: output_dir / "checkpoints" / f"{name}.pt" for name in MODEL_NAMES},
        temperatures,
    )
    primary = json.loads((output_dir / "primary_statistics.json").read_text(encoding="utf-8"))
    if primary.get("n_test_patients") != 1800:
        raise RuntimeError("explanation resume requires the completed 1,800-patient primary evaluation")
    return {
        "config": config,
        "temperatures": {name: float(temperatures[name]) for name in MODEL_NAMES},
        "primary": primary,
    }


def _run_explanation_resume(root: Path) -> None:
    root = Path(root).resolve()
    config = _load_config()
    output_dir = root / str(config["output_directory"])
    if not output_dir.is_dir():
        raise FileNotFoundError("cannot resume explanation stages without the pilot output directory")
    readiness = _explanation_resume_readiness(output_dir)
    runtime_config = copy.deepcopy(readiness["config"])
    runtime_config["output_directory"] = str(output_dir)

    split_record = json.loads(
        (output_dir / "synthetic_patient_splits.json").read_text(encoding="utf-8")
    )
    test_ids = split_record["partition_patient_ids"]["test"]
    test_arrays = _load_npz(output_dir / "test_inputs.npz")
    test_data = _patients_from_archive(test_arrays)
    if [patient.patient_id for patient in test_data] != test_ids:
        raise RuntimeError("saved test inputs do not match the frozen synthetic test split")

    train_ids = split_record["partition_patient_ids"]["train"]
    labels_by_id = {
        patient_id: int(label)
        for patient_id, label in split_record["development_labels"].items()
    }
    train_data = generate_split(train_ids, labels_by_id, "train", runtime_config)

    models = {
        name: _load_checkpoint(output_dir / "checkpoints" / f"{name}.pt", name, runtime_config)
        for name in MODEL_NAMES
    }
    temperatures = readiness["temperatures"]
    predictions = _load_npz(output_dir / "test_predictions.npz")
    explanation_results = _compute_explanation_outputs(
        output_dir,
        test_data,
        train_data,
        models,
        temperatures,
        runtime_config,
    )
    secondary = _secondary_statistics(
        output_dir,
        readiness["primary"],
        predictions,
        explanation_results,
        runtime_config,
    )
    report_path = output_dir / "pilot_report.json"
    atomic_write_json(
        report_path,
        {
            "experiment_id": runtime_config["experiment_id"],
            "status": "executed_balanced_pilot",
            "synthetic_only": True,
            "primary": readiness["primary"],
            "secondary": secondary,
            "interpretation_note": (
                "Cue masking is intervention sensitivity, localization is attribution distribution, "
                "faithfulness is ranked model-response change, and stability is perturbation similarity."
            ),
            "not_a_phase_g_completion": True,
        },
    )
    figure_paths = _save_figures(output_dir, explanation_results)
    after_hashes = hash_protected_inputs(root)
    before = json.loads(
        (output_dir / "protected_inputs_before.json").read_text(encoding="utf-8")
    )["sha256"]
    before = {name.removeprefix("protected:"): digest for name, digest in before.items()}
    if after_hashes != before:
        raise RuntimeError("protected frozen inputs changed during explanation recovery")
    atomic_write_json(
        output_dir / "protected_inputs_after.json",
        {"sha256": after_hashes, "unchanged": True},
    )
    _write_final_manifest(
        output_dir,
        {
            "experiment_id": runtime_config["experiment_id"],
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256_file(CONFIG_PATH),
            "source_and_protected_sha256": _source_hashes(root),
            "protected_inputs_unchanged": True,
            "test_patient_count": int(len(predictions["labels"])),
            "condition": runtime_config["pilot_condition"],
            "seed": int(runtime_config["master_seed"]),
            "resume_type": "explanation_only_after_primary_freeze",
        },
        f"Explanation recovery completed; report: {report_path}",
    )


def _save_figures(
    output_dir: Path,
    explanation_results: dict[str, Any],
) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    faithfulness_path = figures_dir / "faithfulness_auc.png"
    localization_path = figures_dir / "localization_mass.png"
    temporary_faithfulness = faithfulness_path.with_name(faithfulness_path.name + ".tmp")
    temporary_localization = localization_path.with_name(localization_path.name + ".tmp")
    method_colors = {
        "gradcam": "#1f77b4",
        "chefer": "#ff7f0e",
        "occlusion": "#2ca02c",
        "attention_rollout": "#9467bd",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, kind in enumerate(("image_only", "joint")):
        metrics = explanation_results["faithfulness"][kind]
        methods = [str(value) for value in metrics["method_names"]]
        baseline_index = list(metrics["baseline_names"]).index("background_canvas")
        for column, direction_index in enumerate((0, 1)):
            axis = axes[row, column]
            for method_index, method in enumerate(methods):
                mean_curve = metrics["curves"][
                    :, method_index, baseline_index, direction_index, :
                ].mean(axis=0)
                axis.plot(
                    metrics["patch_counts"] / 196.0,
                    mean_curve,
                    label=method,
                    color=method_colors.get(method),
                )
            axis.set_title(f"{kind}: {'deletion' if direction_index == 0 else 'insertion'}")
            axis.set_xlabel("Fraction of patches")
            axis.set_ylabel("Calibrated true-class probability")
            axis.set_ylim(0, 1)
            axis.legend()
    fig.savefig(temporary_faithfulness, format="png", dpi=160)
    plt.close(fig)
    os.replace(temporary_faithfulness, faithfulness_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    methods = ("gradcam", "chefer", "attention_rollout", "occlusion")
    x = np.arange(len(methods))
    width = 0.35
    for axis, kind in zip(axes, ("image_only", "joint"), strict=True):
        metrics = explanation_results["localization"][kind]
        marker = [float(np.mean(metrics[f"{method}_marker_mass"])) for method in methods]
        lesion = [float(np.mean(metrics[f"{method}_lesion_mass"])) for method in methods]
        axis.bar(x - width / 2, marker, width, label="Exterior cue")
        axis.bar(x + width / 2, lesion, width, label="Lesion")
        axis.set_xticks(x, methods, rotation=20)
        axis.set_title(f"{kind} localization mass")
        axis.set_ylabel("Mean normalized attribution mass")
        axis.legend()
    fig.savefig(temporary_localization, format="png", dpi=160)
    plt.close(fig)
    os.replace(temporary_localization, localization_path)
    return [faithfulness_path, localization_path]


def _run_pilot(root: Path, config: dict[str, Any], resume: bool) -> None:
    root = Path(root).resolve()
    validate_config(config)
    output_dir = root / str(config["output_directory"])
    preflight = run_preflight(
        config,
        root,
        check_output_unused=not resume,
    )
    source_hashes = _source_hashes(root)
    config_hash = _sha256_file(CONFIG_PATH)
    prepare_run_dir(output_dir, config_hash, source_hashes, resume=resume)
    if resume:
        stages_dir = output_dir / "stages"
        if stages_dir.exists():
            for stage_path in stages_dir.glob("*.json"):
                _verify_stage(output_dir, stage_path.stem)
    if (output_dir / "run_manifest.json").is_file():
        raise FileExistsError(
            "this run already has a final manifest; use its results instead of rerunning held-out evaluation"
        )

    runtime_config = copy.deepcopy(config)
    runtime_config["output_directory"] = str(output_dir)
    if not resume:
        atomic_write_json(output_dir / "run_config.json", config)
        atomic_write_json(output_dir / "environment.json", preflight)
        atomic_write_json(
            output_dir / "protected_inputs_before.json",
            {"sha256": {key: value for key, value in source_hashes.items() if key.startswith("protected:")}},
        )
    _log(output_dir, f"Started {config['experiment_id']} on {preflight['device']}")
    _log(
        output_dir,
        f"Deterministic batch preflight: {preflight['batch_size']} samples, "
        f"{preflight['forward_backward_seconds']:.3f}s, "
        f"{preflight['peak_memory_bytes']} peak bytes",
    )

    splits, labels_by_id = make_patient_splits(config)
    _validate_splits(splits, labels_by_id, config)
    split_path = output_dir / "synthetic_patient_splits.json"
    split_record = {
        "master_seed": int(config["master_seed"]),
        "partition_patient_ids": splits,
        "development_labels": {
            patient_id: int(labels_by_id[patient_id])
            for partition in (
                "train",
                "validation_model_selection",
                "validation_calibration",
            )
            for patient_id in splits[partition]
        },
        "test_labels_persisted": False,
    }
    if split_path.is_file():
        saved_splits = json.loads(split_path.read_text(encoding="utf-8"))
        if saved_splits != _json_safe(split_record):
            raise ValueError("saved synthetic splits differ from regenerated partitions")
    else:
        atomic_write_json(split_path, split_record)

    train_data = generate_split(splits["train"], labels_by_id, "train", runtime_config)
    selection_data = generate_split(
        splits["validation_model_selection"],
        labels_by_id,
        "validation_model_selection",
        runtime_config,
    )
    calibration_data = generate_split(
        splits["validation_calibration"],
        labels_by_id,
        "validation_calibration",
        runtime_config,
    )
    cue_audit = _validate_training_cues(train_data)
    generation_audit_path = output_dir / "generation_audit.json"
    generation_audit = {
        "synthetic_only": True,
        "pilot_condition": config["pilot_condition"],
        "partition_counts": {name: len(ids) for name, ids in splits.items()},
        "training_cue_reliability": cue_audit,
        "training_image_marker_lesion_overlap": int(
            sum(np.count_nonzero(p.lesion_mask & p.marker_mask) for p in train_data)
        ),
        "training_marker_area_pixels": int(train_data[0].marker_mask.sum()),
        "test_labels_used_before_freeze": False,
    }
    atomic_write_json(generation_audit_path, generation_audit)
    _mark_stage(output_dir, "synthetic_splits_and_training_data", [split_path, generation_audit_path])

    models = _train_models(output_dir, runtime_config, train_data, selection_data)
    temperatures, calibration_logits, calibration_labels = _calibrate_models(
        output_dir, models, calibration_data, runtime_config
    )
    checkpoint_paths = {
        kind: _checkpoint_path(output_dir, kind) for kind in MODEL_NAMES
    }
    assert_test_ready(checkpoint_paths, temperatures)
    frozen_path = output_dir / "checkpoints_and_calibrators_frozen.json"
    if not frozen_path.is_file():
        atomic_write_json(
            frozen_path,
            {
                "frozen_before_test_generation": True,
                "checkpoint_sha256": {
                    kind: _sha256_file(checkpoint_paths[kind]) for kind in MODEL_NAMES
                },
                "temperature_fit_partition": "validation_calibration",
                "temperatures": temperatures,
                "test_labels_used": False,
            },
        )
    _mark_stage(output_dir, "models_and_calibrators_frozen", [*checkpoint_paths.values(), frozen_path])
    _log(output_dir, "All three checkpoints and calibration temperatures are frozen")

    test_data = _load_or_create_test_data(
        output_dir, splits["test"], labels_by_id, runtime_config
    )
    if len(test_data) != 1800 or any(patient.partition != "test" for patient in test_data):
        raise RuntimeError("generated pilot test partition is not exactly 1,800 synthetic patients")
    predictions = _test_predictions(
        output_dir, models, test_data, temperatures, runtime_config
    )
    primary, patient_metrics = _primary_results(
        output_dir,
        predictions,
        calibration_logits,
        calibration_labels,
        temperatures,
        runtime_config,
    )
    _log(
        output_dir,
        f"Primary evaluation finished once: accuracy={primary['accuracy']['joint']:.4f}, "
        f"joint NLL gate={primary['passes_primary_gate']}",
    )

    explanation_results = _compute_explanation_outputs(
        output_dir,
        test_data,
        train_data,
        models,
        temperatures,
        runtime_config,
    )
    secondary = _secondary_statistics(
        output_dir, primary, predictions, explanation_results, runtime_config
    )
    primary["accuracy_by_class"] = {
        kind: {
            str(label): {
                "patient_count": int(np.count_nonzero(predictions["labels"] == label)),
                "accuracy": float(
                    np.mean(
                        predictions[f"probabilities_{kind}"][
                            predictions["labels"] == label
                        ].argmax(axis=1)
                        == label
                    ),
                ),
            }
            for label in range(CLASS_COUNT)
        }
        for kind in MODEL_NAMES
    }
    primary["cue_mask_accuracy_diagnostics"] = secondary["cue_mask_sensitivity"]
    atomic_write_json(output_dir / "primary_statistics.json", primary)
    figure_paths = _save_figures(output_dir, explanation_results)
    _mark_stage(
        output_dir,
        "figures",
        figure_paths,
    )
    report_path = output_dir / "pilot_report.json"
    atomic_write_json(
        report_path,
        {
            "experiment_id": config["experiment_id"],
            "status": "executed_balanced_pilot",
            "synthetic_only": True,
            "primary": primary,
            "secondary": secondary,
            "interpretation_note": (
                "Cue masking is intervention sensitivity, localization is attribution distribution, "
                "faithfulness is ranked model-response change, and stability is perturbation similarity."
            ),
            "not_a_phase_g_completion": True,
        },
    )
    _mark_stage(output_dir, "report", [report_path, output_dir / "primary_statistics.json"])
    after_protected_hashes = hash_protected_inputs(root)
    before_protected_hashes = {
        key.removeprefix("protected:"): value
        for key, value in source_hashes.items()
        if key.startswith("protected:")
    }
    if after_protected_hashes != before_protected_hashes:
        raise RuntimeError("protected frozen Phase F/v1 inputs changed during Phase G execution")
    atomic_write_json(
        output_dir / "protected_inputs_after.json",
        {"sha256": after_protected_hashes, "unchanged": True},
    )
    _log(output_dir, "Protected Phase F/v1 input hashes are unchanged")
    _write_final_manifest(
        output_dir,
        {
            "experiment_id": config["experiment_id"],
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "source_and_protected_sha256": source_hashes,
            "protected_inputs_unchanged": True,
            "test_patient_count": int(len(predictions["labels"])),
            "condition": config["pilot_condition"],
            "seed": int(config["master_seed"]),
        },
        f"Run completed; report: {report_path}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the authorized Phase G synthetic pilot")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument(
        "--resume-explanations",
        action="store_true",
        help="Continue only explanation and secondary-statistic stages using frozen pilot artifacts",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume_explanations and args.resume:
        parser.error("--resume is invalid with --resume-explanations")
    if args.resume and not args.run:
        parser.error("--resume is valid only with --run")
    config = _load_config()
    if args.validate_config:
        output_dir = ROOT / str(config["output_directory"])
        if output_dir.exists():
            raise FileExistsError(f"output path is already in use: {output_dir}")
        print(
            json.dumps(
                {
                    "valid": True,
                    "authorized_conditions": config["authorized_conditions"],
                    "output_directory": str(output_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_only:
        result = run_preflight(config, ROOT)
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return 0
    if args.resume_explanations:
        _run_explanation_resume(ROOT)
        return 0
    _run_pilot(ROOT, config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
