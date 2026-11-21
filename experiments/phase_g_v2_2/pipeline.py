from __future__ import annotations

import copy
import ctypes
import hashlib
import io
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import PIL
import scipy
import skimage
import torch
from torch import nn
import torchvision

from experiments.phase_g.generate import SyntheticPatient
from experiments.phase_g.model import (
    MODEL_NAMES,
    TrainingResult,
    build_model,
    fit_temperature,
    predict_logits,
    train_model,
)
from experiments.phase_g_v2_2.data import (
    REMAINING_CONDITION_NAMES,
    condition_config,
    generate_condition_split,
)
from experiments.phase_g_v2_2.execution import (
    condition_output_directory,
    validate_run_authorization,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_PATIENT_COUNT = 8_400
_SELECTION_PATIENT_COUNT = 900
_PREFLIGHT_TRAIN_PATIENTS = 1_024
_PREFLIGHT_SELECTION_PATIENTS = 256
_PREFLIGHT_BATCH_SIZE = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        Path("experiments/phase_g/config.json"),
        Path("experiments/phase_g/generate.py"),
        Path("experiments/phase_g/model.py"),
        Path("experiments/phase_g/run.py"),
        Path("experiments/phase_g/evaluate.py"),
        Path("experiments/phase_g/explain.py"),
        Path("experiments/phase_g_v2_2/config.json"),
        Path("experiments/phase_g_v2_2/data.py"),
        Path("experiments/phase_g_v2_2/execution.py"),
        Path("experiments/phase_g_v2_2/generator.py"),
        Path("experiments/phase_g_v2_2/pipeline.py"),
        Path("experiments/phase_g_v2_2/render.py"),
        Path("experiments/phase_g_v2_2/runner.py"),
        Path("experiments/phase_g_v2_1/geometry.py"),
        Path("experiments/phase_g_v2_1/localization.py"),
        Path("experiments/phase_g_v2_1/slic.py"),
        Path("docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md"),
    )
    return {
        str(relative_path): _sha256_file(_PROJECT_ROOT / relative_path)
        for relative_path in relative_paths
    }


def require_cuda(device_name: str = "cuda:0") -> torch.device:
    if device_name != "cuda:0":
        raise ValueError("Phase G v2.2 requires cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase G v2.2; CPU fallback is forbidden")
    try:
        torch.cuda.set_device(0)
        torch.cuda.synchronize(0)
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError("CUDA device 0 is unavailable or failed initialization") from error
    return torch.device("cuda:0")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path.resolve()), str(parent.resolve()))) == str(
            parent.resolve()
        )
    except ValueError:
        return False


def prepare_run_directory(
    output_dir: Path,
    config_hash: str,
    source_hashes: dict[str, str],
    resume: bool,
) -> None:
    output_dir = Path(output_dir)
    identity = {
        "config_sha256": config_hash,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    identity_path = output_dir / "run_identity.json"
    validate_run_identity(output_dir, config_hash, source_hashes, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not identity_path.exists():
        _atomic_write_json(identity_path, identity)


def validate_run_identity(
    output_dir: Path,
    config_hash: str,
    source_hashes: dict[str, str],
    resume: bool,
) -> None:
    output_dir = Path(output_dir)
    identity = {
        "config_sha256": config_hash,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    identity_path = output_dir / "run_identity.json"
    if not output_dir.exists() and resume:
        raise FileNotFoundError(f"cannot resume missing output directory: {output_dir}")
    if output_dir.exists():
        if any(output_dir.iterdir()) and not resume:
            raise FileExistsError("output directory already contains a run")
        if resume:
            if not identity_path.is_file():
                raise ValueError("resume requires a matching run identity")
            try:
                existing = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("run identity is unreadable") from error
            if existing != identity:
                raise ValueError("resume config or source hashes do not match")


def validate_run_directory(
    output_dir: Path,
    config: dict[str, Any],
    resume: bool,
) -> None:
    condition = Path(output_dir).name
    runtime_config = condition_config(config, condition)
    validate_run_identity(
        Path(output_dir),
        _json_digest(runtime_config),
        _source_hashes(),
        resume,
    )


def _stage_marker(output_dir: Path, stage_name: str) -> Path:
    if not stage_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in stage_name):
        raise ValueError("invalid stage name")
    return output_dir / f".{stage_name}.stage.json"


def mark_stage(output_dir: Path, stage_name: str, artifacts: Sequence[Path]) -> None:
    root = Path(output_dir).resolve()
    entries = []
    for artifact in artifacts:
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        artifact_path = artifact_path.resolve()
        if not _is_within(artifact_path, root) or not artifact_path.is_file():
            raise ValueError("stage artifact must be a file inside its output directory")
        entries.append(
            {
                "path": str(artifact_path.relative_to(root)),
                "sha256": _sha256_file(artifact_path),
            }
        )
    if not entries:
        raise ValueError("a stage must contain at least one artifact")
    _atomic_write_json(
        _stage_marker(root, stage_name),
        {"stage": stage_name, "artifacts": sorted(entries, key=lambda item: item["path"])},
    )


def verify_stage(output_dir: Path, stage_name: str) -> bool:
    root = Path(output_dir).resolve()
    marker = _stage_marker(root, stage_name)
    if not marker.is_file():
        return False
    try:
        content = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{stage_name} stage manifest is unreadable") from error
    if content.get("stage") != stage_name or not isinstance(content.get("artifacts"), list):
        raise RuntimeError(f"{stage_name} stage manifest is invalid")
    for artifact in content["artifacts"]:
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError(f"{stage_name} stage manifest is invalid")
        path = (root / relative).resolve()
        if not _is_within(path, root) or not path.is_file():
            raise RuntimeError(f"{stage_name} stage artifact is missing or unsafe")
        if _sha256_file(path) != expected:
            raise RuntimeError(f"{stage_name} stage artifact changed: {relative}")
    return True


def _validate_partitions(
    train_data: Sequence[SyntheticPatient],
    selection_data: Sequence[SyntheticPatient],
) -> None:
    if not train_data or not selection_data:
        raise ValueError("training and model-selection data must be nonempty")
    if any(patient.partition != "train" for patient in train_data):
        raise ValueError("training data may contain only the train partition")
    if any(
        patient.partition != "validation_model_selection"
        for patient in selection_data
    ):
        raise ValueError("model selection may use only its validation partition")
    train_ids = [patient.patient_id for patient in train_data]
    selection_ids = [patient.patient_id for patient in selection_data]
    if len(set(train_ids)) != len(train_ids) or len(set(selection_ids)) != len(selection_ids):
        raise ValueError("patient IDs must be unique within each training partition")
    if set(train_ids).intersection(selection_ids):
        raise ValueError("training and model-selection patients must be disjoint")


def validate_protocol_compatibility(
    config: dict[str, Any],
    static_config: dict[str, Any] | None = None,
) -> None:
    issues = []
    if static_config is not None and static_config.get("protocol_status") != "approved":
        issues.append("the checked-in prospective protocol is not approved")

    methods = config.get("explanation_evaluation", {}).get("image_methods", [])
    required_gradcam_variants = {
        "gradcam_final_block_gradient_global",
        "gradcam_final_block_token_local",
        "gradcam_penultimate_block_gradient_global",
        "gradcam_penultimate_block_token_local",
    }
    if not required_gradcam_variants.issubset(set(methods)):
        issues.append(
            "the prospective protocol requires four Grad-CAM variants, while the "
            "current model implementation exposes one gradcam method"
        )

    faithfulness = config.get("explanation_evaluation", {}).get("faithfulness", {})
    if faithfulness.get("superpixel_count") != 128:
        issues.append(
            "the prospective protocol requires SLIC superpixel faithfulness; "
            "the current implementation uses a fixed 14x14 patch grid"
        )
    if faithfulness.get("baselines") != ["background_canvas", "mid_gray"]:
        issues.append(
            "the prospective protocol requires renderer-background and mid-gray "
            "faithfulness baselines"
        )
    primary = config.get("primary_incremental_prediction_criterion", {})
    if "full_accuracy_floor" in primary or "holm_family_size" in primary:
        issues.append(
            "the legacy primary gate includes an accuracy floor and a fixed two-test "
            "family, while the prospective protocol defines NLL as primary and leaves "
            "its threshold and multiplicity interpretation for approval"
        )

    if issues:
        raise ValueError("protocol compatibility blocker(s): " + "; ".join(issues))


def _load_model_stage(
    model_dir: Path,
    kind: str,
    config: dict[str, Any],
) -> nn.Module:
    if not verify_stage(model_dir, "model"):
        raise RuntimeError(f"model stage is incomplete: {kind}")
    checkpoint_path = model_dir / "checkpoints" / f"{kind}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise RuntimeError(f"model checkpoint is unreadable: {kind}") from error
    expected_seed = int(config["model"]["training"]["model_initialization_seeds"][kind])
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("kind") != kind
        or checkpoint.get("seed") != expected_seed
        or not isinstance(checkpoint.get("selected_epoch"), int)
        or checkpoint["selected_epoch"] < 1
        or not isinstance(checkpoint.get("state_dict"), dict)
    ):
        raise RuntimeError(f"model checkpoint metadata is invalid: {kind}")
    model = build_model(kind, config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to("cpu")
    model.eval()
    return model


def _model_stage_artifacts(model_dir: Path, kind: str) -> list[Path]:
    return [
        model_dir / "checkpoints" / f"{kind}.pt",
        model_dir / "training_history.json",
        model_dir / "training_metadata.json",
    ]


def train_condition_models(
    output_dir: Path,
    config: dict[str, Any],
    train_data: list[SyntheticPatient],
    selection_data: list[SyntheticPatient],
    authorization: dict[str, Any] | None = None,
    root: Path | None = None,
    resume: bool = False,
) -> dict[str, nn.Module]:
    if authorization is None:
        raise ValueError("explicit execution authorization is required")
    project_root = Path(root or _PROJECT_ROOT).resolve()
    output_dir = Path(output_dir).resolve()
    condition = output_dir.name
    if condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition output directory: {condition}")
    static_config_path = project_root / "experiments/phase_g_v2_2/config.json"
    try:
        static_config = json.loads(static_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checked-in v2.2 execution config cannot be read") from error
    validated_authorization = validate_run_authorization(
        static_config, authorization, condition, project_root
    )
    if condition not in validated_authorization["authorized_conditions"]:
        raise ValueError(f"condition is not authorized: {condition}")
    if output_dir != condition_output_directory(project_root, condition):
        raise ValueError("condition output must use its isolated v2.2 output directory")

    _validate_partitions(train_data, selection_data)
    runtime_config = condition_config(config, condition)
    validate_protocol_compatibility(runtime_config, static_config)
    source_hashes = _source_hashes()
    config_hash = _json_digest(runtime_config)
    validate_run_identity(output_dir, config_hash, source_hashes, resume)
    require_cuda()
    prepare_run_directory(output_dir, config_hash, source_hashes, resume)
    saved_config = output_dir / "run_config.json"
    if saved_config.is_file():
        if json.loads(saved_config.read_text(encoding="utf-8")) != runtime_config:
            raise ValueError("saved run config differs from the requested condition config")
    else:
        _atomic_write_json(saved_config, runtime_config)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if resume and verify_stage(output_dir, "models_trained"):
        return {
            kind: _load_model_stage(models_dir / kind, kind, runtime_config)
            for kind in MODEL_NAMES
        }

    models: dict[str, nn.Module] = {}
    for kind in MODEL_NAMES:
        require_cuda()
        final_dir = models_dir / kind
        if final_dir.exists():
            if not resume:
                raise FileExistsError(f"model output already exists: {kind}")
            models[kind] = _load_model_stage(final_dir, kind, runtime_config)
            continue

        staging_dir = models_dir / f".{kind}.staging"
        if staging_dir.exists():
            if not resume:
                raise FileExistsError(f"incomplete model stage already exists: {kind}")
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        model_config = copy.deepcopy(runtime_config)
        model_config["output_directory"] = str(staging_dir)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        result: TrainingResult = train_model(
            kind,
            train_data,
            selection_data,
            model_config,
        )
        torch.cuda.synchronize(0)
        elapsed_seconds = time.perf_counter() - started
        checkpoint_path = (staging_dir / "checkpoints" / f"{kind}.pt").resolve()
        if result.checkpoint_path.resolve() != checkpoint_path or not checkpoint_path.is_file():
            raise RuntimeError(f"trainer did not produce the expected checkpoint: {kind}")
        if any(parameter.device.type != "cpu" for parameter in result.model.parameters()):
            raise RuntimeError(f"trainer returned an unexpected model device: {kind}")
        peak_memory = int(torch.cuda.max_memory_allocated(0))
        if peak_memory <= 0:
            raise RuntimeError(f"no CUDA allocation was observed during training: {kind}")
        _atomic_write_json(
            staging_dir / "training_history.json",
            {
                "selected_epoch": result.selected_epoch,
                "history": result.history,
            },
        )
        _atomic_write_json(
            staging_dir / "training_metadata.json",
            {
                "condition": condition,
                "model": kind,
                "device": "cuda:0",
                "batch_size": int(runtime_config["model"]["training"]["batch_size"]),
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "source_sha256": source_hashes,
            },
        )
        mark_stage(staging_dir, "model", _model_stage_artifacts(staging_dir, kind))
        os.replace(staging_dir, final_dir)
        models[kind] = result.model.eval()

    model_manifest = output_dir / "models_trained.json"
    if model_manifest.exists():
        raise FileExistsError("unverified model manifest must not be overwritten")
    _atomic_write_json(
        model_manifest,
        {
            "condition": condition,
            "models": {
                kind: {
                    "checkpoint": str((models_dir / kind / "checkpoints" / f"{kind}.pt").relative_to(output_dir)),
                    "checkpoint_sha256": _sha256_file(
                        models_dir / kind / "checkpoints" / f"{kind}.pt"
                    ),
                }
                for kind in MODEL_NAMES
            },
        },
    )
    mark_stage(output_dir, "models_trained", [model_manifest])
    return models


def calibrate_condition_models(
    output_dir: Path,
    models: dict[str, nn.Module],
    calibration_data: list[SyntheticPatient],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
    if set(models) != set(MODEL_NAMES):
        raise ValueError("calibration requires exactly the three Phase G models")
    if not calibration_data:
        raise ValueError("calibration data cannot be empty")
    if any(patient.partition != "validation_calibration" for patient in calibration_data):
        raise ValueError("temperature fitting may use only calibration data")
    ids = [patient.patient_id for patient in calibration_data]
    if len(set(ids)) != len(ids):
        raise ValueError("calibration patient IDs must be unique")
    require_cuda()

    output_dir = Path(output_dir)
    metadata_path = output_dir / "calibration.json"
    logits_path = output_dir / "calibration_logits.npz"
    if verify_stage(output_dir, "calibration"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("patient_ids") != ids:
            raise ValueError("existing calibration uses different patients")
        with np.load(logits_path, allow_pickle=False) as archive:
            logits_by_model = {kind: archive[kind].copy() for kind in MODEL_NAMES}
            labels = archive["labels"].copy()
        return (
            {kind: float(value) for kind, value in metadata["temperatures"].items()},
            logits_by_model,
            labels,
        )
    if metadata_path.exists() or logits_path.exists():
        raise FileExistsError("incomplete calibration output must not be overwritten")

    labels = np.asarray([patient.label for patient in calibration_data], dtype=np.int64)
    logits_by_model: dict[str, np.ndarray] = {}
    temperatures: dict[str, float] = {}
    batch_size = int(config["model"]["training"]["batch_size"])
    for kind in MODEL_NAMES:
        logits = predict_logits(
            models[kind],
            calibration_data,
            kind,
            batch_size=batch_size,
            device=torch.device("cuda:0"),
        )
        if logits.shape != (len(calibration_data), 6) or not np.isfinite(logits).all():
            raise RuntimeError(f"calibration logits are invalid: {kind}")
        logits_by_model[kind] = logits
        temperatures[kind] = fit_temperature(logits, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, labels=labels, **logits_by_model)
    logits_path.write_bytes(buffer.getvalue())
    _atomic_write_json(
        metadata_path,
        {
            "partition": "validation_calibration",
            "patient_ids": ids,
            "patient_ids_sha256": _json_digest(ids),
            "temperatures": temperatures,
            "batch_size": batch_size,
            "calibration_protocol": config["model"]["calibration"],
        },
    )
    mark_stage(output_dir, "calibration", [metadata_path, logits_path])
    return temperatures, logits_by_model, labels


def _available_system_memory_bytes() -> int | None:
    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except (AttributeError, OSError):
        return None
    return None


def training_dataset_memory_floor(config: dict[str, Any]) -> int:
    dataset = config["dataset"]
    class_counts = dataset["class_counts"]
    train_and_selection = int(class_counts["train_per_class"]) + int(
        class_counts["validation_model_selection_per_class"]
    )
    train_and_selection_count = train_and_selection * len(config["labels"]["values"])
    render = config["render"]["image"]
    pixel_count = int(render["height"]) * int(render["width"])
    image_bytes = pixel_count * int(render["channels"]) * 2
    mask_bytes = pixel_count * 2
    return train_and_selection_count * (image_bytes + mask_bytes)


def require_training_memory(config: dict[str, Any]) -> dict[str, int]:
    available = _available_system_memory_bytes()
    if available is None:
        raise RuntimeError("available system memory could not be measured")
    required = training_dataset_memory_floor(config)
    if available < required:
        raise RuntimeError(
            "insufficient system memory for the raw training and model-selection "
            f"arrays: {available} available, at least {required} required; "
            "this lower bound excludes Python and training overhead"
        )
    return {
        "available_system_memory_bytes": available,
        "raw_training_dataset_memory_floor_bytes": required,
    }


def validate_preflight_output_path(path: Path, project_root: Path) -> Path:
    resolved = Path(path).resolve()
    output_root = Path(project_root).resolve() / "outputs" / "synthetic_phase_g_v2_2"
    for condition in REMAINING_CONDITION_NAMES:
        if _is_within(resolved, output_root / condition):
            raise ValueError("preflight output must be outside condition result directories")
    return resolved


def run_training_preflight(config: dict[str, Any], root: Path) -> dict[str, Any]:
    configured_batch_size = int(config["model"]["training"]["batch_size"])
    if configured_batch_size != _PREFLIGHT_BATCH_SIZE:
        raise ValueError("training preflight requires the approved batch size 64")
    preflight_dir = validate_preflight_output_path(root, _PROJECT_ROOT)
    device = require_cuda()
    if preflight_dir.exists() and any(preflight_dir.iterdir()):
        raise FileExistsError("preflight directory must be empty")
    preflight_dir.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(preflight_dir)
    system_memory_before = _available_system_memory_bytes()
    free_memory_before = int(torch.cuda.mem_get_info(0)[0])
    gpu_properties = torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats(0)

    train_ids = [f"preflight-train-{index:04d}" for index in range(_PREFLIGHT_TRAIN_PATIENTS)]
    selection_ids = [
        f"preflight-selection-{index:04d}"
        for index in range(_PREFLIGHT_SELECTION_PATIENTS)
    ]
    labels = {
        patient_id: index % 6
        for index, patient_id in enumerate(train_ids + selection_ids)
    }
    rendering_started = time.perf_counter()
    train_data = generate_condition_split(
        train_ids,
        labels,
        "train",
        config,
        "image_dominant",
    )
    selection_data = generate_condition_split(
        selection_ids,
        labels,
        "validation_model_selection",
        config,
        "image_dominant",
    )
    rendering_seconds = time.perf_counter() - rendering_started
    _validate_partitions(train_data, selection_data)
    preflight_array_bytes = sum(
        patient.image.nbytes
        + patient.background_canvas.nbytes
        + patient.lesion_mask.nbytes
        + patient.marker_mask.nbytes
        for patient in train_data + selection_data
    )

    preflight_config = copy.deepcopy(condition_config(config, "image_dominant"))
    preflight_config["model"]["training"]["maximum_epochs"] = 1
    preflight_config["output_directory"] = str(preflight_dir / "models")
    per_model: dict[str, dict[str, Any]] = {}
    checkpoint_bytes = 0
    for kind in MODEL_NAMES:
        model_config = copy.deepcopy(preflight_config)
        model_config["output_directory"] = str(preflight_dir / "models" / kind)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        result = train_model(kind, train_data, selection_data, model_config)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated(0))
        if peak_memory <= 0:
            raise RuntimeError(f"no CUDA allocation was observed during preflight: {kind}")
        checkpoint_size = result.checkpoint_path.stat().st_size
        checkpoint_bytes += checkpoint_size
        full_epoch_seconds = elapsed * (
            (_TRAINING_PATIENT_COUNT + _SELECTION_PATIENT_COUNT)
            / (_PREFLIGHT_TRAIN_PATIENTS + _PREFLIGHT_SELECTION_PATIENTS)
        )
        per_model[kind] = {
            "device": str(device),
            "configured_batch_size": configured_batch_size,
            "preflight_train_patients": len(train_data),
            "preflight_selection_patients": len(selection_data),
            "preflight_epoch_seconds": elapsed,
            "estimated_full_data_epoch_seconds": full_epoch_seconds,
            "estimated_maximum_training_seconds": full_epoch_seconds
            * int(config["model"]["training"]["maximum_epochs"]),
            "training_loop_examples_per_second": (
                len(train_data) + len(selection_data)
            )
            / elapsed,
            "peak_cuda_memory_allocated_bytes": peak_memory,
            "peak_cuda_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(0)
            ),
            "checkpoint_bytes": checkpoint_size,
            "selected_epoch": result.selected_epoch,
        }

    disk_after = shutil.disk_usage(preflight_dir)
    projected_one_condition_checkpoint_bytes = checkpoint_bytes
    condition_seconds = sum(
        values["estimated_maximum_training_seconds"]
        for values in per_model.values()
    )
    estimated_full_render_seconds = rendering_seconds * (
        int(config["dataset"]["patient_count"])
        / (len(train_data) + len(selection_data))
    )
    report = {
        "status": "non_evidentiary_training_preflight",
        "condition": None,
        "execution_started": False,
        "device": str(device),
        "gpu_name": gpu_properties.name,
        "gpu_total_memory_bytes": int(gpu_properties.total_memory),
        "gpu_compute_capability": [
            int(gpu_properties.major),
            int(gpu_properties.minor),
        ],
        "cuda_free_memory_before_bytes": free_memory_before,
        "cuda_free_memory_after_bytes": int(torch.cuda.mem_get_info(0)[0]),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pillow_version": PIL.__version__,
        "scikit_image_version": skimage.__version__,
        "available_system_memory_before_bytes": system_memory_before,
        "available_system_memory_after_bytes": _available_system_memory_bytes(),
        "free_disk_before_bytes": int(disk_before.free),
        "free_disk_after_bytes": int(disk_after.free),
        "config_sha256": _json_digest(config),
        "source_sha256": _source_hashes(),
        "preflight_batch_size": configured_batch_size,
        "mixed_precision": False,
        "preflight_data_array_bytes": int(preflight_array_bytes),
        "preflight_data_generation_seconds": rendering_seconds,
        "estimated_full_dataset_render_seconds": estimated_full_render_seconds,
        "raw_training_dataset_memory_floor_bytes": training_dataset_memory_floor(config),
        "raw_all_partitions_memory_floor_bytes": (
            int(config["dataset"]["patient_count"])
            * int(config["render"]["image"]["height"])
            * int(config["render"]["image"]["width"])
            * (int(config["render"]["image"]["channels"]) * 2 + 2)
        ),
        "per_model": per_model,
        "estimated_maximum_seconds_per_condition": condition_seconds,
        "estimated_maximum_seconds_for_five_conditions": condition_seconds * 5,
        "estimated_pre_test_seconds_per_condition": (
            condition_seconds + estimated_full_render_seconds
        ),
        "estimated_pre_test_seconds_for_five_conditions": (
            condition_seconds + estimated_full_render_seconds
        )
        * 5,
        "estimated_checkpoint_bytes_per_condition": projected_one_condition_checkpoint_bytes,
        "estimated_checkpoint_bytes_for_five_conditions": projected_one_condition_checkpoint_bytes * 5,
        "runtime_scope": (
            "Extrapolation uses one measured epoch at the actual batch size and the "
            "configured maximum epoch count. Full-data rendering is extrapolated from "
            "the disposable sample; calibration, test inference, explanations, statistics, "
            "and report generation are not included."
        ),
    }
    report_path = preflight_dir / "preflight_report.json"
    _atomic_write_json(report_path, report)
    return report
