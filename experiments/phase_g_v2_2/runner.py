from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from experiments.phase_g.generate import make_patient_splits
from experiments.phase_g_v2_2.data import (
    REMAINING_CONDITION_NAMES,
    condition_config,
    generate_condition_split,
    make_condition_assignments,
)
from experiments.phase_g_v2_2.execution import (
    condition_output_directory,
    validate_static_config,
    validate_execution_request,
    validate_run_authorization,
)
from experiments.phase_g_v2_2.pipeline import (
    _atomic_write_json,
    calibrate_condition_models,
    mark_stage,
    require_training_memory,
    run_training_preflight,
    train_condition_models,
    validate_protocol_compatibility,
    validate_run_directory,
    verify_stage,
)


def validate_remaining_condition_request(
    root: Path,
    authorized_conditions: list[str],
) -> dict[str, Any]:
    unknown = [
        condition
        for condition in authorized_conditions
        if condition not in REMAINING_CONDITION_NAMES
    ]
    if unknown:
        raise ValueError(f"unknown condition: {unknown[0]}")
    request = {
        "authorized_conditions": list(authorized_conditions),
        "execution_enabled": False,
        "test_data_access": "forbidden",
        "protocol_status": "draft_unapproved",
    }
    validated = validate_execution_request(request)
    config_path = Path(root) / "experiments/phase_g/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("pilot_condition") != "balanced":
        raise ValueError("historical runner is not balanced-only")
    if config.get("authorized_conditions") != ["balanced"]:
        raise ValueError("historical runner authorization does not match the balanced pilot")
    return {
        **validated,
        "historical_config_pilot_condition": "balanced",
        "historical_config_authorized_conditions": ["balanced"],
        "execution_started": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _save_split_record(
    output_dir: Path,
    config: dict[str, Any],
    splits: dict[str, list[str]],
    labels: dict[str, int],
) -> Path:
    record = {
        "master_seed": int(config["master_seed"]),
        "partition_patient_ids": splits,
        "development_labels": {
            patient_id: int(labels[patient_id])
            for partition in (
                "train",
                "validation_model_selection",
                "validation_calibration",
            )
            for patient_id in splits[partition]
        },
        "test_labels_persisted": False,
    }
    path = output_dir / "synthetic_patient_splits.json"
    if path.is_file():
        if _read_json(path) != record:
            raise ValueError("saved synthetic partitions differ from the frozen config")
    else:
        _atomic_write_json(path, record)
    mark_stage(output_dir, "patient_splits", [path])
    return path


def _generate_partition(
    patient_ids: list[str],
    labels: dict[str, int],
    partition: str,
    config: dict[str, Any],
    condition: str,
):
    assignments = None
    if condition == "conflicting":
        assignments = make_condition_assignments(
            patient_ids,
            condition,
            int(config["master_seed"]),
        )
    return generate_condition_split(
        patient_ids,
        labels,
        partition,
        config,
        condition,
        assignments,
    )


def run_condition(
    root: Path,
    condition: str,
    authorization: dict[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    static_config = _read_json(root / "experiments/phase_g_v2_2/config.json")
    validate_static_config(static_config)
    validated = validate_run_authorization(
        static_config,
        authorization,
        condition,
        root,
    )
    if validated.get("execution_enabled") is not True:
        raise ValueError("execution authorization did not enable the requested run")
    config = _read_json(root / "experiments/phase_g/config.json")
    runtime_config = condition_config(config, condition)
    validate_protocol_compatibility(runtime_config, static_config)
    output_dir = condition_output_directory(root, condition)
    validate_run_directory(output_dir, config, resume)
    require_training_memory(config)
    splits, labels = make_patient_splits(config)

    train_data = _generate_partition(
        splits["train"], labels, "train", config, condition
    )
    selection_data = _generate_partition(
        splits["validation_model_selection"],
        labels,
        "validation_model_selection",
        config,
        condition,
    )
    models = train_condition_models(
        output_dir,
        config,
        train_data,
        selection_data,
        authorization=authorization,
        root=root,
        resume=resume,
    )
    del train_data, selection_data

    calibration_data = _generate_partition(
        splits["validation_calibration"],
        labels,
        "validation_calibration",
        config,
        condition,
    )
    temperatures, calibration_logits, calibration_labels = calibrate_condition_models(
        output_dir,
        models,
        calibration_data,
        runtime_config,
    )
    del calibration_data, calibration_logits, calibration_labels
    split_path = _save_split_record(output_dir, config, splits, labels)
    frozen_path = output_dir / "models_and_calibrators_frozen.json"
    if not verify_stage(output_dir, "models_and_calibrators_frozen"):
        if frozen_path.exists():
            raise FileExistsError("unverified frozen-stage record must not be overwritten")
        model_manifest = output_dir / "models_trained.json"
        calibration_metadata = output_dir / "calibration.json"
        _atomic_write_json(
            frozen_path,
            {
                "frozen_before_test_generation": True,
                "condition": condition,
                "temperatures": temperatures,
                "checkpoint_sha256": {
                    kind: hashlib.sha256(
                        (output_dir / "models" / kind / "checkpoints" / f"{kind}.pt").read_bytes()
                    ).hexdigest()
                    for kind in ("image_only", "clinical_only", "joint")
                },
                "calibration_metadata_sha256": hashlib.sha256(
                    calibration_metadata.read_bytes()
                ).hexdigest(),
                "test_labels_used": False,
            },
        )
        mark_stage(
            output_dir,
            "models_and_calibrators_frozen",
            [model_manifest, calibration_metadata, frozen_path, split_path],
        )

    return run_authorized_test_stage(
        output_dir,
        condition,
        runtime_config,
        authorization,
        root=root,
    )


def run_authorized_test_stage(
    output_dir: Path,
    condition: str,
    config: dict[str, Any],
    authorization: dict[str, Any] | None,
    *,
    root: Path,
) -> dict[str, Any]:
    if authorization is None:
        raise ValueError("explicit execution authorization is required")
    root = Path(root).resolve()
    static_config = _read_json(root / "experiments/phase_g_v2_2/config.json")
    validated = validate_run_authorization(
        static_config,
        authorization,
        condition,
        root,
    )
    if validated.get("execution_enabled") is not True:
        raise ValueError("execution authorization did not enable the test stage")
    output_dir = Path(output_dir).resolve()
    if output_dir != condition_output_directory(root, condition):
        raise ValueError("test stage output must use its isolated v2.2 condition directory")
    if not verify_stage(output_dir, "models_and_calibrators_frozen"):
        raise RuntimeError("test access is blocked until models and calibrators are frozen")
    if not verify_stage(output_dir, "models_trained"):
        raise RuntimeError("test access is blocked until all three model checkpoints are verified")
    for kind in ("image_only", "clinical_only", "joint"):
        if not verify_stage(output_dir / "models" / kind, "model"):
            raise RuntimeError(f"test access is blocked by an incomplete {kind} checkpoint")
    if not verify_stage(output_dir, "calibration"):
        raise RuntimeError("test access is blocked until calibration is verified")

    cohort_ids = validated["evaluation_cohort_manifest"]["patient_ids"]
    config_base = _read_json(root / "experiments/phase_g/config.json")
    splits, _ = make_patient_splits(config_base)
    test_ids = splits["test"]
    if len(cohort_ids) != len(test_ids) or set(cohort_ids) != set(test_ids):
        raise ValueError(
            "authorized evaluation cohort must exactly match the frozen synthetic test partition"
        )
    validate_protocol_compatibility(config, static_config)

    raise RuntimeError(
        "test execution is blocked until the v2.1 SLIC and Grad-CAM protocol "
        "mismatches are resolved and a runner implementing those approved methods is available"
    )


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Condition-isolated Phase G v2.2 runner"
    )
    parser.add_argument(
        "--condition",
        choices=REMAINING_CONDITION_NAMES,
        help="remaining predeclared synthetic condition",
    )
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args(argv)

    if args.run and args.preflight_only:
        parser.error("--run and --preflight-only are mutually exclusive")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.authorization_file and not args.run:
        parser.error("--authorization-file requires --run")
    if args.run and (not args.condition or not args.authorization_file):
        parser.error("--run requires --condition and --authorization-file")
    if args.preflight_output and not args.preflight_only:
        parser.error("--preflight-output requires --preflight-only")

    static_config = _read_json(
        project_root / "experiments/phase_g_v2_2/config.json"
    )
    validated_static = validate_static_config(static_config)
    config = _read_json(project_root / "experiments/phase_g/config.json")

    if args.preflight_only:
        for condition in REMAINING_CONDITION_NAMES:
            condition_config(config, condition)
        if args.preflight_output is None:
            output = Path(tempfile.mkdtemp(prefix="phase_g_v2_2_preflight_"))
        else:
            output = args.preflight_output
        try:
            report = run_training_preflight(config, output)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0

    if args.run:
        authorization = _read_json(args.authorization_file)
        try:
            report = run_condition(
                project_root,
                args.condition,
                authorization,
                resume=args.resume,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0

    if args.condition:
        request = validate_remaining_condition_request(project_root, [args.condition])
        try:
            validate_protocol_compatibility(
                condition_config(config, args.condition),
                static_config,
            )
        except ValueError as error:
            print(
                f"Validated {args.condition}; execution disabled "
                f"(started={request['execution_started']}); {error}"
            )
        else:
            print(
                f"Validated {args.condition}; execution disabled "
                f"(started={request['execution_started']})."
            )
        return 0

    try:
        for condition in REMAINING_CONDITION_NAMES:
            validate_protocol_compatibility(
                condition_config(config, condition),
                static_config,
            )
    except ValueError as error:
        compatibility = f"protocol compatibility blocked: {error}"
    else:
        compatibility = "protocol compatibility passed"
    print(
        "Phase G v2.2 configuration valid; execution disabled "
        f"(protocol_status={validated_static['protocol_status']}, "
        f"test_data_access={validated_static['test_data_access']}); "
        f"{compatibility}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
