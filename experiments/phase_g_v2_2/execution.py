from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from experiments.phase_g_v2_2.data import REMAINING_CONDITION_NAMES


_STATIC_CONFIG_RELATIVE_PATH = Path("experiments/phase_g_v2_2/config.json")
_PHASE_G_CONFIG_RELATIVE_PATH = Path("experiments/phase_g/config.json")
_PROTOCOL_RELATIVE_PATH = Path("docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from error
    return value.lower()


def validate_static_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("static config must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported v2.2 config schema")
    if config.get("implementation_version") != "phase_g_v2_2":
        raise ValueError("implementation version must be phase_g_v2_2")
    if config.get("execution_enabled") is not False:
        raise ValueError("checked-in execution_enabled must be false")
    if config.get("test_data_access") != "forbidden":
        raise ValueError("checked-in test_data_access must be forbidden")
    if config.get("authorized_conditions") != []:
        raise ValueError("checked-in authorized_conditions must be empty")
    if config.get("protocol_status") not in {"draft_unapproved", "approved"}:
        raise ValueError("protocol_status must be draft_unapproved or approved")
    if config.get("remaining_conditions") != list(REMAINING_CONDITION_NAMES):
        raise ValueError("remaining_conditions must exactly match the supported conditions")
    return {
        "authorized_conditions": [],
        "execution_enabled": False,
        "test_data_access": "forbidden",
        "protocol_status": config["protocol_status"],
    }


def validate_execution_request(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("execution request must be a mapping")
    if config.get("test_data_access") != "forbidden":
        raise ValueError("test_data_access must be forbidden")

    authorized = config.get("authorized_conditions")
    if not isinstance(authorized, list):
        raise ValueError("authorized_conditions must be a list")
    if any(not isinstance(name, str) for name in authorized):
        raise ValueError("authorized_conditions entries must be strings")
    if len(set(authorized)) != len(authorized):
        raise ValueError("authorized_conditions must not contain duplicates")
    unknown = [name for name in authorized if name not in REMAINING_CONDITION_NAMES]
    if unknown:
        raise ValueError(f"unknown condition: {unknown[0]}")

    if config.get("execution_enabled") is not False:
        raise ValueError("execution_enabled must be explicitly false in v2.2")
    protocol_status = config.get("protocol_status")
    if protocol_status not in {"draft_unapproved", "approved"}:
        raise ValueError("protocol_status must be draft_unapproved or approved")

    return {
        "authorized_conditions": list(authorized),
        "execution_enabled": False,
        "test_data_access": "forbidden",
        "protocol_status": protocol_status,
    }


def validate_run_authorization(
    config: dict[str, Any],
    authorization: dict[str, Any] | None,
    requested_condition: str,
    root: Path,
) -> dict[str, Any]:
    static_state = validate_static_config(config)
    if static_state["protocol_status"] != "approved":
        raise ValueError("checked-in prospective protocol must be approved before execution")
    if not isinstance(authorization, dict):
        raise ValueError("an explicit authorization record is required")
    if authorization.get("schema_version") != 1:
        raise ValueError("unsupported authorization schema")
    if authorization.get("protocol_status") != "approved":
        raise ValueError("an approved protocol is required")
    if authorization.get("execution_enabled") is not True:
        raise ValueError("execution must be explicitly enabled by authorization")
    if authorization.get("test_data_access") != "allowed":
        raise ValueError("authorization must explicitly allow test data access")
    if requested_condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition: {requested_condition}")

    authorized = authorization.get("authorized_conditions")
    if not isinstance(authorized, list):
        raise ValueError("authorized_conditions must be a list")
    if any(not isinstance(name, str) for name in authorized):
        raise ValueError("authorized_conditions entries must be strings")
    if len(set(authorized)) != len(authorized):
        raise ValueError("authorized_conditions must not contain duplicates")
    unknown = [name for name in authorized if name not in REMAINING_CONDITION_NAMES]
    if unknown:
        raise ValueError(f"unknown condition: {unknown[0]}")
    if requested_condition not in authorized:
        raise ValueError(f"condition is not authorized: {requested_condition}")

    root = Path(root).resolve()
    sources = {
        "config_sha256": root / _STATIC_CONFIG_RELATIVE_PATH,
        "phase_g_config_sha256": root / _PHASE_G_CONFIG_RELATIVE_PATH,
        "protocol_sha256": root / _PROTOCOL_RELATIVE_PATH,
    }
    for field, path in sources.items():
        expected = _require_sha256(authorization.get(field), field)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"authorization source hash mismatch: {field}")
    try:
        protocol_text = (
            root / _PROTOCOL_RELATIVE_PATH
        ).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("approved protocol document cannot be read") from error
    status_lines = [
        line.strip()
        for line in protocol_text.splitlines()
        if line.strip().lower().startswith("**status:**")
    ]
    protocol_status_line = status_lines[0].lower() if status_lines else ""
    if (
        "approved for execution" not in protocol_status_line
        or "not approved" in protocol_status_line
    ):
        raise ValueError("protocol document must explicitly be approved for execution")
    try:
        actual_static_config = json.loads(
            (root / _STATIC_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checked-in v2.2 config cannot be read") from error
    if actual_static_config != config:
        raise ValueError("provided static config differs from the checked-in config")

    cohort = authorization.get("evaluation_cohort_manifest")
    if not isinstance(cohort, dict):
        raise ValueError("evaluation cohort manifest is required")
    cohort_relative_path = cohort.get("path")
    if not isinstance(cohort_relative_path, str) or not cohort_relative_path:
        raise ValueError("evaluation cohort manifest path is required")
    cohort_path = (root / cohort_relative_path).resolve()
    if not _is_within(cohort_path, root) or not cohort_path.is_file():
        raise ValueError("evaluation cohort manifest must be a file inside the project root")
    expected_cohort_hash = _require_sha256(
        cohort.get("sha256"), "evaluation cohort sha256"
    )
    if _sha256_file(cohort_path) != expected_cohort_hash:
        raise ValueError("evaluation cohort manifest hash mismatch")
    try:
        cohort_data = json.loads(cohort_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evaluation cohort manifest is not valid JSON") from error
    patient_ids = cohort_data.get("patient_ids") if isinstance(cohort_data, dict) else None
    if (
        not isinstance(patient_ids, list)
        or not patient_ids
        or any(not isinstance(patient_id, str) or not patient_id for patient_id in patient_ids)
        or len(set(patient_ids)) != len(patient_ids)
    ):
        raise ValueError("evaluation cohort manifest must contain unique patient_ids")

    policy = authorization.get("predictive_decision_policy")
    if not isinstance(policy, dict):
        raise ValueError("predictive decision policy is required")
    threshold = policy.get("nll_effect_threshold")
    family_size = policy.get("multiplicity_family_size")
    alpha = policy.get("one_sided_alpha")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0
        or policy.get("multiplicity_method") != "holm"
        or isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size < 1
        or isinstance(alpha, bool)
        or not isinstance(alpha, (float, int))
        or not math.isfinite(float(alpha))
        or not 0 < float(alpha) < 1
    ):
        raise ValueError("predictive decision policy is invalid or unsupported")

    return {
        **authorization,
        "authorized_conditions": list(authorized),
        "evaluation_cohort_manifest": {
            "path": str(cohort_path),
            "sha256": expected_cohort_hash,
            "patient_ids": list(patient_ids),
        },
        "predictive_decision_policy": {
            **policy,
            "nll_effect_threshold": float(threshold),
            "multiplicity_family_size": family_size,
            "one_sided_alpha": float(alpha),
        },
    }


def condition_output_directory(root: Path, condition: str) -> Path:
    if condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition: {condition}")
    return (
        Path(root).resolve()
        / "outputs"
        / "synthetic_phase_g_v2_2"
        / condition
    )
