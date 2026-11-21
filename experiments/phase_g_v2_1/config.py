from __future__ import annotations

from typing import Any


CONDITION_NAMES = (
    "balanced",
    "image_dominant",
    "clinical_dominant",
    "redundant",
    "conflicting",
    "synergistic",
)


def validate_v21_configuration(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("v2.1 configuration must be a mapping")
    if config.get("test_data_access") != "forbidden":
        raise ValueError("test_data_access must be 'forbidden'")
    authorized = config.get("authorized_conditions")
    if not isinstance(authorized, list):
        raise ValueError("authorized_conditions must be a list")
    if len(set(authorized)) != len(authorized):
        raise ValueError("authorized_conditions must not contain duplicates")
    unknown = [name for name in authorized if name not in CONDITION_NAMES]
    if unknown:
        raise ValueError(f"unknown condition: {unknown[0]}")
    if config.get("execution_enabled") is not False:
        raise ValueError("execution_enabled must be explicitly false in v2.1")
    if authorized:
        protocol_status = config.get("protocol_status")
        if protocol_status not in {"draft_unapproved", "approved"}:
            raise ValueError("protocol_status must be draft_unapproved or approved")
    else:
        if "protocol_status" not in config:
            raise ValueError("protocol_status is required")
    return {
        "authorized_conditions": list(authorized),
        "execution_enabled": False,
        "test_data_access": "forbidden",
        "protocol_status": config["protocol_status"],
    }
