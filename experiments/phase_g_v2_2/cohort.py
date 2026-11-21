from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_cohort_manifest(
    manifest: Mapping[str, Any],
    *,
    development_patient_ids: Iterable[str],
    historical_test_patient_ids: Iterable[str] = (),
    historical_validation_patient_ids: Iterable[str] = (),
    protocol_sha256: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "cohort_name",
        "master_seed",
        "patient_ids",
        "labels_by_patient",
        "class_counts",
        "cohort_membership_sha256",
        "test_labels_used",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"cohort manifest missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported cohort manifest schema")
    if manifest["test_labels_used"] is not False:
        raise ValueError("cohort manifest must declare test_labels_used=false")

    patient_ids = manifest["patient_ids"]
    labels = manifest["labels_by_patient"]
    if (
        not isinstance(patient_ids, list)
        or not patient_ids
        or any(not isinstance(value, str) or not value for value in patient_ids)
    ):
        raise ValueError("patient_ids must be a nonempty list of strings")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("cohort patient IDs must be unique")
    if not isinstance(labels, dict) or set(labels) != set(patient_ids):
        raise ValueError("labels_by_patient must contain exactly the cohort IDs")
    for patient_id, label in labels.items():
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 6:
            raise ValueError(f"invalid cohort label for {patient_id}")

    class_counts = manifest["class_counts"]
    if not isinstance(class_counts, dict):
        raise ValueError("class_counts must be a mapping")
    observed = {
        str(label): sum(value == label for value in labels.values())
        for label in range(6)
    }
    if {str(key): int(value) for key, value in class_counts.items()} != observed:
        raise ValueError("class_counts do not match labels_by_patient")

    expected_membership = sha256_json(
        {
            "patient_ids": list(patient_ids),
            "labels_by_patient": labels,
        }
    )
    if manifest["cohort_membership_sha256"] != expected_membership:
        raise ValueError("cohort membership hash mismatch")

    development = set(development_patient_ids)
    historical_test = set(historical_test_patient_ids)
    historical_validation = set(historical_validation_patient_ids)
    cohort = set(patient_ids)
    overlap = sorted(cohort.intersection(development))
    if overlap:
        raise ValueError(f"cohort overlaps development IDs: {overlap[:5]}")
    overlap = sorted(cohort.intersection(historical_test))
    if overlap:
        raise ValueError(f"cohort overlaps historical test IDs: {overlap[:5]}")
    overlap = sorted(cohort.intersection(historical_validation))
    if overlap:
        raise ValueError(f"cohort overlaps historical validation IDs: {overlap[:5]}")

    if protocol_sha256 is not None:
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise ValueError("protocol SHA-256 mismatch")
    if config_sha256 is not None:
        if manifest.get("config_sha256") != config_sha256:
            raise ValueError("configuration SHA-256 mismatch")

    return {
        "cohort_name": manifest["cohort_name"],
        "patient_count": len(patient_ids),
        "class_counts": observed,
        "cohort_membership_sha256": expected_membership,
        "test_labels_used": False,
    }
