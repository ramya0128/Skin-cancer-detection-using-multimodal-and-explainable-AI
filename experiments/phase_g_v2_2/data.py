from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.phase_g.generate import (
    PARTITION_COUNTS_PER_CLASS,
    SyntheticPatient,
    generate_split,
)
from experiments.phase_g_v2_2.generator import make_conflicting_condition_assignments
from experiments.phase_g_v2_2.render import render_conflicting_patient


REMAINING_CONDITION_NAMES = (
    "image_dominant",
    "clinical_dominant",
    "redundant",
    "conflicting",
    "synergistic",
)
_CONFIG_PATH = Path(__file__).with_name("config.json")
_CLASS_COUNT = 6


def _runner_config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Phase G config must be a mapping")
    runner_config = _runner_config()
    if condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition: {condition}")
    definitions = runner_config.get("condition_definitions", {})
    if set(definitions) != set(REMAINING_CONDITION_NAMES):
        raise ValueError("v2.2 condition definitions do not cover the supported conditions")
    source_definition = config.get("conditions", {}).get(condition)
    if not isinstance(source_definition, dict):
        raise ValueError(f"Phase G config does not define condition: {condition}")
    expected_definition = definitions[condition].get("phase_g_condition")
    if not isinstance(expected_definition, dict) or any(
        source_definition.get(key) != value
        for key, value in expected_definition.items()
    ):
        raise ValueError(f"v2.2 condition definition differs from Phase G config: {condition}")

    result = copy.deepcopy(config)
    result["pilot_condition"] = condition
    return result


def make_condition_assignments(
    patient_ids: list[str], condition: str, seed: int
) -> dict[str, str]:
    if condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition: {condition}")
    if condition != "conflicting":
        return {}
    return make_conflicting_condition_assignments(
        patient_ids,
        np.random.default_rng(seed),
    )


def generate_condition_split(
    patient_ids: list[str],
    labels: dict[str, int],
    partition: str,
    config: dict[str, Any],
    condition: str,
    assignments: dict[str, str] | None = None,
) -> list[SyntheticPatient]:
    if partition not in PARTITION_COUNTS_PER_CLASS:
        raise ValueError(f"unknown partition: {partition}")
    if partition == "test":
        raise ValueError("test rendering is forbidden by the v2.2 condition generator")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient IDs must be unique within a partition")
    missing = [patient_id for patient_id in patient_ids if patient_id not in labels]
    if missing:
        raise KeyError(f"missing labels for patient IDs: {missing[:5]}")
    runtime_config = condition_config(config, condition)

    if condition == "conflicting":
        if assignments is None or set(assignments) != set(patient_ids):
            raise ValueError("conflicting rendering requires an assignment for every patient")
        allowed_orientations = {"image_correct", "clinical_correct", "none"}
        invalid = [value for value in assignments.values() if value not in allowed_orientations]
        if invalid:
            raise ValueError(f"invalid conflicting orientation: {invalid[0]}")
        return [
            render_conflicting_patient(
                patient_id,
                labels[patient_id],
                partition,
                runtime_config,
                assignments[patient_id],
            )
            for patient_id in patient_ids
        ]
    if assignments is not None:
        raise ValueError("population assignments are only valid for conflicting condition")
    return generate_split(patient_ids, labels, partition, runtime_config)


def audit_condition_cues(
    patients: Sequence[SyntheticPatient], condition: str
) -> dict[str, Any]:
    if condition not in REMAINING_CONDITION_NAMES:
        raise ValueError(f"unsupported condition: {condition}")
    if not patients:
        raise ValueError("cue audit requires at least one patient")
    labels = np.asarray([patient.label for patient in patients], dtype=np.int64)
    image = np.asarray([patient.image_cue for patient in patients], dtype=np.int64)
    clinical = np.asarray([patient.clinical_value for patient in patients], dtype=np.int64)
    if (
        np.any((labels < 0) | (labels >= _CLASS_COUNT))
        or np.any((image < 0) | (image >= _CLASS_COUNT))
        or np.any((clinical < 0) | (clinical >= _CLASS_COUNT))
    ):
        raise ValueError("patient labels and cue values must be six-class indices")

    joint_counts: dict[str, list[list[int]]] = {}
    for label in range(_CLASS_COUNT):
        class_mask = labels == label
        counts = np.zeros((_CLASS_COUNT, _CLASS_COUNT), dtype=np.int64)
        np.add.at(counts, (image[class_mask], clinical[class_mask]), 1)
        joint_counts[str(label)] = counts.tolist()

    result: dict[str, Any] = {
        "condition": condition,
        "patient_count": len(patients),
        "image_reliability": float(np.mean(image == labels)),
        "clinical_reliability": float(np.mean(clinical == labels)),
        "joint_cue_counts_by_label": joint_counts,
    }
    if condition == "conflicting":
        orientation_counts = {
            orientation: 0
            for orientation in ("image_correct", "clinical_correct", "none")
        }
        for patient in patients:
            orientation = patient.render_metadata.get("conflict_orientation")
            if orientation not in orientation_counts:
                raise ValueError("conflicting patient is missing a valid orientation")
            orientation_counts[orientation] += 1
        result["population_counts"] = orientation_counts
    if condition == "redundant":
        result["cues_identical"] = bool(np.array_equal(image, clinical))
    if condition == "synergistic":
        result["modulo_six_decoding_accuracy"] = float(
            np.mean((image + clinical) % _CLASS_COUNT == labels)
        )
    return result
