from __future__ import annotations

import numpy as np


CLASS_COUNT = 6


def _draw_other_class(label: int, rng: np.random.Generator) -> int:
    draw = int(rng.integers(0, CLASS_COUNT - 1))
    return draw + 1 if draw >= label else draw


def _corrupt_class(
    label: int,
    reliability: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < reliability:
        return label
    return _draw_other_class(label, rng)


def make_conflicting_condition_assignments(
    patient_ids: list[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    if len(patient_ids) % 4:
        raise ValueError("conflicting population requires a patient count divisible by four")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient IDs must be unique")

    count = len(patient_ids)
    shuffled = np.asarray(patient_ids, dtype=object)[rng.permutation(count)]
    assignments = {patient_id: "none" for patient_id in patient_ids}
    conflict_count = count // 2
    image_correct_count = conflict_count // 2

    for patient_id in shuffled[:image_correct_count]:
        assignments[str(patient_id)] = "image_correct"
    for patient_id in shuffled[image_correct_count:conflict_count]:
        assignments[str(patient_id)] = "clinical_correct"
    return assignments


def generate_conflicting_cues(
    label: int,
    orientation: str,
    rng: np.random.Generator,
    clinical_rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    if label not in range(CLASS_COUNT):
        raise ValueError("label must lie in [0, 5]")
    if orientation not in {"none", "image_correct", "clinical_correct"}:
        raise ValueError("unknown conflicting orientation")
    if clinical_rng is None:
        clinical_rng = rng
    if orientation == "image_correct":
        return int(label), _draw_other_class(label, clinical_rng)
    if orientation == "clinical_correct":
        return _draw_other_class(label, rng), int(label)
    return (
        _corrupt_class(label, 0.85, rng),
        _corrupt_class(label, 0.85, clinical_rng),
    )
