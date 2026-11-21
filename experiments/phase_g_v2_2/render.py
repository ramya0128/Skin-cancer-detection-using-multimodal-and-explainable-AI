from __future__ import annotations

from experiments.phase_g.generate import SyntheticPatient, render_patient


def render_conflicting_patient(
    patient_id: str,
    label: int,
    partition: str,
    config: dict,
    orientation: str,
) -> SyntheticPatient:
    if orientation not in {"image_correct", "clinical_correct", "none"}:
        raise ValueError(f"invalid conflicting orientation: {orientation}")
    if partition == "test":
        raise ValueError("test rendering is forbidden by the v2.2 condition generator")
    if config.get("pilot_condition") != "conflicting":
        raise ValueError("conflicting rendering requires the conflicting condition config")
    return render_patient(
        patient_id,
        label,
        partition,
        config,
        conflict_orientation=orientation,
    )
