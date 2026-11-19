from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.phase_f.io import PROTECTED_INPUTS


EXPECTED_COUNTS_PER_CLASS = {
    "train_per_class": 1400,
    "validation_model_selection_per_class": 150,
    "validation_calibration_per_class": 150,
    "test_per_class": 300,
}
EXPECTED_CONDITIONS = {
    "balanced",
    "image_dominant",
    "clinical_dominant",
    "redundant",
    "conflicting",
    "synergistic",
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("pilot_condition") != "balanced":
        raise ValueError("only the balanced pilot condition is authorized")
    if config.get("authorized_conditions") != ["balanced"]:
        raise ValueError("authorized conditions must contain only balanced")
    if int(config.get("master_seed", -1)) != 20261119:
        raise ValueError("master seed differs from the frozen proposal")
    if config.get("labels", {}).get("values") != [0, 1, 2, 3, 4, 5]:
        raise ValueError("the proposal requires six synthetic labels in fixed order")
    dataset = config.get("dataset", {})
    if int(dataset.get("patient_count", -1)) != 12000:
        raise ValueError("patient count differs from the frozen proposal")
    if dataset.get("class_counts") != EXPECTED_COUNTS_PER_CLASS:
        raise ValueError("partition counts differ from the frozen proposal")

    balanced = config.get("balanced_pilot", {})
    if balanced.get("image_cue_reliability") != 0.85:
        raise ValueError("image cue reliability must be 0.85")
    if balanced.get("clinical_cue_reliability") != 0.85:
        raise ValueError("clinical cue reliability must be 0.85")
    if balanced.get("incorrect_cue_semantics") != "uniform_among_other_five_classes":
        raise ValueError("cue errors must be uniform among the other five classes")
    if balanced.get("cue_noise_independence") != "independent_conditional_on_label":
        raise ValueError("cue errors must be conditionally independent")

    conditions = config.get("conditions", {})
    if set(conditions) != EXPECTED_CONDITIONS:
        raise ValueError("config must define exactly the six predeclared conditions")
    for condition in EXPECTED_CONDITIONS - {"balanced"}:
        if conditions[condition].get("status") != "defined_not_authorized_to_run":
            raise ValueError(f"{condition} must remain unauthorized")
    if conditions["balanced"].get("status") != "pilot_only":
        raise ValueError("balanced must be the sole pilot condition")

    primary = config.get("primary_incremental_prediction_criterion", {})
    if primary.get("nll_effect_threshold") != 0.1:
        raise ValueError("primary NLL practical threshold must be 0.10 nats")
    if primary.get("holm_family_size") != 2 or primary.get("one_sided_alpha") != 0.05:
        raise ValueError("primary NLL multiplicity settings differ from the proposal")
    if primary.get("bootstrap_replicates") != 10000:
        raise ValueError("primary NLL bootstrap count must be 10,000")
    if primary.get("full_accuracy_floor") != 0.8:
        raise ValueError("joint accuracy floor must be 0.80")
    if config.get("output_directory") != (
        "outputs/synthetic_phase_g_v1/pilot_balanced_085_085_v1"
    ):
        raise ValueError("output directory differs from the versioned pilot path")
    if config.get("overwrite_existing_output") is not False:
        raise ValueError("overwriting an existing output directory is prohibited")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_protected_inputs(root: Path) -> dict[str, str]:
    root = Path(root)
    paths = {str(relative): root / relative for relative in PROTECTED_INPUTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing protected inputs: " + ", ".join(missing))
    return {
        relative: _sha256_file(path)
        for relative, path in paths.items()
    }


def prepare_run_dir(
    output_dir: Path,
    config_hash: str,
    source_hashes: dict[str, str],
    resume: bool,
) -> None:
    output_dir = Path(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = {
        "config_sha256": config_hash,
        "source_sha256": dict(sorted(source_hashes.items())),
    }

    if not output_dir.exists():
        if resume:
            raise FileNotFoundError(f"cannot resume missing output directory: {output_dir}")
        output_dir.mkdir(parents=True)
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if not resume:
        raise FileExistsError(
            f"output directory already exists; pass resume only for matching hashes: {output_dir}"
        )
    if not identity_path.is_file():
        raise ValueError(f"resume directory has no run identity: {output_dir}")
    saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if saved_identity != identity:
        raise ValueError("resume config or source hashes do not match")
