from __future__ import annotations

import math


CLASS_COUNT = 6


def _joint_metrics(joint: dict[tuple[int, int, int], float]) -> dict[str, float]:
    accuracy = 0.0
    nll = 0.0
    for image_cue in range(CLASS_COUNT):
        for clinical_cue in range(CLASS_COUNT):
            observation = sum(
                joint[(label, image_cue, clinical_cue)]
                for label in range(CLASS_COUNT)
            )
            if observation <= 0:
                continue
            posterior = [
                joint[(label, image_cue, clinical_cue)] / observation
                for label in range(CLASS_COUNT)
            ]
            accuracy += observation * max(posterior)
            for label in range(CLASS_COUNT):
                nll -= joint[(label, image_cue, clinical_cue)] * math.log(
                    posterior[label]
                )
    return {"accuracy": accuracy, "nll": nll}


def _independent_baseline(
    image_reliability: float,
    clinical_reliability: float,
) -> dict[str, float]:
    image_error = (1.0 - image_reliability) / (CLASS_COUNT - 1)
    clinical_error = (1.0 - clinical_reliability) / (CLASS_COUNT - 1)
    joint = {}
    for label in range(CLASS_COUNT):
        for image_cue in range(CLASS_COUNT):
            for clinical_cue in range(CLASS_COUNT):
                image_probability = (
                    image_reliability
                    if image_cue == label
                    else image_error
                )
                clinical_probability = (
                    clinical_reliability
                    if clinical_cue == label
                    else clinical_error
                )
                joint[(label, image_cue, clinical_cue)] = (
                    image_probability * clinical_probability / CLASS_COUNT
                )
    return _joint_metrics(joint)


def _redundant_baseline() -> dict[str, float]:
    reliability = 0.85
    error = 0.03
    joint = {}
    for label in range(CLASS_COUNT):
        for image_cue in range(CLASS_COUNT):
            for clinical_cue in range(CLASS_COUNT):
                joint[(label, image_cue, clinical_cue)] = (
                    (reliability if image_cue == label else error) / CLASS_COUNT
                    if clinical_cue == image_cue
                    else 0.0
                )
    return _joint_metrics(joint)


def _conflicting_baseline() -> dict[str, float]:
    reliability = 0.85
    error = 0.03
    joint = {}
    for label in range(CLASS_COUNT):
        for image_cue in range(CLASS_COUNT):
            for clinical_cue in range(CLASS_COUNT):
                independent = 0.5 * (
                    (reliability if image_cue == label else error)
                    * (reliability if clinical_cue == label else error)
                )
                image_correct = (
                    0.25
                    if image_cue == label and clinical_cue != label
                    else 0.0
                )
                image_correct = (
                    image_correct * (1.0 / 5.0)
                    if image_cue == label and clinical_cue != label
                    else 0.0
                )
                clinical_correct = (
                    0.25
                    if clinical_cue == label and image_cue != label
                    else 0.0
                )
                clinical_correct = (
                    clinical_correct * (1.0 / 5.0)
                    if clinical_cue == label and image_cue != label
                    else 0.0
                )
                joint[(label, image_cue, clinical_cue)] = (
                    independent + image_correct + clinical_correct
                ) / CLASS_COUNT
    return _joint_metrics(joint)


def _synergistic_baseline() -> dict[str, float]:
    reliability = 0.85
    error = 0.03
    joint = {}
    for label in range(CLASS_COUNT):
        for observed_image_share in range(CLASS_COUNT):
            for observed_clinical_share in range(CLASS_COUNT):
                probability = 0.0
                for latent_u in range(CLASS_COUNT):
                    latent_v = (label - latent_u) % CLASS_COUNT
                    probability += (
                        (
                            reliability
                            if observed_image_share == latent_u
                            else error
                        )
                        * (
                            reliability
                            if observed_clinical_share == latent_v
                            else error
                        )
                        / (CLASS_COUNT * CLASS_COUNT)
                    )
                joint[(label, observed_image_share, observed_clinical_share)] = probability
    return _joint_metrics(joint)


def calculate_exact_baselines() -> dict[str, dict[str, float]]:
    return {
        "balanced": _independent_baseline(0.85, 0.85),
        "image_dominant": _independent_baseline(0.95, 0.60),
        "clinical_dominant": _independent_baseline(0.60, 0.95),
        "redundant": _redundant_baseline(),
        "conflicting": _conflicting_baseline(),
        "synergistic": _synergistic_baseline(),
    }
