from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


CLASS_COUNT = 6
PARTITION_COUNTS_PER_CLASS = {
    "train": 1400,
    "validation_model_selection": 150,
    "validation_calibration": 150,
    "test": 300,
}
GLYPHS = (
    ("1100", "0000", "0000", "0000"),
    ("0011", "0000", "0000", "0000"),
    ("0000", "1100", "0000", "0000"),
    ("0000", "0011", "0000", "0000"),
    ("0000", "0000", "1100", "0000"),
    ("0000", "0000", "0011", "0000"),
)
RENDER_STREAMS = (
    "lesion_geometry",
    "lesion_texture",
    "background",
    "marker_location",
    "image_cue_noise",
    "clinical_cue_noise",
    "explanation_baselines",
)


@dataclass(frozen=True)
class SyntheticPatient:
    patient_id: str
    label: int
    partition: str
    image: np.ndarray
    image_cue: int
    clinical_value: int
    background_canvas: np.ndarray
    lesion_mask: np.ndarray
    marker_mask: np.ndarray
    render_metadata: dict[str, Any]


def _seed(master_seed: int, patient_id: str, stream_name: str) -> int:
    payload = f"{master_seed}:{patient_id}:{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _global_rng(master_seed: int, stream_name: str) -> np.random.Generator:
    return np.random.default_rng(_seed(master_seed, "GLOBAL", stream_name))


def _stream_rngs(master_seed: int, patient_id: str) -> tuple[dict[str, np.random.Generator], dict[str, int]]:
    seeds = {
        stream_name: _seed(master_seed, patient_id, stream_name)
        for stream_name in RENDER_STREAMS
    }
    return (
        {name: np.random.default_rng(seed) for name, seed in seeds.items()},
        seeds,
    )


def _draw_other_class(label: int, rng: np.random.Generator) -> int:
    draw = int(rng.integers(0, CLASS_COUNT - 1))
    return draw + 1 if draw >= label else draw


def _corrupt_class(label: int, reliability: float, rng: np.random.Generator) -> int:
    if rng.random() < reliability:
        return label
    return _draw_other_class(label, rng)


def make_conflicting_assignments(
    patient_ids: list[str], rng: np.random.Generator
) -> dict[str, str | None]:
    if len(patient_ids) % 4:
        raise ValueError("conflicting assignment requires a patient count divisible by four")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("conflicting assignment requires unique patient IDs")

    n_conflict = len(patient_ids) // 2
    shuffled = np.asarray(patient_ids, dtype=object)[rng.permutation(len(patient_ids))]
    assignments: dict[str, str | None] = {patient_id: None for patient_id in patient_ids}
    conflict_ids = shuffled[:n_conflict].tolist()
    rng.shuffle(conflict_ids)
    n_per_orientation = n_conflict // 2
    for patient_id in conflict_ids[:n_per_orientation]:
        assignments[str(patient_id)] = "image"
    for patient_id in conflict_ids[n_per_orientation:]:
        assignments[str(patient_id)] = "clinical"
    return assignments


def generate_cues(
    label: int,
    condition: str,
    rngs: dict[str, np.random.Generator],
    conflict_orientation: str | None = None,
) -> tuple[int, int]:
    if label not in range(CLASS_COUNT):
        raise ValueError(f"label must be in [0, {CLASS_COUNT - 1}]")
    if condition not in {
        "balanced",
        "image_dominant",
        "clinical_dominant",
        "redundant",
        "conflicting",
        "synergistic",
    }:
        raise ValueError(f"unknown condition: {condition}")

    image_rng = rngs["image_cue_noise"]
    clinical_rng = rngs["clinical_cue_noise"]
    if condition == "balanced":
        if conflict_orientation is not None:
            raise ValueError("conflict orientation is only valid for conflicting condition")
        return (
            _corrupt_class(label, 0.85, image_rng),
            _corrupt_class(label, 0.85, clinical_rng),
        )
    if condition == "image_dominant":
        return (
            _corrupt_class(label, 0.95, image_rng),
            _corrupt_class(label, 0.60, clinical_rng),
        )
    if condition == "clinical_dominant":
        return (
            _corrupt_class(label, 0.60, image_rng),
            _corrupt_class(label, 0.95, clinical_rng),
        )
    if condition == "redundant":
        shared = _corrupt_class(label, 0.85, rngs["shared"])
        return shared, shared
    if condition == "conflicting":
        if conflict_orientation == "image":
            return label, _draw_other_class(label, clinical_rng)
        if conflict_orientation == "clinical":
            return _draw_other_class(label, image_rng), label
        if conflict_orientation is not None:
            raise ValueError("conflict orientation must be 'image', 'clinical', or None")
        return (
            _corrupt_class(label, 0.85, image_rng),
            _corrupt_class(label, 0.85, clinical_rng),
        )

    image_share = int(rngs["shared"].integers(0, CLASS_COUNT))
    clinical_share = (label - image_share) % CLASS_COUNT
    image_share = _corrupt_class(
        image_share, 0.85, rngs["image_share_noise"]
    )
    clinical_share = _corrupt_class(
        clinical_share, 0.85, rngs["clinical_share_noise"]
    )
    return image_share, clinical_share


def make_patient_splits(
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    dataset = config["dataset"]
    class_counts = dataset["class_counts"]
    master_seed = int(config["master_seed"])
    labels = np.repeat(np.asarray(config["labels"]["values"], dtype=np.int64), 2000)
    split_rng = _global_rng(master_seed, "split")
    split_rng.shuffle(labels)

    opaque_rng = _global_rng(master_seed, "opaque_patient_id")
    id_bytes = opaque_rng.integers(0, 256, size=(len(labels), 12), dtype=np.uint8)
    patient_ids = [f"p{row.tobytes().hex()}" for row in id_bytes]
    label_by_id = {
        patient_id: int(label) for patient_id, label in zip(patient_ids, labels, strict=True)
    }

    splits = {name: [] for name in PARTITION_COUNTS_PER_CLASS}
    for label in config["labels"]["values"]:
        class_ids = [patient_id for patient_id in patient_ids if label_by_id[patient_id] == label]
        split_rng.shuffle(class_ids)
        start = 0
        for partition, count_key in (
            ("train", "train_per_class"),
            ("validation_model_selection", "validation_model_selection_per_class"),
            ("validation_calibration", "validation_calibration_per_class"),
            ("test", "test_per_class"),
        ):
            count = int(class_counts[count_key])
            splits[partition].extend(class_ids[start : start + count])
            start += count
        if start != len(class_ids):
            raise ValueError(f"class {label} partition counts do not sum to class size")

    if len(label_by_id) != int(dataset["patient_count"]):
        raise ValueError("generated patient ID count does not match config")
    return splits, label_by_id


def _upsample_grid(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    source = Image.fromarray(np.asarray(grid, dtype=np.float32))
    return np.asarray(
        source.resize(size, resample=Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def _textured_canvas(
    base_rgb: tuple[int, int, int],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    channels = []
    for base in base_rgb:
        grid = rng.normal(0.0, noise_std, size=(16, 16))
        channels.append(np.clip(base + _upsample_grid(grid, (224, 224)), 0, 255))
    return np.rint(np.stack(channels, axis=2)).astype(np.uint8)


def _make_lesion(
    geometry_rng: np.random.Generator,
    texture_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center_x = float(geometry_rng.uniform(96, 128))
    center_y = float(geometry_rng.uniform(96, 128))
    radius_x = float(geometry_rng.uniform(36, 52))
    radius_y = float(geometry_rng.uniform(36, 52))
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    radial_scale = np.ones_like(theta)
    harmonics = []
    for harmonic in (2, 3, 4, 5):
        amplitude = float(geometry_rng.uniform(-0.02, 0.02))
        phase = float(geometry_rng.uniform(0.0, 2.0 * math.pi))
        radial_scale += amplitude * np.sin(harmonic * theta + phase)
        harmonics.append({"harmonic": harmonic, "amplitude": amplitude, "phase": phase})

    points = [
        (
            center_x + radius_x * radial_scale[index] * math.cos(angle),
            center_y + radius_y * radial_scale[index] * math.sin(angle),
        )
        for index, angle in enumerate(theta)
    ]
    mask_image = Image.new("L", (224, 224), 0)
    ImageDraw.Draw(mask_image).polygon(points, fill=1)
    lesion_mask = np.asarray(mask_image, dtype=np.uint8).astype(bool)

    color = texture_rng.integers(100, 191, size=3, dtype=np.uint8)
    texture = np.stack(
        [
            _upsample_grid(texture_rng.normal(0.0, 5.0, size=(16, 16)), (224, 224))
            for _ in range(3)
        ],
        axis=2,
    )
    lesion = np.rint(np.clip(color.reshape(1, 1, 3) + texture, 0, 255)).astype(np.uint8)
    metadata = {
        "center": [center_x, center_y],
        "radii": [radius_x, radius_y],
        "harmonics": harmonics,
        "base_rgb": color.tolist(),
    }
    return lesion_mask, lesion, metadata


def _make_glyph(class_index: int) -> np.ndarray:
    glyph = np.empty((16, 16, 3), dtype=np.uint8)
    code = GLYPHS[class_index]
    for row in range(4):
        for column in range(4):
            value = 0 if code[row][column] == "1" else 255
            glyph[row * 4 : (row + 1) * 4, column * 4 : (column + 1) * 4] = value
    return glyph


def render_patient(
    patient_id: str,
    label: int,
    partition: str,
    config: dict[str, Any],
) -> SyntheticPatient:
    if partition not in PARTITION_COUNTS_PER_CLASS:
        raise ValueError(f"unknown partition: {partition}")
    if label not in config["labels"]["values"]:
        raise ValueError(f"label is not configured: {label}")

    master_seed = int(config["master_seed"])
    rngs, stream_seeds = _stream_rngs(master_seed, patient_id)
    background = _textured_canvas(
        tuple(config["render"]["background"]["base_rgb"]),
        float(config["render"]["background"]["texture_noise_std_per_channel"]),
        rngs["background"],
    )
    lesion_mask, lesion_image, lesion_metadata = _make_lesion(
        rngs["lesion_geometry"], rngs["lesion_texture"]
    )
    image = background.copy()
    image[lesion_mask] = lesion_image[lesion_mask]

    condition = str(config["pilot_condition"])
    cue_rngs = {
        "image_cue_noise": rngs["image_cue_noise"],
        "clinical_cue_noise": rngs["clinical_cue_noise"],
        "shared": rngs["image_cue_noise"],
        "image_share_noise": rngs["image_cue_noise"],
        "clinical_share_noise": rngs["clinical_cue_noise"],
    }
    image_cue, clinical_value = generate_cues(label, condition, cue_rngs)

    marker_options = config["render"]["exterior_marker"]["allowed_top_left_pixels"]
    marker_y, marker_x = marker_options[
        int(rngs["marker_location"].integers(0, len(marker_options)))
    ]
    image[marker_y : marker_y + 16, marker_x : marker_x + 16] = _make_glyph(image_cue)
    marker_mask = np.zeros((224, 224), dtype=bool)
    marker_mask[marker_y : marker_y + 16, marker_x : marker_x + 16] = True
    marker_distance = ndimage.distance_transform_edt(~lesion_mask)[marker_mask].min()
    minimum_gap = float(config["render"]["exterior_marker"]["minimum_gap_from_lesion_pixels"])
    if marker_distance < minimum_gap:
        raise RuntimeError(
            f"generated marker-lesion gap {marker_distance} is below {minimum_gap} pixels"
        )

    replacement_image: list[int] = []
    replacement_clinical: list[int] = []
    if partition == "test":
        replacement_rng = rngs["explanation_baselines"]
        draws = int(config["cue_replacements"]["draws_per_test_patient"])
        replacement_image = replacement_rng.integers(0, CLASS_COUNT, size=draws).tolist()
        replacement_clinical = replacement_rng.integers(0, CLASS_COUNT, size=draws).tolist()

    metadata = {
        "marker_top_left": [int(marker_y), int(marker_x)],
        "lesion": lesion_metadata,
        "stream_seeds": stream_seeds,
        "replacement_image_cues": replacement_image,
        "replacement_clinical_cues": replacement_clinical,
    }
    return SyntheticPatient(
        patient_id=patient_id,
        label=int(label),
        partition=partition,
        image=image,
        image_cue=image_cue,
        clinical_value=clinical_value,
        background_canvas=background,
        lesion_mask=lesion_mask,
        marker_mask=marker_mask,
        render_metadata=metadata,
    )


def generate_split(
    patient_ids: list[str],
    labels: dict[str, int],
    partition: str,
    config: dict[str, Any],
) -> list[SyntheticPatient]:
    if partition not in PARTITION_COUNTS_PER_CLASS:
        raise ValueError(f"unknown partition: {partition}")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient IDs must be unique within a partition")
    missing = [patient_id for patient_id in patient_ids if patient_id not in labels]
    if missing:
        raise KeyError(f"missing labels for patient IDs: {missing[:5]}")
    return [
        render_patient(patient_id, labels[patient_id], partition, config)
        for patient_id in patient_ids
    ]
