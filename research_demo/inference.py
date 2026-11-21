from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


_BINARY_LEVELS = {"True", "False", "UNK"}
_REGIONS = {
    "ABDOMEN",
    "ARM",
    "BACK",
    "CHEST",
    "EAR",
    "FACE",
    "FOOT",
    "FOREARM",
    "HAND",
    "LIP",
    "NECK",
    "NOSE",
    "SCALP",
    "THIGH",
}
_CATEGORICAL_FIELDS = (
    "grew",
    "itch",
    "changed",
    "elevation",
    "bleed",
    "hurt",
    "region",
)
_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class ProbabilisticModel(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ClinicalInputs:
    grew: str
    itch: str
    changed: str
    elevation: str
    bleed: str
    hurt: str
    region: str
    age: int

    def __post_init__(self) -> None:
        for field in _CATEGORICAL_FIELDS[:-1]:
            if getattr(self, field) not in _BINARY_LEVELS:
                raise ValueError(f"{field} must be True, False, or UNK")
        if self.region not in _REGIONS:
            raise ValueError(f"region must be one of the observed categories")
        if type(self.age) is not int:
            raise ValueError("age must be an integer")

    @property
    def age_out_of_range(self) -> bool:
        return not 6 <= self.age <= 94


@dataclass(frozen=True)
class ClinicalRuntime:
    model: ProbabilisticModel
    columns: tuple[str, ...]
    class_names: tuple[str, ...]
    background: pd.DataFrame
    groups: dict[str, list[int]]
    background_class_counts: tuple[int, ...]


@dataclass(frozen=True)
class ClinicalPrediction:
    probabilities: np.ndarray
    predicted_class_index: int
    grouped_shapley_values: np.ndarray


@dataclass(frozen=True)
class CaseRuntime:
    clinical: ClinicalRuntime
    vit_model: torch.nn.Module
    device: torch.device
    xgb_weights: np.ndarray


@dataclass(frozen=True)
class CasePrediction:
    clinical: ClinicalPrediction
    vit_probabilities: np.ndarray
    vit_predicted_class_index: int
    cwf_probabilities: np.ndarray
    cwf_predicted_class_index: int
    gradcam_map: np.ndarray
    xgb_weight: float
    xgb_weighted_term: float
    vit_weighted_term: float
    cwf_normalization_constant: float


def prepare_uploaded_image(image_bytes: bytes) -> torch.Tensor:
    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("uploaded image must be PNG or JPEG")
            image.load()
            rgb_image = image.convert("RGB")
            image_tensor = _IMAGE_TRANSFORM(rgb_image)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("uploaded image is not a valid PNG or JPEG") from error
    return image_tensor.unsqueeze(0)


def encode_clinical_inputs(
    inputs: ClinicalInputs,
    columns: Sequence[str],
) -> pd.DataFrame:
    saved_columns = list(columns)
    if len(saved_columns) != 40 or len(set(saved_columns)) != len(saved_columns):
        raise ValueError("saved clinical schema must contain 40 unique columns")
    if "age" not in saved_columns:
        raise ValueError("saved clinical schema must contain the age column")

    values = {field: getattr(inputs, field) for field in _CATEGORICAL_FIELDS}
    values["age"] = inputs.age
    encoded = pd.get_dummies(
        pd.DataFrame([values]),
        columns=list(_CATEGORICAL_FIELDS),
        dummy_na=True,
    )
    return encoded.reindex(columns=saved_columns, fill_value=0).astype(np.float32)


def fuse_cwf(
    xgb_probabilities: np.ndarray,
    vit_probabilities: np.ndarray,
    xgb_weights: np.ndarray,
) -> np.ndarray:
    xgb_values = np.asarray(xgb_probabilities, dtype=np.float64)
    vit_values = np.asarray(vit_probabilities, dtype=np.float64)
    weights = np.asarray(xgb_weights, dtype=np.float64)
    if xgb_values.shape != (6,) or vit_values.shape != (6,):
        raise ValueError("branch probabilities must each contain six classes")
    if weights.shape != (6,):
        raise ValueError("CWF weights must contain one value per class")
    if not (
        np.isfinite(xgb_values).all()
        and np.isfinite(vit_values).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("CWF inputs must be finite")
    if (
        np.any((xgb_values < 0) | (xgb_values > 1))
        or np.any((vit_values < 0) | (vit_values > 1))
        or not np.isclose(xgb_values.sum(), 1.0, atol=1e-6, rtol=0.0)
        or not np.isclose(vit_values.sum(), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("branch probabilities must be normalized values in [0, 1]")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("CWF weights must lie in [0, 1]")

    fused = weights * xgb_values + (1.0 - weights) * vit_values
    total = fused.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("CWF probability vector has no positive mass")
    return fused / total


def _validate_probability_matrix(
    probabilities: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (expected_rows, 6):
        raise ValueError("clinical model must return one six-class row per input")
    if (
        not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("clinical model returned invalid class probabilities")
    return values


def load_clinical_runtime(
    project_root: Path,
    metadata_path: Path,
) -> ClinicalRuntime:
    from joblib import load

    from experiments.phase_f.io import EXPECTED_CLASSES, EXPECTED_VARIABLES
    from experiments.phase_f.shapley import build_feature_groups

    project_root = Path(project_root)
    metadata_path = Path(metadata_path)
    outputs = project_root / "outputs"
    model_path = outputs / "xgb_final_v1.joblib"
    split_path = outputs / "split_v1.json"
    explanation_path = outputs / "treeshap_v1.npz"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"clinical metadata is missing: {metadata_path}")
    for path in (model_path, split_path, explanation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required clinical runtime artifact is missing: {path}")

    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    train_ids = split.get("train_img_ids")
    if (
        not isinstance(train_ids, list)
        or not train_ids
        or any(not isinstance(image_id, str) for image_id in train_ids)
        or len(set(train_ids)) != len(train_ids)
    ):
        raise ValueError("split must contain unique training image IDs")

    metadata_ids = pd.read_csv(metadata_path, usecols=["img_id"])["img_id"]
    if metadata_ids.isna().any():
        raise ValueError("metadata contains missing image IDs")
    metadata_ids = metadata_ids.astype(str).tolist()
    if len(set(metadata_ids)) != len(metadata_ids):
        raise ValueError("metadata contains duplicate image IDs")
    missing_ids = sorted(set(train_ids).difference(metadata_ids))
    if missing_ids:
        raise ValueError(f"training split IDs are absent from metadata: {missing_ids[:5]}")

    training_ids = set(train_ids)
    skip_rows = [
        row_number
        for row_number, image_id in enumerate(metadata_ids, start=1)
        if image_id not in training_ids
    ]
    metadata_columns = ["img_id", "diagnostic", *EXPECTED_VARIABLES]
    training_metadata = pd.read_csv(
        metadata_path,
        skiprows=skip_rows,
        usecols=metadata_columns,
    )
    training_metadata["img_id"] = training_metadata["img_id"].astype(str)
    indexed = training_metadata.set_index("img_id")
    train_df = indexed.loc[train_ids].reset_index(drop=True)
    if train_df["diagnostic"].isna().any():
        raise ValueError("training metadata contains missing diagnoses")

    diagnoses = train_df["diagnostic"].replace({"BOD": "SCC"}).astype(str)
    class_indices = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
    unknown_classes = sorted(set(diagnoses).difference(class_indices))
    if unknown_classes:
        raise ValueError(f"training metadata has unknown diagnoses: {unknown_classes}")
    y_train = diagnoses.map(class_indices).to_numpy(dtype=np.int64)

    artifact = load(model_path)
    if not isinstance(artifact, dict) or not {"model", "columns"} <= artifact.keys():
        raise ValueError("saved XGBoost artifact must contain model and columns")
    model = artifact["model"]
    raw_columns = artifact["columns"]
    if (
        not isinstance(raw_columns, (list, tuple))
        or len(raw_columns) != 40
        or any(not isinstance(column, str) for column in raw_columns)
        or len(set(raw_columns)) != len(raw_columns)
    ):
        raise ValueError("saved XGBoost schema must contain 40 unique string columns")
    columns = tuple(raw_columns)
    if getattr(model, "n_features_in_", None) != len(columns):
        raise ValueError("saved XGBoost model feature count does not match its schema")
    if not np.array_equal(
        np.asarray(getattr(model, "classes_", [])),
        np.arange(len(EXPECTED_CLASSES)),
    ):
        raise ValueError("saved XGBoost classes do not match the frozen class order")

    groups = build_feature_groups(columns)
    x_train = pd.get_dummies(
        train_df[EXPECTED_VARIABLES],
        dummy_na=True,
    ).reindex(columns=columns, fill_value=0)
    x_train = x_train.astype(np.float32)
    if x_train.isna().any().any() or not np.isfinite(x_train.to_numpy()).all():
        raise ValueError("reconstructed training matrix contains invalid values")

    with np.load(explanation_path, allow_pickle=False) as saved:
        if "test_background_indices" not in saved.files:
            raise ValueError("saved clinical explanations lack background indices")
        background_indices = np.asarray(saved["test_background_indices"])
    if (
        background_indices.shape != (200,)
        or background_indices.dtype.kind not in "iu"
        or len(np.unique(background_indices)) != 200
        or np.any(background_indices < 0)
        or np.any(background_indices >= len(x_train))
    ):
        raise ValueError("saved background indices must select 200 unique training rows")
    background_class_counts = np.bincount(
        y_train[background_indices],
        minlength=len(EXPECTED_CLASSES),
    )
    expected_counts = np.asarray([34, 34, 33, 33, 33, 33])
    if not np.array_equal(background_class_counts, expected_counts):
        raise ValueError("saved background indices are not the validated class-stratified sample")

    return ClinicalRuntime(
        model=model,
        columns=columns,
        class_names=tuple(EXPECTED_CLASSES),
        background=x_train.iloc[background_indices].reset_index(drop=True),
        groups=groups,
        background_class_counts=tuple(int(value) for value in background_class_counts),
    )


def predict_clinical(
    inputs: ClinicalInputs,
    runtime: ClinicalRuntime,
) -> ClinicalPrediction:
    from experiments.phase_f.shapley import (
        construct_coalition_batch,
        exact_group_shapley,
    )

    encoded = encode_clinical_inputs(inputs, runtime.columns)
    direct_probabilities = _validate_probability_matrix(
        runtime.model.predict_proba(encoded),
        expected_rows=1,
    )[0]
    case = encoded.to_numpy(dtype=np.float32)[0]
    background = runtime.background.to_numpy(dtype=np.float32)
    coalition_count = 1 << len(runtime.groups)
    rows_per_case = len(background)
    candidates = np.empty(
        (coalition_count * rows_per_case, len(runtime.columns)),
        dtype=np.float32,
    )
    for coalition in range(coalition_count):
        start = coalition * rows_per_case
        candidates[start:start + rows_per_case] = construct_coalition_batch(
            case,
            background,
            runtime.groups,
            coalition,
        )

    predictions = _validate_probability_matrix(
        runtime.model.predict_proba(
            pd.DataFrame(candidates, columns=runtime.columns)
        ),
        expected_rows=coalition_count * rows_per_case,
    )
    coalition_values = predictions.reshape(
        coalition_count,
        rows_per_case,
        len(runtime.class_names),
    ).mean(axis=1)
    if not np.allclose(
        coalition_values[-1],
        direct_probabilities,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("full-coalition predictions do not match direct XGBoost output")

    contributions = exact_group_shapley(
        coalition_values,
        n_groups=len(runtime.groups),
    )
    target_class = int(np.argmax(direct_probabilities))
    efficiency_error = abs(
        float(contributions[:, target_class].sum())
        - float(coalition_values[-1, target_class] - coalition_values[0, target_class])
    )
    if efficiency_error > 1e-5:
        raise ValueError(
            f"clinical grouped-Shapley efficiency error is {efficiency_error:.3g}"
        )
    return ClinicalPrediction(
        probabilities=direct_probabilities,
        predicted_class_index=target_class,
        grouped_shapley_values=contributions[:, target_class],
    )


def _load_vit_model(
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    from experiments.phase_f.image_eval import _load_model

    return _load_model(checkpoint_path, n_classes, device)


def load_case_runtime(
    project_root: Path,
    metadata_path: Path,
    device: torch.device | str | None = None,
) -> CaseRuntime:
    project_root = Path(project_root)
    outputs = project_root / "outputs"
    checkpoint_path = outputs / "vit_final_v1.pt"
    fusion_path = outputs / "fusion_v1.json"
    for path in (checkpoint_path, fusion_path):
        if not path.is_file():
            raise FileNotFoundError(f"required image runtime artifact is missing: {path}")

    with fusion_path.open("r", encoding="utf-8") as file:
        fusion = json.load(file)
    if not isinstance(fusion, dict) or fusion.get("selected") != "CWF":
        raise ValueError("saved fusion artifact must select CWF")
    saved_classes = fusion.get("classes")
    if not isinstance(saved_classes, list) or any(
        not isinstance(class_name, str) for class_name in saved_classes
    ):
        raise ValueError("saved fusion artifact must contain the ordered class names")
    try:
        xgb_weights = np.asarray(fusion["w_xgb"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("saved CWF artifact must contain six numeric XGBoost weights") from error
    if (
        xgb_weights.shape != (6,)
        or not np.isfinite(xgb_weights).all()
        or np.any((xgb_weights < 0) | (xgb_weights > 1))
    ):
        raise ValueError("saved CWF artifact must contain six finite weights in [0, 1]")

    clinical_runtime = load_clinical_runtime(project_root, Path(metadata_path))
    if tuple(saved_classes) != clinical_runtime.class_names:
        raise ValueError("saved fusion classes do not match the frozen model class order")

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for ViT inference but is unavailable")
    vit_model = _load_vit_model(
        checkpoint_path,
        n_classes=len(clinical_runtime.class_names),
        device=resolved_device,
    )
    return CaseRuntime(
        clinical=clinical_runtime,
        vit_model=vit_model,
        device=resolved_device,
        xgb_weights=xgb_weights,
    )


def predict_uploaded_case(
    image: torch.Tensor,
    inputs: ClinicalInputs,
    runtime: CaseRuntime,
) -> CasePrediction:
    from experiments.phase_f.image_eval import (
        _gradcam_maps,
        _normalized_model_inputs,
    )

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a prepared torch tensor")
    if image.shape != (1, 3, 224, 224):
        raise ValueError("image must contain one RGB 224x224 case")

    clinical_prediction = predict_clinical(inputs, runtime.clinical)
    normalized_image = _normalized_model_inputs(image, runtime.device)
    runtime.vit_model.eval()
    with torch.inference_mode():
        logits = runtime.vit_model(normalized_image)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.shape != (1, len(runtime.clinical.class_names))
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("ViT model must return one finite logit per saved class")
    vit_probabilities = _validate_probability_matrix(
        torch.softmax(logits, dim=1).cpu().numpy(),
        expected_rows=1,
    )[0]
    cwf_probabilities = fuse_cwf(
        clinical_prediction.probabilities,
        vit_probabilities,
        runtime.xgb_weights,
    )
    cwf_normalization_constant = float(
        (
            runtime.xgb_weights * clinical_prediction.probabilities
            + (1.0 - runtime.xgb_weights) * vit_probabilities
        ).sum()
    )
    cwf_class_index = int(np.argmax(cwf_probabilities))
    gradcam = np.asarray(
        _gradcam_maps(
            runtime.vit_model,
            normalized_image,
            np.asarray([cwf_class_index], dtype=np.int64),
        ),
        dtype=np.float32,
    )
    if (
        gradcam.shape != (1, 14, 14)
        or not np.isfinite(gradcam).all()
        or np.any((gradcam < 0) | (gradcam > 1))
    ):
        raise ValueError("ViT Grad-CAM must return one finite normalized 14x14 map")
    xgb_weight = float(runtime.xgb_weights[cwf_class_index])
    return CasePrediction(
        clinical=clinical_prediction,
        vit_probabilities=vit_probabilities,
        vit_predicted_class_index=int(np.argmax(vit_probabilities)),
        cwf_probabilities=cwf_probabilities,
        cwf_predicted_class_index=cwf_class_index,
        gradcam_map=gradcam[0],
        xgb_weight=xgb_weight,
        xgb_weighted_term=float(
            xgb_weight * clinical_prediction.probabilities[cwf_class_index]
        ),
        vit_weighted_term=float(
            (1.0 - xgb_weight) * vit_probabilities[cwf_class_index]
        ),
        cwf_normalization_constant=cwf_normalization_constant,
    )
