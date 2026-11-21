from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

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
        or not np.isclose(xgb_values.sum(), 1.0, atol=1e-6)
        or not np.isclose(vit_values.sum(), 1.0, atol=1e-6)
    ):
        raise ValueError("branch probabilities must be normalized values in [0, 1]")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("CWF weights must lie in [0, 1]")

    fused = weights * xgb_values + (1.0 - weights) * vit_values
    total = fused.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("CWF probability vector has no positive mass")
    return fused / total
