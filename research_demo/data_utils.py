from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_npz_safely(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load readable arrays without unpickling object-valued NPZ members."""
    arrays: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}

    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            try:
                arrays[key] = archive[key]
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise
                skipped[key] = "Object array skipped; pickle loading is disabled."

    return arrays, skipped


def validate_probabilities(
    probabilities: np.ndarray,
    *,
    atol: float = 1e-4,
) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("probabilities must contain only finite numeric values")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities must be within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=atol, rtol=0):
        raise ValueError("probability rows must sum to 1")
    return values.astype(np.float64, copy=False)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str] | tuple[str, ...] | np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(labels)
    names = [str(name) for name in classes]
    probs = validate_probabilities(probabilities)

    if truth.ndim != 1 or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if len(truth) != len(probs):
        raise ValueError("label and probability row counts must match")
    if probs.shape[1] != len(names):
        raise ValueError("probability columns must match the class count")
    if truth.size == 0:
        raise ValueError("metrics require at least one aligned row")
    if np.any(truth < 0) or np.any(truth >= len(names)):
        raise ValueError("labels contain class indices outside the class list")

    predicted = probs.argmax(axis=1)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)

    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )
    row_indices = np.arange(len(truth))
    nll = -np.log(np.clip(probs[row_indices, truth], 1e-12, 1.0)).mean()

    return {
        "accuracy": float(np.mean(predicted == truth)),
        "macro_f1": float(per_class_f1.mean()),
        "nll": float(nll),
        "support": {
            name: int(confusion[index].sum())
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion,
        "predicted": predicted,
    }


def artifact_record(root: Path, relative_path: str, category: str) -> dict[str, Any]:
    path = root / relative_path
    record: dict[str, Any] = {
        "path": relative_path,
        "category": category,
        "exists": path.is_file(),
    }
    if path.is_file():
        stat = path.stat()
        record.update(
            size_bytes=stat.st_size,
            modified_local=datetime.fromtimestamp(stat.st_mtime)
            .astimezone()
            .isoformat(timespec="seconds"),
        )
    return record
