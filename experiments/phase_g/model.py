from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vision_transformer import VisionTransformer

from experiments.phase_g.generate import SyntheticPatient


MODEL_NAMES = ("image_only", "clinical_only", "joint")


@dataclass
class TrainingResult:
    model: nn.Module
    selected_epoch: int
    history: list[dict[str, float]]
    checkpoint_path: Path


class _PatientDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, patients: Sequence[SyntheticPatient]):
        if not patients:
            raise ValueError("model dataset cannot be empty")
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patient = self.patients[index]
        image = torch.from_numpy(patient.image.transpose(2, 0, 1).copy()).float().div_(255.0)
        category = torch.tensor(patient.clinical_value, dtype=torch.long)
        label = torch.tensor(patient.label, dtype=torch.long)
        return image, category, label


class _ImageOnlyModel(nn.Module):
    def __init__(self, image_encoder: VisionTransformer, feature_dim: int):
        super().__init__()
        self.image_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.image_encoder(images))


class _ClinicalOnlyModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(6, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, 6)

    def forward(self, categories: torch.Tensor) -> torch.Tensor:
        if categories.dtype != torch.long:
            raise TypeError("clinical categories must be integer indices")
        return self.classifier(self.activation(self.hidden(self.embedding(categories))))


class _JointModel(nn.Module):
    def __init__(
        self,
        image_encoder: VisionTransformer,
        image_dim: int,
        clinical_embedding_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.clinical_embedding = nn.Embedding(6, clinical_embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + clinical_embedding_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 6),
        )

    def forward(self, images: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        if categories.dtype != torch.long:
            raise TypeError("clinical categories must be integer indices")
        image_features = self.image_encoder(images)
        clinical_features = self.clinical_embedding(categories)
        return self.fusion(torch.cat((image_features, clinical_features), dim=1))


def _build_image_encoder(config: dict[str, Any]) -> VisionTransformer:
    if config["initialization"] != "scratch" or config["pretrained_weights"] is not False:
        raise ValueError("Phase G models must be initialized from scratch")
    image = config["image_encoder"]
    encoder = VisionTransformer(
        image_size=int(image["image_size"]),
        patch_size=int(image["patch_size"]),
        num_layers=int(image["encoder_layers"]),
        num_heads=int(image["attention_heads"]),
        hidden_dim=int(image["hidden_dimension"]),
        mlp_dim=int(image["mlp_dimension"]),
        num_classes=6,
    )
    encoder.heads = nn.Identity()
    return encoder


def build_model(kind: str, config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    image_dim = int(model_config["image_encoder"]["hidden_dimension"])
    clinical_embedding_dim = int(model_config["clinical_encoder"]["embedding_dimension"])

    if kind == "image_only":
        return _ImageOnlyModel(_build_image_encoder(model_config), image_dim)
    if kind == "clinical_only":
        clinical = model_config["clinical_only"]["architecture"]
        if clinical != "clinical_embedding_16_mlp_32_gelu_six_class_head":
            raise ValueError(f"unsupported clinical-only architecture: {clinical}")
        return _ClinicalOnlyModel(clinical_embedding_dim, hidden_dim=32)
    if kind == "joint":
        if model_config["joint_auxiliary_modality_classification_losses"] is not False:
            raise ValueError("joint training must not use auxiliary modality losses")
        fusion = model_config["fusion"]
        return _JointModel(
            _build_image_encoder(model_config),
            image_dim=image_dim,
            clinical_embedding_dim=clinical_embedding_dim,
            fusion_hidden_dim=int(fusion["hidden_dimension"]),
            dropout=float(fusion["dropout"]),
        )
    raise ValueError(f"unknown model kind: {kind}")


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def _model_logits(
    model: nn.Module,
    kind: str,
    images: torch.Tensor,
    categories: torch.Tensor,
) -> torch.Tensor:
    if kind == "image_only":
        return model(images)
    if kind == "clinical_only":
        return model(categories)
    if kind == "joint":
        return model(images, categories)
    raise ValueError(f"unknown model kind: {kind}")


def _evaluate_loss(
    model: nn.Module,
    kind: str,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, categories, labels in data_loader:
            images = images.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = _model_logits(model, kind, images, categories)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            total_count += int(labels.numel())
    return total_loss / total_count


def train_model(
    kind: str,
    train_data: list[SyntheticPatient],
    selection_data: list[SyntheticPatient],
    config: dict[str, Any],
) -> TrainingResult:
    if kind not in MODEL_NAMES:
        raise ValueError(f"unknown model kind: {kind}")
    if not train_data or not selection_data:
        raise ValueError("training and model-selection data must be nonempty")
    train_ids = {patient.patient_id for patient in train_data}
    selection_ids = {patient.patient_id for patient in selection_data}
    if train_ids.intersection(selection_ids):
        raise ValueError("training and model-selection patient IDs overlap")
    if any(patient.partition != "train" for patient in train_data):
        raise ValueError("train_data contains a non-training patient")
    if any(patient.partition != "validation_model_selection" for patient in selection_data):
        raise ValueError("selection_data contains a non-selection patient")

    training = config["model"]["training"]
    seed = int(training["model_initialization_seeds"][kind])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind, config).to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _PatientDataset(train_data),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    selection_loader = DataLoader(
        _PatientDataset(selection_data),
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    checkpoint_path = (
        Path(config["output_directory"]) / "checkpoints" / f"{kind}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    maximum_epochs = int(training["maximum_epochs"])
    patience = int(training["early_stopping"]["patience"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, categories, labels in train_loader:
            images = images.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model, kind, images, categories)
            loss = classification_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * int(labels.numel())
            total_count += int(labels.numel())

        validation_loss = _evaluate_loss(model, kind, selection_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_nll": total_loss / total_count,
                "selection_nll": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training ended without a finite model-selection checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "kind": kind,
            "seed": seed,
            "selected_epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint_path,
    )
    model.to("cpu")
    return TrainingResult(
        model=model,
        selected_epoch=best_epoch,
        history=history,
        checkpoint_path=checkpoint_path,
    )


def _validate_logits_labels(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 6:
        raise ValueError("logits must have shape (N, 6)")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= 6)):
        raise ValueError("logits or labels contain invalid values")
    return logits, labels


def _numpy_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(scaled).sum(axis=1))
    return float(np.mean(log_normalizer - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits, labels = _validate_logits_labels(logits, labels)
    result = minimize_scalar(
        lambda log_temperature: _numpy_nll(logits, labels, math.exp(log_temperature)),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    temperature = math.exp(float(result.x))
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature optimization returned a non-positive value")
    return temperature


def predict_logits(
    model: nn.Module,
    data: list[SyntheticPatient],
    kind: str,
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> np.ndarray:
    if not data:
        raise ValueError("prediction data cannot be empty")
    if kind not in MODEL_NAMES:
        raise ValueError(f"unknown model kind: {kind}")
    prediction_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(prediction_device)
    model.eval()
    loader = DataLoader(
        _PatientDataset(data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=prediction_device.type == "cuda",
    )
    outputs = []
    with torch.no_grad():
        for images, categories, _ in loader:
            logits = _model_logits(
                model,
                kind,
                images.to(prediction_device, non_blocking=True),
                categories.to(prediction_device, non_blocking=True),
            )
            outputs.append(logits.detach().cpu().numpy())
    model.to("cpu")
    return np.concatenate(outputs, axis=0)


def predict_probabilities(
    model: nn.Module,
    data: list[SyntheticPatient],
    temperature: float,
) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(model, _ClinicalOnlyModel):
        kind = "clinical_only"
    elif isinstance(model, _JointModel):
        kind = "joint"
    elif isinstance(model, _ImageOnlyModel):
        kind = "image_only"
    else:
        raise TypeError(f"unsupported Phase G model: {type(model).__name__}")
    logits = predict_logits(model, data, kind)
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
