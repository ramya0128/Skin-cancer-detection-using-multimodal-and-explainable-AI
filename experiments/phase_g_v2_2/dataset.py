from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from experiments.phase_g.generate import SyntheticPatient, generate_split
from experiments.phase_g_v2_2.data import (
    make_condition_assignments,
)


class LazySyntheticPatientDataset(Dataset[SyntheticPatient]):
    def __init__(
        self,
        patient_ids: list[str],
        labels: dict[str, int],
        partition: str,
        config: dict[str, Any],
        condition: str,
        *,
        cache_size: int = 64,
        seed: int = 20261119,
    ):
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError("patient IDs must be unique")
        if not patient_ids:
            raise ValueError("lazy dataset cannot be empty")
        if cache_size < 1:
            raise ValueError("cache size must be positive")
        if partition in {"test"}:
            raise ValueError("test rendering is forbidden in the lazy development dataset")
        if condition not in {
            "balanced",
            "image_dominant",
            "clinical_dominant",
            "redundant",
            "conflicting",
            "synergistic",
        }:
            raise ValueError(f"unknown condition: {condition}")
        missing = [patient_id for patient_id in patient_ids if patient_id not in labels]
        if missing:
            raise KeyError(f"missing labels for patient IDs: {missing[:5]}")

        self.patient_ids = tuple(patient_ids)
        self.labels = dict(labels)
        self.partition = partition
        self.config = dict(config)
        self.condition = condition
        self.cache_size = int(cache_size)
        self.seed = int(seed)
        self._cache: OrderedDict[str, SyntheticPatient] = OrderedDict()
        self._assignments = (
            make_condition_assignments(list(patient_ids), condition, seed)
            if condition == "conflicting"
            else None
        )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> SyntheticPatient:
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("lazy dataset index out of range")
        patient_id = self.patient_ids[index]
        cached = self._cache.pop(patient_id, None)
        if cached is not None:
            self._cache[patient_id] = cached
            return cached

        runtime_config = dict(self.config)
        runtime_config["pilot_condition"] = self.condition
        if self.condition == "conflicting":
            from experiments.phase_g_v2_2.render import render_conflicting_patient

            patient = render_conflicting_patient(
                patient_id,
                self.labels[patient_id],
                self.partition,
                runtime_config,
                self._assignments[patient_id],
            )
        else:
            patient = generate_split(
                [patient_id],
                {patient_id: self.labels[patient_id]},
                self.partition,
                runtime_config,
            )[0]

        self._cache[patient_id] = patient
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return patient

    def cache_metrics(self) -> dict[str, int]:
        return {
            "cache_size": self.cache_size,
            "cached_patients": len(self._cache),
        }
