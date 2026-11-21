from __future__ import annotations

import copy
import json
import tracemalloc
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.phase_g_v2_2.dataset import LazySyntheticPatientDataset


ROOT = Path(__file__).resolve().parents[3]


class PhaseGV22LazyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )

    def test_repeated_requests_are_deterministic_and_cache_bounded(self):
        patient_ids = [f"lazy-{index}" for index in range(8)]
        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        dataset = LazySyntheticPatientDataset(
            patient_ids,
            labels,
            "train",
            self.config,
            "image_dominant",
            cache_size=3,
        )

        first = dataset[0]
        repeated = dataset[0]
        np.testing.assert_array_equal(first.image, repeated.image)
        dataset[1]
        dataset[2]
        dataset[3]
        self.assertLessEqual(dataset.cache_metrics()["cached_patients"], 3)
        rerendered = dataset[0]
        np.testing.assert_array_equal(first.image, rerendered.image)

    def test_dataloader_uses_one_worker_and_batch_level_ids(self):
        patient_ids = [f"lazy-batch-{index}" for index in range(4)]
        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        dataset = LazySyntheticPatientDataset(
            patient_ids,
            labels,
            "train",
            self.config,
            "balanced",
            cache_size=2,
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda values: values,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)

    def test_test_partition_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "test"):
            LazySyntheticPatientDataset(
                ["test-0"],
                {"test-0": 0},
                "test",
                self.config,
                "balanced",
            )

    def test_disposable_smoke_cache_peak_is_measureable(self):
        patient_ids = [f"lazy-memory-{index}" for index in range(12)]
        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        dataset = LazySyntheticPatientDataset(
            patient_ids,
            labels,
            "train",
            self.config,
            "balanced",
            cache_size=4,
        )
        tracemalloc.start()
        for index in range(12):
            dataset[index]
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertGreater(peak, 0)
        self.assertLessEqual(dataset.cache_metrics()["cached_patients"], 4)


if __name__ == "__main__":
    unittest.main()
