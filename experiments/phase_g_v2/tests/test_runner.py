from __future__ import annotations

import unittest

import numpy as np

from experiments.phase_g_v2.run_validation import (
    flatten_npz_arrays,
    select_validation_patient_ids,
    summarize_faithfulness,
)


class PhaseGV2RunnerTests(unittest.TestCase):
    def test_stratified_selection_uses_only_model_selection_labels(self):
        split = {
            "development_labels": {
                "a0": 0,
                "a1": 0,
                "b0": 1,
                "b1": 1,
            },
            "partition_patient_ids": {
                "validation_model_selection": ["a0", "a1", "b0", "b1"],
                "test": ["test0", "test1"],
            },
        }

        selected = select_validation_patient_ids(split, per_class=1)

        self.assertEqual(selected, ["a0", "b0"])

    def test_selection_rejects_test_partition_and_unavailable_class_count(self):
        split = {
            "development_labels": {"a0": 0, "a1": 0, "b0": 1},
            "partition_patient_ids": {
                "validation_model_selection": ["a0", "a1", "b0", "test0"],
                "test": ["test0"],
            },
        }

        with self.assertRaisesRegex(ValueError, "overlap"):
            select_validation_patient_ids(split, per_class=1)

        split["partition_patient_ids"]["validation_model_selection"] = ["a0", "a1", "b0"]
        with self.assertRaisesRegex(ValueError, "class 1"):
            select_validation_patient_ids(split, per_class=2)

    def test_faithfulness_summary_pairs_each_method_with_patient_controls(self):
        result = {
            "patient_ids": np.asarray(["p0", "p1"]),
            "baseline_names": np.asarray(["background"]),
            "directions": np.asarray(["deletion", "insertion"]),
            "valid_masks": {"method": np.asarray([True, False])},
            "auc": {"method": np.asarray([[[0.2, 0.8]], [[0.4, 0.6]]])},
            "random_auc": np.asarray(
                [
                    [[[0.5, 0.5]], [[0.7, 0.7]]],
                    [[[0.1, 0.9]], [[0.3, 0.9]]],
                ]
            ),
            "center_auc": np.asarray([[[0.6, 0.4]], [[0.2, 0.8]]]),
        }

        summary = summarize_faithfulness(result, replicates=100, seed=13)

        deletion = summary["methods"]["method"]["background"]["deletion"]
        self.assertEqual(deletion["n"], 1)
        self.assertAlmostEqual(deletion["method_mean_auc"], 0.2)
        self.assertAlmostEqual(deletion["random_mean_auc"], 0.6)
        self.assertAlmostEqual(deletion["method_minus_random_mean"], -0.4)
        self.assertAlmostEqual(deletion["method_minus_center_mean"], -0.4)

    def test_npz_serialization_flattens_nested_arrays_without_object_dtype(self):
        arrays = flatten_npz_arrays(
            {"curves": {"method": np.ones((2, 3))}, "count": 2, "optional": None},
            prefix="image_only__",
        )

        self.assertIn("image_only__curves__method", arrays)
        self.assertEqual(arrays["image_only__curves__method"].shape, (2, 3))
        self.assertEqual(int(arrays["image_only__count"]), 2)
        self.assertNotIn("image_only__optional", arrays)
        self.assertTrue(all(not value.dtype.hasobject for value in arrays.values()))


if __name__ == "__main__":
    unittest.main()
