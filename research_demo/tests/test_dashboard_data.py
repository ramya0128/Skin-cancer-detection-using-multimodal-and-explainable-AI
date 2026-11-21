import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_demo.data_utils import classification_metrics, load_npz_safely


class SafeArtifactLoadingTests(unittest.TestCase):
    def test_skips_object_arrays_without_blocking_numeric_artifact_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed.npz"
            np.savez(
                path,
                probabilities=np.asarray([[0.7, 0.3]], dtype=np.float32),
                object_metadata=np.asarray(["metadata"], dtype=object),
            )

            arrays, skipped = load_npz_safely(path)

        np.testing.assert_allclose(arrays["probabilities"], [[0.7, 0.3]])
        self.assertIn("object_metadata", skipped)


class ClassificationMetricsTests(unittest.TestCase):
    def test_reports_metrics_and_support_from_aligned_saved_probabilities(self):
        labels = np.asarray([0, 1, 1])
        probabilities = np.asarray(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]],
            dtype=np.float64,
        )

        result = classification_metrics(labels, probabilities, ["A", "B"])

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(
            result["nll"],
            -(math.log(0.9) + math.log(0.8) + math.log(0.2)) / 3,
        )
        self.assertEqual(result["support"], {"A": 1, "B": 2})

    def test_rejects_misaligned_or_invalid_probability_arrays(self):
        with self.assertRaisesRegex(ValueError, "row counts"):
            classification_metrics(
                np.asarray([0, 1]),
                np.asarray([[0.8, 0.2]]),
                ["A", "B"],
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            classification_metrics(
                np.asarray([0]),
                np.asarray([[np.nan, 1.0]]),
                ["A", "B"],
            )


if __name__ == "__main__":
    unittest.main()
