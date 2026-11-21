from __future__ import annotations

import unittest

from experiments.phase_g_v2_1.reproducibility import verify_protected_artifacts


class PhaseGV21PreservationTests(unittest.TestCase):
    def test_v1_v2_protected_artifacts_are_unchanged(self):
        result = verify_protected_artifacts()

        self.assertTrue(result["unchanged"])
        self.assertEqual(result["file_count"], 100)
        self.assertEqual(result["mismatches"], [])

    def test_test_partition_is_not_referenced_by_v21_static_paths(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        package = root / "experiments" / "phase_g_v2_1"
        text = "\n".join([
            (package / "config.py").read_text(encoding="utf-8"),
            (package / "reproducibility.py").read_text(encoding="utf-8"),
            (package / "slic.py").read_text(encoding="utf-8"),
        ])

        self.assertNotIn("test_inputs.npz", text)
        self.assertNotIn("test_predictions.npz", text)
        self.assertNotIn("test_logits", text)
        self.assertNotIn("test_labels", text)


if __name__ == "__main__":
    unittest.main()
