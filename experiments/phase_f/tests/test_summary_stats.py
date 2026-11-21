import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


try:
    summary = importlib.import_module("experiments.phase_f.summarize")
except ModuleNotFoundError as error:
    if error.name != "experiments.phase_f.summarize":
        raise
    summary = None


class SummaryStatisticsTests(unittest.TestCase):
    def require_summary(self):
        self.assertIsNotNone(
            summary,
            "experiments.phase_f.summarize has not been implemented",
        )
        return summary

    def test_cluster_bootstrap_resamples_patients_not_rows(self):
        module = self.require_summary()
        values = np.asarray([0.0] * 100 + [100.0])
        patient_ids = np.asarray(["many-images"] * 100 + ["one-image"])

        lower, upper = module.cluster_bootstrap(
            values,
            patient_ids,
            n_resamples=500,
            seed=42,
        )
        repeat_lower, repeat_upper = module.cluster_bootstrap(
            values,
            patient_ids,
            n_resamples=500,
            seed=42,
        )

        self.assertEqual(float(lower), 0.0)
        self.assertEqual(float(upper), 100.0)
        self.assertEqual(float(repeat_lower), float(lower))
        self.assertEqual(float(repeat_upper), float(upper))

    def test_sign_flip_uses_patient_aggregated_differences(self):
        module = self.require_summary()
        patient_ids = np.asarray(["many-images"] * 100 + ["patient-b", "patient-c"])
        values_a = np.asarray([2.0] * 100 + [-1.0, -1.0])
        values_b = np.zeros_like(values_a)

        p_value = module.paired_cluster_signflip_p(
            values_a,
            values_b,
            patient_ids,
            n_permutations=500,
            seed=42,
        )

        self.assertEqual(p_value, 1.0)

    def test_holm_adjustments_are_monotone_and_not_below_raw_p_values(self):
        module = self.require_summary()
        raw = np.asarray([0.01, 0.04, 0.03])

        adjusted = module.holm_adjust(raw)

        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06], atol=1e-12)
        self.assertTrue(np.all(adjusted >= raw))
        ordered = adjusted[np.argsort(raw)]
        self.assertTrue(np.all(np.diff(ordered) >= 0))

    def test_report_reads_holm_family_count_from_statistics_summary(self):
        module = self.require_summary()
        report_summary = {
            "test_rows": 3,
            "test_patients": 2,
            "class_counts": {"ACK": 1},
            "control": {
                "test_accuracy": 0.5,
                "test_macro_f1": 0.4,
                "rank_spearman_xgb_mean": 0.2,
                "rank_spearman_cwf_mean": 0.3,
            },
            "statistics": {
                "holm_families": {
                    "image_deletion": 48,
                    "image_insertion": 48,
                    "tabular_deletion": 2,
                    "tabular_insertion": 2,
                },
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            module._write_report(path, report_summary, [], [])

            report = path.read_text(encoding="utf-8")

        self.assertIn("4 predeclared Holm families.", report)

    def test_code_hash_manifest_uses_workspace_relative_paths(self):
        module = self.require_summary()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiments" / "phase_f" / "summarize.py"
            source.parent.mkdir(parents=True)
            source.write_text("source\n", encoding="utf-8")

            hashes = module._code_hashes(root, [source])
            expected = hashlib.sha256(source.read_bytes()).hexdigest()

        self.assertEqual(hashes, {"experiments/phase_f/summarize.py": expected})


if __name__ == "__main__":
    unittest.main()
