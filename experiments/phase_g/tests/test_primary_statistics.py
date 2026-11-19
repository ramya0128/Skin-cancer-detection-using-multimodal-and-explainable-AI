import unittest

import numpy as np

from experiments.phase_g.evaluate import (
    centered_bootstrap_p_value,
    holm_adjusted_p_values,
    holm_lower_bounds,
    oracle_metrics,
    paired_nll_gains,
    primary_nll_inference,
)
from experiments.phase_g.model import fit_temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class PrimaryStatisticsTests(unittest.TestCase):
    def test_oracle_metrics_match_the_six_class_symmetric_channel(self):
        metrics = oracle_metrics()

        self.assertAlmostEqual(metrics["joint_accuracy"], 0.85, places=10)
        self.assertAlmostEqual(metrics["unimodal_accuracy"], 0.85, places=10)
        self.assertAlmostEqual(metrics["joint_nll"], 0.302653, places=6)
        self.assertAlmostEqual(metrics["unimodal_nll"], 0.664125, places=5)
        self.assertAlmostEqual(metrics["ideal_nll_gain"], 0.361472, places=6)

    def test_paired_nll_gain_sign_is_unimodal_minus_joint(self):
        labels = np.asarray([0, 1])
        image_only = np.full((2, 6), 0.02)
        clinical_only = np.full((2, 6), 0.02)
        joint = np.full((2, 6), 0.01)
        image_only[0, 0], image_only[1, 1] = 0.80, 0.50
        clinical_only[0, 0], clinical_only[1, 1] = 0.70, 0.60
        joint[0, 0], joint[1, 1] = 0.90, 0.70
        image_only /= image_only.sum(axis=1, keepdims=True)
        clinical_only /= clinical_only.sum(axis=1, keepdims=True)
        joint /= joint.sum(axis=1, keepdims=True)

        gains = paired_nll_gains(
            {
                "image_only": image_only,
                "clinical_only": clinical_only,
                "joint": joint,
            },
            labels,
        )
        np.testing.assert_allclose(
            gains["image_only_minus_joint"],
            np.log(joint[np.arange(2), labels] / image_only[np.arange(2), labels]),
        )
        np.testing.assert_allclose(
            gains["clinical_only_minus_joint"],
            np.log(joint[np.arange(2), labels] / clinical_only[np.arange(2), labels]),
        )

    def test_temperature_is_positive_and_uses_only_supplied_calibration_rows(self):
        calibration_logits = np.asarray(
            [[4.0, 0.0, -1.0, -2.0, -3.0, -4.0], [0.0, 3.0, -1.0, -2.0, -3.0, -4.0]]
        )
        calibration_labels = np.asarray([0, 1])

        temperature = fit_temperature(calibration_logits, calibration_labels)

        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_centered_bootstrap_tail_uses_add_one_formula(self):
        estimates = np.asarray([1.0, 1.1, 0.9])

        self.assertEqual(centered_bootstrap_p_value(estimates, 1.0, 0.8), 0.25)
        self.assertEqual(centered_bootstrap_p_value(estimates, 1.0, 0.9), 0.5)

    def test_holm_adjustment_is_monotone_and_restores_original_order(self):
        np.testing.assert_allclose(
            holm_adjusted_p_values(np.asarray([0.04, 0.01, 0.03])),
            np.asarray([0.06, 0.03, 0.06]),
        )
        np.testing.assert_allclose(
            holm_adjusted_p_values(np.asarray([0.001, 0.2])),
            np.asarray([0.002, 0.2]),
        )

    def test_holm_inversion_returns_finite_simultaneous_lower_bounds(self):
        estimates = {
            "image_only_minus_joint": 1.0,
            "clinical_only_minus_joint": 1.0,
        }
        bootstrap_estimates = {
            "image_only_minus_joint": np.linspace(0.90, 1.10, 1001),
            "clinical_only_minus_joint": np.linspace(0.95, 1.05, 1001),
        }

        bounds = holm_lower_bounds(bootstrap_estimates, estimates, alpha=0.05)

        self.assertTrue(all(np.isfinite(value) for value in bounds.values()))
        self.assertTrue(all(0.8 < value < 1.0 for value in bounds.values()))

    def test_primary_bootstrap_is_paired_reproducible_and_returns_both_gates(self):
        labels = np.arange(6, dtype=np.int64)
        calibration_logits = {}
        test_logits = {}
        probabilities = {}
        for name, strength in (
            ("image_only", 1.8),
            ("clinical_only", 1.6),
            ("joint", 2.4),
        ):
            logits = np.full((6, 6), -1.0)
            logits[np.arange(6), labels] = strength
            calibration_logits[name] = logits.copy()
            test_logits[name] = logits.copy()
            probabilities[name] = softmax(logits)

        kwargs = dict(
            probabilities=probabilities,
            test_logits=test_logits,
            test_labels=labels,
            calibration_logits=calibration_logits,
            calibration_labels=labels,
            test_classes=labels,
            replicates=25,
            seed=717,
            threshold=0.1,
        )
        result = primary_nll_inference(**kwargs)
        repeated = primary_nll_inference(**kwargs)

        self.assertEqual(result["bootstrap_estimates"]["image_only_minus_joint"].shape, (25,))
        self.assertEqual(
            result["bootstrap_estimates"]["clinical_only_minus_joint"].shape, (25,)
        )
        np.testing.assert_array_equal(
            result["bootstrap_estimates"]["image_only_minus_joint"],
            repeated["bootstrap_estimates"]["image_only_minus_joint"],
        )
        self.assertEqual(set(result["raw_p_values"]), {"image_only_minus_joint", "clinical_only_minus_joint"})
        self.assertEqual(
            set(result["holm_adjusted_p_values"]),
            {"image_only_minus_joint", "clinical_only_minus_joint"},
        )
        self.assertEqual(set(result["lower_bounds"]), {"image_only_minus_joint", "clinical_only_minus_joint"})
        self.assertIsInstance(result["passes"], bool)


if __name__ == "__main__":
    unittest.main()
