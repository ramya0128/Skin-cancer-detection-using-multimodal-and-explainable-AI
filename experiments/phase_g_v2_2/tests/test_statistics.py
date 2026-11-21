from __future__ import annotations

import unittest

import numpy as np
from unittest import mock

from experiments.phase_g_v2_2.statistics import (
    PRIMARY_CONTRAST_NAMES,
    condition_success,
    experiment_success,
    holm_adjusted_p_values,
    monte_carlo_contrast_power,
    normal_contrast_power,
    paired_stratified_bootstrap,
    summarize_bootstrap,
)


class PhaseGV22StatisticsTests(unittest.TestCase):
    def test_primary_family_has_exactly_ten_contrast_names(self):
        self.assertEqual(len(PRIMARY_CONTRAST_NAMES), 10)
        self.assertEqual(len(set(PRIMARY_CONTRAST_NAMES)), 10)

    def test_holm_adjustment_is_monotone_and_matches_known_example(self):
        p_values = np.asarray([0.01, 0.04, 0.03])
        adjusted = holm_adjusted_p_values(p_values)
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_marginal_intervals_are_not_marked_simultaneous(self):
        estimates = {"a": 0.12, "b": 0.04}
        rng = np.random.default_rng(20261121)
        bootstrap = {
            "a": rng.normal(0.12, 0.05, 1000),
            "b": rng.normal(0.04, 0.05, 1000),
        }
        summary = summarize_bootstrap(estimates, bootstrap)
        self.assertIn("marginal_percentile_95", summary)
        self.assertFalse(summary["simultaneous_interpretation"])
        self.assertEqual(set(summary["holm_adjusted_p_values"]), set(estimates))

    def test_bootstrap_threshold_p_value_is_hand_calculable(self):
        from experiments.phase_g_v2_2.statistics import bootstrap_centered_p_value

        p = bootstrap_centered_p_value(
            np.asarray([0.11, 0.12, 0.13, 0.14]),
            estimate=0.125,
            threshold=0.10,
        )
        self.assertAlmostEqual(p, 1.0 / 5.0)

        p_positive = bootstrap_centered_p_value(
            np.asarray([0.11, 0.12, 0.13, 0.14]),
            estimate=0.125,
            threshold=-1.0,
        )
        self.assertAlmostEqual(p_positive, 1.0 / 5.0)

    def test_adjusted_lower_bounds_are_present_but_marginal_intervals_are_separate(self):
        rng = np.random.default_rng(91)
        estimates = {"a": 0.12, "b": 0.08}
        bootstrap = {
            "a": rng.normal(0.12, 0.02, 1000),
            "b": rng.normal(0.08, 0.02, 1000),
        }
        summary = summarize_bootstrap(estimates, bootstrap, alpha=0.05)
        self.assertEqual(set(summary["marginal_percentile_95"]), set(estimates))
        self.assertEqual(set(summary["holm_adjusted_one_sided_lower_bounds"]), set(estimates))
        self.assertFalse(summary["simultaneous_interpretation"])

    def test_normal_power_is_scenario_only_and_threshold_sensitive(self):
        zero = normal_contrast_power(0.10, 0.50, 300)
        threshold = normal_contrast_power(0.15, 0.50, 195, threshold=0.10)
        self.assertGreater(zero["contrast_power"], 0.0)
        self.assertLess(threshold["per_class_n"], 300)
        self.assertEqual(threshold["margin"], 0.04999999999999999)

    def test_monte_carlo_power_reports_uncertainty_and_seed(self):
        result = monte_carlo_contrast_power(
            effect=0.10,
            sd=0.50,
            per_class_n=300,
            replicates=1000,
            seed=7,
        )
        self.assertEqual(result["replicates"], 1000)
        self.assertEqual(result["seed"], 7)
        self.assertGreaterEqual(result["monte_carlo_standard_error"], 0.0)
        self.assertEqual(len(result["approximate_95_interval"]), 2)

    def test_paired_bootstrap_keeps_three_models_and_calibration_resampling(self):
        rng = np.random.default_rng(3)
        test_labels = np.repeat(np.arange(6), 20)
        calibration_labels = np.repeat(np.arange(6), 30)
        test_logits = {}
        calibration_logits = {}
        for name in ("image_only", "clinical_only", "joint"):
            test_logits[name] = rng.normal(size=(len(test_labels), 6))
            calibration_logits[name] = rng.normal(size=(len(calibration_labels), 6))
        result = paired_stratified_bootstrap(
            test_logits,
            test_labels,
            calibration_logits,
            calibration_labels,
            contrasts=("image_dominant.image_minus_joint",),
            replicates=100,
            test_seed=11,
            calibration_seed=12,
        )
        self.assertTrue(result["temperature_calibration_resampled"])
        self.assertTrue(result["test_patient_resampled"])
        self.assertEqual(result["bootstrap"]["image_dominant.image_minus_joint"].shape, (100,))

    def test_calibration_refit_is_called_once_per_model_per_replicate(self):
        rng = np.random.default_rng(4)
        test_labels = np.repeat(np.arange(6), 10)
        calibration_labels = np.repeat(np.arange(6), 12)
        test_logits = {
            name: rng.normal(size=(len(test_labels), 6))
            for name in ("image_only", "clinical_only", "joint")
        }
        calibration_logits = {
            name: rng.normal(size=(len(calibration_labels), 6))
            for name in ("image_only", "clinical_only", "joint")
        }
        with mock.patch(
            "experiments.phase_g_v2_2.statistics.fit_temperature",
            side_effect=lambda logits, labels: 1.0,
        ) as temperature:
            paired_stratified_bootstrap(
                test_logits,
                test_labels,
                calibration_logits,
                calibration_labels,
                contrasts=("image_dominant.image_minus_joint",),
                replicates=100,
            )
        self.assertEqual(temperature.call_count, 300)
        for call in temperature.call_args_list:
            logits, labels = call.args
            self.assertEqual(logits.shape[1], 6)
            self.assertEqual(len(logits), len(labels))
            self.assertEqual(set(np.unique(labels)), set(range(6)))

    def test_shared_cohort_rejects_different_patient_order(self):
        rng = np.random.default_rng(5)
        test_labels = np.repeat(np.arange(6), 10)
        test_ids = np.asarray([f"p{index}" for index in range(len(test_labels))])
        test_logits = {
            condition: {
                name: rng.normal(size=(len(test_labels), 6))
                for name in ("image_only", "clinical_only", "joint")
            }
            for condition in ("image_dominant", "clinical_dominant")
        }
        calibration_labels = np.repeat(np.arange(6), 12)
        calibration_logits = {
            name: rng.normal(size=(len(calibration_labels), 6))
            for name in ("image_only", "clinical_only", "joint")
        }
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_stratified_bootstrap(
                test_logits,
                {
                    "image_dominant": test_labels,
                    "clinical_dominant": test_labels,
                },
                calibration_logits,
                calibration_labels,
                contrasts=(
                    "image_dominant.image_minus_joint",
                    "clinical_dominant.image_minus_joint",
                ),
                replicates=100,
                test_patient_ids={
                    "image_dominant": test_ids,
                    "clinical_dominant": test_ids[::-1],
                },
            )

    def test_condition_and_experiment_success_require_all_appropriate_bounds(self):
        bounds = {
            "image_dominant.image_minus_joint": 0.01,
            "image_dominant.clinical_minus_joint": 0.02,
            "clinical_dominant.image_minus_joint": -0.01,
            "clinical_dominant.clinical_minus_joint": 0.03,
        }
        self.assertTrue(condition_success(bounds, "image_dominant"))
        self.assertFalse(condition_success(bounds, "clinical_dominant"))
        summary = experiment_success(
            bounds,
            ("image_dominant", "clinical_dominant"),
        )
        self.assertTrue(summary["some_conditions"])
        self.assertFalse(summary["all_conditions"])
        self.assertTrue(summary["no_conditions"] is False)


if __name__ == "__main__":
    unittest.main()
