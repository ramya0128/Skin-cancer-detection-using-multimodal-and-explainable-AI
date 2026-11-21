from __future__ import annotations

import unittest

import numpy as np

from experiments.phase_g_v2_1.localization import translated_control_summary


class TranslatedControlAvailabilityTests(unittest.TestCase):
    def test_all_controls_available_uses_each_valid_repeat(self):
        result = translated_control_summary(
            np.asarray([0.25, 0.75]),
            np.asarray([True, True]),
        )

        self.assertTrue(result["patient_available"])
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertEqual(result["actual_denominator"], 2)
        self.assertEqual(result["total_requested"], 2)
        self.assertEqual(result["available_translations"], 2)
        self.assertEqual(result["unavailable_translations"], 0)

    def test_some_controls_unavailable_excludes_them_from_denominator(self):
        result = translated_control_summary(
            np.asarray([0.2, np.nan, 0.4]),
            np.asarray([True, False, True]),
        )

        self.assertTrue(result["patient_available"])
        self.assertAlmostEqual(result["mean"], 0.3)
        self.assertEqual(result["actual_denominator"], 2)
        self.assertEqual(result["total_requested"], 3)
        self.assertEqual(result["available_translations"], 2)
        self.assertEqual(result["unavailable_translations"], 1)

    def test_no_valid_controls_returns_nan_not_zero(self):
        result = translated_control_summary(
            np.asarray([np.nan, np.nan]),
            np.asarray([False, False]),
        )

        self.assertFalse(result["patient_available"])
        self.assertTrue(np.isnan(result["mean"]))
        self.assertEqual(result["actual_denominator"], 0)
        self.assertEqual(result["available_translations"], 0)
        self.assertEqual(result["unavailable_translations"], 2)

    def test_valid_control_with_zero_attribution_is_not_unavailable(self):
        result = translated_control_summary(
            np.asarray([0.0, 0.0]),
            np.asarray([True, True]),
        )

        self.assertTrue(result["patient_available"])
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["actual_denominator"], 2)
        self.assertTrue(np.isfinite(result["mean"]))

    def test_available_geometry_with_nan_attribution_is_not_numeric_denominator(self):
        result = translated_control_summary(
            np.asarray([np.nan, 0.4]),
            np.asarray([True, True]),
        )

        self.assertTrue(result["patient_available"])
        self.assertEqual(result["available_translations"], 2)
        self.assertEqual(result["unavailable_translations"], 0)
        self.assertEqual(result["actual_denominator"], 1)
        self.assertAlmostEqual(result["mean"], 0.4)

    def test_batch_aggregate_uses_only_available_patients_and_controls(self):
        masses = np.asarray([
            [0.2, np.nan, 0.4],
            [np.nan, np.nan, np.nan],
            [0.0, 0.6, np.nan],
        ])
        availability = np.asarray([
            [True, False, True],
            [False, False, False],
            [True, True, False],
        ])

        result = translated_control_summary(masses, availability)

        self.assertEqual(result["total_requested"], 9)
        self.assertEqual(result["available_translations"], 4)
        self.assertEqual(result["unavailable_translations"], 5)
        self.assertEqual(result["patients_with_valid_translation"], 2)
        self.assertEqual(result["patients_with_no_valid_translation"], 1)
        self.assertEqual(result["actual_denominator"], 4)
        self.assertAlmostEqual(result["mean"], 0.3)
        self.assertTrue(np.isnan(result["patient_means"][1]))

    def test_downstream_summary_preserves_nan_and_has_explicit_counts(self):
        patient_means = np.asarray([0.2, np.nan, 0.4])

        valid = np.isfinite(patient_means)
        values = patient_means[valid]

        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(float(values.mean()), 0.3)
        self.assertTrue(np.isnan(patient_means[1]))


if __name__ == "__main__":
    unittest.main()
