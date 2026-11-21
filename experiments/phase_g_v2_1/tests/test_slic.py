from __future__ import annotations

import unittest

import numpy as np

from experiments.phase_g_v2_1.geometry import shape_preserving_controls
from experiments.phase_g_v2_1.slic import slic_parameters, slic_segment


class PhaseGV21SLICAndGeometryTests(unittest.TestCase):
    def test_repeated_slic_calls_are_identical_with_fixed_seed(self):
        y, x = np.mgrid[:224, :224]
        image = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)

        first = slic_segment(image, seed=20261144)
        second = slic_segment(image, seed=20261144)

        np.testing.assert_array_equal(first["labels"], second["labels"])

    def test_slic_metadata_records_exact_v2_settings_and_seed(self):
        metadata = slic_parameters(seed=20261144)

        self.assertEqual(metadata["n_segments"], 128)
        self.assertEqual(metadata["compactness"], 10.0)
        self.assertEqual(metadata["sigma"], 0)
        self.assertEqual(metadata["seed"], 20261144)
        self.assertEqual(metadata["start_label"], 0)
        self.assertEqual(metadata["channel_axis"], -1)
        self.assertFalse(metadata["convert2lab"])
        self.assertTrue(metadata["enforce_connectivity"])
        self.assertEqual(metadata["scikit_image"], "0.26.0")

    def test_geometry_returns_availability_and_unavailable_counts(self):
        lesions = np.zeros((2, 12, 12), dtype=bool)
        lesions[0, 4:7, 4:7] = True
        lesions[1] = True
        markers = np.zeros_like(lesions)

        result = shape_preserving_controls(
            lesions,
            markers,
            seed=20261143,
            repeats=3,
        )

        self.assertTrue(result["available"][0].all())
        self.assertFalse(result["available"][1].any())
        self.assertEqual(int(result["total_requested"]), 6)
        self.assertEqual(int(result["available"].sum()), 3)
        self.assertEqual(int((~result["available"]).sum()), 3)
        self.assertEqual(int(result["patient_available"].sum()), 1)
        self.assertEqual(int((~result["patient_available"]).sum()), 1)

    def test_valid_zero_control_remains_available(self):
        lesion = np.zeros((1, 12, 12), dtype=bool)
        lesion[0, 4:7, 4:7] = True
        marker = np.zeros_like(lesion)

        result = shape_preserving_controls(
            lesion,
            marker,
            seed=91,
            repeats=1,
        )

        self.assertTrue(result["available"][0, 0])
        self.assertEqual(int(result["control_masks"][0, 0].sum()), 9)


if __name__ == "__main__":
    unittest.main()
