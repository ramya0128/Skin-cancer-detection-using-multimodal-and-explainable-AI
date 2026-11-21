from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.phase_g.tests.test_explain import _TinyVisionModel
from experiments.phase_g_v2.cam import compute_gradcam, normalize_cam_maps
from experiments.phase_g_v2.geometry import (
    area_matched_random_pixel_null,
    shape_preserving_controls,
    translate_mask,
)
from experiments.phase_g_v2.localization import localization_metrics
from experiments.phase_g_v2.independent import replace_marker_glyph
from experiments.phase_g.generate import _make_glyph


class PhaseGV2CoreTests(unittest.TestCase):
    def test_cam_normalization_marks_unusable_maps_invalid(self):
        raw = np.asarray(
            [
                [[-2.0, -1.0], [-4.0, -3.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 2.0], [2.0, 2.0]],
                [[0.0, np.nan], [1.0, 2.0]],
            ],
            dtype=np.float64,
        )

        result = normalize_cam_maps(raw)

        np.testing.assert_array_equal(result.valid, [False, True, False, False])
        self.assertEqual(result.invalid_reasons[0], "no_positive_support")
        self.assertEqual(result.invalid_reasons[1], "")
        self.assertEqual(result.invalid_reasons[2], "constant_positive_map")
        self.assertEqual(result.invalid_reasons[3], "non_finite_raw_map")
        self.assertTrue(np.all(result.maps[[0, 2, 3]] == 0))

    def test_vit_gradcam_returns_validity_and_gradient_diagnostics(self):
        torch.manual_seed(23)
        model = _TinyVisionModel(joint=False).eval()
        images = torch.rand((2, 3, 224, 224), dtype=torch.float32)

        result = compute_gradcam(
            model,
            images,
            torch.tensor([1, 4]),
            "image_only",
            weighting="token_local",
            method_id="gradcam_v2_token_local",
        )

        self.assertEqual(result.maps.shape, (2, 14, 14))
        self.assertEqual(result.valid.shape, (2,))
        self.assertEqual(result.invalid_reasons.shape, (2,))
        self.assertTrue(np.isfinite(result.diagnostics["gradient_l2"]).all())
        self.assertTrue(np.all(result.diagnostics["gradient_l2"] > 0))
        for index in range(2):
            if result.valid[index]:
                self.assertGreater(float(result.maps[index].sum()), 0.0)
                self.assertEqual(result.invalid_reasons[index], "")
            else:
                self.assertTrue(np.all(result.maps[index] == 0))

    def test_localization_excludes_invalid_maps_and_preserves_valid_mass(self):
        maps = np.zeros((3, 14, 14), dtype=np.float32)
        maps[0, 7, 7] = 1.0
        maps[1, 0, 0] = np.nan
        lesion = np.zeros((3, 224, 224), dtype=bool)
        marker = np.zeros_like(lesion)
        lesion[:, 96:128, 96:128] = True
        marker[:, 0:16, 0:16] = True
        result = localization_metrics(
            maps,
            np.asarray(["p0", "p1", "p2"]),
            lesion,
            marker,
        )

        self.assertEqual(result["patient_count"], 3)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["invalid_count"], 2)
        self.assertAlmostEqual(float(result["valid_fraction"]), 1.0 / 3.0)
        self.assertEqual(result["invalid_reasons"][1], "non_finite_map")
        self.assertTrue(np.isnan(result["marker_share"][1:]).all())
        self.assertAlmostEqual(
            float(
                result["marker_share"][0]
                + result["lesion_share"][0]
                + result["background_share"][0]
            ),
            1.0,
            places=6,
        )

    def test_translated_control_mass_is_paired_by_patient(self):
        maps = np.zeros((1, 14, 14), dtype=np.float32)
        maps[0, 7, 7] = 1.0
        lesion = np.zeros((1, 224, 224), dtype=bool)
        marker = np.zeros_like(lesion)
        lesion[0, 112:128, 112:128] = True
        marker[0, 0:16, 0:16] = True
        controls = np.zeros((1, 2, 224, 224), dtype=bool)
        controls[0, 0, 112:128, 112:128] = True
        controls[0, 1, 96:112, 96:112] = True

        result = localization_metrics(
            maps,
            np.asarray(["p0"]),
            lesion,
            marker,
            translated_control_masks=controls,
        )

        self.assertEqual(result["translated_control_mass"].shape, (1, 2))
        self.assertAlmostEqual(float(result["translated_control_mass"][0, 0]), 1.0)
        self.assertAlmostEqual(float(result["translated_control_mass"][0, 1]), 0.0)

    def test_localization_does_not_mutate_upstream_validity(self):
        upstream_valid = np.asarray([True])
        lesion = np.zeros((1, 224, 224), dtype=bool)

        localization_metrics(
            np.full((1, 14, 14), np.nan),
            np.asarray(["p0"]),
            lesion,
            lesion,
            upstream_valid=upstream_valid,
        )

        np.testing.assert_array_equal(upstream_valid, [True])

    def test_shape_preserving_control_translates_mask_and_is_reproducible(self):
        lesion = np.zeros((1, 12, 12), dtype=bool)
        lesion[0, 4:7, 4:7] = True
        marker = np.zeros_like(lesion)
        marker[0, 0:2, 0:2] = True

        first = shape_preserving_controls(lesion, marker, seed=91, repeats=3)
        second = shape_preserving_controls(lesion, marker, seed=91, repeats=3)

        self.assertTrue(first["available"].all())
        self.assertEqual(first["control_masks"].shape, (1, 3, 12, 12))
        for repeat in range(3):
            control = first["control_masks"][0, repeat]
            dy, dx = first["selected_offsets"][0, repeat]
            expected = np.zeros((12, 12), dtype=bool)
            expected[4 + int(dy) : 7 + int(dy), 4 + int(dx) : 7 + int(dx)] = True
            np.testing.assert_array_equal(control, expected)
            self.assertEqual(int(control.sum()), int(lesion[0].sum()))
            self.assertFalse(np.any(control & lesion[0]))
            self.assertFalse(np.any(control & marker[0]))
        np.testing.assert_array_equal(first["selected_offsets"], second["selected_offsets"])
        np.testing.assert_array_equal(
            first["valid_placement_counts"], second["valid_placement_counts"]
        )

    def test_shape_preserving_control_reports_unavailable_without_fallback(self):
        lesion = np.ones((1, 4, 4), dtype=bool)
        marker = np.zeros_like(lesion)

        result = shape_preserving_controls(lesion, marker, seed=3, repeats=2)

        self.assertFalse(result["available"].any())
        self.assertTrue(np.all(result["valid_placement_counts"] == 0))
        self.assertTrue(np.all(result["selected_offsets"] == -1))
        self.assertFalse(result["control_masks"].any())

    def test_random_pixel_null_is_explicit_and_area_matched(self):
        lesion = np.zeros((10, 10), dtype=bool)
        lesion[3:6, 3:6] = True
        occupied = lesion.copy()
        occupied[0:2, 0:2] = True

        control = area_matched_random_pixel_null(
            lesion, occupied, np.random.default_rng(7)
        )

        self.assertEqual(int(control.sum()), int(lesion.sum()))
        self.assertFalse(np.any(control & occupied))

    def test_marker_counterfactual_changes_only_the_glyph_pixels(self):
        image = np.full((1, 224, 224, 3), 127, dtype=np.uint8)
        masks = np.zeros((1, 224, 224), dtype=bool)
        masks[0, 0:16, 208:224] = True

        result = replace_marker_glyph(image, masks, np.asarray([3]))

        np.testing.assert_array_equal(result[0, 0:16, 208:224], _make_glyph(3))
        np.testing.assert_array_equal(result[0, 32:64, 32:64], image[0, 32:64, 32:64])
        self.assertEqual(int(np.count_nonzero(result != image)), 16 * 16 * 3)


if __name__ == "__main__":
    unittest.main()
