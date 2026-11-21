import importlib
import unittest

import numpy as np
import torch


try:
    image_eval = importlib.import_module("experiments.phase_f.image_eval")
except ModuleNotFoundError:
    image_eval = None


class ImageCurveHelperTests(unittest.TestCase):
    def require_module(self):
        if image_eval is None:
            self.fail("experiments.phase_f.image_eval has not been implemented")
        return image_eval

    def test_patch_schedule_covers_the_image_in_fixed_five_percent_steps(self):
        module = self.require_module()
        expected = np.asarray([
            0, 10, 20, 29, 39, 49, 59, 69, 78, 88, 98,
            108, 118, 127, 137, 147, 157, 167, 176, 186, 196,
        ])
        actual = module.make_patch_schedule()
        np.testing.assert_array_equal(actual, expected)

    def test_full_deletion_and_insertion_recover_their_endpoints(self):
        module = self.require_module()
        original = torch.ones((3, 224, 224), dtype=torch.uint8)
        baseline = torch.zeros_like(original)
        order = np.arange(196, dtype=np.int64)

        deleted = module.apply_ranked_patches(
            original, baseline, order, patch_count=196, mode="deletion"
        )
        inserted = module.apply_ranked_patches(
            original, baseline, order, patch_count=196, mode="insertion"
        )

        self.assertTrue(torch.equal(deleted, baseline))
        self.assertTrue(torch.equal(inserted, original))

    def test_fusion_normalizes_probabilities_and_fixed_target_is_not_argmax(self):
        module = self.require_module()
        xgb_probability = np.asarray([[0.90, 0.05, 0.05]], dtype=np.float64)
        vit_probability = np.asarray([[0.10, 0.85, 0.05]], dtype=np.float64)
        weights = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
        fused = module.cwf_probabilities(xgb_probability, vit_probability, weights)

        np.testing.assert_allclose(fused, [[0.50, 0.45, 0.05]], atol=1e-12)
        self.assertAlmostEqual(float(fused.sum()), 1.0, places=12)
        self.assertEqual(int(fused.argmax(axis=1)[0]), 0)
        self.assertEqual(module.score_fixed_target(fused, np.asarray([0]))[0], fused[0, 0])
        self.assertEqual(module.score_fixed_target(fused, np.asarray([1]))[0], fused[0, 1])

    def test_tied_map_scores_have_a_deterministic_patch_order(self):
        module = self.require_module()
        patch_scores = np.zeros((14, 14), dtype=np.float32)

        order = module.map_to_patch_order(patch_scores)

        np.testing.assert_array_equal(order, np.arange(196))


if __name__ == "__main__":
    unittest.main()
