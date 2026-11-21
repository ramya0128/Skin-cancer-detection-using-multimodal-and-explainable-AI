import importlib
import unittest

import numpy as np
import torch
from torch import nn


image_eval = importlib.import_module("experiments.phase_f.image_eval")


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.normalization = nn.LayerNorm(4)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.ones(1, 2, 4))
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.ln = nn.LayerNorm(4)
        with torch.no_grad():
            self.ln.weight.fill_(2.0)
            self.ln.bias.fill_(0.25)


class TinyHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.heads = TinyHeads()


class MeanLogitModel(nn.Module):
    def forward(self, images):
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack([mean, -mean], dim=1)


class ImageSanityHelperTests(unittest.TestCase):
    def require_function(self, name):
        function = getattr(image_eval, name, None)
        self.assertTrue(callable(function), f"{name} has not been implemented")
        return function

    def test_horizontal_flip_map_is_restored_to_original_orientation(self):
        map_back = self.require_function("map_back_horizontal_flip")
        original = np.arange(196, dtype=np.float32).reshape(14, 14)

        restored = map_back(original[:, ::-1])

        np.testing.assert_array_equal(restored, original)

    def test_brightness_is_applied_in_raw_unit_space_before_clipping(self):
        transform = self.require_function("brightness_transform")
        pixels = torch.tensor([[[[0, 64, 200, 255]]]], dtype=torch.uint8)

        transformed = transform(pixels, 1.1)

        expected = torch.tensor([[[[0.0, 64 / 255 * 1.1, 200 / 255 * 1.1, 1.0]]]])
        self.assertEqual(transformed.dtype, torch.float32)
        torch.testing.assert_close(transformed, expected)

    def test_float_raw_brightness_images_are_not_divided_by_255_twice(self):
        images = torch.full((2, 3, 8, 8), 128, dtype=torch.uint8)
        raw_float = image_eval.brightness_transform(images, 1.0)
        model = MeanLogitModel()

        from_uint8 = image_eval._predict_vit(model, images, torch.device("cpu"))
        from_float = image_eval._predict_vit(model, raw_float, torch.device("cpu"))

        np.testing.assert_allclose(from_float, from_uint8, atol=1e-7)

    def test_unchanged_prediction_mask_preserves_the_exact_denominator(self):
        unchanged_mask = self.require_function("unchanged_prediction_mask")
        original = np.asarray([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.7, 0.3],
        ])
        transformed = np.asarray([
            [0.6, 0.4],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.1, 0.9],
        ])

        mask = unchanged_mask(original, transformed)

        np.testing.assert_array_equal(mask, [True, False, False, False])
        self.assertEqual(int(mask.sum()), 1)

    def test_spearman_map_similarity_detects_reversed_rank_order(self):
        similarity = self.require_function("spearman_map_similarity")
        original = np.arange(196, dtype=np.float32).reshape(14, 14)

        score = similarity(original, original[::-1, ::-1])

        self.assertAlmostEqual(score, -1.0, places=12)

    def test_spearman_is_undefined_when_either_map_has_no_rank_variation(self):
        similarity = self.require_function("spearman_map_similarity")

        score = similarity(np.zeros((14, 14)), np.zeros((14, 14)))

        self.assertTrue(np.isnan(score))

    def test_ssim_is_one_for_identical_maps_and_finite_for_different_maps(self):
        similarity = self.require_function("ssim_map_similarity")
        original = np.arange(196, dtype=np.float32).reshape(14, 14) / 195

        identical = similarity(original, original.copy())
        different = similarity(original, original[::-1, ::-1])

        self.assertAlmostEqual(identical, 1.0, places=12)
        self.assertTrue(np.isfinite(different))
        self.assertLess(different, 1.0)

    def test_seeded_randomization_is_repeatable_and_changes_only_its_stage(self):
        randomize = self.require_function("randomize_vit_parameters")
        original = TinyViT()
        original_state = original.state_dict()

        for stage in (
            "head",
            "final_encoder_block_and_head",
            "full_encoder_and_head",
        ):
            first = randomize(original, stage, seed=17).state_dict()
            second = randomize(original, stage, seed=17).state_dict()
            self.assertEqual(first.keys(), original_state.keys())
            self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))
            self.assertTrue(any(not torch.equal(first[key], original_state[key]) for key in first))

            if stage == "head":
                self.assertTrue(torch.equal(first["encoder.layers.0.projection.weight"],
                                            original_state["encoder.layers.0.projection.weight"]))
            elif stage == "final_encoder_block_and_head":
                self.assertTrue(torch.equal(first["encoder.layers.0.projection.weight"],
                                            original_state["encoder.layers.0.projection.weight"]))
                self.assertFalse(torch.equal(first["encoder.layers.1.projection.weight"],
                                             original_state["encoder.layers.1.projection.weight"]))
                self.assertTrue(torch.equal(first["encoder.ln.weight"],
                                            original_state["encoder.ln.weight"]))
            else:
                self.assertFalse(torch.equal(first["encoder.layers.0.projection.weight"],
                                             original_state["encoder.layers.0.projection.weight"]))
                self.assertFalse(torch.equal(first["encoder.pos_embedding"],
                                             original_state["encoder.pos_embedding"]))
                self.assertFalse(torch.equal(first["encoder.ln.weight"],
                                             original_state["encoder.ln.weight"]))


if __name__ == "__main__":
    unittest.main()
