from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from experiments.phase_g_v2.independent import (
    independent_superpixel_faithfulness,
    segment_superpixels,
)


class _RedPixelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red_signal = images[:, 0].mean(dim=(1, 2)) * self.scale
        logits = torch.zeros((len(images), 6), device=images.device)
        logits[:, 0] = red_signal
        logits[:, 1] = -red_signal
        return logits


class PhaseGV2IndependentTests(unittest.TestCase):
    def test_slic_regions_are_connected_labels_and_reproducible(self):
        y, x = np.mgrid[:224, :224]
        image = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)

        first = segment_superpixels(image, n_segments=32, compactness=8.0)
        second = segment_superpixels(image, n_segments=32, compactness=8.0)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (224, 224))
        self.assertEqual(set(np.unique(first)), set(range(len(np.unique(first)))))

    def test_superpixel_evaluation_uses_patient_level_curves_and_endpoints(self):
        model = _RedPixelModel().eval()
        model.to("cpu")
        images = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        images[:, :, 112:, 0] = 255
        baselines = {"renderer_background": np.zeros_like(images)}
        maps = {"fixture_map": np.zeros((1, 14, 14), dtype=np.float32)}
        maps["fixture_map"][0, :, 7:] = 1.0
        valid = {"fixture_map": np.asarray([True])}

        result = independent_superpixel_faithfulness(
            model,
            "image_only",
            images,
            baselines,
            np.asarray([0]),
            maps,
            valid,
            np.asarray(["validation-patient-0"]),
            n_segments=32,
            random_orders=3,
            seed=15,
            batch_size=16,
            device=torch.device("cpu"),
        )

        self.assertEqual(result["patient_count"], 1)
        self.assertEqual(result["valid_patient_count"]["fixture_map"], 1)
        self.assertEqual(result["curves"]["fixture_map"].shape, (1, 1, 2, 21))
        np.testing.assert_allclose(
            result["curves"]["fixture_map"][0, 0, 0, 0],
            result["curves"]["fixture_map"][0, 0, 1, -1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["curves"]["fixture_map"][0, 0, 0, -1],
            result["curves"]["fixture_map"][0, 0, 1, 0],
            atol=1e-6,
        )
        self.assertEqual(result["random_curves"].shape, (1, 3, 1, 2, 21))
        self.assertEqual(result["center_curves"].shape, (1, 1, 2, 21))


if __name__ == "__main__":
    unittest.main()
