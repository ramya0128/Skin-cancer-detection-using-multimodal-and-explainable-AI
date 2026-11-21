from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments.phase_g_v2_2.faithfulness import (
    PROTOCOL_BASELINES,
    build_protocol_baselines,
)


class PhaseGV22FaithfulnessTests(unittest.TestCase):
    def test_protocol_baselines_are_exact_and_patient_specific(self):
        images = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        images[0, :, :, 0] = 10
        images[1, :, :, 1] = 20
        backgrounds = np.full_like(images, 224)

        baselines = build_protocol_baselines(images, backgrounds)

        self.assertEqual(tuple(baselines), PROTOCOL_BASELINES)
        np.testing.assert_array_equal(baselines["renderer_background"], backgrounds)
        np.testing.assert_array_equal(
            baselines["mid_gray_128"],
            np.full_like(images, 128),
        )

    def test_baseline_builder_rejects_shape_or_dtype_mismatch(self):
        images = np.zeros((1, 224, 224, 3), dtype=np.float32)
        backgrounds = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "uint8"):
            build_protocol_baselines(images, backgrounds)

    def test_protocol_wrapper_passes_exact_slic_and_control_settings(self):
        from experiments.phase_g_v2_2.faithfulness import (
            independent_protocol_faithfulness,
        )

        model = nn.Linear(1, 1)
        model.to("cpu")
        images = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        backgrounds = np.zeros_like(images)
        maps = {"method": np.zeros((1, 14, 14), dtype=np.float32)}
        valid = {"method": np.asarray([True])}
        fake_result = {
            "baseline_names": np.asarray(PROTOCOL_BASELINES),
            "region_counts": np.asarray([128]),
        }
        with mock.patch(
            "experiments.phase_g_v2_2.faithfulness.independent_superpixel_faithfulness",
            return_value=fake_result,
        ) as implementation:
            with self.assertRaisesRegex(ValueError, "requires cuda:0"):
                independent_protocol_faithfulness(
                    model,
                    "image_only",
                    images,
                    backgrounds,
                    np.asarray([0]),
                    maps,
                    valid,
                    np.asarray(["p0"]),
                    device=torch.device("cpu"),
                )
        implementation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
