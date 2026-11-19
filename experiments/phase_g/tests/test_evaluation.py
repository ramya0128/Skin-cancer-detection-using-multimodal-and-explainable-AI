from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.phase_g.evaluate import (
    bootstrap_mean_interval,
    build_localization_controls,
    deletion_insertion_metrics,
    localization_metrics,
    paired_signflip_p,
    stability_metrics,
)
from experiments.phase_g.generate import render_patient
from experiments.phase_g.tests.test_explain import _TinyVisionModel


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: torch.Tensor, categories: torch.Tensor | None = None
    ) -> torch.Tensor:
        signal = images.mean(dim=(1, 2, 3)) * self.scale
        logits = torch.zeros((len(images), 6), device=images.device)
        logits[:, 0] = signal
        if categories is not None:
            logits.scatter_add_(
                1,
                categories.clamp_max(5).unsqueeze(1),
                torch.ones((len(images), 1), device=images.device),
            )
        return logits


class PhaseGEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patient = render_patient("evaluation-patient-0", 0, "train", load_config())
        cls.image = torch.from_numpy(
            cls.patient.image.transpose(2, 0, 1).copy()
        ).unsqueeze(0)
        cls.background = torch.from_numpy(
            cls.patient.background_canvas.transpose(2, 0, 1).copy()
        ).unsqueeze(0)
        cls.masks = {
            "lesion": cls.patient.lesion_mask[None, ...],
            "marker": cls.patient.marker_mask[None, ...],
        }

    def test_localization_reports_mass_area_enrichment_and_random_regions(self):
        controls = build_localization_controls(self.masks, seed=13, repeats=3)
        repeated_controls = build_localization_controls(self.masks, seed=13, repeats=3)
        np.testing.assert_array_equal(
            controls["marker_random_masks"], repeated_controls["marker_random_masks"]
        )
        marker_patch = np.zeros((1, 14, 14), dtype=np.float32)
        marker_y, marker_x = self.patient.render_metadata["marker_top_left"]
        marker_patch[0, marker_y // 16, marker_x // 16] = 1.0

        result = localization_metrics(
            marker_patch, self.masks, controls, np.asarray([self.patient.patient_id])
        )

        self.assertAlmostEqual(float(result["marker_mass"][0]), 1.0)
        self.assertAlmostEqual(float(result["lesion_mass"][0]), 0.0)
        self.assertIn("background_mass", result)
        self.assertIn("marker_area_enrichment", result)
        self.assertEqual(result["marker_random_mass"].shape, (1, 3))
        self.assertEqual(result["outcome_type"].item(), "localization")
        self.assertNotIn("curves", result)

    def test_deletion_insertion_curves_have_faithfulness_endpoints(self):
        model = _ToyModel().eval()
        maps = {"gradcam": np.arange(196, dtype=np.float32).reshape(1, 14, 14)}
        result = deletion_insertion_metrics(
            model,
            "joint",
            self.image,
            np.asarray([self.patient.clinical_value]),
            self.background,
            maps,
            np.asarray([self.patient.label]),
            {
                "random_order_count": 2,
                "seed": 19,
                "gaussian_blur_radius": 4.0,
                "curve_batch_size": 64,
                "temperature": 2.0,
            },
        )
        repeated = deletion_insertion_metrics(
            model,
            "joint",
            self.image,
            np.asarray([self.patient.clinical_value]),
            self.background,
            maps,
            np.asarray([self.patient.label]),
            {
                "random_order_count": 2,
                "seed": 19,
                "gaussian_blur_radius": 4.0,
                "curve_batch_size": 64,
                "temperature": 2.0,
            },
        )

        curves = result["curves"]
        self.assertEqual(curves.shape, (1, 1, 2, 2, 21))
        for baseline in range(2):
            np.testing.assert_allclose(
                curves[0, 0, baseline, 0, 0],
                curves[0, 0, baseline, 1, -1],
            )
            np.testing.assert_allclose(
                curves[0, 0, baseline, 0, -1],
                curves[0, 0, baseline, 1, 0],
            )
        np.testing.assert_array_equal(result["target_classes"], [self.patient.label])
        self.assertEqual(result["random_order_curves"].shape, (1, 2, 2, 2, 21))
        self.assertEqual(result["random_order_seeds"].shape, (2,))
        np.testing.assert_array_equal(
            result["random_order_curves"], repeated["random_order_curves"]
        )
        self.assertAlmostEqual(float(result["temperature"]), 2.0)
        self.assertNotIn("marker_mass", result)
        self.assertEqual(result["outcome_type"].item(), "faithfulness")

    def test_stability_keeps_prediction_changes_and_map_similarity_separate(self):
        torch.manual_seed(29)
        model = _TinyVisionModel(joint=True).eval()
        maps = {
            "gradcam": np.ones((1, 14, 14), dtype=np.float32),
            "attention_rollout": np.eye(14, dtype=np.float32)[None, ...],
        }

        result = stability_metrics(
            model,
            "joint",
            self.image,
            np.asarray([self.patient.clinical_value]),
            maps,
            np.asarray([self.patient.label]),
            {
                "brightness_factors": [0.9, 1.1],
                "image_batch_size": 16,
                "temperature": 2.0,
            },
        )

        self.assertEqual(result["prediction_changed"].shape, (3, 1))
        self.assertEqual(result["spearman"].shape, (3, 1, 2))
        self.assertEqual(result["valid_subset"].shape, (3, 1))
        self.assertEqual(result["outcome_type"].item(), "stability")

    def test_patient_bootstrap_and_signflip_are_seeded(self):
        values = np.asarray([0.3, 0.4, 0.5, 0.6])
        first = bootstrap_mean_interval(values, seed=7)
        second = bootstrap_mean_interval(values, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], float(values.mean()))
        self.assertGreaterEqual(first[1], float(values.mean()))
        p_value = paired_signflip_p(values, np.zeros_like(values), seed=8)
        self.assertGreater(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
