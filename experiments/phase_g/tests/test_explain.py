from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from experiments.phase_g.explain import (
    compute_clinical_attributions,
    compute_image_maps,
    compute_joint_shapley,
    cue_mask_sensitivity,
)
from experiments.phase_g.generate import render_patient


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class _TinyVisionModel(nn.Module):
    def __init__(self, joint: bool):
        super().__init__()
        self.image_encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=1,
            num_heads=3,
            hidden_dim=24,
            mlp_dim=48,
            num_classes=6,
        )
        self.image_encoder.heads = nn.Identity()
        self.joint = joint
        self.clinical_embedding = nn.Embedding(7, 8)
        self.classifier = nn.Linear(24 + (8 if joint else 0), 6)

    def forward(
        self, images: torch.Tensor, categories: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.image_encoder(images)
        if self.joint:
            if categories is None:
                raise ValueError("joint model requires fixed clinical categories")
            clinical = self.clinical_embedding(categories.clamp_max(5))
            clinical = clinical.masked_fill((categories == 6).unsqueeze(1), 0.0)
            features = torch.cat((features, clinical), dim=1)
        return self.classifier(features)


class _ToyJointModel(nn.Module):
    def forward(self, images: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        image_signal = images.mean(dim=(1, 2, 3))
        clinical_signal = (categories == 0).to(images.dtype)
        class_zero = 0.15 + 0.15 * image_signal + 0.1 * clinical_signal
        probabilities = torch.full(
            (len(images), 6), 0.0, dtype=images.dtype, device=images.device
        )
        probabilities[:, 1:] = ((1.0 - class_zero) / 5.0).unsqueeze(1)
        probabilities[:, 0] = class_zero
        return probabilities.log()


class _ToyClinicalModel(nn.Module):
    def forward(self, categories: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(categories), 6), dtype=torch.float32)
        logits.scatter_(1, categories.unsqueeze(1), 2.0)
        return logits


class PhaseGExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        cls.patient = render_patient("explain-patient-0", 0, "train", config)
        cls.patient.render_metadata["replacement_image_cues"] = [index % 6 for index in range(20)]
        cls.patient.render_metadata["replacement_clinical_cues"] = [
            (index + 3) % 6 for index in range(20)
        ]
        cls.donor = render_patient("explain-patient-1", 1, "train", config)

    def test_image_methods_return_finite_fixed_target_patch_maps(self):
        torch.manual_seed(23)
        model = _TinyVisionModel(joint=True).eval()
        images = torch.full((1, 3, 224, 224), 0.5)

        maps = compute_image_maps(
            model,
            "joint",
            images,
            np.asarray([2]),
            np.asarray([3]),
            ("gradcam", "chefer", "attention_rollout", "occlusion"),
        )

        self.assertEqual(set(maps), {"gradcam", "chefer", "attention_rollout", "occlusion"})
        for value in maps.values():
            self.assertEqual(value.shape, (1, 14, 14))
            self.assertTrue(np.isfinite(value).all())
            self.assertTrue(np.all(value >= 0.0))

    def test_image_only_explanation_does_not_consume_clinical_categories(self):
        torch.manual_seed(24)
        model = _TinyVisionModel(joint=False).eval()
        maps = compute_image_maps(
            model,
            "image_only",
            torch.full((1, 3, 224, 224), 0.5),
            np.asarray([1]),
            None,
            ("gradcam",),
        )

        self.assertEqual(maps["gradcam"].shape, (1, 14, 14))
        with self.assertRaises(ValueError):
            compute_image_maps(
                model,
                "image_only",
                torch.full((1, 3, 224, 224), 0.5),
                np.asarray([1]),
                np.asarray([0]),
                ("gradcam",),
            )

    def test_clinical_attribution_is_observed_probability_minus_reference(self):
        model = _ToyClinicalModel().eval()
        categories = np.asarray([0, 2], dtype=np.int64)
        targets = np.asarray([0, 1], dtype=np.int64)
        reference = np.full(6, 1.0 / 6.0)

        result = compute_clinical_attributions(
            model, categories, targets, reference, temperature=2.0
        )

        self.assertEqual(result["signed_effect"].shape, (2,))
        self.assertTrue(np.allclose(
            result["signed_effect"],
            result["observed_value"] - result["reference_value"],
        ))
        self.assertTrue(np.allclose(
            result["reference_value"],
            result["category_values"] @ reference,
        ))
        self.assertAlmostEqual(float(result["temperature"]), 2.0)
        self.assertTrue(np.all(result["absolute_effect"] >= 0.0))

    def test_additive_model_shapley_values_satisfy_efficiency_in_both_games(self):
        model = _ToyJointModel().eval()
        targets = np.asarray([0], dtype=np.int64)

        modality = compute_joint_shapley(
            model, [self.patient], targets, [self.donor], "modality"
        )
        cue = compute_joint_shapley(
            model, [self.patient], targets, [self.donor], "cue", temperature=2.0
        )

        self.assertLess(float(np.max(np.abs(modality["efficiency_residual"]))), 1e-6)
        self.assertLess(float(np.max(np.abs(cue["efficiency_residual"]))), 1e-6)
        self.assertEqual(modality["image"].shape, (1,))
        self.assertEqual(modality["clinical"].shape, (1,))
        expected_image = 0.15 * (
            self.patient.image.mean() / 255.0
            - self.donor.image.mean() / 255.0
        )
        expected_clinical = 0.1 * (
            float(self.patient.clinical_value == 0)
            - float(self.donor.clinical_value == 0)
        )
        self.assertAlmostEqual(float(modality["image"][0]), expected_image, places=5)
        self.assertAlmostEqual(float(modality["clinical"][0]), expected_clinical, places=5)
        self.assertAlmostEqual(float(cue["image"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(cue["temperature"]), 2.0)

    def test_cue_sensitivity_keeps_replacement_and_literal_absence_separate(self):
        result = cue_mask_sensitivity(
            _ToyJointModel(), [self.patient], {"temperature": 2.0}
        )

        self.assertIn("image_replacement_target_probability", result)
        self.assertIn("clinical_replacement_target_probability", result)
        self.assertIn("literal_absence_target_probability", result)
        self.assertIn("image_replacement_nll_change", result)
        self.assertIn("clinical_replacement_accuracy_change", result)
        self.assertIn("literal_absence_target_probability_change", result)
        self.assertEqual(result["literal_absence_type"].item(), "out_of_support_diagnostic")
        self.assertAlmostEqual(float(result["temperature"]), 2.0)
        self.assertEqual(result["image_replacement_target_probability"].shape, (1,))
        self.assertEqual(result["clinical_replacement_target_probability"].shape, (1,))
        self.assertEqual(result["literal_absence_target_probability"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
