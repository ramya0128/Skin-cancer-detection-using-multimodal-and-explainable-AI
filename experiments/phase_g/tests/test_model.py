import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.phase_g.generate import render_patient
from experiments.phase_g.model import build_model, classification_loss, train_model


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class PhaseGModelTests(unittest.TestCase):
    def test_all_model_branches_return_finite_six_class_logits(self):
        config = load_config()
        image_only = build_model("image_only", config)
        clinical_only = build_model("clinical_only", config)
        joint = build_model("joint", config)
        images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        categories = torch.tensor([0, 5], dtype=torch.long)

        with torch.no_grad():
            outputs = (
                image_only(images),
                clinical_only(categories),
                joint(images, categories),
            )
        for logits in outputs:
            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertTrue(torch.isfinite(logits).all())

    def test_image_only_and_joint_use_the_prespecified_vit_dimensions(self):
        config = load_config()
        image_only = build_model("image_only", config)
        joint = build_model("joint", config)

        for model in (image_only, joint):
            encoder = model.image_encoder
            self.assertEqual(encoder.image_size, 224)
            self.assertEqual(encoder.patch_size, 16)
            self.assertEqual(encoder.conv_proj.kernel_size, (16, 16))
            self.assertEqual(encoder.conv_proj.out_channels, 192)
            self.assertEqual(len(encoder.encoder.layers), 4)
            self.assertEqual(encoder.encoder.layers[0].self_attention.num_heads, 3)
            self.assertEqual(encoder.encoder.layers[0].mlp[0].out_features, 384)

    def test_clinical_branch_requires_categorical_indices(self):
        model = build_model("clinical_only", load_config())
        logits = model(torch.tensor([0, 5], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 6))
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            model(torch.tensor([0.0, 5.0], dtype=torch.float32))

    def test_joint_model_accepts_reserved_masked_category_without_embedding_lookup(self):
        model = build_model("joint", load_config())
        images = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        logits = model(images, torch.tensor([6], dtype=torch.long))

        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertTrue(torch.isfinite(logits).all())

    def test_joint_objective_is_only_six_class_cross_entropy(self):
        logits = torch.tensor(
            [[2.0, 0.0, -1.0, -2.0, -3.0, -4.0], [0.0, 2.0, -1.0, -2.0, -3.0, -4.0]]
        )
        labels = torch.tensor([0, 1], dtype=torch.long)

        loss = classification_loss(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.equal(loss, expected))

    def test_interrupted_checkpoint_write_does_not_leave_a_partial_checkpoint(self):
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_directory"] = temp_dir
            config["model"]["training"]["batch_size"] = 2
            config["model"]["training"]["maximum_epochs"] = 1
            config["model"]["training"]["early_stopping"]["patience"] = 1
            train_data = [
                render_patient(f"atomic-train-{index}", index, "train", config)
                for index in range(2)
            ]
            selection_data = [
                render_patient(f"atomic-selection-{index}", index, "validation_model_selection", config)
                for index in range(2)
            ]

            checkpoint_path = Path(temp_dir) / "checkpoints" / "clinical_only.pt"

            def interrupted_save(_payload, destination):
                Path(destination).write_bytes(b"partial")
                raise OSError("simulated interrupted checkpoint write")

            with patch("torch.cuda.is_available", return_value=False):
                with patch("torch.save", side_effect=interrupted_save):
                    with self.assertRaisesRegex(OSError, "simulated interrupted"):
                        train_model("clinical_only", train_data, selection_data, config)

            self.assertFalse(checkpoint_path.exists())
            self.assertEqual(list(checkpoint_path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
