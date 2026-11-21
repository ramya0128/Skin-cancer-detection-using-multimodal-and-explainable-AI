from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.phase_g.model import build_model


class PhaseGModelArchitectureTests(unittest.TestCase):
    def test_three_phase_g_models_have_expected_branches(self):
        root = Path(__file__).resolve().parents[3]
        config = json.loads(
            (root / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )

        image_only = build_model("image_only", config)
        clinical_only = build_model("clinical_only", config)
        joint = build_model("joint", config)

        self.assertIn("image_encoder", dict(image_only.named_modules()))
        self.assertIn("embedding", dict(clinical_only.named_modules()))
        self.assertIn("fusion", dict(joint.named_modules()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_phase_g_vit_forward_backward_on_cuda(self):
        root = Path(__file__).resolve().parents[3]
        config = json.loads(
            (root / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        device = torch.device("cuda:0")

        for kind in ("image_only", "joint"):
            torch.cuda.empty_cache()
            model = build_model(kind, config).to(device).train()
            images = torch.randn((2, 3, 224, 224), device=device)
            categories = torch.tensor([0, 5], dtype=torch.long, device=device)
            labels = torch.tensor([0, 5], dtype=torch.long, device=device)

            logits = model(images) if kind == "image_only" else model(images, categories)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.cuda.synchronize(device)

            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(next(model.parameters()).device, device)


if __name__ == "__main__":
    unittest.main()
