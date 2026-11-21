from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.phase_g_v2.cam import compute_gradcam
from experiments.phase_g_v2_2.explain import (
    GRADCAM_PROTOCOL_METHODS,
    gradcam_method_specs,
)


ROOT = Path(__file__).resolve().parents[3]


class PhaseGV22ExplanationTests(unittest.TestCase):
    def test_four_protocol_gradcam_specs_map_to_v2_implementation(self):
        specs = gradcam_method_specs()
        self.assertEqual(
            tuple(spec["method_id"] for spec in specs),
            GRADCAM_PROTOCOL_METHODS,
        )
        self.assertEqual(
            {(spec["layer_index"], spec["weighting"]) for spec in specs},
            {(-1, "global"), (-1, "token_local"), (-2, "global"), (-2, "token_local")},
        )

    def test_joint_gradcam_requires_categories_and_keeps_test_identity(self):
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        from experiments.phase_g.model import build_model

        model = build_model("joint", config).eval()
        images = torch.rand((2, 3, 224, 224), dtype=torch.float32)
        targets = torch.tensor([0, 5], dtype=torch.long)
        categories = torch.tensor([2, 4], dtype=torch.long)

        with self.assertRaises(ValueError):
            compute_gradcam(
                model,
                images,
                targets,
                "joint",
                None,
                layer_index=-1,
                weighting="global",
            )
        result = compute_gradcam(
            model,
            images,
            targets,
            "joint",
            categories,
            layer_index=-1,
            weighting="global",
        )
        self.assertEqual(result.maps.shape, (2, 14, 14))
        self.assertEqual(result.valid.dtype, np.dtype(bool))

    def test_checkpoint_hashing_is_deterministic_and_only_records_files(self):
        from experiments.phase_g_v2_2.explain import sha256_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"fixture")
            first = sha256_checkpoint(checkpoint)
            second = sha256_checkpoint(checkpoint)
            self.assertEqual(first, second)
            self.assertEqual(first, hashlib.sha256(b"fixture").hexdigest())

    def test_four_maps_are_reproducible_for_identical_cpu_smoke_inputs(self):
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        from experiments.phase_g.model import build_model

        model = build_model("image_only", config).eval()
        images = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        targets = torch.tensor([3], dtype=torch.long)
        first = {}
        second = {}
        for spec in gradcam_method_specs():
            first_result = compute_gradcam(
                model,
                images,
                targets,
                "image_only",
                None,
                layer_index=int(spec["layer_index"]),
                weighting=str(spec["weighting"]),
                method_id=str(spec["method_id"]),
            )
            second_result = compute_gradcam(
                model,
                images,
                targets,
                "image_only",
                None,
                layer_index=int(spec["layer_index"]),
                weighting=str(spec["weighting"]),
                method_id=str(spec["method_id"]),
            )
            first[str(spec["method_id"])] = first_result.maps
            second[str(spec["method_id"])] = second_result.maps
        for method_id in GRADCAM_PROTOCOL_METHODS:
            np.testing.assert_array_equal(first[method_id], second[method_id])


if __name__ == "__main__":
    unittest.main()
