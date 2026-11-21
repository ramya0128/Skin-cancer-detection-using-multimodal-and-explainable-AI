from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import experiments.phase_g.run as phase_g_run
from experiments.phase_g.model import MODEL_NAMES
from experiments.phase_g.run import (
    _explanation_resume_readiness,
    _mark_stage,
    _secondary_statistics,
    assert_test_ready,
    atomic_save_npz,
    atomic_write_json,
    run_preflight,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class PhaseGRunnerTests(unittest.TestCase):
    def test_final_manifest_hashes_the_last_completion_log_line(self):
        write_manifest = getattr(phase_g_run, "_write_final_manifest", None)
        self.assertTrue(callable(write_manifest), "final manifest writer is required")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "run.log").write_text("started\n", encoding="utf-8")
            (output_dir / "result.bin").write_bytes(b"result")

            write_manifest(
                output_dir,
                {"status": "completed"},
                "completion logged before manifest hashing",
            )

            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            for relative_path, expected_hash in manifest["artifact_sha256"].items():
                import hashlib

                actual_hash = hashlib.sha256(
                    (output_dir / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(expected_hash, actual_hash)

    def test_explanation_resume_requires_frozen_primary_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                _explanation_resume_readiness(output_dir)

    def test_resume_rejects_artifacts_that_changed_after_stage_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifact = output_dir / "predictions.npz"
            artifact.write_bytes(b"frozen")
            _mark_stage(output_dir, "predictions", [artifact])

            artifact.write_bytes(b"changed")

            verify_stage = getattr(phase_g_run, "_verify_stage", None)
            self.assertTrue(callable(verify_stage), "resume stage verification is required")
            with self.assertRaisesRegex(RuntimeError, "changed after stage completion"):
                verify_stage(output_dir, "predictions")

    def test_secondary_statistics_handles_auc_dimensions_and_nonfinite_stability(self):
        labels = np.arange(6, dtype=np.int64)
        cue = {
            f"{name}_{metric}": np.zeros(6, dtype=np.float32)
            for name in ("image_replacement", "clinical_replacement")
            for metric in ("accuracy_change", "target_probability_change", "nll_change")
        }
        localization = {}
        faithfulness = {}
        stability = {}
        methods = ("gradcam", "chefer", "attention_rollout", "occlusion")
        for kind in ("image_only", "joint"):
            localization[kind] = {
                f"{method}_{region}{suffix}": np.full(
                    (6, 3) if suffix == "_random_mass" else (6,),
                    0.1,
                    dtype=np.float32,
                )
                for method in methods
                for region in ("marker", "lesion")
                for suffix in ("_mass", "_random_mass")
            }
            faithfulness[kind] = {
                "method_names": np.asarray(methods),
                "baseline_names": np.asarray(("background_canvas", "gaussian_blur")),
                "auc": np.zeros((6, 4, 2, 2), dtype=np.float32),
                "random_order_auc": np.zeros((6, 2, 2, 2), dtype=np.float32),
                "center_gaussian_auc": np.zeros((6, 2, 2), dtype=np.float32),
            }
            spearman = np.ones((3, 6, 4), dtype=np.float32)
            spearman[0, 0, 0] = np.nan
            stability[kind] = {
                "transform_names": np.asarray(
                    ("horizontal_flip", "brightness_0.9", "brightness_1.1")
                ),
                "map_names": np.asarray(methods),
                "prediction_changed": np.zeros((3, 6), dtype=bool),
                "valid_subset": np.ones((3, 6), dtype=bool),
                "spearman": spearman,
                "ssim": np.ones((3, 6, 4), dtype=np.float32),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = _secondary_statistics(
                Path(temp_dir),
                {"passes_primary_gate": False},
                {"labels": labels},
                {
                    "cue_sensitivity": cue,
                    "localization": localization,
                    "faithfulness": faithfulness,
                    "stability": stability,
                },
                {
                    "explanation_evaluation": {
                        "statistics": {
                            "descriptive_bootstrap_replicates": 25,
                            "descriptive_bootstrap_seed": 3,
                            "paired_signflip_draws": 32,
                            "paired_signflip_seed": 4,
                        }
                    }
                },
            )

        self.assertEqual(result["cue_mask_per_class"]["image_replacement"]["0"]["patient_count"], 1)
        self.assertEqual(
            result["stability"]["image_only"]["horizontal_flip"]["maps"]["gradcam"][
                "spearman_valid_count"
            ],
            5,
        )
        self.assertIn("image_only_deletion", result["faithfulness"])

    def test_test_evaluation_requires_all_checkpoints_and_frozen_temperatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_paths = {}
            for name in MODEL_NAMES:
                path = Path(temp_dir) / f"{name}.pt"
                path.write_bytes(b"checkpoint")
                checkpoint_paths[name] = path
            temperatures = {name: 1.2 for name in MODEL_NAMES}

            assert_test_ready(checkpoint_paths, temperatures)

            with self.assertRaises(RuntimeError):
                assert_test_ready(
                    checkpoint_paths,
                    {name: 1.2 for name in MODEL_NAMES if name != "joint"},
                )
            with self.assertRaises(RuntimeError):
                assert_test_ready(
                    {name: path for name, path in checkpoint_paths.items() if name != "joint"},
                    temperatures,
                )

    def test_preflight_is_read_only_and_emits_no_test_metrics(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / str(config["output_directory"])

            result = run_preflight(config, root, device_override=torch.device("cpu"))

            self.assertFalse(output_dir.exists())
            self.assertNotIn("test_predictions", result)
            self.assertNotIn("test_metrics", result)
            self.assertIn("device", result)
            self.assertIn("forward_backward_seconds", result)

    def test_atomic_writers_persist_complete_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "manifest.json"
            npz_path = root / "arrays.npz"

            atomic_write_json(json_path, {"stage": "preflight"})
            atomic_save_npz(npz_path, {"values": np.arange(4)})

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"stage": "preflight"})
            with np.load(npz_path) as saved:
                np.testing.assert_array_equal(saved["values"], np.arange(4))
            self.assertEqual({path.name for path in root.iterdir()}, {"manifest.json", "arrays.npz"})


if __name__ == "__main__":
    unittest.main()
