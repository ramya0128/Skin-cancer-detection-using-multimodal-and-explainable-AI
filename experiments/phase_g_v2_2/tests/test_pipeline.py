from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.phase_g.generate import SyntheticPatient
from experiments.phase_g.model import (
    MODEL_NAMES,
    TrainingResult,
    build_model,
    fit_temperature,
)
from experiments.phase_g_v2_2.pipeline import (
    calibrate_condition_models,
    mark_stage,
    prepare_run_directory,
    require_cuda,
    run_training_preflight,
    train_condition_models,
    validate_preflight_output_path,
    verify_stage,
)


ROOT = Path(__file__).resolve().parents[3]


def _patient(patient_id: str, label: int, partition: str) -> SyntheticPatient:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=bool)
    return SyntheticPatient(
        patient_id=patient_id,
        label=label,
        partition=partition,
        image=image,
        image_cue=label,
        clinical_value=label,
        background_canvas=image,
        lesion_mask=mask,
        marker_mask=mask,
        render_metadata={},
    )


class PhaseGV22PipelineTests(unittest.TestCase):
    def test_cuda_is_mandatory_and_only_cuda_zero_is_accepted(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                require_cuda()
        with self.assertRaisesRegex(ValueError, "cuda:0"):
            require_cuda("cpu")
        self.assertEqual(require_cuda(), torch.device("cuda:0"))

    def test_run_identity_and_stage_hashes_reject_mismatch_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "condition"
            source_hashes = {"model.py": "a" * 64}
            prepare_run_directory(
                output_dir,
                "b" * 64,
                source_hashes,
                resume=False,
            )
            artifact = output_dir / "training" / "history.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"selected_epoch":1}\n', encoding="utf-8")
            mark_stage(output_dir, "training", [artifact])

            self.assertTrue(verify_stage(output_dir, "training"))
            prepare_run_directory(
                output_dir,
                "b" * 64,
                source_hashes,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "hashes"):
                prepare_run_directory(
                    output_dir,
                    "c" * 64,
                    source_hashes,
                    resume=True,
                )
            artifact.write_text('{"selected_epoch":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                verify_stage(output_dir, "training")

    def test_training_refuses_missing_authorization_before_creating_outputs(self):
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "outputs" / "synthetic_phase_g_v2_2" / "conflicting"
            with self.assertRaisesRegex(ValueError, "authorization"):
                train_condition_models(
                    output_dir,
                    config,
                    [_patient("train-0", 0, "train")],
                    [_patient("selection-0", 0, "validation_model_selection")],
                    root=root,
                )
            self.assertFalse(output_dir.exists())

    def test_training_api_enforces_protocol_compatibility_before_cuda_or_output(self):
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static_config_path = root / "experiments" / "phase_g_v2_2" / "config.json"
            static_config_path.parent.mkdir(parents=True)
            static_config_path.write_bytes(
                (ROOT / "experiments/phase_g_v2_2/config.json").read_bytes()
            )
            output_dir = root / "outputs" / "synthetic_phase_g_v2_2" / "conflicting"
            with mock.patch(
                "experiments.phase_g_v2_2.pipeline.validate_run_authorization",
                return_value={
                    "execution_enabled": True,
                    "authorized_conditions": ["conflicting"],
                },
            ), mock.patch(
                "experiments.phase_g_v2_2.pipeline.require_cuda"
            ) as require_cuda, mock.patch(
                "experiments.phase_g_v2_2.pipeline.train_model",
                side_effect=AssertionError("training reached"),
            ):
                with self.assertRaisesRegex(ValueError, "protocol compatibility"):
                    train_condition_models(
                        output_dir,
                        config,
                        [_patient("train-0", 0, "train")],
                        [_patient("selection-0", 0, "validation_model_selection")],
                        authorization={"fixture": True},
                        root=root,
                    )

            require_cuda.assert_not_called()
            self.assertFalse(output_dir.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "requires RTX/CUDA execution")
    def test_interrupted_model_stage_resumes_without_overwriting_completed_models(self):
        config_path = ROOT / "experiments/phase_g/config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["pilot_condition"] = "conflicting"
        train_data = [
            _patient("train-0", 0, "train"),
            _patient("train-1", 1, "train"),
        ]
        selection_data = [
            _patient("selection-0", 0, "validation_model_selection"),
            _patient("selection-1", 1, "validation_model_selection"),
        ]
        calls = []
        interrupted = {"value": False}

        def fake_train_model(kind, actual_train, actual_selection, model_config):
            calls.append((kind, actual_train, actual_selection))
            model = build_model(kind, model_config)
            checkpoint_path = (
                Path(model_config["output_directory"])
                / "checkpoints"
                / f"{kind}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "kind": kind,
                    "seed": model_config["model"]["training"]["model_initialization_seeds"][kind],
                    "selected_epoch": 1,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            torch.zeros(1, device="cuda:0")
            torch.cuda.synchronize()
            if kind == "clinical_only" and not interrupted["value"]:
                interrupted["value"] = True
                checkpoint_path.with_name(checkpoint_path.name + ".tmp").write_text(
                    "interrupted", encoding="utf-8"
                )
                raise RuntimeError("simulated interruption")
            return TrainingResult(
                model=model,
                selected_epoch=1,
                history=[{"epoch": 1.0, "train_nll": 1.0, "selection_nll": 1.0}],
                checkpoint_path=checkpoint_path,
            )

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            static_config_path = project_root / "experiments" / "phase_g_v2_2" / "config.json"
            static_config_path.parent.mkdir(parents=True)
            static_config_path.write_bytes(
                (ROOT / "experiments/phase_g_v2_2/config.json").read_bytes()
            )
            output_dir = (
                project_root
                / "outputs"
                / "synthetic_phase_g_v2_2"
                / "conflicting"
            )
            with mock.patch(
                "experiments.phase_g_v2_2.pipeline.train_model",
                side_effect=fake_train_model,
            ), mock.patch(
                "experiments.phase_g_v2_2.pipeline.validate_run_authorization",
                return_value={"execution_enabled": True, "authorized_conditions": ["conflicting"]},
            ), mock.patch(
                "experiments.phase_g_v2_2.pipeline.validate_protocol_compatibility"
            ):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    train_condition_models(
                        output_dir,
                        config,
                        train_data,
                        selection_data,
                        authorization={"test_fixture": True},
                        root=project_root,
                        resume=False,
                    )
                image_checkpoint = (
                    output_dir
                    / "models"
                    / "image_only"
                    / "checkpoints"
                    / "image_only.pt"
                )
                image_hash = hashlib.sha256(
                    image_checkpoint.read_bytes()
                ).hexdigest()
                models = train_condition_models(
                    output_dir,
                    config,
                    train_data,
                    selection_data,
                    authorization={"test_fixture": True},
                    root=project_root,
                    resume=True,
                )

            self.assertEqual(set(models), set(MODEL_NAMES))
            self.assertEqual([kind for kind, _, _ in calls].count("image_only"), 1)
            self.assertEqual([kind for kind, _, _ in calls].count("clinical_only"), 2)
            self.assertEqual([kind for kind, _, _ in calls].count("joint"), 1)
            self.assertTrue(all(actual_selection is selection_data for _, _, actual_selection in calls))
            self.assertEqual(
                hashlib.sha256(image_checkpoint.read_bytes()).hexdigest(),
                image_hash,
            )

    def test_calibration_temperature_uses_only_supplied_calibration_labels(self):
        calibration_data = [
            _patient(f"cal-{index}", index, "validation_calibration")
            for index in range(6)
        ]
        labels = np.arange(6, dtype=np.int64)
        logits = np.eye(6, dtype=np.float64) * 2.0
        models = {kind: torch.nn.Linear(1, 1) for kind in MODEL_NAMES}
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "experiments.phase_g_v2_2.pipeline.predict_logits",
                side_effect=lambda model, data, kind, batch_size, device: logits.copy(),
            ) as predict:
                temperatures, saved_logits, saved_labels = calibrate_condition_models(
                    Path(temporary),
                    models,
                    calibration_data,
                    config,
                )

        self.assertTrue(np.array_equal(saved_labels, labels))
        for kind in MODEL_NAMES:
            np.testing.assert_array_equal(saved_logits[kind], logits)
            self.assertAlmostEqual(
                temperatures[kind], fit_temperature(logits, labels), places=8
            )
            self.assertIs(
                predict.call_args_list[MODEL_NAMES.index(kind)].args[1],
                calibration_data,
            )

    def test_training_preflight_rejects_batch_size_substitution(self):
        config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )
        config["model"]["training"]["batch_size"] = 32
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "batch size 64"):
                run_training_preflight(config, Path(temporary))

    def test_preflight_output_path_cannot_be_inside_a_condition_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_path = (
                Path(temporary)
                / "outputs"
                / "synthetic_phase_g_v2_2"
                / "image_dominant"
                / "preflight"
            )
            with self.assertRaisesRegex(ValueError, "outside condition result"):
                validate_preflight_output_path(result_path, Path(temporary))


if __name__ == "__main__":
    unittest.main()
