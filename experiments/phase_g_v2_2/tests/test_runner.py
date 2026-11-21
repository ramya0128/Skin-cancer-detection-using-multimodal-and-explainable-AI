from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from experiments.phase_g_v2_2.data import REMAINING_CONDITION_NAMES
from experiments.phase_g_v2_2.runner import (
    main,
    run_authorized_test_stage,
    run_condition,
)


class PhaseGV22RunnerCliTests(unittest.TestCase):
    def test_default_command_validates_without_running(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch(
            "experiments.phase_g_v2_2.runner.run_condition"
        ) as run_condition:
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("execution disabled", output.getvalue().lower())
        self.assertIn("protocol compatibility blocked", output.getvalue().lower())
        run_condition.assert_not_called()

    def test_all_remaining_conditions_can_be_dry_run_validated(self):
        with mock.patch("experiments.phase_g_v2_2.runner.run_condition") as run_condition:
            for condition in REMAINING_CONDITION_NAMES:
                with self.subTest(condition=condition):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(main(["--condition", condition]), 0)
                    self.assertIn(condition, output.getvalue())
        run_condition.assert_not_called()

    def test_execution_requires_an_authorization_file(self):
        with self.assertRaises(SystemExit) as error, mock.patch(
            "experiments.phase_g_v2_2.runner.run_condition"
        ) as run_condition:
            main(["--run", "--condition", "conflicting"])

        self.assertEqual(error.exception.code, 2)
        run_condition.assert_not_called()

    def test_unknown_condition_is_rejected_by_cli(self):
        with self.assertRaises(SystemExit) as error:
            main(["--condition", "balanced"])
        self.assertEqual(error.exception.code, 2)

    def test_resume_is_only_valid_with_run(self):
        with self.assertRaises(SystemExit) as error:
            main(["--condition", "conflicting", "--resume"])
        self.assertEqual(error.exception.code, 2)

    def test_test_stage_requires_frozen_models_before_test_data_access(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static_config_path = root / "experiments" / "phase_g_v2_2" / "config.json"
            static_config_path.parent.mkdir(parents=True)
            static_config_path.write_bytes(
                (
                    Path(__file__).resolve().parents[3]
                    / "experiments/phase_g_v2_2/config.json"
                ).read_bytes()
            )
            output_dir = (
                root / "outputs" / "synthetic_phase_g_v2_2" / "conflicting"
            )
            with mock.patch(
                "experiments.phase_g_v2_2.runner.validate_run_authorization",
                return_value={
                    "authorized_conditions": ["conflicting"],
                    "execution_enabled": True,
                    "evaluation_cohort_manifest": {"patient_ids": ["p1"]},
                },
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.generate_condition_split"
            ) as generate_test_data:
                with self.assertRaisesRegex(RuntimeError, "frozen"):
                    run_authorized_test_stage(
                        output_dir,
                        "conflicting",
                        {},
                        {"fixture": True},
                        root=root,
                    )

            generate_test_data.assert_not_called()
            self.assertFalse(output_dir.exists())

    def test_protocol_mismatch_blocks_training_before_patient_generation(self):
        import tempfile
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "experiments/phase_g_v2_2/config.json",
                "experiments/phase_g/config.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((project_root / relative).read_bytes())
            authorization = {"fixture": True}
            with mock.patch(
                "experiments.phase_g_v2_2.runner.validate_run_authorization",
                return_value={
                    "authorized_conditions": ["conflicting"],
                    "execution_enabled": True,
                },
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.make_patient_splits"
            ) as make_splits:
                with self.assertRaisesRegex(ValueError, "protocol compatibility"):
                    run_condition(root, "conflicting", authorization)

            make_splits.assert_not_called()

    def test_training_refuses_insufficient_ram_before_patient_generation(self):
        import tempfile
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "experiments/phase_g_v2_2/config.json",
                "experiments/phase_g/config.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((project_root / relative).read_bytes())
            with mock.patch(
                "experiments.phase_g_v2_2.runner.validate_run_authorization",
                return_value={
                    "authorized_conditions": ["conflicting"],
                    "execution_enabled": True,
                },
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.validate_protocol_compatibility"
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.require_training_memory",
                side_effect=RuntimeError("insufficient system memory"),
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.make_patient_splits"
            ) as make_splits:
                with self.assertRaisesRegex(RuntimeError, "insufficient"):
                    run_condition(root, "conflicting", {"fixture": True})

            make_splits.assert_not_called()

    def test_existing_condition_output_is_rejected_before_rendering_or_overwrite(self):
        import tempfile
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "experiments/phase_g_v2_2/config.json",
                "experiments/phase_g/config.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((project_root / relative).read_bytes())
            output_dir = (
                root / "outputs" / "synthetic_phase_g_v2_2" / "conflicting"
            )
            output_dir.mkdir(parents=True)
            sentinel = output_dir / "user-file.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with mock.patch(
                "experiments.phase_g_v2_2.runner.validate_run_authorization",
                return_value={
                    "authorized_conditions": ["conflicting"],
                    "execution_enabled": True,
                },
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.validate_protocol_compatibility"
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.require_training_memory",
                return_value={},
            ), mock.patch(
                "experiments.phase_g_v2_2.runner.make_patient_splits"
            ) as make_splits:
                with self.assertRaises(FileExistsError):
                    run_condition(root, "conflicting", {"fixture": True})

            make_splits.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
