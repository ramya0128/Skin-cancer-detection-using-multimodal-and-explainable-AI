import hashlib
import json
import tempfile
import unittest
import copy
from pathlib import Path
from unittest.mock import patch

from experiments.phase_g.io import (
    hash_protected_inputs,
    prepare_run_dir,
    validate_config,
)
from experiments.phase_f.io import PROTECTED_INPUTS


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class PhaseGIOTests(unittest.TestCase):
    def test_new_run_directory_records_identity_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pilot"
            prepare_run_dir(
                output_dir,
                config_hash="config-sha",
                source_hashes={"generate.py": "source-sha"},
                resume=False,
            )
            identity_path = output_dir / "run_identity.json"
            self.assertTrue(identity_path.is_file())
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            self.assertEqual(identity["config_sha256"], "config-sha")
            self.assertEqual(identity["source_sha256"], {"generate.py": "source-sha"})

            with self.assertRaises(FileExistsError):
                prepare_run_dir(
                    output_dir,
                    config_hash="config-sha",
                    source_hashes={"generate.py": "source-sha"},
                    resume=False,
                )

    def test_resume_requires_exact_config_and_source_hash_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pilot"
            prepare_run_dir(
                output_dir,
                config_hash="config-sha",
                source_hashes={"generate.py": "source-sha"},
                resume=False,
            )
            prepare_run_dir(
                output_dir,
                config_hash="config-sha",
                source_hashes={"generate.py": "source-sha"},
                resume=True,
            )
            with self.assertRaises(ValueError):
                prepare_run_dir(
                    output_dir,
                    config_hash="different-config",
                    source_hashes={"generate.py": "source-sha"},
                    resume=True,
                )
            with self.assertRaises(ValueError):
                prepare_run_dir(
                    output_dir,
                    config_hash="config-sha",
                    source_hashes={"generate.py": "different-source"},
                    resume=True,
                )

    def test_interrupted_run_identity_write_leaves_no_partial_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pilot"

            with patch.object(Path, "write_text", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    prepare_run_dir(
                        output_dir,
                        config_hash="config-sha",
                        source_hashes={"run.py": "source-sha"},
                        resume=False,
                    )

            self.assertFalse((output_dir / "run_identity.json").exists())
            self.assertFalse(output_dir.exists())

    def test_protected_hashing_reads_raw_bytes_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected: dict[str, str] = {}
            for index, relative_path in enumerate(PROTECTED_INPUTS):
                payload = f"protected-payload-{index}".encode("ascii")
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                expected[str(relative_path)] = hashlib.sha256(payload).hexdigest()

            with patch("numpy.load", side_effect=AssertionError("must hash bytes only")):
                hashes = hash_protected_inputs(root)

            self.assertEqual(hashes, expected)

    def test_protected_hashing_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                hash_protected_inputs(Path(temp_dir))

    def test_only_balanced_condition_is_valid_for_execution(self):
        config = load_config()
        validate_config(config)

        unauthorized = dict(config)
        unauthorized["pilot_condition"] = "image_dominant"
        with self.assertRaises(ValueError):
            validate_config(unauthorized)

    def test_explanation_and_secondary_statistics_are_frozen_in_config(self):
        config = load_config()

        invalid = copy.deepcopy(config)
        invalid["explanation_evaluation"]["faithfulness"]["random_order_count"] = 19
        with self.assertRaises(ValueError):
            validate_config(invalid)

        invalid = copy.deepcopy(config)
        invalid["explanation_evaluation"]["statistics"]["paired_signflip_draws"] = 9999
        with self.assertRaises(ValueError):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
