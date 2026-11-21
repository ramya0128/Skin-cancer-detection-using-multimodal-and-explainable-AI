from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase_g_v2_2.execution import validate_execution_request


class ExecutionAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.static_config = {
            "schema_version": 1,
            "implementation_version": "phase_g_v2_2",
            "protocol_status": "approved",
            "execution_enabled": False,
            "test_data_access": "forbidden",
            "authorized_conditions": [],
            "remaining_conditions": [
                "image_dominant",
                "clinical_dominant",
                "redundant",
                "conflicting",
                "synergistic",
            ],
        }
        self._write_json(
            "experiments/phase_g_v2_2/config.json", self.static_config
        )
        self._write_json("experiments/phase_g/config.json", {"training": "frozen"})
        self._write_text(
            "docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md",
            "**Status:** Approved for execution.\n",
        )
        self._write_json("approved_cohort.json", {"patient_ids": ["p1", "p2"]})
        self.authorization = {
            "schema_version": 1,
            "protocol_status": "approved",
            "execution_enabled": True,
            "test_data_access": "allowed",
            "authorized_conditions": ["conflicting"],
            "config_sha256": self._sha256(
                self.root / "experiments/phase_g_v2_2/config.json"
            ),
            "phase_g_config_sha256": self._sha256(
                self.root / "experiments/phase_g/config.json"
            ),
            "protocol_sha256": self._sha256(
                self.root / "docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md"
            ),
            "evaluation_cohort_manifest": {
                "path": "approved_cohort.json",
                "sha256": self._sha256(self.root / "approved_cohort.json"),
            },
            "predictive_decision_policy": {
                "nll_effect_threshold": 0.1,
                "multiplicity_method": "holm",
                "multiplicity_family_size": 2,
                "one_sided_alpha": 0.05,
            },
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_text(self, relative_path: str, value: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def _write_json(self, relative_path: str, value: dict) -> Path:
        return self._write_text(
            relative_path, json.dumps(value, sort_keys=True) + "\n"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_execution_is_disabled_by_default(self):
        result = validate_execution_request({
            "authorized_conditions": [],
            "execution_enabled": False,
            "test_data_access": "forbidden",
            "protocol_status": "draft_unapproved",
        })
        self.assertFalse(result["execution_enabled"])

    def test_nonempty_authorization_requires_approved_protocol_and_explicit_enable(self):
        with self.assertRaisesRegex(ValueError, "execution_enabled"):
            validate_execution_request({
                "authorized_conditions": ["conflicting"],
                "execution_enabled": True,
                "test_data_access": "forbidden",
                "protocol_status": "approved",
            })

    def test_only_exact_supported_conditions_can_be_authorized(self):
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            validate_execution_request({
                "authorized_conditions": ["balanced"],
                "execution_enabled": False,
                "test_data_access": "forbidden",
                "protocol_status": "approved",
            })

    def test_test_data_access_must_be_forbidden_for_planning(self):
        with self.assertRaisesRegex(ValueError, "test_data_access"):
            validate_execution_request({
                "authorized_conditions": [],
                "execution_enabled": False,
                "test_data_access": "allowed",
                "protocol_status": "draft_unapproved",
            })

    def test_runner_request_never_starts_execution(self):
        from experiments.phase_g_v2_2.runner import validate_remaining_condition_request

        result = validate_remaining_condition_request(
            Path(__file__).resolve().parents[3],
            ["conflicting"],
        )

        self.assertFalse(result["execution_started"])
        self.assertFalse(result["execution_enabled"])

    def test_static_config_requires_disabled_defaults(self):
        from experiments.phase_g_v2_2.execution import validate_static_config

        result = validate_static_config(self.static_config)
        self.assertEqual(result["authorized_conditions"], [])
        self.assertFalse(result["execution_enabled"])
        self.assertEqual(result["test_data_access"], "forbidden")

    def test_authorized_condition_requires_matching_hashes_and_policy(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        result = validate_run_authorization(
            self.static_config,
            self.authorization,
            "conflicting",
            self.root,
        )
        self.assertEqual(result["authorized_conditions"], ["conflicting"])
        self.assertTrue(result["execution_enabled"])

    def test_authorization_cannot_override_a_draft_checked_in_protocol_status(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        draft_config = {**self.static_config, "protocol_status": "draft_unapproved"}
        draft_config_path = self._write_json(
            "experiments/phase_g_v2_2/config.json", draft_config
        )
        authorization = {
            **self.authorization,
            "config_sha256": self._sha256(draft_config_path),
        }

        with self.assertRaisesRegex(ValueError, "protocol.*approved"):
            validate_run_authorization(
                draft_config, authorization, "conflicting", self.root
            )

    def test_authorization_requires_the_hashed_protocol_document_to_be_approved(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        protocol_path = self._write_text(
            "docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md",
            "**Status:** Frozen as protocol structure; not approved for execution.\n",
        )
        authorization = {
            **self.authorization,
            "protocol_sha256": self._sha256(protocol_path),
        }

        with self.assertRaisesRegex(ValueError, "protocol document"):
            validate_run_authorization(
                self.static_config, authorization, "conflicting", self.root
            )

    def test_authorization_rejects_draft_protocol_and_missing_approvals(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        invalid = dict(self.authorization)
        invalid["protocol_status"] = "draft_unapproved"
        with self.assertRaisesRegex(ValueError, "approved"):
            validate_run_authorization(
                self.static_config, invalid, "conflicting", self.root
            )

        for field in ("predictive_decision_policy", "evaluation_cohort_manifest"):
            invalid = dict(self.authorization)
            invalid.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_run_authorization(
                    self.static_config, invalid, "conflicting", self.root
                )

    def test_authorization_rejects_mismatched_source_and_cohort_hashes(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        for field in ("config_sha256", "phase_g_config_sha256", "protocol_sha256"):
            invalid = dict(self.authorization)
            invalid[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hash"):
                validate_run_authorization(
                    self.static_config, invalid, "conflicting", self.root
                )

        invalid = dict(self.authorization)
        invalid["evaluation_cohort_manifest"] = {
            **self.authorization["evaluation_cohort_manifest"],
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_run_authorization(
                self.static_config, invalid, "conflicting", self.root
            )

    def test_authorization_rejects_unlisted_duplicate_and_unknown_conditions(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        for conditions, requested in (
            (["image_dominant"], "conflicting"),
            (["conflicting", "conflicting"], "conflicting"),
            (["balanced"], "balanced"),
        ):
            invalid = dict(self.authorization)
            invalid["authorized_conditions"] = conditions
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                validate_run_authorization(
                    self.static_config, invalid, requested, self.root
                )

    def test_authorization_rejects_non_string_condition_entries(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        invalid = dict(self.authorization)
        invalid["authorized_conditions"] = [{"condition": "conflicting"}]
        with self.assertRaisesRegex(ValueError, "strings"):
            validate_run_authorization(
                self.static_config, invalid, "conflicting", self.root
            )

    def test_output_directory_is_condition_local_and_rejects_unknown_conditions(self):
        from experiments.phase_g_v2_2.execution import condition_output_directory

        expected = (
            self.root / "outputs" / "synthetic_phase_g_v2_2" / "conflicting"
        )
        self.assertEqual(
            condition_output_directory(self.root, "conflicting"), expected
        )
        with self.assertRaisesRegex(ValueError, "condition"):
            condition_output_directory(self.root, "balanced")

    def test_authorization_rejects_cohort_path_escape_and_invalid_policy(self):
        from experiments.phase_g_v2_2.execution import validate_run_authorization

        invalid = dict(self.authorization)
        invalid["evaluation_cohort_manifest"] = {
            "path": "..\\outside.json",
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_run_authorization(
                self.static_config, invalid, "conflicting", self.root
            )

        invalid = dict(self.authorization)
        invalid["predictive_decision_policy"] = {
            **self.authorization["predictive_decision_policy"],
            "multiplicity_family_size": 0,
        }
        with self.assertRaisesRegex(ValueError, "policy"):
            validate_run_authorization(
                self.static_config, invalid, "conflicting", self.root
            )


if __name__ == "__main__":
    unittest.main()
