from __future__ import annotations

import unittest

from experiments.phase_g_v2_1.config import validate_v21_configuration


class PhaseGV21ConfigurationTests(unittest.TestCase):
    def test_no_condition_authorization_is_valid_and_safe(self):
        result = validate_v21_configuration({
            "authorized_conditions": [],
            "execution_enabled": False,
            "test_data_access": "forbidden",
            "protocol_status": "draft_unapproved",
        })

        self.assertEqual(result["authorized_conditions"], [])
        self.assertFalse(result["execution_enabled"])

    def test_balanced_only_allowlist_is_valid(self):
        result = validate_v21_configuration({
            "authorized_conditions": ["balanced"],
            "execution_enabled": False,
            "test_data_access": "forbidden",
            "protocol_status": "draft_unapproved",
        })

        self.assertEqual(result["authorized_conditions"], ["balanced"])

    def test_nonempty_authorization_requires_explicit_execution_enable(self):
        with self.assertRaisesRegex(ValueError, "execution_enabled"):
            validate_v21_configuration({
                "authorized_conditions": ["balanced"],
                "execution_enabled": True,
                "test_data_access": "forbidden",
                "protocol_status": "draft_unapproved",
            })

    def test_unknown_or_multiple_conditions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            validate_v21_configuration({
                "authorized_conditions": ["unknown"],
                "execution_enabled": False,
                "test_data_access": "forbidden",
                "protocol_status": "draft_unapproved",
            })
        with self.assertRaisesRegex(ValueError, "execution_enabled"):
            validate_v21_configuration({
                "authorized_conditions": ["balanced", "image_dominant"],
                "execution_enabled": True,
                "test_data_access": "forbidden",
                "protocol_status": "approved",
            })

    def test_test_data_access_must_be_forbidden(self):
        with self.assertRaisesRegex(ValueError, "test_data_access"):
            validate_v21_configuration({
                "authorized_conditions": [],
                "execution_enabled": False,
                "test_data_access": "allowed",
                "protocol_status": "draft_unapproved",
            })

if __name__ == "__main__":
    unittest.main()
