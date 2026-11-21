from __future__ import annotations

import unittest

from experiments.phase_g_v2_2.cohort import (
    sha256_json,
    validate_cohort_manifest,
)


def _manifest(per_class: int = 2):
    patient_ids = [f"fresh-{label}-{index}" for label in range(6) for index in range(per_class)]
    labels = {
        patient_id: int(patient_id.split("-")[1])
        for patient_id in patient_ids
    }
    membership = sha256_json(
        {
            "patient_ids": patient_ids,
            "labels_by_patient": labels,
        }
    )
    return {
        "schema_version": 1,
        "cohort_name": "fresh_fixture",
        "master_seed": 20261121,
        "patient_ids": patient_ids,
        "labels_by_patient": labels,
        "class_counts": {str(label): per_class for label in range(6)},
        "cohort_membership_sha256": membership,
        "test_labels_used": False,
        "protocol_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }


class PhaseGV22CohortTests(unittest.TestCase):
    def test_fresh_balanced_cohort_is_configurable_not_hardcoded(self):
        manifest = _manifest(per_class=3)
        result = validate_cohort_manifest(
            manifest,
            development_patient_ids=["old-1"],
            historical_test_patient_ids=["pilot-1"],
            historical_validation_patient_ids=["v2-72-1"],
            protocol_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        self.assertEqual(result["patient_count"], 18)
        self.assertEqual(result["class_counts"], {str(label): 3 for label in range(6)})

    def test_duplicate_and_overlap_ids_are_rejected(self):
        manifest = _manifest()
        manifest["patient_ids"][1] = manifest["patient_ids"][0]
        manifest["labels_by_patient"] = {
            patient_id: label
            for patient_id, label in manifest["labels_by_patient"].items()
        }
        manifest["cohort_membership_sha256"] = sha256_json(
            {
                "patient_ids": manifest["patient_ids"],
                "labels_by_patient": manifest["labels_by_patient"],
            }
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_cohort_manifest(manifest, development_patient_ids=[])

        manifest = _manifest()
        with self.assertRaisesRegex(ValueError, "development"):
            validate_cohort_manifest(
                manifest,
                development_patient_ids=[manifest["patient_ids"][0]],
            )

    def test_test_labels_and_hashes_must_be_frozen(self):
        manifest = _manifest()
        manifest["test_labels_used"] = True
        with self.assertRaisesRegex(ValueError, "test_labels_used"):
            validate_cohort_manifest(manifest, development_patient_ids=[])

        manifest = _manifest()
        manifest["cohort_membership_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "membership"):
            validate_cohort_manifest(manifest, development_patient_ids=[])


if __name__ == "__main__":
    unittest.main()
