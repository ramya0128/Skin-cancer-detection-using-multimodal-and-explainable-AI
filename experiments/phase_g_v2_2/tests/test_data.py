from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.phase_g.generate import (
    GLYPHS,
    generate_cues,
    generate_split,
    make_patient_splits,
)
from experiments.phase_g_v2_2.data import (
    audit_condition_cues,
    condition_config,
    generate_condition_split,
    make_condition_assignments,
)


ROOT = Path(__file__).resolve().parents[3]
REMAINING_CONDITIONS = (
    "image_dominant",
    "clinical_dominant",
    "redundant",
    "conflicting",
    "synergistic",
)


class PhaseGV22ConditionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "experiments/phase_g/config.json").read_text(encoding="utf-8")
        )

    def test_condition_config_selects_only_remaining_conditions(self):
        original = copy.deepcopy(self.config)

        for condition in REMAINING_CONDITIONS:
            runtime = condition_config(self.config, condition)
            self.assertEqual(runtime["pilot_condition"], condition)
            self.assertEqual(runtime["model"], self.config["model"])
        self.assertEqual(self.config, original)
        with self.assertRaisesRegex(ValueError, "condition"):
            condition_config(self.config, "balanced")

    def test_test_partition_generation_is_forbidden(self):
        patient_id = "forbidden-test-patient"
        with self.assertRaisesRegex(ValueError, "test rendering is forbidden"):
            generate_condition_split(
                [patient_id],
                {patient_id: 0},
                "test",
                self.config,
                "image_dominant",
            )

    def test_patient_splits_remain_class_balanced_and_disjoint(self):
        splits, labels = make_patient_splits(self.config)
        all_ids = []
        expected = {
            "train": 1400,
            "validation_model_selection": 150,
            "validation_calibration": 150,
            "test": 300,
        }

        for partition, per_class in expected.items():
            ids = splits[partition]
            self.assertEqual(len(ids), per_class * 6)
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(
                np.bincount([labels[patient_id] for patient_id in ids], minlength=6).tolist(),
                [per_class] * 6,
            )
            all_ids.extend(ids)
        self.assertEqual(len(set(all_ids)), 12000)

    def test_conflict_assignments_are_deterministic_exact_and_reach_renderer(self):
        patient_ids = [f"patient-{index}" for index in range(400)]
        assignments = make_condition_assignments(
            patient_ids, "conflicting", seed=20261119
        )
        self.assertEqual(
            assignments,
            make_condition_assignments(patient_ids, "conflicting", seed=20261119),
        )
        counts = {
            orientation: sum(value == orientation for value in assignments.values())
            for orientation in ("image_correct", "clinical_correct", "none")
        }
        self.assertEqual(
            counts,
            {"image_correct": 100, "clinical_correct": 100, "none": 200},
        )

        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        rendered = generate_condition_split(
            patient_ids,
            labels,
            "train",
            self.config,
            "conflicting",
            assignments,
        )
        records = {patient.patient_id: patient for patient in rendered}
        for patient_id, orientation in assignments.items():
            patient = records[patient_id]
            self.assertEqual(patient.partition, "train")
            self.assertEqual(patient.render_metadata["conflict_orientation"], orientation)
            if orientation == "image_correct":
                self.assertEqual(patient.image_cue, patient.label)
                self.assertNotEqual(patient.clinical_value, patient.label)
            elif orientation == "clinical_correct":
                self.assertNotEqual(patient.image_cue, patient.label)
                self.assertEqual(patient.clinical_value, patient.label)

        audit = audit_condition_cues(rendered, "conflicting")
        self.assertAlmostEqual(audit["image_reliability"], 0.675, delta=0.05)
        self.assertAlmostEqual(audit["clinical_reliability"], 0.675, delta=0.05)
        self.assertEqual(audit["population_counts"], counts)
        independent = [
            records[patient_id]
            for patient_id, orientation in assignments.items()
            if orientation == "none"
        ]
        joint_correctness = np.asarray(
            [
                (
                    patient.image_cue == patient.label,
                    patient.clinical_value == patient.label,
                )
                for patient in independent
            ],
            dtype=bool,
        )
        observed_joint = np.asarray(
            [
                np.mean(joint_correctness[:, 0] & joint_correctness[:, 1]),
                np.mean(joint_correctness[:, 0] & ~joint_correctness[:, 1]),
                np.mean(~joint_correctness[:, 0] & joint_correctness[:, 1]),
                np.mean(~joint_correctness[:, 0] & ~joint_correctness[:, 1]),
            ]
        )
        expected_joint = np.asarray([0.85**2, 0.85 * 0.15, 0.15 * 0.85, 0.15**2])
        tolerance = 5 * np.sqrt(expected_joint * (1 - expected_joint) / len(independent))
        self.assertTrue(np.all(np.abs(observed_joint - expected_joint) <= tolerance))

    def test_base_split_requires_and_passes_conflict_orientations(self):
        patient_ids = [f"base-conflict-{index}" for index in range(24)]
        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        config = condition_config(self.config, "conflicting")
        assignments = make_condition_assignments(
            patient_ids, "conflicting", seed=20261155
        )

        with self.assertRaisesRegex(ValueError, "orientation for every patient"):
            generate_split(patient_ids, labels, "train", config)

        patients = generate_split(
            patient_ids,
            labels,
            "train",
            config,
            conflict_orientations=assignments,
        )
        self.assertEqual(
            {
                orientation: sum(
                    patient.render_metadata["conflict_orientation"] == orientation
                    for patient in patients
                )
                for orientation in ("image_correct", "clinical_correct", "none")
            },
            {"image_correct": 6, "clinical_correct": 6, "none": 12},
        )
        for patient in patients:
            orientation = patient.render_metadata["conflict_orientation"]
            if orientation == "image_correct":
                self.assertEqual(patient.image_cue, patient.label)
                self.assertNotEqual(patient.clinical_value, patient.label)
            elif orientation == "clinical_correct":
                self.assertNotEqual(patient.image_cue, patient.label)
                self.assertEqual(patient.clinical_value, patient.label)

    def test_generated_patients_match_each_conditions_cue_distribution(self):
        patient_count = 1200
        labels = {
            f"generated-{condition}-{index}": index % 6
            for condition in REMAINING_CONDITIONS
            for index in range(patient_count)
        }

        for condition_index, condition in enumerate(REMAINING_CONDITIONS):
            with self.subTest(condition=condition):
                condition_ids = [
                    f"generated-{condition}-{index}"
                    for index in range(patient_count)
                ]
                assignments = (
                    make_condition_assignments(
                        condition_ids, condition, seed=20261160 + condition_index
                    )
                    if condition == "conflicting"
                    else None
                )
                patients = generate_condition_split(
                    condition_ids,
                    labels,
                    "train",
                    self.config,
                    condition,
                    assignments,
                )
                audit = audit_condition_cues(patients, condition)
                self.assertEqual(audit["patient_count"], patient_count)
                self.assertAlmostEqual(
                    sum(
                        sum(sum(label_counts) for label_counts in matrix)
                        for matrix in audit["joint_cue_counts_by_label"].values()
                    ),
                    patient_count,
                )

                rendered_labels = np.asarray([patient.label for patient in patients])
                image_cues = np.asarray([patient.image_cue for patient in patients])
                clinical_cues = np.asarray(
                    [patient.clinical_value for patient in patients]
                )
                if condition in {"image_dominant", "clinical_dominant"}:
                    expected_image, expected_clinical = {
                        "image_dominant": (0.95, 0.60),
                        "clinical_dominant": (0.60, 0.95),
                    }[condition]
                    for observed, expected in (
                        (audit["image_reliability"], expected_image),
                        (audit["clinical_reliability"], expected_clinical),
                    ):
                        tolerance = 5 * np.sqrt(expected * (1 - expected) / patient_count)
                        self.assertAlmostEqual(observed, expected, delta=tolerance)

                    image_correct = image_cues == rendered_labels
                    clinical_correct = clinical_cues == rendered_labels
                    joint_events = np.asarray(
                        [
                            image_correct & clinical_correct,
                            image_correct & ~clinical_correct,
                            ~image_correct & clinical_correct,
                            ~image_correct & ~clinical_correct,
                        ]
                    )
                    expected_events = np.asarray(
                        [
                            expected_image * expected_clinical,
                            expected_image * (1 - expected_clinical),
                            (1 - expected_image) * expected_clinical,
                            (1 - expected_image) * (1 - expected_clinical),
                        ]
                    )
                    observed_events = joint_events.mean(axis=1)
                    tolerances = 5 * np.sqrt(
                        expected_events * (1 - expected_events) / patient_count
                    )
                    self.assertTrue(
                        np.all(np.abs(observed_events - expected_events) <= tolerances)
                    )
                elif condition == "redundant":
                    np.testing.assert_array_equal(image_cues, clinical_cues)
                    tolerance = 5 * np.sqrt(0.85 * 0.15 / patient_count)
                    self.assertAlmostEqual(
                        audit["image_reliability"], 0.85, delta=tolerance
                    )
                elif condition == "conflicting":
                    self.assertEqual(
                        audit["population_counts"],
                        {
                            "image_correct": patient_count // 4,
                            "clinical_correct": patient_count // 4,
                            "none": patient_count // 2,
                        },
                    )
                    self.assertAlmostEqual(
                        audit["image_reliability"], 0.675, delta=0.05
                    )
                    self.assertAlmostEqual(
                        audit["clinical_reliability"], 0.675, delta=0.05
                    )
                    independent = [
                        patient
                        for patient in patients
                        if patient.render_metadata["conflict_orientation"] == "none"
                    ]
                    both_correct = np.mean(
                        [
                            patient.image_cue == patient.label
                            and patient.clinical_value == patient.label
                            for patient in independent
                        ]
                    )
                    tolerance = 5 * np.sqrt(0.85**2 * (1 - 0.85**2) / len(independent))
                    self.assertAlmostEqual(both_correct, 0.85**2, delta=tolerance)
                else:
                    self.assertAlmostEqual(
                        audit["modulo_six_decoding_accuracy"],
                        0.727,
                        delta=5 * np.sqrt(0.727 * (1 - 0.727) / patient_count),
                    )
                    for label in range(6):
                        class_patients = [
                            patient for patient in patients if patient.label == label
                        ]
                        for cue_name in ("image_cue", "clinical_value"):
                            counts = np.bincount(
                                [
                                    getattr(patient, cue_name)
                                    for patient in class_patients
                                ],
                                minlength=6,
                            )
                            expected = len(class_patients) / 6
                            tolerance = 5 * np.sqrt(
                                expected * (1 - 1 / 6)
                            )
                            self.assertTrue(
                                np.all(np.abs(counts - expected) <= tolerance)
                            )
                if condition == "conflicting":
                    del independent
                if condition == "synergistic":
                    del class_patients
                del patients

    def test_independent_condition_reliabilities_match_specification(self):
        count_per_class = 5000
        labels = np.repeat(np.arange(6), count_per_class)
        expected = {
            "image_dominant": (0.95, 0.60),
            "clinical_dominant": (0.60, 0.95),
        }
        for index, (condition, reliabilities) in enumerate(expected.items()):
            rngs = {
                "image_cue_noise": np.random.default_rng(20261130 + index),
                "clinical_cue_noise": np.random.default_rng(20261230 + index),
                "shared": np.random.default_rng(20261330 + index),
                "image_share_noise": np.random.default_rng(20261430 + index),
                "clinical_share_noise": np.random.default_rng(20261530 + index),
            }
            cues = np.asarray(
                [generate_cues(int(label), condition, rngs) for label in labels]
            )
            observed = tuple(
                float(np.mean(cues[:, column] == labels))
                for column in range(2)
            )
            for actual, target in zip(observed, reliabilities, strict=True):
                self.assertAlmostEqual(actual, target, delta=0.01)

    def test_redundant_cues_are_identical_and_have_shared_reliability(self):
        count_per_class = 5000
        labels = np.repeat(np.arange(6), count_per_class)
        rngs = {
            "image_cue_noise": np.random.default_rng(20261140),
            "clinical_cue_noise": np.random.default_rng(20261141),
            "shared": np.random.default_rng(20261142),
            "image_share_noise": np.random.default_rng(20261143),
            "clinical_share_noise": np.random.default_rng(20261144),
        }
        cues = np.asarray(
            [generate_cues(int(label), "redundant", rngs) for label in labels]
        )

        np.testing.assert_array_equal(cues[:, 0], cues[:, 1])
        self.assertAlmostEqual(float(np.mean(cues[:, 0] == labels)), 0.85, delta=0.01)

    def test_synergistic_share_errors_preserve_only_other_five_values(self):
        count_per_class = 10000
        labels = np.repeat(np.arange(6), count_per_class)
        rngs = {
            "image_cue_noise": np.random.default_rng(20261150),
            "clinical_cue_noise": np.random.default_rng(20261151),
            "shared": np.random.default_rng(20261152),
            "image_share_noise": np.random.default_rng(20261153),
            "clinical_share_noise": np.random.default_rng(20261154),
        }
        cues = np.asarray(
            [generate_cues(int(label), "synergistic", rngs) for label in labels]
        )
        decoded = (cues[:, 0] + cues[:, 1]) % 6

        self.assertTrue(np.all((cues >= 0) & (cues < 6)))
        self.assertAlmostEqual(float(np.mean(decoded == labels)), 0.727, delta=0.01)
        for label in range(6):
            image_share_counts = np.bincount(
                cues[labels == label, 0], minlength=6
            ) / count_per_class
            clinical_share_counts = np.bincount(
                cues[labels == label, 1], minlength=6
            ) / count_per_class
            np.testing.assert_allclose(image_share_counts, np.full(6, 1 / 6), atol=0.02)
            np.testing.assert_allclose(
                clinical_share_counts, np.full(6, 1 / 6), atol=0.02
            )

    def test_nonconflicting_records_are_rendered_by_the_requested_condition(self):
        patient_id = "patient-rendered"
        for condition in (
            "image_dominant",
            "clinical_dominant",
            "redundant",
            "synergistic",
        ):
            patient = generate_condition_split(
                [patient_id],
                {patient_id: 4},
                "train",
                self.config,
                condition,
            )[0]
            self.assertEqual(patient.partition, "train")
            self.assertEqual(patient.image.shape, (224, 224, 3))
            self.assertIn(patient.image_cue, range(6))
            self.assertIn(patient.clinical_value, range(6))
            self.assertEqual(condition_config(self.config, condition)["pilot_condition"], condition)

    def test_every_condition_encodes_image_and_clinical_cues_without_altering_lesions(self):
        patient_ids = [f"visual-cue-{index}" for index in range(24)]
        labels = {patient_id: index % 6 for index, patient_id in enumerate(patient_ids)}
        lesion_masks = {}

        for condition in REMAINING_CONDITIONS:
            assignments = (
                make_condition_assignments(
                    patient_ids, condition, seed=20261170
                )
                if condition == "conflicting"
                else None
            )
            patients = generate_condition_split(
                patient_ids,
                labels,
                "train",
                self.config,
                condition,
                assignments,
            )
            for patient in patients:
                marker_y, marker_x = patient.render_metadata["marker_top_left"]
                marker = patient.image[marker_y:marker_y + 16, marker_x:marker_x + 16]
                decoded_glyph = tuple(
                    "".join(
                        "1"
                        if marker[row * 4:(row + 1) * 4, column * 4:(column + 1) * 4]
                        .mean()
                        < 128
                        else "0"
                        for column in range(4)
                    )
                    for row in range(4)
                )
                self.assertEqual(decoded_glyph, GLYPHS[patient.image_cue])
                self.assertIn(patient.clinical_value, range(6))
                self.assertEqual(patient.lesion_mask.shape, (224, 224))
                self.assertEqual(patient.lesion_mask.dtype, np.bool_)
                self.assertTrue(patient.lesion_mask.any())
                self.assertEqual(patient.marker_mask.dtype, np.bool_)
                self.assertEqual(int(patient.marker_mask.sum()), 16 * 16)
                self.assertFalse(np.any(patient.lesion_mask & patient.marker_mask))
                self.assertTrue(patient.marker_mask.any())

                if patient.patient_id in lesion_masks:
                    np.testing.assert_array_equal(
                        lesion_masks[patient.patient_id], patient.lesion_mask
                    )
                else:
                    lesion_masks[patient.patient_id] = patient.lesion_mask.copy()

            repeated_id = patient_ids[0]
            repeated_assignment = (
                {repeated_id: assignments[repeated_id]}
                if assignments is not None
                else None
            )
            repeated = generate_condition_split(
                [repeated_id],
                labels,
                "train",
                self.config,
                condition,
                repeated_assignment,
            )[0]
            original = patients[0]
            np.testing.assert_array_equal(repeated.image, original.image)
            self.assertEqual(repeated.clinical_value, original.clinical_value)
            self.assertEqual(repeated.image_cue, original.image_cue)
            del patients


if __name__ == "__main__":
    unittest.main()
