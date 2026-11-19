import json
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from experiments.phase_g.generate import (
    SyntheticPatient,
    generate_cues,
    make_conflicting_assignments,
    make_patient_splits,
    render_patient,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
CLASS_COUNT = 6


def load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_rngs(seed: int) -> dict[str, np.random.Generator]:
    names = (
        "image_cue_noise",
        "clinical_cue_noise",
        "condition",
        "conflict_assignment",
        "shared",
        "image_share_noise",
        "clinical_share_noise",
    )
    return {
        name: np.random.default_rng(seed + index)
        for index, name in enumerate(names)
    }


def sample_condition(condition: str, seed: int, n: int = 8400) -> tuple[np.ndarray, ...]:
    labels = np.arange(n, dtype=np.int64) % CLASS_COUNT
    rngs = make_rngs(seed)
    orientations = None
    if condition == "conflicting":
        patient_ids = [f"patient-{index}" for index in range(n)]
        orientations = make_conflicting_assignments(
            patient_ids, rngs["conflict_assignment"]
        )
    image_cues = np.empty(n, dtype=np.int64)
    clinical_cues = np.empty(n, dtype=np.int64)
    for index, label in enumerate(labels):
        image_cues[index], clinical_cues[index] = generate_cues(
            int(label),
            condition,
            rngs,
            conflict_orientation=None if orientations is None else orientations[f"patient-{index}"],
        )
    return labels, image_cues, clinical_cues


class GenerationTests(unittest.TestCase):
    def test_patient_splits_are_class_balanced_disjoint_and_labels_are_separate(self):
        config = load_config()
        splits, labels = make_patient_splits(config)

        self.assertEqual(
            set(splits),
            {"train", "validation_model_selection", "validation_calibration", "test"},
        )
        expected_per_class = {
            "train": 1400,
            "validation_model_selection": 150,
            "validation_calibration": 150,
            "test": 300,
        }
        all_ids: list[str] = []
        for partition, expected_count in expected_per_class.items():
            patient_ids = splits[partition]
            all_ids.extend(patient_ids)
            self.assertEqual(len(patient_ids), 6 * expected_count)
            self.assertEqual(len(set(patient_ids)), len(patient_ids))
            counts = np.bincount([labels[patient_id] for patient_id in patient_ids], minlength=6)
            np.testing.assert_array_equal(counts, np.full(6, expected_count))
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(labels), set(all_ids))
        self.assertEqual(set(labels.values()), set(range(CLASS_COUNT)))

    def test_rendered_patient_is_deterministic_and_has_exact_disjoint_masks(self):
        config = load_config()
        patient = render_patient("opaque-patient-17", 2, "train", config)
        repeated = render_patient("opaque-patient-17", 2, "train", config)

        self.assertIsInstance(patient, SyntheticPatient)
        self.assertEqual(patient.partition, "train")
        self.assertEqual(patient.label, 2)
        self.assertIn(patient.image_cue, range(CLASS_COUNT))
        self.assertIn(patient.clinical_value, range(CLASS_COUNT))
        self.assertEqual(patient.image.dtype, np.uint8)
        self.assertEqual(patient.image.shape, (224, 224, 3))
        self.assertEqual(patient.background_canvas.dtype, np.uint8)
        self.assertEqual(patient.background_canvas.shape, (224, 224, 3))
        for mask in (patient.lesion_mask, patient.marker_mask):
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(mask.shape, (224, 224))
            self.assertTrue(mask.any())
        self.assertEqual(int(patient.marker_mask.sum()), 16 * 16)
        self.assertFalse(np.any(patient.lesion_mask & patient.marker_mask))
        marker_corner = tuple(patient.render_metadata["marker_top_left"])
        self.assertIn(marker_corner, {(0, 0), (0, 208), (208, 0), (208, 208)})
        marker_y, marker_x = marker_corner
        glyph = patient.image[marker_y : marker_y + 16, marker_x : marker_x + 16]
        self.assertEqual(int(np.all(glyph == 0, axis=2).sum()), 32)
        self.assertTrue(np.all(glyph[:, :, 0] == glyph[:, :, 1]))
        self.assertTrue(np.all(glyph[:, :, 1] == glyph[:, :, 2]))
        distance = ndimage.distance_transform_edt(~patient.lesion_mask)
        self.assertGreaterEqual(float(distance[patient.marker_mask].min()), 16.0)

        np.testing.assert_array_equal(patient.image, repeated.image)
        np.testing.assert_array_equal(patient.background_canvas, repeated.background_canvas)
        np.testing.assert_array_equal(patient.lesion_mask, repeated.lesion_mask)
        np.testing.assert_array_equal(patient.marker_mask, repeated.marker_mask)
        self.assertEqual(patient.render_metadata, repeated.render_metadata)

    def test_lesion_and_background_rendering_do_not_depend_on_label(self):
        config = load_config()
        first = render_patient("same-opaque-id", 0, "train", config)
        second = render_patient("same-opaque-id", 4, "train", config)

        np.testing.assert_array_equal(first.background_canvas, second.background_canvas)
        np.testing.assert_array_equal(first.lesion_mask, second.lesion_mask)
        self.assertEqual(first.render_metadata["lesion"], second.render_metadata["lesion"])
        self.assertEqual(
            first.render_metadata["marker_top_left"],
            second.render_metadata["marker_top_left"],
        )

    def test_balanced_cues_have_independent_point_eight_five_reliability(self):
        labels, image_cues, clinical_cues = sample_condition("balanced", seed=101)
        image_correct = image_cues == labels
        clinical_correct = clinical_cues == labels
        standard_deviation = np.sqrt(len(labels) * 0.85 * 0.15)
        allowed_deviation = 5 * standard_deviation

        self.assertLessEqual(abs(image_correct.sum() - 0.85 * len(labels)), allowed_deviation)
        self.assertLessEqual(
            abs(clinical_correct.sum() - 0.85 * len(labels)), allowed_deviation
        )
        expected_both_correct = len(labels) * 0.85**2
        both_correct_sd = np.sqrt(len(labels) * 0.85**2 * (1 - 0.85**2))
        self.assertLessEqual(
            abs(np.logical_and(image_correct, clinical_correct).sum() - expected_both_correct),
            5 * both_correct_sd,
        )
        for label in range(CLASS_COUNT):
            label_rows = labels == label
            for cue in (image_cues, clinical_cues):
                counts = np.bincount(cue[label_rows], minlength=CLASS_COUNT)
                probabilities = np.full(CLASS_COUNT, 0.03)
                probabilities[label] = 0.85
                expected = label_rows.sum() * probabilities
                standard_deviations = np.sqrt(
                    label_rows.sum() * probabilities * (1 - probabilities)
                )
                self.assertTrue(
                    np.all(np.abs(counts - expected) <= 5 * standard_deviations)
                )
        self.assertTrue(np.all((image_cues >= 0) & (image_cues < CLASS_COUNT)))
        self.assertTrue(np.all((clinical_cues >= 0) & (clinical_cues < CLASS_COUNT)))

    def test_dominant_conditions_have_the_declared_marginal_reliabilities(self):
        for condition, expected in (
            ("image_dominant", (0.95, 0.60)),
            ("clinical_dominant", (0.60, 0.95)),
        ):
            with self.subTest(condition=condition):
                labels, image_cues, clinical_cues = sample_condition(condition, seed=202)
                expected_image, expected_clinical = expected
                tolerance = 5 * np.sqrt(len(labels) * 0.25) / len(labels)
                self.assertAlmostEqual(np.mean(image_cues == labels), expected_image, delta=tolerance)
                self.assertAlmostEqual(
                    np.mean(clinical_cues == labels), expected_clinical, delta=tolerance
                )

    def test_redundant_cues_are_identical_and_point_eight_five_reliable(self):
        labels, image_cues, clinical_cues = sample_condition("redundant", seed=303)
        self.assertTrue(np.array_equal(image_cues, clinical_cues))
        tolerance = 5 * np.sqrt(len(labels) * 0.85 * 0.15) / len(labels)
        self.assertAlmostEqual(np.mean(image_cues == labels), 0.85, delta=tolerance)

    def test_conflicting_condition_has_balanced_half_conflicts_and_point_six_seven_five_margins(self):
        n = 1200
        patient_ids = [f"patient-{index}" for index in range(n)]
        assignments = make_conflicting_assignments(
            patient_ids, make_rngs(404)["conflict_assignment"]
        )
        self.assertEqual(sum(value is not None for value in assignments.values()), n // 2)
        self.assertEqual(sum(value == "image" for value in assignments.values()), n // 4)
        self.assertEqual(sum(value == "clinical" for value in assignments.values()), n // 4)

        labels, image_cues, clinical_cues = sample_condition("conflicting", seed=404, n=n)
        image_correct = image_cues == labels
        clinical_correct = clinical_cues == labels

        for index, patient_id in enumerate(patient_ids):
            orientation = assignments[patient_id]
            if orientation == "image":
                self.assertTrue(image_correct[index])
                self.assertFalse(clinical_correct[index])
            elif orientation == "clinical":
                self.assertFalse(image_correct[index])
                self.assertTrue(clinical_correct[index])
        tolerance = 5 * np.sqrt(0.675 * 0.325 / len(labels))
        self.assertAlmostEqual(np.mean(image_correct), 0.675, delta=tolerance)
        self.assertAlmostEqual(np.mean(clinical_correct), 0.675, delta=tolerance)

    def test_synergistic_shares_are_individually_uninformative_and_jointly_correlated(self):
        labels, image_shares, clinical_shares = sample_condition("synergistic", seed=505)
        image_accuracy = np.mean(image_shares == labels)
        clinical_accuracy = np.mean(clinical_shares == labels)
        pair_decoding_accuracy = np.mean(
            (image_shares + clinical_shares) % CLASS_COUNT == labels
        )
        marginal_tolerance = 5 * np.sqrt((1 / 6) * (5 / 6) / len(labels))
        joint_tolerance = 5 * np.sqrt(0.727 * (1 - 0.727) / len(labels))

        self.assertTrue(np.all((image_shares >= 0) & (image_shares < CLASS_COUNT)))
        self.assertTrue(np.all((clinical_shares >= 0) & (clinical_shares < CLASS_COUNT)))
        self.assertAlmostEqual(image_accuracy, 1 / CLASS_COUNT, delta=marginal_tolerance)
        self.assertAlmostEqual(clinical_accuracy, 1 / CLASS_COUNT, delta=marginal_tolerance)
        self.assertAlmostEqual(pair_decoding_accuracy, 0.727, delta=joint_tolerance)

    def test_named_random_streams_reproduce_each_condition(self):
        for condition in (
            "balanced",
            "image_dominant",
            "clinical_dominant",
            "redundant",
            "conflicting",
            "synergistic",
        ):
            with self.subTest(condition=condition):
                first_rngs = make_rngs(606)
                second_rngs = make_rngs(606)
                first = [
                    generate_cues(index % CLASS_COUNT, condition, first_rngs)
                    for index in range(120)
                ]
                second = [
                    generate_cues(index % CLASS_COUNT, condition, second_rngs)
                    for index in range(120)
                ]
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
