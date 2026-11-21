from __future__ import annotations

import unittest

import numpy as np

from experiments.phase_g_v2_2.generator import (
    generate_conflicting_cues,
    make_conflicting_condition_assignments,
)
from experiments.phase_g_v2_2.render import (
    make_conflicting_split_assignments,
    render_conflicting_patient,
)


class ConflictingGeneratorTests(unittest.TestCase):
    def test_rendered_patient_uses_orientation_from_full_split_assignment(self):
        import json
        from pathlib import Path

        config = json.loads(
            (Path(__file__).resolve().parents[3] / "experiments/phase_g/config.json").read_text(
                encoding="utf-8"
            )
        )
        config["pilot_condition"] = "conflicting"
        patient_ids = [f"p{i}" for i in range(120)]
        assignments = make_conflicting_split_assignments(
            patient_ids,
            master_seed=20261119,
        )

        first = render_conflicting_patient(
            patient_ids[0],
            2,
            "train",
            config,
            assignments[patient_ids[0]],
        )
        repeated = render_conflicting_patient(
            patient_ids[0],
            2,
            "train",
            config,
            assignments[patient_ids[0]],
        )

        np.testing.assert_array_equal(first.image, repeated.image)
        np.testing.assert_array_equal(first.lesion_mask, repeated.lesion_mask)
        np.testing.assert_array_equal(first.marker_mask, repeated.marker_mask)
        orientation = assignments[patient_ids[0]]
        if orientation == "image_correct":
            self.assertEqual(first.image_cue, 2)
            self.assertNotEqual(first.clinical_value, 2)
        elif orientation == "clinical_correct":
            self.assertNotEqual(first.image_cue, 2)
            self.assertEqual(first.clinical_value, 2)

    def test_exact_population_components_and_balanced_orientations(self):
        patient_ids = [f"p{i}" for i in range(1200)]
        assignments = make_conflicting_condition_assignments(
            patient_ids,
            np.random.default_rng(404),
        )

        counts = {
            key: sum(value == key for value in assignments.values())
            for key in ("none", "image_correct", "clinical_correct")
        }

        self.assertEqual(counts["none"], 600)
        self.assertEqual(counts["image_correct"], 300)
        self.assertEqual(counts["clinical_correct"], 300)

    def test_rendered_cue_pairs_match_all_three_components(self):
        rng = np.random.default_rng(405)
        labels = np.repeat(np.arange(6), 200)
        assignments = make_conflicting_condition_assignments(
            [f"p{i}" for i in range(len(labels))],
            rng,
        )

        image_correct = 0
        clinical_correct = 0
        independent_both_correct = 0
        independently_correct_total = 0

        for index, label in enumerate(labels):
            orientation = assignments[f"p{index}"]
            image_cue, clinical_cue = generate_conflicting_cues(
                int(label),
                orientation,
                rng,
            )

            if orientation == "image_correct":
                self.assertEqual(image_cue, int(label))
                self.assertNotEqual(clinical_cue, int(label))
                image_correct += 1
            elif orientation == "clinical_correct":
                self.assertNotEqual(image_cue, int(label))
                self.assertEqual(clinical_cue, int(label))
                clinical_correct += 1
            else:
                independently_correct_total += 1
                independent_both_correct += int(
                    image_cue == int(label) and clinical_cue == int(label)
                )

        self.assertEqual(image_correct, 300)
        self.assertEqual(clinical_correct, 300)
        self.assertAlmostEqual(
            independently_correct_total,
            600,
        )
        expected_both = independently_correct_total * 0.85**2
        self.assertLess(
            abs(independent_both_correct - expected_both),
            5 * np.sqrt(independently_correct_total * 0.85**2 * (1 - 0.85**2)),
        )

    def test_marginal_reliability_is_point_six_seven_five(self):
        rng = np.random.default_rng(406)
        labels = np.repeat(np.arange(6), 1000)
        assignments = make_conflicting_condition_assignments(
            [f"p{i}" for i in range(len(labels))],
            rng,
        )
        image_correct = 0
        clinical_correct = 0

        for index, label in enumerate(labels):
            image_cue, clinical_cue = generate_conflicting_cues(
                int(label),
                assignments[f"p{index}"],
                rng,
            )
            image_correct += int(image_cue == int(label))
            clinical_correct += int(clinical_cue == int(label))

        tolerance = 5 * np.sqrt(len(labels) * 0.675 * 0.325) / len(labels)
        self.assertAlmostEqual(image_correct / len(labels), 0.675, delta=tolerance)
        self.assertAlmostEqual(clinical_correct / len(labels), 0.675, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
