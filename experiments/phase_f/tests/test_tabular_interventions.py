import unittest

import numpy as np
from experiments.phase_f import tabular_eval


MODEL_COLUMNS = [
    "age",
    "grew_False", "grew_True", "grew_UNK", "grew_nan",
    "itch_False", "itch_True", "itch_UNK", "itch_nan",
    "changed_False", "changed_True", "changed_UNK", "changed_nan",
    "elevation_False", "elevation_True", "elevation_UNK", "elevation_nan",
    "bleed_False", "bleed_True", "bleed_UNK", "bleed_nan",
    "region_ABDOMEN", "region_ARM", "region_BACK", "region_CHEST",
    "region_EAR", "region_FACE", "region_FOOT", "region_FOREARM",
    "region_HAND", "region_LIP", "region_NECK", "region_NOSE",
    "region_SCALP", "region_THIGH", "region_nan",
    "hurt_False", "hurt_True", "hurt_UNK", "hurt_nan",
]


class GroupedInterventionTests(unittest.TestCase):
    def require_module(self):
        return tabular_eval

    def test_all_forty_model_columns_belong_to_one_expected_variable(self):
        module = self.require_module()

        groups = module.build_feature_groups(MODEL_COLUMNS)

        self.assertEqual(
            list(groups),
            ["grew", "itch", "changed", "elevation", "bleed", "hurt", "region", "age"],
        )
        flattened = [index for indices in groups.values() for index in indices]
        self.assertEqual(sorted(flattened), list(range(40)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(groups["age"], [0])
        self.assertEqual(groups["region"], list(range(21, 36)))

    def test_one_coalition_keeps_each_absent_variable_from_one_background_row(self):
        module = self.require_module()
        groups = module.build_feature_groups([
            "grew_False", "grew_True", "itch_False", "itch_True", "age",
        ], variables=["grew", "itch", "age"])
        test_row = np.asarray([1.0, 0.0, 0.0, 1.0, 55.0])
        background = np.asarray([
            [0.0, 1.0, 1.0, 0.0, 20.0],
            [1.0, 0.0, 0.0, 1.0, 30.0],
        ])

        intervened = module.construct_coalition_batch(
            test_row,
            background,
            groups,
            coalition_mask=1,
        )

        np.testing.assert_array_equal(intervened, [
            [1.0, 0.0, 1.0, 0.0, 20.0],
            [1.0, 0.0, 0.0, 1.0, 30.0],
        ])

    def test_label_permutation_is_seeded_and_preserves_training_label_counts(self):
        module = self.require_module()
        training_labels = np.arange(6, dtype=np.int64)
        original = training_labels.copy()

        first = module.permute_training_labels(training_labels, seed=42)
        second = module.permute_training_labels(training_labels, seed=42)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.sort(original))
        np.testing.assert_array_equal(training_labels, original)
        self.assertFalse(np.array_equal(first, training_labels))

    def test_exact_group_shapley_recovers_additive_feature_effects(self):
        module = self.require_module()
        effects = np.asarray([2.0, -1.0, 0.5])
        coalition_values = np.asarray([
            [sum(effects[index] for index in range(3) if mask & (1 << index))]
            for mask in range(8)
        ])

        shapley = module.exact_group_shapley(coalition_values, n_groups=3)

        np.testing.assert_allclose(shapley[:, 0], effects, atol=1e-12)
        self.assertAlmostEqual(float(shapley[:, 0].sum()), 1.5)

    def test_deletion_and_insertion_curves_have_opposite_endpoints(self):
        module = self.require_module()
        coalition_values = np.asarray([[0.0], [1.0], [2.0], [3.0]])

        deletion, insertion = module.deletion_insertion_curves(
            coalition_values,
            group_order=[0, 1],
        )

        np.testing.assert_array_equal(deletion[:, 0], [3.0, 2.0, 0.0])
        np.testing.assert_array_equal(insertion[:, 0], [0.0, 1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
