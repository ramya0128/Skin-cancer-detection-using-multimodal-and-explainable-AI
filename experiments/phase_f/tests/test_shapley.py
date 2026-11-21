import unittest

import numpy as np

from experiments.phase_f.shapley import (
    build_feature_groups,
    construct_coalition_batch,
    exact_group_shapley,
)


class GroupedShapleyTests(unittest.TestCase):
    def test_build_feature_groups_maps_each_model_column_once(self):
        groups = build_feature_groups(
            ["age", "grew_True", "grew_False", "region_FACE"],
            variables=("grew", "region", "age"),
        )

        self.assertEqual(groups, {"grew": [1, 2], "region": [3], "age": [0]})

    def test_coalition_intervenes_on_every_column_in_a_group(self):
        intervened = construct_coalition_batch(
            test_row=np.asarray([9.0, 8.0, 7.0]),
            background=np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
            groups={"grew": [0, 1], "itch": [2]},
            coalition_mask=1,
        )

        np.testing.assert_array_equal(
            intervened,
            [[9.0, 8.0, 1.0], [9.0, 8.0, 0.0]],
        )

    def test_exact_group_shapley_recovers_additive_contributions(self):
        expected = np.arange(1, 9, dtype=np.float64) / 10.0
        coalition_values = np.asarray(
            [
                [sum(expected[index] for index in range(8) if mask & (1 << index))]
                for mask in range(1 << 8)
            ]
        )

        contributions = exact_group_shapley(coalition_values, n_groups=8)

        np.testing.assert_allclose(contributions[:, 0], expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
