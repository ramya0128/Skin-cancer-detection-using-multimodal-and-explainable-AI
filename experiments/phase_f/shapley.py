import math

import numpy as np

from experiments.phase_f.io import EXPECTED_VARIABLES


def build_feature_groups(columns, variables=EXPECTED_VARIABLES):
    groups = {variable: [] for variable in variables}
    for column_index, column in enumerate(columns):
        matches = [
            variable for variable in variables
            if column == variable or str(column).startswith(variable + "_")
        ]
        if len(matches) != 1:
            raise ValueError(f"column {column!r} maps to {len(matches)} variables")
        groups[matches[0]].append(column_index)
    if any(not indices for indices in groups.values()):
        raise ValueError("every expected variable must map to at least one model column")
    return groups


def construct_coalition_batch(test_row, background, groups, coalition_mask):
    test_row = np.asarray(test_row)
    background = np.asarray(background)
    if test_row.ndim != 1 or background.ndim != 2:
        raise ValueError("test_row must be 1D and background must be 2D")
    if background.shape[1] != len(test_row):
        raise ValueError("test row and background feature counts differ")
    n_groups = len(groups)
    if not isinstance(coalition_mask, (int, np.integer)) or not (
        0 <= coalition_mask < (1 << n_groups)
    ):
        raise ValueError("coalition mask is outside the valid group range")

    assigned = [index for indices in groups.values() for index in indices]
    if sorted(assigned) != list(range(len(test_row))):
        raise ValueError("groups must partition all model columns exactly once")

    intervened = background.copy()
    for group_index, indices in enumerate(groups.values()):
        if coalition_mask & (1 << group_index):
            intervened[:, indices] = test_row[indices]
    return intervened


def exact_group_shapley(coalition_values, n_groups):
    values = np.asarray(coalition_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != (1 << n_groups):
        raise ValueError("coalition values must have one row per group subset")

    phi = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    denominator = math.factorial(n_groups)
    for group_index in range(n_groups):
        bit = 1 << group_index
        for coalition in range(1 << n_groups):
            if coalition & bit:
                continue
            size = coalition.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_groups - size - 1)
                / denominator
            )
            phi[group_index] += weight * (
                values[coalition | bit] - values[coalition]
            )
    return phi
