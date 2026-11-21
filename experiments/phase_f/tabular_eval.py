import argparse
import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from experiments.phase_f.io import EXPECTED_CLASSES, EXPECTED_VARIABLES, load_frozen_arrays
from experiments.phase_f.shapley import (
    build_feature_groups,
    construct_coalition_batch,
    exact_group_shapley,
)


DEFAULT_META = Path(r"D:\D DRIVE DOWNLOADS\zr7vgbcyr2-1\metadata.csv")
DEFAULT_OUTPUT = Path("outputs") / "phase_f_v1" / "tabular_task4_v1"
SEED = 42
BATCH_CASES = 2
RANDOM_ORDER_SEEDS = tuple(range(42, 62))


def permute_training_labels(training_labels, seed=SEED):
    labels = np.asarray(training_labels)
    if labels.ndim != 1:
        raise ValueError("training labels must be one-dimensional")
    return np.random.default_rng(seed).permutation(labels)


def deletion_insertion_curves(coalition_values, group_order):
    values = np.asarray(coalition_values, dtype=np.float64)
    order = np.asarray(group_order, dtype=int)
    n_groups = len(order)
    if values.shape[0] != (1 << n_groups):
        raise ValueError("coalition count does not match group order")
    if sorted(order.tolist()) != list(range(n_groups)):
        raise ValueError("group_order must be a permutation of all groups")

    full_mask = (1 << n_groups) - 1
    insertion_masks = [0]
    for group_index in order:
        insertion_masks.append(insertion_masks[-1] | (1 << int(group_index)))
    deletion_masks = [full_mask ^ mask for mask in insertion_masks]
    return values[deletion_masks], values[insertion_masks]


def recompute_cwf(xgb_probabilities, vit_probabilities, xgb_weights):
    xgb_probabilities = np.asarray(xgb_probabilities, dtype=np.float64)
    vit_probabilities = np.asarray(vit_probabilities, dtype=np.float64)
    weights = np.asarray(xgb_weights, dtype=np.float64)
    fused = xgb_probabilities * weights + vit_probabilities * (1.0 - weights)
    totals = fused.sum(axis=-1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("CWF produced an invalid normalization denominator")
    return fused / totals


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "shap": shap.__version__,
        "xgboost": xgb.__version__,
    }


def _load_inputs(root, metadata_path):
    frozen = load_frozen_arrays(root)
    split = frozen["split"]
    base = frozen["base"]
    fusion = frozen["fusion"]
    images = frozen["images"]
    tree = frozen["tree"]
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"clinical metadata is missing: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if metadata["img_id"].duplicated().any():
        raise ValueError("metadata contains duplicate image IDs")
    metadata["diagnostic"] = metadata["diagnostic"].replace({"BOD": "SCC"})
    indexed = metadata.set_index("img_id", drop=False)
    train_ids = split["train_img_ids"]
    test_ids = split["test_img_ids"]
    missing_ids = sorted((set(train_ids) | set(test_ids)) - set(indexed.index))
    if missing_ids:
        raise ValueError(f"split IDs are absent from metadata: {missing_ids[:5]}")
    train_df = indexed.loc[train_ids].reset_index(drop=True)
    test_df = indexed.loc[test_ids].reset_index(drop=True)

    if train_df["patient_id"].isna().any() or test_df["patient_id"].isna().any():
        raise ValueError("patient IDs must be present to verify split independence")
    train_patients = set(train_df["patient_id"].astype(str))
    test_patients = set(test_df["patient_id"].astype(str))
    patient_overlap = train_patients.intersection(test_patients)
    if patient_overlap:
        raise ValueError(f"training and test splits share {len(patient_overlap)} patient IDs")

    label_encoder = LabelEncoder().fit(train_df["diagnostic"])
    if label_encoder.classes_.astype(str).tolist() != EXPECTED_CLASSES:
        raise ValueError("metadata labels do not match the frozen class order")
    y_train = label_encoder.transform(train_df["diagnostic"])
    y_test = label_encoder.transform(test_df["diagnostic"])
    if not np.array_equal(y_train, base["y_train"].astype(int)):
        raise ValueError("training labels do not align with frozen training labels")
    if not np.array_equal(y_test, base["y_test"].astype(int)):
        raise ValueError("test labels do not align with frozen test labels")

    model_path = Path(root) / "outputs" / "xgb_final_v1.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]
    columns = list(artifact["columns"])
    groups = build_feature_groups(columns)
    features = list(EXPECTED_VARIABLES)
    x_train = pd.get_dummies(train_df[features], dummy_na=True).astype(float)
    x_train = x_train.reindex(columns=columns)
    x_test = pd.get_dummies(test_df[features], dummy_na=True).reindex(
        columns=columns, fill_value=0
    ).astype(float)
    if x_train.shape != (len(train_ids), len(columns)):
        raise ValueError(f"unexpected training matrix shape: {x_train.shape}")
    if x_test.shape != (len(test_ids), len(columns)):
        raise ValueError(f"unexpected test matrix shape: {x_test.shape}")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("reconstructed matrices contain missing values")
    if len(columns) != 40:
        raise ValueError(f"expected 40 saved model columns, found {len(columns)}")

    reproduced = model.predict_proba(x_test)
    prediction_error = float(np.max(np.abs(reproduced - base["test_xgb"])))
    if prediction_error >= 1e-6:
        raise ValueError(f"frozen XGBoost reproduction error is {prediction_error}")

    background_indices = np.asarray(tree["test_background_indices"], dtype=int)
    background = x_train.iloc[background_indices].copy()
    background_classes = np.bincount(y_train[background_indices], minlength=6)
    if not np.array_equal(background_classes, [34, 34, 33, 33, 33, 33]):
        raise ValueError("saved background indices are not the validated stratified sample")
    return {
        "frozen": frozen,
        "train_df": train_df,
        "test_df": test_df,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "columns": columns,
        "groups": groups,
        "background": background,
        "background_indices": background_indices,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata_sha256": _sha256(metadata_path),
        "prediction_reproduction_max_abs_error": prediction_error,
        "train_patient_count": len(train_patients),
        "test_patient_count": len(test_patients),
        "patient_overlap_count": len(patient_overlap),
    }


def _predict_coalitions(model, cases, background, groups, columns, batch_cases):
    case_array = np.asarray(cases, dtype=np.float32)
    background_array = np.asarray(background, dtype=np.float32)
    n_cases, n_features = case_array.shape
    n_background = len(background_array)
    n_groups = len(groups)
    n_classes = len(EXPECTED_CLASSES)
    n_coalitions = 1 << n_groups
    values = np.empty((n_cases, n_coalitions, n_classes), dtype=np.float64)
    rows_per_case = n_coalitions * n_background
    first_batch_seconds = None
    start_time = time.perf_counter()

    for batch_start in range(0, n_cases, batch_cases):
        current_cases = case_array[batch_start:batch_start + batch_cases]
        current_count = len(current_cases)
        candidate_matrix = np.empty(
            (current_count * rows_per_case, n_features),
            dtype=np.float32,
        )
        for local_index, case_row in enumerate(current_cases):
            case_start = local_index * rows_per_case
            for coalition in range(n_coalitions):
                begin = case_start + coalition * n_background
                candidate_matrix[begin:begin + n_background] = construct_coalition_batch(
                    case_row,
                    background_array,
                    groups,
                    coalition,
                )

        candidate_frame = pd.DataFrame(candidate_matrix, columns=columns)
        predictions = np.asarray(model.predict_proba(candidate_frame), dtype=np.float64)
        if predictions.shape != (len(candidate_matrix), n_classes):
            raise ValueError(f"unexpected prediction shape: {predictions.shape}")
        values[batch_start:batch_start + current_count] = predictions.reshape(
            current_count, n_coalitions, n_background, n_classes
        ).mean(axis=2)

        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - start_time
            estimate_minutes = (
                first_batch_seconds * n_cases / current_count / 60.0
            )
            logging.info(
                "first coalition batch: %d cases in %.2fs; estimated full pass %.2f min",
                current_count,
                first_batch_seconds,
                estimate_minutes,
            )
        completed = min(batch_start + current_count, n_cases)
        if completed == n_cases or completed % 40 == 0:
            logging.info("coalitions complete: %d/%d cases", completed, n_cases)

    return values, time.perf_counter() - start_time, first_batch_seconds


def _group_shapley(coalition_values):
    grouped = np.empty(
        (len(coalition_values), len(EXPECTED_CLASSES), len(EXPECTED_VARIABLES)),
        dtype=np.float64,
    )
    for row_index, row_values in enumerate(coalition_values):
        grouped[row_index] = exact_group_shapley(
            row_values,
            n_groups=len(EXPECTED_VARIABLES),
        ).T
    return grouped


def _efficiency_error(coalitions, shapley):
    residual = shapley.sum(axis=2) - (coalitions[:, -1, :] - coalitions[:, 0, :])
    return float(np.max(np.abs(residual)))


def _orders(seeds, n_groups):
    return np.asarray([
        np.random.default_rng(seed).permutation(n_groups)
        for seed in seeds
    ], dtype=np.int64)


def _rank_correlations(frozen_shapley, control_shapley, target_classes):
    correlations = np.full(len(target_classes), np.nan, dtype=np.float64)
    for row_index, target_class in enumerate(target_classes):
        frozen_importance = np.abs(frozen_shapley[row_index, target_class])
        control_importance = np.abs(control_shapley[row_index, target_class])
        correlation = spearmanr(frozen_importance, control_importance).statistic
        if np.isfinite(correlation):
            correlations[row_index] = float(correlation)
    return correlations


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_evaluation(root, metadata_path, output_dir):
    root = Path(root)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    inputs = _load_inputs(root, metadata_path)
    if output_dir.exists():
        raise FileExistsError(
            f"versioned output already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    file_handler = logging.FileHandler(output_dir / "execution.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
        force=True,
    )

    frozen = inputs["frozen"]
    base = frozen["base"]
    fusion = frozen["fusion"]
    images = frozen["images"]
    tree = frozen["tree"]
    x_test = inputs["x_test"]
    x_train = inputs["x_train"]
    y_train = inputs["y_train"]
    y_test = inputs["y_test"]
    test_vit = np.asarray(base["test_vit"], dtype=np.float64)
    xgb_weights = np.asarray(fusion["w_xgb"], dtype=np.float64)
    target_classes = np.asarray(images["target_fused_class"], dtype=int)
    saved_tree_shapley = np.asarray(tree["test_full_probability"], dtype=np.float64)
    if not np.array_equal(images["test_row_keys"].astype(str), np.asarray(frozen["split"]["test_img_ids"], dtype=str)):
        raise ValueError("image explanation targets are misaligned with frozen test IDs")

    config = {
        "protocol": "grouped_interventional_256_coalitions",
        "background_rows": 200,
        "background_indices_sha256": hashlib.sha256(
            inputs["background_indices"].astype("<i8").tobytes()
        ).hexdigest(),
        "batch_cases": BATCH_CASES,
        "random_order_seeds": list(RANDOM_ORDER_SEEDS),
        "seed": SEED,
        "groups": list(EXPECTED_VARIABLES),
        "control": "XGBClassifier with 01 notebook recipe trained on permuted training labels only",
        "control_label_permutation_seed": SEED,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info("input validation passed: train=%s test=%s", x_train.shape, x_test.shape)
    logging.info(
        "patient-level separation: train=%d patients test=%d patients overlap=%d",
        inputs["train_patient_count"],
        inputs["test_patient_count"],
        inputs["patient_overlap_count"],
    )
    logging.info(
        "frozen model prediction reproduction max absolute error: %.12g",
        inputs["prediction_reproduction_max_abs_error"],
    )
    logging.info("versions: %s", json.dumps(_versions(), sort_keys=True))

    frozen_xgb_coalitions, frozen_seconds, _ = _predict_coalitions(
        inputs["model"],
        x_test,
        inputs["background"],
        inputs["groups"],
        inputs["columns"],
        BATCH_CASES,
    )
    frozen_cwf_coalitions = recompute_cwf(
        frozen_xgb_coalitions,
        test_vit[:, None, :],
        xgb_weights,
    )
    frozen_xgb_shapley = _group_shapley(frozen_xgb_coalitions)
    frozen_cwf_shapley = _group_shapley(frozen_cwf_coalitions)
    frozen_xgb_efficiency = _efficiency_error(
        frozen_xgb_coalitions, frozen_xgb_shapley
    )
    frozen_cwf_efficiency = _efficiency_error(
        frozen_cwf_coalitions, frozen_cwf_shapley
    )
    if max(frozen_xgb_efficiency, frozen_cwf_efficiency) >= 1e-5:
        raise ValueError("frozen-model grouped Shapley efficiency check failed")

    frozen_full = np.asarray(inputs["model"].predict_proba(x_test), dtype=np.float64)
    if not np.allclose(frozen_xgb_coalitions[:, -1], frozen_full, atol=1e-6):
        raise ValueError("full-coalition XGBoost predictions do not match direct predictions")
    if not np.allclose(frozen_cwf_coalitions[:, -1], fusion["cwf_test"], atol=1e-6):
        raise ValueError("full-coalition CWF predictions do not match frozen CWF")

    orders = _orders(RANDOM_ORDER_SEEDS, len(EXPECTED_VARIABLES))
    deletion_insertion = np.empty(
        (len(x_test), 2, 2, len(EXPECTED_VARIABLES) + 1), dtype=np.float64
    )
    random_curves = np.empty(
        (len(x_test), len(orders), 2, 2, len(EXPECTED_VARIABLES) + 1),
        dtype=np.float64,
    )
    random_auc_xgb = np.empty((len(x_test), len(orders), 2), dtype=np.float64)
    random_auc_cwf = np.empty_like(random_auc_xgb)
    x_axis = np.linspace(0.0, 1.0, len(EXPECTED_VARIABLES) + 1)
    for row_index in range(len(x_test)):
        target_class = int(target_classes[row_index])
        attribution_order = np.argsort(
            -np.abs(saved_tree_shapley[row_index, target_class]),
            kind="stable",
        )
        del_xgb, ins_xgb = deletion_insertion_curves(
            frozen_xgb_coalitions[row_index], attribution_order
        )
        del_cwf, ins_cwf = deletion_insertion_curves(
            frozen_cwf_coalitions[row_index], attribution_order
        )
        deletion_insertion[row_index, 0, 0] = del_xgb[:, target_class]
        deletion_insertion[row_index, 0, 1] = ins_xgb[:, target_class]
        deletion_insertion[row_index, 1, 0] = del_cwf[:, target_class]
        deletion_insertion[row_index, 1, 1] = ins_cwf[:, target_class]
        for order_index, order in enumerate(orders):
            random_del_xgb, random_ins_xgb = deletion_insertion_curves(
                frozen_xgb_coalitions[row_index], order
            )
            random_del_cwf, random_ins_cwf = deletion_insertion_curves(
                frozen_cwf_coalitions[row_index], order
            )
            random_curves[row_index, order_index, 0, 0] = random_del_xgb[:, target_class]
            random_curves[row_index, order_index, 0, 1] = random_ins_xgb[:, target_class]
            random_curves[row_index, order_index, 1, 0] = random_del_cwf[:, target_class]
            random_curves[row_index, order_index, 1, 1] = random_ins_cwf[:, target_class]
            random_auc_xgb[row_index, order_index] = [
                np.trapezoid(random_del_xgb[:, target_class], x=x_axis),
                np.trapezoid(random_ins_xgb[:, target_class], x=x_axis),
            ]
            random_auc_cwf[row_index, order_index] = [
                np.trapezoid(random_del_cwf[:, target_class], x=x_axis),
                np.trapezoid(random_ins_cwf[:, target_class], x=x_axis),
            ]

    permuted_y_train = permute_training_labels(y_train, seed=SEED)
    control = XGBClassifier(
        objective="multi:softprob",
        num_class=len(EXPECTED_CLASSES),
        eval_metric="mlogloss",
        random_state=SEED,
    )
    logging.info("fitting label-permuted control using training rows only")
    control.fit(x_train, permuted_y_train)
    control_path = output_dir / "xgb_permuted_control.joblib"
    joblib.dump(
        {
            "model": control,
            "columns": inputs["columns"],
            "training_label_permutation_seed": SEED,
            "training_row_count": len(x_train),
        },
        control_path,
    )
    control_test_xgb = np.asarray(control.predict_proba(x_test), dtype=np.float64)
    control_test_cwf = recompute_cwf(control_test_xgb, test_vit, xgb_weights)
    control_xgb_coalitions, control_seconds, _ = _predict_coalitions(
        control,
        x_test,
        inputs["background"],
        inputs["groups"],
        inputs["columns"],
        BATCH_CASES,
    )
    control_cwf_coalitions = recompute_cwf(
        control_xgb_coalitions,
        test_vit[:, None, :],
        xgb_weights,
    )
    control_xgb_shapley = _group_shapley(control_xgb_coalitions)
    control_cwf_shapley = _group_shapley(control_cwf_coalitions)
    control_xgb_efficiency = _efficiency_error(
        control_xgb_coalitions, control_xgb_shapley
    )
    control_cwf_efficiency = _efficiency_error(
        control_cwf_coalitions, control_cwf_shapley
    )
    if max(control_xgb_efficiency, control_cwf_efficiency) >= 1e-5:
        raise ValueError("control grouped Shapley efficiency check failed")
    if not np.allclose(control_xgb_coalitions[:, -1], control_test_xgb, atol=1e-6):
        raise ValueError("control full-coalition predictions do not match direct predictions")
    if not np.allclose(control_cwf_coalitions[:, -1], control_test_cwf, atol=1e-6):
        raise ValueError("control full-coalition CWF predictions do not match direct fusion")

    rank_spearman_xgb = _rank_correlations(
        frozen_xgb_shapley, control_xgb_shapley, target_classes
    )
    rank_spearman_cwf = _rank_correlations(
        frozen_cwf_shapley, control_cwf_shapley, target_classes
    )
    probabilities_valid = all(
        np.isfinite(array).all()
        and np.allclose(array.sum(axis=-1), 1.0, atol=1e-6)
        for array in (
            frozen_xgb_coalitions,
            frozen_cwf_coalitions,
            control_xgb_coalitions,
            control_cwf_coalitions,
            control_test_xgb,
            control_test_cwf,
        )
    )
    if not probabilities_valid:
        raise ValueError("one or more saved probability arrays are invalid")
    if not all(
        np.isfinite(array).all()
        for array in (
            frozen_xgb_shapley,
            frozen_cwf_shapley,
            control_xgb_shapley,
            control_cwf_shapley,
            deletion_insertion,
            random_curves,
            random_auc_xgb,
            random_auc_cwf,
        )
    ):
        raise ValueError("one or more attribution or curve arrays are non-finite")
    if not np.allclose(
        deletion_insertion[:, 0, 0, 0], frozen_xgb_coalitions[:, -1, :][
            np.arange(len(x_test)), target_classes
        ], atol=1e-7
    ):
        raise ValueError("deletion curves do not start at the full-model XGBoost score")
    if not np.allclose(
        deletion_insertion[:, 0, 0, -1], frozen_xgb_coalitions[:, 0, :][
            np.arange(len(x_test)), target_classes
        ], atol=1e-7
    ):
        raise ValueError("deletion curves do not end at the empty-coalition XGBoost score")
    if not np.allclose(
        deletion_insertion[:, 1, 1, -1], fusion["cwf_test"][
            np.arange(len(x_test)), target_classes
        ], atol=1e-7
    ):
        raise ValueError("insertion curves do not end at the full-model CWF score")
    test_row_keys = np.asarray(frozen["split"]["test_img_ids"], dtype=str)
    training_ids = frozen["split"]["train_img_ids"]
    if set(training_ids).intersection(test_row_keys.tolist()):
        raise ValueError("training and test image IDs overlap")

    result_path = output_dir / "tabular_task4_results.npz"
    np.savez_compressed(
        result_path,
        classes=np.asarray(EXPECTED_CLASSES),
        variables=np.asarray(EXPECTED_VARIABLES),
        test_row_keys=test_row_keys,
        target_cwf_class=target_classes,
        random_order_seeds=np.asarray(RANDOM_ORDER_SEEDS, dtype=np.int64),
        random_orders=orders,
        frozen_xgb_coalition_probability=frozen_xgb_coalitions,
        frozen_cwf_coalition_probability=frozen_cwf_coalitions,
        control_xgb_coalition_probability=control_xgb_coalitions,
        control_cwf_coalition_probability=control_cwf_coalitions,
        frozen_xgb_group_shapley=frozen_xgb_shapley,
        frozen_cwf_group_shapley=frozen_cwf_shapley,
        control_xgb_group_shapley=control_xgb_shapley,
        control_cwf_group_shapley=control_cwf_shapley,
        frozen_deletion_insertion_target_curves=deletion_insertion,
        frozen_random_order_target_curves=random_curves,
        frozen_random_order_auc_xgb=random_auc_xgb,
        frozen_random_order_auc_cwf=random_auc_cwf,
        control_test_xgb_probability=control_test_xgb,
        control_test_cwf_probability=control_test_cwf,
        rank_spearman_target_xgb=rank_spearman_xgb,
        rank_spearman_target_cwf=rank_spearman_cwf,
        efficiency_errors=np.asarray([
            frozen_xgb_efficiency,
            frozen_cwf_efficiency,
            control_xgb_efficiency,
            control_cwf_efficiency,
        ]),
        test_y=y_test,
    )

    output_hashes = {
        path.name: _sha256(path)
        for path in (result_path, control_path, output_dir / "config.json")
    }
    manifest = {
        "status": "completed",
        "experiment": "Phase F Task 4 grouped clinical interventions and label-permuted control",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": _versions(),
        "input_sha256": {
            **frozen["input_hashes"],
            str(inputs["metadata_path"]): inputs["metadata_sha256"],
        },
        "code_sha256": _sha256(Path(__file__)),
        "output_sha256": output_hashes,
        "data": {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "train_patients": inputs["train_patient_count"],
            "test_patients": inputs["test_patient_count"],
            "patient_overlap_count": inputs["patient_overlap_count"],
            "training_test_image_id_overlap_count": 0,
            "model_columns": inputs["columns"],
            "groups": list(EXPECTED_VARIABLES),
            "background_indices": inputs["background_indices"].tolist(),
            "background_class_counts": np.bincount(
                y_train[inputs["background_indices"]], minlength=6
            ).tolist(),
            "frozen_prediction_reproduction_max_abs_error": (
                inputs["prediction_reproduction_max_abs_error"]
            ),
        },
        "protocol": config,
        "control": {
            "training_rows": len(x_train),
            "training_labels_only": True,
            "permutation_seed": SEED,
            "test_labels_used_for_training_or_selection": False,
            "test_accuracy": float(accuracy_score(y_test, control_test_xgb.argmax(axis=1))),
            "test_macro_f1": float(f1_score(
                y_test,
                control_test_xgb.argmax(axis=1),
                labels=np.arange(len(EXPECTED_CLASSES)),
                average="macro",
                zero_division=0,
            )),
            "rank_spearman_xgb_mean": float(np.nanmean(rank_spearman_xgb)),
            "rank_spearman_xgb_valid_rows": int(np.isfinite(rank_spearman_xgb).sum()),
            "rank_spearman_cwf_mean": float(np.nanmean(rank_spearman_cwf)),
            "rank_spearman_cwf_valid_rows": int(np.isfinite(rank_spearman_cwf).sum()),
            "model_environment_xgboost_version": xgb.__version__,
        },
        "validation": {
            "probabilities_finite_and_normalized": probabilities_valid,
            "all_eight_groups_present": len(inputs["groups"]) == 8,
            "frozen_xgb_efficiency_max_abs": frozen_xgb_efficiency,
            "frozen_cwf_efficiency_max_abs": frozen_cwf_efficiency,
            "control_xgb_efficiency_max_abs": control_xgb_efficiency,
            "control_cwf_efficiency_max_abs": control_cwf_efficiency,
            "frozen_coalition_runtime_seconds": frozen_seconds,
            "control_coalition_runtime_seconds": control_seconds,
            "random_order_count": len(orders),
        },
        "interpretation_note": (
            "All held-out test-set metrics are exploratory because a test image was "
            "previously used for debugging; no test result was used to tune this run."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    logging.info("output SHA-256: %s", json.dumps(output_hashes, sort_keys=True))
    logging.info(
        "efficiency max abs error (frozen XGB/CWF, control XGB/CWF): %.3g / %.3g / %.3g / %.3g",
        frozen_xgb_efficiency,
        frozen_cwf_efficiency,
        control_xgb_efficiency,
        control_cwf_efficiency,
    )
    logging.info(
        "control test exploratory accuracy=%.4f macro_f1=%.4f; rank Spearman XGB/CWF=%.4f/%.4f",
        manifest["control"]["test_accuracy"],
        manifest["control"]["test_macro_f1"],
        manifest["control"]["rank_spearman_xgb_mean"],
        manifest["control"]["rank_spearman_cwf_mean"],
    )
    logging.info("completed Task 4 in %s", output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        inputs = _load_inputs(args.root, args.metadata)
        print(json.dumps({
            "train_shape": list(inputs["x_train"].shape),
            "test_shape": list(inputs["x_test"].shape),
            "groups": {key: len(value) for key, value in inputs["groups"].items()},
            "background_class_counts": np.bincount(
                inputs["y_train"][inputs["background_indices"]], minlength=6
            ).tolist(),
            "patient_overlap_count": inputs["patient_overlap_count"],
            "prediction_reproduction_max_abs_error": (
                inputs["prediction_reproduction_max_abs_error"]
            ),
            "input_sha256": {
                **inputs["frozen"]["input_hashes"],
                str(inputs["metadata_path"]): inputs["metadata_sha256"],
            },
        }, indent=2, sort_keys=True))
        return
    run_evaluation(args.root, args.metadata, args.output_dir)


if __name__ == "__main__":
    main()
