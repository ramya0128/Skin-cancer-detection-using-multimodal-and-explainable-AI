import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from experiments.phase_f.io import EXPECTED_CLASSES, EXPECTED_VARIABLES, PROTECTED_INPUTS, load_frozen_arrays


ROOT = Path(__file__).resolve().parents[2]
META_PATH = Path(r"D:\D DRIVE DOWNLOADS\zr7vgbcyr2-1\metadata.csv")
DEFAULT_OUTPUT = Path("outputs") / "phase_f_v1" / "summary_v2"
BOOTSTRAP_RESAMPLES = 2000
SIGNFLIP_PERMUTATIONS = 10000
SEED = 42

_BOOTSTRAP_INDEX_CACHE = {}
_SIGNFLIP_MATRIX_CACHE = {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes(root, source_paths):
    root = Path(root).resolve()
    hashes = {}
    for source_path in source_paths:
        source_path = Path(source_path).resolve()
        try:
            relative_path = source_path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"source file is outside the experiment root: {source_path}") from error
        hashes[relative_path] = _sha256(source_path)
    return hashes


def _bootstrap_indices(n_clusters, n_resamples, seed):
    key = (n_clusters, n_resamples, seed)
    if key not in _BOOTSTRAP_INDEX_CACHE:
        rng = np.random.default_rng(seed)
        _BOOTSTRAP_INDEX_CACHE[key] = rng.integers(
            0,
            n_clusters,
            size=(n_resamples, n_clusters),
            dtype=np.int32,
        )
    return _BOOTSTRAP_INDEX_CACHE[key]


def _cluster_means(values, patient_ids):
    values = np.asarray(values, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    if values.ndim == 0 or values.shape[0] != len(patient_ids):
        raise ValueError("values and patient_ids must have matching row counts")
    if len(patient_ids) == 0:
        raise ValueError("at least one patient cluster is required")
    if not np.isfinite(values).all():
        raise ValueError("cluster statistics require finite values")
    unique_ids, inverse = np.unique(patient_ids, return_inverse=True)
    cluster_values = np.stack([
        values[inverse == cluster_index].mean(axis=0)
        for cluster_index in range(len(unique_ids))
    ])
    return unique_ids, cluster_values


def cluster_bootstrap(values, patient_ids, n_resamples, seed):
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    _, patient_means = _cluster_means(values, patient_ids)
    indices = _bootstrap_indices(len(patient_means), n_resamples, seed)
    bootstrap_means = patient_means[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
    return np.asarray(lower), np.asarray(upper)


def _signflip_matrix(n_patients, n_permutations, seed):
    key = (n_patients, n_permutations, seed)
    if key not in _SIGNFLIP_MATRIX_CACHE:
        rng = np.random.default_rng(seed)
        _SIGNFLIP_MATRIX_CACHE[key] = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(n_permutations, n_patients),
        )
    return _SIGNFLIP_MATRIX_CACHE[key]


def paired_cluster_signflip_p(values_a, values_b, patient_ids, n_permutations, seed):
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("paired sign-flip inputs must be one-dimensional")
    if values_a.shape != values_b.shape or len(values_a) != len(patient_ids):
        raise ValueError("paired values and patient_ids must have matching shapes")
    if not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
        raise ValueError("paired sign-flip inputs must be finite")
    _, patient_differences = _cluster_means(values_a - values_b, patient_ids)
    patient_differences = patient_differences.reshape(-1)
    signs = _signflip_matrix(len(patient_differences), n_permutations, seed)
    observed = abs(float(patient_differences.mean()))
    null_statistics = np.abs(signs @ patient_differences / len(patient_differences))
    exceedances = int(np.count_nonzero(null_statistics >= observed - 1e-15))
    return float((exceedances + 1) / (n_permutations + 1))


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("finite p-values must be in [0, 1]")
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(finite)
    if not len(valid_indices):
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices], kind="stable")]
    ranked = values[order] * np.arange(len(order), 0, -1, dtype=np.float64)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ranked))
    return adjusted


def _string_list(values):
    return [str(value) for value in np.asarray(values).tolist()]


def _verify_manifest_chunks(root, manifest):
    entries = manifest.get("result_chunks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("execution manifest has no result chunk inventory")
    for entry in entries:
        path = root / Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"saved result chunk is missing: {path}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"saved result chunk size changed: {path}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"saved result chunk hash changed: {path}")


def _validate_run_inputs(root, metadata_path):
    root = Path(root)
    phase_output = root / "outputs" / "phase_f_v1"
    image_manifest = json.loads(
        (phase_output / "image_experiment_manifest.json").read_text(encoding="utf-8")
    )
    sanity_manifest = json.loads(
        (phase_output / "image_sanity_manifest.json").read_text(encoding="utf-8")
    )
    tabular_dir = phase_output / "tabular_task4_v1"
    tabular_manifest = json.loads(
        (tabular_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if image_manifest.get("status") != "executed_and_validated":
        raise ValueError("image curves manifest is not marked executed_and_validated")
    if sanity_manifest.get("status") != "executed_and_validated":
        raise ValueError("image sanity manifest is not marked executed_and_validated")
    if tabular_manifest.get("status") != "completed":
        raise ValueError("tabular intervention manifest is not complete")
    if image_manifest.get("rows") != 461 or sanity_manifest.get("rows") != 461:
        raise ValueError("image run manifests do not cover the frozen 461 test rows")
    if tabular_manifest.get("data", {}).get("test_rows") != 461:
        raise ValueError("tabular run manifest does not cover the frozen test rows")

    image_inputs = image_manifest["frozen_run_identity"]["input_sha256"]
    sanity_inputs = sanity_manifest["protected_input_sha256"]
    tabular_inputs = tabular_manifest["input_sha256"]
    if image_inputs != sanity_inputs:
        raise ValueError("image stability used different frozen inputs than image curves")
    for relative, expected_hash in image_inputs.items():
        if tabular_inputs.get(relative) != expected_hash:
            raise ValueError(f"tabular run used a different frozen input: {relative}")
        path = root / Path(relative)
        if _sha256(path) != expected_hash:
            raise ValueError(f"protected input changed since execution: {relative}")

    frozen = load_frozen_arrays(root)
    split = frozen["split"]
    test_ids = [str(value) for value in split["test_img_ids"]]
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    if metadata["img_id"].duplicated().any():
        raise ValueError("clinical metadata contains duplicate image IDs")
    metadata["diagnostic"] = metadata["diagnostic"].replace({"BOD": "SCC"})
    indexed = metadata.set_index("img_id", drop=False)
    if not set(test_ids).issubset(set(indexed.index)):
        raise ValueError("one or more frozen test IDs are missing from clinical metadata")
    test_df = indexed.loc[test_ids]
    if test_df["patient_id"].isna().any():
        raise ValueError("patient IDs are required for cluster summaries")
    if metadata_path.resolve().as_posix() not in tabular_inputs:
        matching = [
            key for key in tabular_inputs
            if Path(key).resolve() == metadata_path.resolve()
        ]
    else:
        matching = [metadata_path.resolve().as_posix()]
    if not matching:
        matching = [key for key in tabular_inputs if Path(key).name == metadata_path.name]
    if len(matching) != 1:
        raise ValueError("could not uniquely identify the metadata hash in the tabular manifest")
    metadata_hash = tabular_inputs[matching[0]]
    if _sha256(metadata_path) != metadata_hash:
        raise ValueError("clinical metadata hash differs from the Task 4 run")

    _verify_manifest_chunks(root, image_manifest)
    _verify_manifest_chunks(root, sanity_manifest)
    for filename, expected_hash in tabular_manifest["output_sha256"].items():
        path = tabular_dir / filename
        if _sha256(path) != expected_hash:
            raise ValueError(f"tabular result output hash changed: {path}")

    phase_config = json.loads(
        (root / "experiments" / "phase_f" / "config.json").read_text(encoding="utf-8")
    )
    sanity_config = json.loads(
        (root / "experiments" / "phase_f" / "sanity_config.json").read_text(encoding="utf-8")
    )
    return {
        "root": root,
        "phase_output": phase_output,
        "tabular_dir": tabular_dir,
        "image_manifest": image_manifest,
        "sanity_manifest": sanity_manifest,
        "tabular_manifest": tabular_manifest,
        "frozen": frozen,
        "test_ids": test_ids,
        "test_df": test_df,
        "patient_ids": test_df["patient_id"].astype(str).to_numpy(),
        "y_test": frozen["base"]["y_test"].astype(int),
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_hash,
        "input_sha256_before": {
            **image_inputs,
            str(metadata_path): metadata_hash,
        },
        "phase_config": phase_config,
        "sanity_config": sanity_config,
    }


def _collect_image_auc(inputs):
    phase_output = inputs["phase_output"]
    paths = sorted(phase_output.glob("image_curves_*.npz"))
    if len(paths) != int(inputs["image_manifest"]["chunk_count"]):
        raise ValueError("image curve chunk count differs from the execution manifest")

    n_rows = len(inputs["test_ids"])
    row_indices = {image_id: index for index, image_id in enumerate(inputs["test_ids"])}
    seen = np.zeros(n_rows, dtype=bool)
    auc_values = None
    order_names = None
    target_sources = None
    score_spaces = None
    directions = None
    target_classes = None
    expected_images = inputs["frozen"]["images"]

    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            keys = _string_list(saved["test_row_keys"])
            positions = np.asarray([row_indices[key] for key in keys], dtype=int)
            if seen[positions].any():
                raise ValueError(f"duplicate image-curve rows found in {path}")
            names = _string_list(saved["order_names"])
            sources = _string_list(saved["target_sources"])
            spaces = _string_list(saved["score_spaces"])
            curve_directions = _string_list(saved["directions"])
            chunk_auc = np.asarray(saved["auc"], dtype=np.float64)
            chunk_targets = np.asarray(saved["target_classes"], dtype=int)
            expected_shape = (
                len(keys), len(names), len(spaces), len(curve_directions)
            )
            if chunk_auc.shape != expected_shape or chunk_targets.shape != (
                len(keys), len(names)
            ):
                raise ValueError(f"unexpected image AUC/target shapes in {path}")
            if not np.isfinite(chunk_auc).all() or np.any(
                (chunk_auc < 0) | (chunk_auc > 1)
            ):
                raise ValueError(f"invalid image AUC values in {path}")
            if order_names is None:
                order_names = names
                target_sources = sources
                score_spaces = spaces
                directions = curve_directions
                auc_values = np.empty(
                    (n_rows, len(names), len(spaces), len(curve_directions)),
                    dtype=np.float64,
                )
                target_classes = np.empty((n_rows, len(names)), dtype=int)
            elif (
                names != order_names
                or sources != target_sources
                or spaces != score_spaces
                or curve_directions != directions
            ):
                raise ValueError(f"image curve schema differs between chunks: {path}")

            for spec_index, source in enumerate(target_sources):
                target_key = (
                    "target_fused_class" if source == "fused"
                    else "target_vit_top_class"
                )
                expected_targets = expected_images[target_key][positions]
                if not np.array_equal(chunk_targets[:, spec_index], expected_targets):
                    raise ValueError(f"saved target class changed in {path}")
            auc_values[positions] = chunk_auc
            target_classes[positions] = chunk_targets
            seen[positions] = True

    if not seen.all():
        raise ValueError(f"image curve outputs cover only {int(seen.sum())}/{n_rows} rows")
    return {
        "auc": auc_values,
        "order_names": order_names,
        "target_sources": target_sources,
        "score_spaces": score_spaces,
        "directions": directions,
        "target_classes": target_classes,
    }


def _collect_sanity(inputs):
    phase_output = inputs["phase_output"]
    row_indices = {image_id: index for index, image_id in enumerate(inputs["test_ids"])}
    n_rows = len(inputs["test_ids"])
    map_keys = None
    stability = {}
    randomization = {}

    def fill_state(state, key, values, positions, total_rows):
        if key not in state:
            state[key] = np.full(total_rows, np.nan, dtype=np.float64)
        if np.isfinite(state[key][positions]).any():
            raise ValueError("sanity chunk rows overlap")
        state[key][positions] = np.asarray(values, dtype=np.float64)

    stability_paths = sorted(phase_output.glob("image_stability_*.npz"))
    randomization_paths = sorted(phase_output.glob("image_randomization_*.npz"))
    if len(stability_paths) + len(randomization_paths) != int(
        inputs["sanity_manifest"]["chunk_count"]
    ):
        raise ValueError("image sanity chunk count differs from its manifest")

    for path in stability_paths:
        with np.load(path, allow_pickle=False) as saved:
            keys = _string_list(saved["test_row_keys"])
            positions = np.asarray([row_indices[key] for key in keys], dtype=int)
            if map_keys is None:
                map_keys = _string_list(saved["map_keys"])
            elif _string_list(saved["map_keys"]) != map_keys:
                raise ValueError("image map-key order changes between sanity chunks")
            transform = str(np.asarray(saved["transform_name"]).item())
            state = stability.setdefault(transform, {
                "seen": np.zeros(n_rows, dtype=bool),
                "spearman": np.full((n_rows, len(map_keys)), np.nan, dtype=np.float64),
                "spearman_valid": np.zeros((n_rows, len(map_keys)), dtype=bool),
                "ssim": np.full((n_rows, len(map_keys)), np.nan, dtype=np.float64),
                "ssim_valid": np.zeros((n_rows, len(map_keys)), dtype=bool),
                "fused_unchanged": np.zeros(n_rows, dtype=bool),
                "vit_unchanged": np.zeros(n_rows, dtype=bool),
            })
            if state["seen"][positions].any():
                raise ValueError(f"duplicate stability rows found in {path}")
            for metric, validity in (
                ("spearman", "spearman_valid"),
                ("ssim", "ssim_valid"),
            ):
                values = np.asarray(saved[f"map_{metric}"], dtype=np.float64)
                valid = np.asarray(saved[f"map_{metric}_valid"], dtype=bool)
                if values.shape != (len(keys), len(map_keys)) or valid.shape != values.shape:
                    raise ValueError(f"invalid {metric} arrays in {path}")
                if not np.isfinite(values[valid]).all():
                    raise ValueError(f"non-finite valid {metric} values in {path}")
                state[metric][positions] = values
                state[validity][positions] = valid
            state["fused_unchanged"][positions] = saved["fused_unchanged_mask"]
            state["vit_unchanged"][positions] = saved["vit_unchanged_mask"]
            state["seen"][positions] = True

    for path in randomization_paths:
        with np.load(path, allow_pickle=False) as saved:
            keys = _string_list(saved["test_row_keys"])
            positions = np.asarray([row_indices[key] for key in keys], dtype=int)
            if map_keys is None:
                map_keys = _string_list(saved["map_keys"])
            elif _string_list(saved["map_keys"]) != map_keys:
                raise ValueError("image map-key order changes between randomization chunks")
            stage = str(np.asarray(saved["randomization_stage"]).item())
            state = randomization.setdefault(stage, {
                "seen": np.zeros(n_rows, dtype=bool),
                "spearman": np.full((n_rows, len(map_keys)), np.nan, dtype=np.float64),
                "spearman_valid": np.zeros((n_rows, len(map_keys)), dtype=bool),
                "ssim": np.full((n_rows, len(map_keys)), np.nan, dtype=np.float64),
                "ssim_valid": np.zeros((n_rows, len(map_keys)), dtype=bool),
                "vit_argmax_changed": np.zeros(n_rows, dtype=bool),
                "cwf_argmax_changed": np.zeros(n_rows, dtype=bool),
                "vit_delta": np.full(n_rows, np.nan, dtype=np.float64),
                "cwf_delta": np.full(n_rows, np.nan, dtype=np.float64),
            })
            if state["seen"][positions].any():
                raise ValueError(f"duplicate randomization rows found in {path}")
            for metric, validity in (
                ("spearman", "spearman_valid"),
                ("ssim", "ssim_valid"),
            ):
                values = np.asarray(saved[f"map_{metric}"], dtype=np.float64)
                valid = np.asarray(saved[f"map_{metric}_valid"], dtype=bool)
                if values.shape != (len(keys), len(map_keys)) or valid.shape != values.shape:
                    raise ValueError(f"invalid randomized {metric} arrays in {path}")
                if not np.isfinite(values[valid]).all():
                    raise ValueError(f"non-finite valid randomized {metric} in {path}")
                state[metric][positions] = values
                state[validity][positions] = valid
            state["vit_argmax_changed"][positions] = saved["vit_argmax_changed"]
            state["cwf_argmax_changed"][positions] = saved["cwf_argmax_changed"]
            state["vit_delta"][positions] = saved["vit_max_abs_probability_delta"]
            state["cwf_delta"][positions] = saved["cwf_max_abs_probability_delta"]
            state["seen"][positions] = True

    if map_keys is None:
        raise ValueError("no image sanity chunks were found")
    for state_group in (stability, randomization):
        for name, state in state_group.items():
            if not state["seen"].all():
                raise ValueError(
                    f"{name} covers only {int(state['seen'].sum())}/{n_rows} rows"
                )
    expected_transforms = {"horizontal_flip", "brightness_0.9", "brightness_1.1"}
    if set(stability) != expected_transforms:
        raise ValueError(f"unexpected image stability conditions: {sorted(stability)}")
    expected_stages = {
        str(stage["name"]) for stage in inputs["sanity_config"]["randomization_stages"]
    }
    if set(randomization) != expected_stages:
        raise ValueError(f"unexpected ViT randomization stages: {sorted(randomization)}")
    return {"map_keys": map_keys, "stability": stability, "randomization": randomization}


def _patient_mean(values, patient_ids):
    _, means = _cluster_means(values, patient_ids)
    return np.mean(means, axis=0)


def _metric_rows(
    pooled_rows,
    class_rows,
    inputs,
    experiment,
    metric,
    method,
    values,
    target_source="",
    score_space="",
    direction="",
    baseline="",
    condition="",
):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(inputs["test_ids"]),):
        raise ValueError(f"metric {metric} has unexpected row shape {values.shape}")

    def append(destination, class_name, selection):
        selected = selection & np.isfinite(values)
        row_values = values[selected]
        patient_ids = inputs["patient_ids"][selected]
        if len(row_values):
            estimate = float(_patient_mean(row_values, patient_ids))
            lower, upper = cluster_bootstrap(
                row_values,
                patient_ids,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            )
            lower, upper = float(lower), float(upper)
        else:
            estimate = lower = upper = None
        destination.append({
            "experiment": experiment,
            "metric": metric,
            "method": method,
            "target_source": target_source,
            "score_space": score_space,
            "direction": direction,
            "baseline": baseline,
            "condition": condition,
            "class": class_name,
            "n_rows": int(len(row_values)),
            "n_patients": int(len(np.unique(patient_ids))),
            "estimate": estimate,
            "ci_low_95": lower,
            "ci_high_95": upper,
        })

    append(pooled_rows, "ALL", np.ones(len(values), dtype=bool))
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        append(class_rows, class_name, inputs["y_test"] == class_index)


def _paired_comparison(inputs, comparison_rows, family, experiment, direction,
                       method, baseline, score_space, target_source, values_a, values_b):
    differences = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    if differences.shape != (len(inputs["test_ids"]),) or not np.isfinite(differences).all():
        raise ValueError("paired comparison arrays must be finite and test-row aligned")
    estimate = float(_patient_mean(differences, inputs["patient_ids"]))
    lower, upper = cluster_bootstrap(
        differences,
        inputs["patient_ids"],
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=SEED,
    )
    p_value = paired_cluster_signflip_p(
        values_a,
        values_b,
        inputs["patient_ids"],
        n_permutations=SIGNFLIP_PERMUTATIONS,
        seed=SEED,
    )
    comparison_rows.append({
        "family": family,
        "experiment": experiment,
        "direction": direction,
        "method": method,
        "baseline": baseline,
        "target_source": target_source,
        "score_space": score_space,
        "n_rows": len(differences),
        "n_patients": len(np.unique(inputs["patient_ids"])),
        "mean_paired_delta_method_minus_baseline": estimate,
        "delta_ci_low_95": float(lower),
        "delta_ci_high_95": float(upper),
        "raw_p_value": p_value,
        "holm_adjusted_p_value": None,
    })


def _build_summary_tables(inputs, image_curves, sanity, tabular):
    pooled_rows = []
    class_rows = []
    comparison_rows = []
    image_auc = image_curves["auc"]
    order_names = image_curves["order_names"]
    sources = image_curves["target_sources"]
    spaces = image_curves["score_spaces"]
    directions = image_curves["directions"]
    names_to_index = {name: index for index, name in enumerate(order_names)}
    sources_by_name = dict(zip(order_names, sources))

    random_auc = {}
    for source in ("fused", "vit_top"):
        indices = [
            index for index, name in enumerate(order_names)
            if name.startswith(f"random_order_{source}_")
        ]
        if len(indices) != 20:
            raise ValueError(f"expected 20 random image orders for {source}")
        random_auc[source] = image_auc[:, indices].mean(axis=1)

    map_methods = [
        name for name in order_names
        if name in set(sanity["map_keys"])
    ]
    reference_methods = [
        name for name in order_names
        if name.startswith(("random_reference_", "center_reference_"))
    ]
    if len(map_methods) != 8 or len(reference_methods) != 4:
        raise ValueError("saved image methods or map-reference baselines are incomplete")

    image_summary_methods = map_methods + reference_methods
    for method in image_summary_methods:
        source = sources_by_name[method]
        method_values = image_auc[:, names_to_index[method]]
        for score_index, score_space in enumerate(spaces):
            for direction_index, direction in enumerate(directions):
                values = method_values[:, score_index, direction_index]
                _metric_rows(
                    pooled_rows, class_rows, inputs,
                    "image", "target_probability_auc", method, values,
                    target_source=source, score_space=score_space, direction=direction,
                )
    for source in ("fused", "vit_top"):
        for score_index, score_space in enumerate(spaces):
            for direction_index, direction in enumerate(directions):
                values = random_auc[source][:, score_index, direction_index]
                _metric_rows(
                    pooled_rows, class_rows, inputs,
                    "image", "target_probability_auc", "random_order_mean", values,
                    target_source=source, score_space=score_space, direction=direction,
                    baseline="mean_of_20_seeded_orders",
                )

    for method in map_methods:
        source = sources_by_name[method]
        method_values = image_auc[:, names_to_index[method]]
        baseline_names = (
            f"random_reference_{source}",
            f"center_reference_{source}",
            "random_order_mean",
        )
        for score_index, score_space in enumerate(spaces):
            for direction_index, direction in enumerate(directions):
                values = method_values[:, score_index, direction_index]
                family = f"image_{direction}"
                for baseline in baseline_names:
                    if baseline == "random_order_mean":
                        baseline_values = random_auc[source][
                            :, score_index, direction_index
                        ]
                    else:
                        baseline_values = image_auc[
                            :, names_to_index[baseline], score_index, direction_index
                        ]
                    _paired_comparison(
                        inputs,
                        comparison_rows,
                        family,
                        "image",
                        direction,
                        method,
                        baseline,
                        score_space,
                        source,
                        values,
                        baseline_values,
                    )

    curve_steps = np.linspace(0.0, 1.0, tabular["frozen_deletion_insertion_target_curves"].shape[-1])
    tabular_auc = np.trapezoid(
        tabular["frozen_deletion_insertion_target_curves"],
        x=curve_steps,
        axis=-1,
    )
    random_tabular_auc = np.stack([
        tabular["frozen_random_order_auc_xgb"].mean(axis=1),
        tabular["frozen_random_order_auc_cwf"].mean(axis=1),
    ], axis=1)
    if tabular_auc.shape != (len(inputs["test_ids"]), 2, 2):
        raise ValueError(f"unexpected clinical AUC shape: {tabular_auc.shape}")
    for score_index, score_space in enumerate(("xgb", "cwf")):
        for direction_index, direction in enumerate(("deletion", "insertion")):
            ordered_values = tabular_auc[:, score_index, direction_index]
            random_values = random_tabular_auc[:, score_index, direction_index]
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "clinical", "target_probability_auc", "TreeSHAP_group_order",
                ordered_values, target_source="cwf_selected_class",
                score_space=score_space, direction=direction,
            )
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "clinical", "target_probability_auc", "random_order_mean",
                random_values, target_source="cwf_selected_class",
                score_space=score_space, direction=direction,
                baseline="mean_of_20_seeded_orders",
            )
            _paired_comparison(
                inputs,
                comparison_rows,
                f"tabular_{direction}",
                "clinical",
                direction,
                "TreeSHAP_group_order",
                "random_order_mean",
                score_space,
                "cwf_selected_class",
                ordered_values,
                random_values,
            )

    class_masks = {
        "fused": "fused_unchanged",
        "vit_top": "vit_unchanged",
    }
    for source, mask_key in class_masks.items():
        for transform, state in sanity["stability"].items():
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "image_stability", "unchanged_prediction_rate",
                f"{source}_prediction", state[mask_key].astype(float),
                target_source=source, condition=transform,
            )
    for transform, state in sanity["stability"].items():
        for map_index, map_key in enumerate(sanity["map_keys"]):
            source = "fused" if map_key.endswith("_fused") else "vit_top"
            for metric in ("spearman", "ssim"):
                valid = state[f"{metric}_valid"][:, map_index]
                values = np.where(valid, state[metric][:, map_index], np.nan)
                _metric_rows(
                    pooled_rows, class_rows, inputs,
                    "image_stability", f"map_{metric}", map_key, values,
                    target_source=source, condition=transform,
                )

    for stage, state in sanity["randomization"].items():
        for map_index, map_key in enumerate(sanity["map_keys"]):
            source = "fused" if map_key.endswith("_fused") else "vit_top"
            for metric in ("spearman", "ssim"):
                valid = state[f"{metric}_valid"][:, map_index]
                values = np.where(valid, state[metric][:, map_index], np.nan)
                _metric_rows(
                    pooled_rows, class_rows, inputs,
                    "vit_randomization", f"map_{metric}", map_key, values,
                    target_source=source, condition=stage,
                )
        for score_space, key in (
            ("vit", "vit_argmax_changed"),
            ("cwf", "cwf_argmax_changed"),
        ):
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "vit_randomization", "argmax_changed_rate", stage,
                state[key].astype(float), score_space=score_space, condition=stage,
            )
        for score_space, key in (("vit", "vit_delta"), ("cwf", "cwf_delta")):
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "vit_randomization", "mean_max_probability_delta", stage,
                state[key], score_space=score_space, condition=stage,
            )

    targets = np.asarray(tabular["target_cwf_class"], dtype=int)
    row_indices = np.arange(len(targets))
    for score_space, key in (
        ("xgb", "frozen_xgb_group_shapley"),
        ("cwf", "frozen_cwf_group_shapley"),
    ):
        selected = np.asarray(tabular[key])[row_indices, targets, :]
        for group_index, group_name in enumerate(EXPECTED_VARIABLES):
            _metric_rows(
                pooled_rows, class_rows, inputs,
                "clinical", "mean_absolute_group_shapley",
                group_name, np.abs(selected[:, group_index]),
                target_source="cwf_selected_class", score_space=score_space,
            )

    family_members = {}
    for row in comparison_rows:
        family_members.setdefault(row["family"], []).append(row)
    for rows in family_members.values():
        corrected = holm_adjust([row["raw_p_value"] for row in rows])
        for row, adjusted in zip(rows, corrected):
            row["holm_adjusted_p_value"] = float(adjusted)
    return {
        "pooled_metrics": pd.DataFrame(pooled_rows),
        "per_class_metrics": pd.DataFrame(class_rows),
        "paired_comparisons": pd.DataFrame(comparison_rows),
        "tabular_auc": tabular_auc,
        "random_tabular_auc": random_tabular_auc,
        "image_random_auc": random_auc,
        "families": family_members,
    }


def _draw_auc_figure(comparisons, output_path):
    width, height = 1500, 1740
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    draw.text((40, 24), "Phase F: paired AUC difference vs random-order control", fill="black", font=title_font)
    draw.text(
        (40, 44),
        "Delta = explanation order minus mean of 20 seeded random orders; all values are exploratory.",
        fill="black",
        font=font,
    )
    panel_width, panel_height = 710, 390
    lefts = (35, 755)
    top = 85
    specs = []
    for direction in ("deletion", "insertion"):
        for score_space in ("vit", "cwf"):
            specs.append(("image", direction, score_space, "image"))
    for direction in ("deletion", "insertion"):
        for score_space in ("xgb", "cwf"):
            specs.append(("clinical", direction, score_space, "clinical"))

    colors = {"fused": (45, 99, 170), "vit_top": (220, 118, 38), "clinical": (50, 135, 85)}
    for panel_index, (experiment, direction, score_space, label) in enumerate(specs):
        col = panel_index % 2
        row = panel_index // 2
        x0 = lefts[col]
        y0 = top + row * (panel_height + 12)
        panel_rows = [
            item for item in comparisons
            if item["experiment"] == experiment
            and item["direction"] == direction
            and item["score_space"] == score_space
            and item["baseline"] == "random_order_mean"
        ]
        title = f"{label.title()} {direction} | {score_space.upper()} score"
        draw.text((x0 + 8, y0 + 5), title, fill="black", font=title_font)
        if not panel_rows:
            draw.text((x0 + 10, y0 + 35), "No comparisons available", fill="black", font=font)
            continue
        maximum = max(
            max(abs(item["delta_ci_low_95"]), abs(item["delta_ci_high_95"]))
            for item in panel_rows
        )
        maximum = max(maximum * 1.15, 0.02)
        label_x = x0 + 8
        plot_left = x0 + 260
        plot_right = x0 + panel_width - 20
        plot_width = plot_right - plot_left
        zero_x = int(plot_left + plot_width / 2)
        plot_top = y0 + 38
        plot_bottom = y0 + panel_height - 18
        draw.line((zero_x, plot_top, zero_x, plot_bottom), fill=(120, 120, 120), width=1)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(170, 170, 170), width=1)
        row_height = max(28, (plot_bottom - plot_top) // max(len(panel_rows), 1))
        for item_index, item in enumerate(panel_rows):
            yy = plot_top + item_index * row_height + row_height // 2
            if experiment == "image":
                source = item["target_source"]
                color = colors.get(source, (80, 80, 80))
                method = item["method"].replace("_fused", "").replace("_vit_top", "")
                target_label = "CWF target" if source == "fused" else "ViT target"
                text = f"{method} ({target_label})"
            else:
                color = colors["clinical"]
                text = "TreeSHAP group order"
            draw.text((label_x, yy - 5), text, fill="black", font=font)
            scale = plot_width / (2 * maximum)
            low_x = int(zero_x + item["delta_ci_low_95"] * scale)
            high_x = int(zero_x + item["delta_ci_high_95"] * scale)
            value_x = int(zero_x + item["mean_paired_delta_method_minus_baseline"] * scale)
            draw.line((low_x, yy, high_x, yy), fill=color, width=2)
            draw.line((low_x, yy - 4, low_x, yy + 4), fill=color, width=1)
            draw.line((high_x, yy - 4, high_x, yy + 4), fill=color, width=1)
            draw.ellipse((value_x - 4, yy - 4, value_x + 4, yy + 4), fill=color)
        draw.text((plot_left, plot_bottom + 2), f"-{maximum:.3f}", fill="black", font=font)
        draw.text((zero_x - 8, plot_bottom + 2), "0", fill="black", font=font)
        draw.text((plot_right - 35, plot_bottom + 2), f"+{maximum:.3f}", fill="black", font=font)
    draw.text(
        (40, height - 23),
        "Negative deletion deltas and positive insertion deltas indicate stronger target-score perturbation; not a method ranking.",
        fill="black",
        font=font,
    )
    image.save(output_path, format="PNG", optimize=True)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_report(path, summary, comparisons, pooled_metrics):
    lines = [
        "# Phase F exploratory faithfulness and stability report",
        "",
        "## Scope and caveat",
        "",
        "This is one fixed exploratory evaluation of the frozen 461-row PAD-UFES-20 test split. "
        "A test image was previously used during explanation debugging, so these results are "
        "descriptive and are not confirmatory. No test result was used to tune this run.",
        "",
        "Image maps are ViT-branch explanations even when the target class was selected by CWF; "
        "CWF scores are recomputed under image interventions. Clinical TreeSHAP values explain "
        "the XGBoost branch; clinical CWF scores are recomputed with the ViT probabilities held fixed.",
        "",
        "## Data and execution",
        "",
        f"- Frozen rows: {summary['test_rows']} across {summary['test_patients']} patients.",
        f"- True-class counts: `{json.dumps(summary['class_counts'], sort_keys=True)}`.",
        "- Image curves and sanity runs cover all 461 ordered test IDs; grouped clinical calculations use "
        "256 coalitions and the same class-stratified 200-row training background.",
        "- The label-permuted XGBoost control was trained on training rows only (seed 42, XGBoost 3.2.0).",
        "",
        "## Findings",
        "",
        "Patient-cluster confidence intervals and Holm-adjusted paired comparisons are in the CSV files. "
        "AUC deltas are method minus baseline; lower deletion and higher insertion scores indicate "
        "stronger target-score perturbation, but no method is selected or ranked from this test set.",
        "",
        f"- The control's exploratory accuracy was {summary['control']['test_accuracy']:.3f} "
        f"and macro-F1 was {summary['control']['test_macro_f1']:.3f}.",
        f"- Mean grouped-attribution rank Spearman versus the frozen model was "
        f"{summary['control']['rank_spearman_xgb_mean']:.3f} for XGBoost and "
        f"{summary['control']['rank_spearman_cwf_mean']:.3f} for CWF.",
        f"- Paired comparisons: {len(comparisons)} total across "
        f"{len(summary['statistics']['holm_families'])} predeclared Holm families.",
        f"- Summary metrics: {len(pooled_metrics)} pooled rows, including AUC, image stability, "
        "ViT randomization, and clinical attribution magnitude.",
        "",
        "## Limitations",
        "",
        "The test split is exposed and must not be used for further method development. The clinical "
        "randomization control is a single seed and its test behavior is only a sanity check. An "
        "untouched external cohort and independent validation remain necessary for confirmatory claims.",
        "",
        "## Reproducibility",
        "",
        "Configurations, exact environment records, execution logs, input hashes, output hashes, and "
        "validation status are saved alongside this report.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_summary(root=ROOT, metadata_path=META_PATH, output_dir=DEFAULT_OUTPUT):
    root = Path(root)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite report output: {output_dir}")

    start_time = time.perf_counter()
    inputs = _validate_run_inputs(root, metadata_path)
    image_curves = _collect_image_auc(inputs)
    sanity = _collect_sanity(inputs)
    result_path = inputs["tabular_dir"] / "tabular_task4_results.npz"
    with np.load(result_path, allow_pickle=False) as saved:
        expected_ids = inputs["test_ids"]
        if _string_list(saved["test_row_keys"]) != expected_ids:
            raise ValueError("tabular result IDs do not align with frozen test IDs")
        if not np.array_equal(saved["test_y"].astype(int), inputs["y_test"]):
            raise ValueError("tabular result labels do not align with frozen test labels")
        tabular = {key: saved[key].copy() for key in saved.files}

    tables = _build_summary_tables(inputs, image_curves, sanity, tabular)
    class_counts = {
        class_name: int(np.sum(inputs["y_test"] == class_index))
        for class_index, class_name in enumerate(EXPECTED_CLASSES)
    }
    class_patient_counts = {
        class_name: int(len(np.unique(
            inputs["patient_ids"][inputs["y_test"] == class_index]
        )))
        for class_index, class_name in enumerate(EXPECTED_CLASSES)
    }
    tabular_control = inputs["tabular_manifest"]["control"]
    family_counts = {
        family: len(rows) for family, rows in tables["families"].items()
    }
    summary = {
        "status": "complete_exploratory",
        "test_rows": len(inputs["test_ids"]),
        "test_patients": int(len(np.unique(inputs["patient_ids"]))),
        "class_counts": class_counts,
        "patient_counts_by_class": class_patient_counts,
        "image": {
            "curve_chunks": int(inputs["image_manifest"]["chunk_count"]),
            "sanity_chunks": int(inputs["sanity_manifest"]["chunk_count"]),
            "curve_methods": len(image_curves["order_names"]),
            "target_sources": ["fused", "vit_top"],
            "score_spaces": image_curves["score_spaces"],
            "directions": image_curves["directions"],
            "stability_conditions": sorted(sanity["stability"]),
            "randomization_stages": sorted(sanity["randomization"]),
        },
        "tabular": {
            "coalitions_per_row": 256,
            "background_rows": 200,
            "groups": list(EXPECTED_VARIABLES),
            "efficiency_errors": inputs["tabular_manifest"]["validation"],
        },
        "control": tabular_control,
        "statistics": {
            "bootstrap_unit": "patient",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "paired_test": "two-sided patient-aggregated sign-flip",
            "signflip_permutations": SIGNFLIP_PERMUTATIONS,
            "holm_families": family_counts,
            "direction_interpretation": {
                "deletion": "lower AUC means the target score falls faster",
                "insertion": "higher AUC means the target score rises faster",
            },
        },
        "interpretation_scope": {
            "image_maps": "ViT branch maps; fused names denote a CWF-selected target, not an end-to-end CWF pixel attribution",
            "image_cwf_score": "recomputed CWF probability from intervened ViT and frozen XGBoost probabilities",
            "clinical_shapley": "grouped interventional probability-Shapley values for the XGBoost branch",
            "clinical_cwf_score": "recomputed CWF probability from intervened XGBoost and frozen ViT probabilities",
            "test_set": "exploratory only due disclosed test-image debugging deviation",
        },
        "report_files": [
            "report.md",
            "pooled_metrics.csv",
            "per_class_metrics.csv",
            "paired_comparisons.csv",
            "phase_f_auc_deltas.png",
            "environment.json",
            "input_hashes_final.json",
        ],
    }
    environment = {
        "summary": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": Image.__version__,
        },
        "image": inputs["image_manifest"]["environment"],
        "tabular": inputs["tabular_manifest"]["versions"],
        "commands": {
            "summary": "python -m experiments.phase_f.summarize",
            "image": inputs["image_manifest"]["command"],
            "image_sanity": inputs["sanity_manifest"]["command"],
            "tabular": "python -m experiments.phase_f.tabular_eval",
        },
        "config": {
            "image": "experiments/phase_f/config.json",
            "image_sanity": "experiments/phase_f/sanity_config.json",
            "tabular": "outputs/phase_f_v1/tabular_task4_v1/config.json",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "signflip_permutations": SIGNFLIP_PERMUTATIONS,
            "seed": SEED,
        },
    }

    output_dir.mkdir(parents=True)
    log_path = output_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("command: %s", " ".join([sys.executable, *sys.argv]))
    logging.info("validated image/tabular chunks and protected input hashes")
    logging.info("rows=%d patients=%d class_counts=%s", len(inputs["test_ids"]), summary["test_patients"], class_counts)

    pooled = tables["pooled_metrics"]
    per_class = tables["per_class_metrics"]
    comparisons = tables["paired_comparisons"]
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    _draw_auc_figure(
        comparisons.to_dict(orient="records"),
        output_dir / "phase_f_auc_deltas.png",
    )
    summary["elapsed_summary_seconds"] = float(time.perf_counter() - start_time)
    summary["comparison_count"] = int(len(comparisons))
    summary["pooled_metric_count"] = int(len(pooled))
    summary["per_class_metric_count"] = int(len(per_class))
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "report.md",
        summary,
        comparisons.to_dict(orient="records"),
        pooled.to_dict(orient="records"),
    )

    input_hashes_after = {}
    for relative, expected_hash in inputs["frozen"]["input_hashes"].items():
        path = root / Path(relative)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"protected input changed during summary: {relative}")
        input_hashes_after[relative] = actual_hash
    metadata_after = _sha256(inputs["metadata_path"])
    if metadata_after != inputs["metadata_sha256"]:
        raise ValueError("clinical metadata changed during summary")
    input_hashes_after[str(inputs["metadata_path"])] = metadata_after
    input_hashes_record = {
        "before": inputs["input_sha256_before"],
        "after": input_hashes_after,
        "protected_hashes_unchanged": (
            inputs["input_sha256_before"] == input_hashes_after
        ),
    }
    if not input_hashes_record["protected_hashes_unchanged"]:
        raise ValueError("before/after protected input hash snapshots differ")
    (output_dir / "input_hashes_final.json").write_text(
        json.dumps(input_hashes_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = {
        "phase": "F",
        "status": "executed_and_validated_exploratory",
        "test_results_confirmatory": False,
        "experiments": {
            "image_deletion_insertion_random_controls": "executed_and_validated",
            "image_stability": "executed_and_validated",
            "vit_model_randomization": "executed_and_validated",
            "grouped_clinical_interventions": "executed_and_validated",
            "xgboost_label_permutation_control": "executed_and_validated",
            "patient_cluster_statistics_and_report": "executed_and_validated",
        },
        "limitations": [
            "one test image was used during explanation debugging",
            "no untouched external validation cohort is included",
            "Phase G/I/H and remaining Phase J work are separate pending workstreams",
        ],
        "protected_inputs_unchanged": True,
    }
    summary["status"] = "complete_exploratory"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info(
        "created %d pooled metrics, %d per-class metrics, and %d paired comparisons",
        len(pooled), len(per_class), len(comparisons),
    )
    logging.info("final protected input hashes verified")
    logging.shutdown()

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "status": "complete_exploratory",
        "code_sha256": _code_hashes(root, [Path(__file__)]),
        "output_sha256": output_hashes,
        "input_sha256_before": inputs["input_sha256_before"],
        "input_sha256_after": input_hashes_after,
        "protected_inputs_unchanged": True,
        "image_chunk_count": len(list(inputs["phase_output"].glob("image_curves_*.npz"))),
        "sanity_chunk_count": (
            len(list(inputs["phase_output"].glob("image_stability_*.npz")))
            + len(list(inputs["phase_output"].glob("image_randomization_*.npz")))
        ),
        "tabular_results_sha256": _sha256(result_path),
        "elapsed_seconds": summary["elapsed_summary_seconds"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metadata", type=Path, default=META_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_summary(args.root, args.metadata, args.output_dir)
    print(json.dumps({
        "status": summary["status"],
        "test_rows": summary["test_rows"],
        "comparison_count": summary["comparison_count"],
        "pooled_metric_count": summary["pooled_metric_count"],
        "per_class_metric_count": summary["per_class_metric_count"],
        "output_dir": str(Path(args.root) / args.output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
