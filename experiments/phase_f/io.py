import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
EXPECTED_VARIABLES = [
    "grew",
    "itch",
    "changed",
    "elevation",
    "bleed",
    "hurt",
    "region",
    "age",
]
IMAGE_MAP_KEYS = (
    "gradcam_fused",
    "gradcam_vit_top",
    "attention_rollout_fused",
    "attention_rollout_vit_top",
    "chefer_fused",
    "chefer_vit_top",
    "occlusion_fused",
    "occlusion_vit_top",
)
PROTECTED_INPUTS = (
    Path("outputs") / "split_v1.json",
    Path("outputs") / "base_outputs_v1.npz",
    Path("outputs") / "fusion_v1.npz",
    Path("outputs") / "image_explainers_v1.npz",
    Path("outputs") / "treeshap_v1.npz",
    Path("outputs") / "xgb_final_v1.joblib",
    Path("outputs") / "vit_final_v1.pt",
    Path("cache") / "images_uint8_224.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_npz(path: Path, allow_object_arrays: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_object_arrays) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _require_keys(arrays: dict[str, np.ndarray], keys: tuple[str, ...], name: str) -> None:
    missing = sorted(set(keys).difference(arrays))
    if missing:
        raise ValueError(f"{name} is missing required arrays: {missing}")


def _as_strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).tolist()]


def _validate_frozen_arrays(
    split: dict[str, object],
    base: dict[str, np.ndarray],
    fusion: dict[str, np.ndarray],
    images: dict[str, np.ndarray],
    tree: dict[str, np.ndarray],
) -> None:
    required = {
        "base": (base, ("classes", "y_train", "y_test", "test_xgb", "test_vit")),
        "fusion": (fusion, ("selected", "classes", "w_xgb", "cwf_test")),
        "image": (
            images,
            ("classes", "test_row_keys", "target_fused_class", "target_vit_top_class")
            + IMAGE_MAP_KEYS,
        ),
        "TreeSHAP": (
            tree,
            ("classes", "variables", "test_row_keys", "test_full_probability",
             "test_background_indices"),
        ),
    }
    for name, (arrays, keys) in required.items():
        _require_keys(arrays, keys, name)

    train_ids = split.get("train_img_ids")
    test_ids = split.get("test_img_ids")
    if not isinstance(train_ids, list) or not isinstance(test_ids, list):
        raise ValueError("split must contain train_img_ids and test_img_ids lists")
    if len(train_ids) != 1837 or len(test_ids) != 461:
        raise ValueError(f"unexpected split sizes: train={len(train_ids)}, test={len(test_ids)}")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("split image IDs are not unique")
    if set(train_ids).intersection(test_ids):
        raise ValueError("training and test image IDs overlap")

    row_keys = _as_strings(images["test_row_keys"])
    tree_keys = _as_strings(tree["test_row_keys"])
    if row_keys != test_ids or tree_keys != test_ids:
        raise ValueError("image or TreeSHAP test row keys do not align with frozen split IDs")
    if _as_strings(base["classes"]) != EXPECTED_CLASSES:
        raise ValueError("base class order differs from frozen class order")
    if _as_strings(fusion["classes"]) != EXPECTED_CLASSES:
        raise ValueError("fusion class order differs from frozen class order")
    if _as_strings(images["classes"]) != EXPECTED_CLASSES:
        raise ValueError("image-explainer class order differs from frozen class order")
    if _as_strings(tree["classes"]) != EXPECTED_CLASSES:
        raise ValueError("TreeSHAP class order differs from frozen class order")
    if _as_strings(tree["variables"]) != EXPECTED_VARIABLES:
        raise ValueError("TreeSHAP variable order differs from the eight grouped variables")
    if str(np.asarray(fusion["selected"]).item()) != "CWF":
        raise ValueError("frozen fusion selection is not CWF")

    n_test = len(test_ids)
    if base["y_train"].shape != (1837,) or base["y_test"].shape != (n_test,):
        raise ValueError("base label array has an unexpected row count")
    if (
        np.any((base["y_train"] < 0) | (base["y_train"] >= len(EXPECTED_CLASSES)))
        or np.any((base["y_test"] < 0) | (base["y_test"] >= len(EXPECTED_CLASSES)))
    ):
        raise ValueError("base labels contain class indices outside the frozen classes")
    for key in ("test_xgb", "test_vit"):
        if base[key].shape != (n_test, len(EXPECTED_CLASSES)):
            raise ValueError(f"{key} has an unexpected probability shape: {base[key].shape}")
        if not np.isfinite(base[key]).all() or not np.allclose(
            base[key].sum(axis=1), 1.0, atol=1e-4
        ):
            raise ValueError(f"{key} contains invalid probabilities")
    if fusion["w_xgb"].shape != (len(EXPECTED_CLASSES),):
        raise ValueError("CWF class weights have an unexpected shape")
    if not np.isfinite(fusion["w_xgb"]).all() or np.any(
        (fusion["w_xgb"] < 0) | (fusion["w_xgb"] > 1)
    ):
        raise ValueError("CWF class weights must be finite values in [0, 1]")
    if fusion["cwf_test"].shape != (n_test, len(EXPECTED_CLASSES)):
        raise ValueError("CWF test probabilities have an unexpected shape")
    if not np.isfinite(fusion["cwf_test"]).all() or not np.allclose(
        fusion["cwf_test"].sum(axis=1), 1.0, atol=1e-4
    ):
        raise ValueError("CWF test probabilities are invalid")

    for key in ("target_fused_class", "target_vit_top_class"):
        targets = images[key]
        if targets.shape != (n_test,) or np.any(
            (targets < 0) | (targets >= len(EXPECTED_CLASSES))
        ):
            raise ValueError(f"{key} has invalid target classes")
    for key in IMAGE_MAP_KEYS:
        maps = images[key]
        if maps.shape != (n_test, 14, 14):
            raise ValueError(f"{key} has an unexpected map shape: {maps.shape}")
        if not np.isfinite(maps).all() or np.any((maps < 0) | (maps > 1)):
            raise ValueError(f"{key} contains invalid map values")
    if tree["test_full_probability"].shape != (n_test, len(EXPECTED_CLASSES), 8):
        raise ValueError("grouped TreeSHAP probabilities have an unexpected shape")
    background = tree["test_background_indices"]
    if (
        background.shape != (200,)
        or not np.issubdtype(background.dtype, np.integer)
        or len(np.unique(background)) != 200
        or np.any((background < 0) | (background >= len(train_ids)))
    ):
        raise ValueError("TreeSHAP background indices must be 200 unique training rows")
    background_class_counts = np.bincount(
        base["y_train"][background],
        minlength=len(EXPECTED_CLASSES),
    )
    if not np.array_equal(background_class_counts, np.asarray([34, 34, 33, 33, 33, 33])):
        raise ValueError(
            "TreeSHAP background must be class-stratified across the six training classes"
        )
    if not np.isfinite(tree["test_full_probability"]).all():
        raise ValueError("grouped TreeSHAP probabilities contain non-finite values")


def load_frozen_arrays(root: Path) -> dict[str, object]:
    root = Path(root)
    paths = {relative: root / relative for relative in PROTECTED_INPUTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))

    outputs = root / "outputs"
    split = json.loads((outputs / "split_v1.json").read_text(encoding="utf-8"))
    # This project artifact stores its class labels as a NumPy object array.
    base = _read_npz(outputs / "base_outputs_v1.npz", allow_object_arrays=True)
    fusion = _read_npz(outputs / "fusion_v1.npz")
    images = _read_npz(outputs / "image_explainers_v1.npz")
    tree = _read_npz(outputs / "treeshap_v1.npz")
    _validate_frozen_arrays(split, base, fusion, images, tree)

    input_hashes = {
        str(relative): sha256_file(path)
        for relative, path in paths.items()
    }
    return {
        "split": split,
        "base": base,
        "fusion": fusion,
        "images": images,
        "tree": tree,
        "input_hashes": input_hashes,
    }


def prepare_run_dir(
    output_dir: Path,
    config_hash: str,
    input_hashes: dict[str, str],
    resume: bool,
) -> None:
    output_dir = Path(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = {
        "config_sha256": config_hash,
        "input_sha256": dict(sorted(input_hashes.items())),
    }

    if not output_dir.exists():
        if resume:
            raise FileNotFoundError(f"cannot resume missing output directory: {output_dir}")
        output_dir.mkdir(parents=True)
        identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        return

    if not resume:
        raise FileExistsError(
            f"output directory already exists; pass resume only for matching hashes: {output_dir}"
        )
    if not identity_path.is_file():
        raise ValueError(f"resume directory has no run identity: {output_dir}")
    saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if saved_identity != identity:
        raise ValueError("resume config or protected input hashes do not match")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("only --validate-only is available in Task 1")
    inputs = load_frozen_arrays(args.root)
    print("Validated rows:", len(inputs["split"]["test_img_ids"]))
    print("Classes:", len(EXPECTED_CLASSES))
    print("Clinical groups:", len(EXPECTED_VARIABLES))
    print("Protected input hashes:", json.dumps(inputs["input_hashes"], sort_keys=True))


if __name__ == "__main__":
    main()
