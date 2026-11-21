import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


try:
    io_module = importlib.import_module("experiments.phase_f.io")
except ModuleNotFoundError:
    io_module = None


def build_fixture(root: Path, *, bad_id: bool = False, missing_map: bool = False,
                  bad_shape: bool = False, bad_background: bool = False) -> None:
    outputs = root / "outputs"
    cache = root / "cache"
    outputs.mkdir(parents=True)
    cache.mkdir()
    row_keys = [f"image-{index:03d}" for index in range(461)]
    split = {
        "test_img_ids": row_keys,
        "train_img_ids": [f"train-{index:04d}" for index in range(1837)],
    }
    (outputs / "split_v1.json").write_text(json.dumps(split), encoding="utf-8")

    classes = np.asarray(["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"])
    test_prob = np.full((461, 6), 1 / 6, dtype=np.float32)
    base = {
        "classes": classes,
        "y_train": np.repeat(
            np.arange(6, dtype=np.int64),
            [603, 671, 45, 195, 143, 180],
        ),
        "y_test": np.zeros(461, dtype=np.int64),
        "test_xgb": test_prob,
        "test_vit": test_prob,
    }
    np.savez(outputs / "base_outputs_v1.npz", **base)
    np.savez(
        outputs / "fusion_v1.npz",
        selected=np.asarray("CWF"),
        classes=classes,
        w_xgb=np.full(6, 0.5, dtype=np.float32),
        cwf_test=test_prob,
    )
    image_maps = {
        "classes": classes,
        "test_row_keys": np.asarray(row_keys),
        "target_fused_class": np.zeros(461, dtype=np.int64),
        "target_vit_top_class": np.zeros(461, dtype=np.int64),
    }
    for map_name in (
        "gradcam_fused",
        "gradcam_vit_top",
        "attention_rollout_fused",
        "attention_rollout_vit_top",
        "chefer_fused",
        "chefer_vit_top",
        "occlusion_fused",
        "occlusion_vit_top",
    ):
        image_maps[map_name] = np.zeros((461, 14, 14), dtype=np.float16)
    if bad_id:
        image_maps["test_row_keys"][0] = "wrong-image"
    if missing_map:
        del image_maps["occlusion_vit_top"]
    np.savez(outputs / "image_explainers_v1.npz", **image_maps)

    background_indices = np.concatenate([
        np.flatnonzero(base["y_train"] == class_index)[:quota]
        for class_index, quota in enumerate([34, 34, 33, 33, 33, 33])
    ])
    if bad_background:
        background_indices = np.arange(200)
    tree = {
        "classes": classes,
        "variables": np.asarray(
            ["grew", "itch", "changed", "elevation", "bleed", "hurt", "region", "age"]
        ),
        "test_row_keys": np.asarray(row_keys),
        "test_full_probability": np.zeros((461, 6, 8), dtype=np.float32),
        "test_background_indices": background_indices,
    }
    if bad_shape:
        tree["test_full_probability"] = np.zeros((460, 6, 8), dtype=np.float32)
    np.savez(outputs / "treeshap_v1.npz", **tree)

    for name in ("xgb_final_v1.joblib", "vit_final_v1.pt"):
        (outputs / name).write_bytes(b"fixture checkpoint bytes")
    (cache / "images_uint8_224.pt").write_bytes(b"fixture cache bytes")


class FrozenInputValidationTests(unittest.TestCase):
    def require_module(self):
        if io_module is None:
            self.fail("experiments.phase_f.io has not been implemented")
        return io_module

    def test_rejects_misaligned_test_image_ids(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture(root, bad_id=True)
            with self.assertRaisesRegex(ValueError, "row|ID|key|align"):
                module.load_frozen_arrays(root)

    def test_rejects_missing_maps_and_wrong_row_count(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture(root, missing_map=True)
            with self.assertRaisesRegex(ValueError, "missing|required|map"):
                module.load_frozen_arrays(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture(root, bad_shape=True)
            with self.assertRaisesRegex(ValueError, "shape|461|row"):
                module.load_frozen_arrays(root)

    def test_rejects_non_stratified_training_background(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture(root, bad_background=True)
            with self.assertRaisesRegex(ValueError, "background.*strat|strat.*background"):
                module.load_frozen_arrays(root)

    def test_run_directory_requires_matching_resume_identity(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "phase_f_v1"
            input_hashes = {"vit": "a" * 64, "split": "b" * 64}
            module.prepare_run_dir(output_dir, "config-hash", input_hashes, resume=False)
            with self.assertRaisesRegex(FileExistsError, "resume|exists|overwrite"):
                module.prepare_run_dir(output_dir, "config-hash", input_hashes, resume=False)
            module.prepare_run_dir(output_dir, "config-hash", input_hashes, resume=True)
            with self.assertRaisesRegex(ValueError, "hash|config|resume"):
                module.prepare_run_dir(
                    output_dir,
                    "different-config",
                    input_hashes,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
