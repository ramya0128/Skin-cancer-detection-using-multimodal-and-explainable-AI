from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from research_demo.inference import (
    ClinicalInputs,
    encode_clinical_inputs,
    fuse_cwf,
    load_clinical_runtime,
    prepare_uploaded_image,
    predict_clinical,
)


REGIONS = (
    "ABDOMEN",
    "ARM",
    "BACK",
    "CHEST",
    "EAR",
    "FACE",
    "FOOT",
    "FOREARM",
    "HAND",
    "LIP",
    "NECK",
    "NOSE",
    "SCALP",
    "THIGH",
)


def image_bytes(image_format: str = "PNG") -> bytes:
    image = Image.new("RGBA", (16, 8), (12, 34, 56, 255))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def saved_columns() -> list[str]:
    categorical = ("grew", "itch", "changed", "elevation", "bleed", "hurt")
    levels = ("False", "True", "UNK", "nan")
    columns = [
        f"{variable}_{level}"
        for variable in categorical
        for level in levels
    ]
    columns.extend(f"region_{region}" for region in REGIONS)
    columns.append("region_nan")
    columns.append("age")
    return list(reversed(columns))


CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
BACKGROUND_COUNTS = (34, 34, 33, 33, 33, 33)


class ToyClassifier:
    n_features_in_ = 40
    classes_ = np.arange(6)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        age = frame["age"].to_numpy(dtype=np.float64) / 100.0
        scores = np.column_stack(
            [
                0.02 * age + frame["grew_True"].to_numpy(),
                0.03 * age + frame["itch_True"].to_numpy(),
                0.4 * frame["changed_True"].to_numpy(),
                0.3 * frame["elevation_True"].to_numpy(),
                0.2 * frame["bleed_True"].to_numpy(),
                0.1 * frame["hurt_True"].to_numpy(),
            ]
        )
        scores -= scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)


def clinical_inputs() -> ClinicalInputs:
    return ClinicalInputs(
        grew="True",
        itch="False",
        changed="UNK",
        elevation="True",
        bleed="False",
        hurt="UNK",
        region="FACE",
        age=35,
    )


def create_clinical_project(
    root: Path,
    columns: list[str] | None = None,
    background_indices: np.ndarray | None = None,
) -> Path:
    from joblib import dump

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = [f"train-{index:03d}" for index in range(sum(BACKGROUND_COUNTS))]
    labels = [
        class_name
        for class_name, count in zip(CLASSES, BACKGROUND_COUNTS)
        for _ in range(count)
    ]
    rows = []
    levels = ("True", "False", "UNK")
    for index, (image_id, label) in enumerate(zip(train_ids, labels)):
        rows.append(
            {
                "img_id": image_id,
                "diagnostic": label,
                "grew": levels[index % 3],
                "itch": levels[(index + 1) % 3],
                "changed": levels[(index + 2) % 3],
                "elevation": levels[index % 3],
                "bleed": levels[(index + 1) % 3],
                "hurt": levels[(index + 2) % 3],
                "region": REGIONS[index % len(REGIONS)],
                "age": 6 + (index % 89),
            }
        )
    rows.append(
        {
            "img_id": "test-only",
            "diagnostic": "TEST_LABEL_MUST_NOT_BE_USED",
            "grew": "True",
            "itch": "True",
            "changed": "True",
            "elevation": "True",
            "bleed": "True",
            "hurt": "True",
            "region": "FACE",
            "age": 1000,
        }
    )
    metadata_path = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    (output_dir / "split_v1.json").write_text(
        json.dumps({"train_img_ids": train_ids, "test_img_ids": ["test-only"]}),
        encoding="utf-8",
    )
    indices = (
        np.arange(200, dtype=np.int64)
        if background_indices is None
        else np.asarray(background_indices)
    )
    np.savez(output_dir / "treeshap_v1.npz", test_background_indices=indices)
    saved = saved_columns() if columns is None else columns
    dump(
        {"model": ToyClassifier(), "columns": saved},
        output_dir / "xgb_final_v1.joblib",
    )
    return metadata_path


class UploadedInputTests(unittest.TestCase):
    def test_prepares_uploaded_png_as_raw_rgb_tensor(self):
        image = prepare_uploaded_image(image_bytes())

        self.assertEqual(tuple(image.shape), (1, 3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        np.testing.assert_allclose(
            image[0, :, 100, 100].numpy(),
            np.asarray([12, 34, 56], dtype=np.float32) / 255.0,
            atol=1e-6,
        )

    def test_rejects_corrupt_and_unsupported_image_bytes(self):
        with self.assertRaises(ValueError):
            prepare_uploaded_image(b"not an image")
        with self.assertRaises(ValueError):
            prepare_uploaded_image(image_bytes("GIF"))

    def test_validates_clinical_categories_and_preserves_out_of_range_age(self):
        inputs = ClinicalInputs(
            grew="True",
            itch="False",
            changed="UNK",
            elevation="True",
            bleed="False",
            hurt="UNK",
            region="FACE",
            age=100,
        )

        self.assertEqual(inputs.age, 100)
        self.assertTrue(inputs.age_out_of_range)
        self.assertFalse(
            ClinicalInputs(
                grew="True",
                itch="False",
                changed="UNK",
                elevation="True",
                bleed="False",
                hurt="UNK",
                region="FACE",
                age=94,
            ).age_out_of_range
        )
        with self.assertRaises(ValueError):
            ClinicalInputs(
                grew="yes",
                itch="False",
                changed="UNK",
                elevation="True",
                bleed="False",
                hurt="UNK",
                region="FACE",
                age=35,
            )
        with self.assertRaises(ValueError):
            ClinicalInputs(
                grew="True",
                itch="False",
                changed="UNK",
                elevation="True",
                bleed="False",
                hurt="UNK",
                region="FACE",
                age=35.5,
            )

    def test_encodes_all_columns_in_saved_order(self):
        inputs = ClinicalInputs(
            grew="True",
            itch="False",
            changed="UNK",
            elevation="True",
            bleed="False",
            hurt="UNK",
            region="FACE",
            age=35,
        )
        columns = saved_columns()

        encoded = encode_clinical_inputs(inputs, columns)

        self.assertEqual(encoded.shape, (1, 40))
        self.assertEqual(encoded.columns.tolist(), columns)
        self.assertEqual(encoded.iloc[0]["grew_True"], 1.0)
        self.assertEqual(encoded.iloc[0]["grew_False"], 0.0)
        self.assertEqual(encoded.iloc[0]["grew_nan"], 0.0)
        self.assertEqual(encoded.iloc[0]["region_FACE"], 1.0)
        self.assertEqual(encoded.iloc[0]["region_nan"], 0.0)
        self.assertEqual(encoded.iloc[0]["age"], 35.0)

    def test_fuses_class_weighted_probabilities_and_rejects_invalid_values(self):
        xgb = np.asarray([0.6, 0.4, 0, 0, 0, 0], dtype=np.float64)
        vit = np.asarray([0.2, 0.8, 0, 0, 0, 0], dtype=np.float64)
        weights = np.asarray([0.75, 0.25, 0.5, 0.5, 0.5, 0.5])

        fused = fuse_cwf(xgb, vit, weights)

        np.testing.assert_allclose(fused, [5 / 12, 7 / 12, 0, 0, 0, 0])
        self.assertAlmostEqual(float(fused.sum()), 1.0)
        with self.assertRaises(ValueError):
            fuse_cwf(np.asarray([-0.1, 1.1, 0, 0, 0, 0]), vit, weights)
        with self.assertRaises(ValueError):
            fuse_cwf(np.asarray([0.5, 0.50001, 0, 0, 0, 0]), vit, weights)
        with self.assertRaises(ValueError):
            fuse_cwf(xgb, vit, np.asarray([0.75, 0.25, 0.5]))
        with self.assertRaises(ValueError):
            fuse_cwf(xgb, vit, np.asarray([0.75, 0.25, 0.5, 0.5, 0.5, 1.1]))


class ClinicalRuntimeTests(unittest.TestCase):
    def test_loads_only_stratified_training_background(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(root)

            runtime = load_clinical_runtime(root, metadata_path)

        self.assertEqual(len(runtime.columns), 40)
        self.assertEqual(runtime.class_names, CLASSES)
        self.assertEqual(runtime.background.shape, (200, 40))
        self.assertEqual(runtime.background_class_counts, BACKGROUND_COUNTS)
        self.assertLess(float(runtime.background["age"].max()), 1000.0)

    def test_predicts_probabilities_with_efficient_grouped_shapley(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(root)
            runtime = load_clinical_runtime(root, metadata_path)

            prediction = predict_clinical(clinical_inputs(), runtime)

            baseline = runtime.model.predict_proba(runtime.background).mean(axis=0)
        self.assertEqual(prediction.probabilities.shape, (6,))
        self.assertTrue(np.isfinite(prediction.probabilities).all())
        self.assertAlmostEqual(float(prediction.probabilities.sum()), 1.0, places=6)
        self.assertEqual(
            prediction.predicted_class_index,
            int(prediction.probabilities.argmax()),
        )
        self.assertEqual(prediction.grouped_shapley_values.shape, (8,))
        self.assertTrue(np.isfinite(prediction.grouped_shapley_values).all())
        expected_delta = (
            prediction.probabilities[prediction.predicted_class_index]
            - baseline[prediction.predicted_class_index]
        )
        self.assertAlmostEqual(
            float(prediction.grouped_shapley_values.sum()),
            float(expected_delta),
            delta=1e-5,
        )

    def test_rejects_missing_metadata_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                load_clinical_runtime(root, root / "missing.csv")

            output_dir = root / "outputs"
            output_dir.mkdir()
            metadata_path = root / "metadata.csv"
            metadata_path.write_text("img_id\n", encoding="utf-8")
            (output_dir / "split_v1.json").write_text(
                json.dumps({"train_img_ids": []}),
                encoding="utf-8",
            )
            np.savez(
                output_dir / "treeshap_v1.npz",
                test_background_indices=np.arange(200),
            )
            with self.assertRaises(FileNotFoundError):
                load_clinical_runtime(root, metadata_path)

    def test_rejects_wrong_saved_model_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(
                root,
                columns=saved_columns()[:-1],
            )

            with self.assertRaises(ValueError):
                load_clinical_runtime(root, metadata_path)

    def test_rejects_background_index_outside_training_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(
                root,
                background_indices=np.asarray([*range(199), 200]),
            )

            with self.assertRaises(ValueError):
                load_clinical_runtime(root, metadata_path)


if __name__ == "__main__":
    unittest.main()
