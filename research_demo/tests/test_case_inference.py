from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_demo.inference import (
    CasePrediction,
    CaseRuntime,
    ClinicalInputs,
    encode_clinical_inputs,
    fuse_cwf,
    load_case_runtime,
    load_clinical_runtime,
    prepare_uploaded_image,
    predict_clinical,
    predict_uploaded_case,
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
FUSION_WEIGHTS = np.asarray(
    [
        0.5147864184008762,
        0.5261437908496732,
        0.4146341463414634,
        0.49825783972125426,
        0.4520547945205479,
        0.3730569948186528,
    ]
)


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


class FakeViTLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.ln_1 = nn.Identity()
        with torch.no_grad():
            self.projection.weight.fill_(1.0)
            self.projection.bias.zero_()


class FakeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([FakeViTLayer()])
        self.heads = nn.Module()
        self.heads.head = nn.Linear(1, 6)
        with torch.no_grad():
            self.heads.head.weight.copy_(
                torch.arange(1, 7, dtype=torch.float32).reshape(6, 1) / 1000.0
            )
            self.heads.head.bias.copy_(
                torch.log(torch.tensor([0.13, 0.15, 0.16, 0.17, 0.18, 0.21]))
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = F.adaptive_avg_pool2d(
            image.mean(dim=1, keepdim=True),
            output_size=(14, 14),
        ).flatten(2).transpose(1, 2)
        cls_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat((cls_token, patches), dim=1)
        tokens = self.encoder.layers[-1].projection(tokens)
        tokens = self.encoder.layers[-1].ln_1(tokens)
        return self.heads.head(tokens[:, 0, :])


class InvalidLogitViT(FakeViT):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (image.shape[0], 6),
            float("nan"),
            dtype=image.dtype,
            device=image.device,
        )


class InvalidProbabilityClassifier(ToyClassifier):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = super().predict_proba(frame)
        probabilities[:, 0] += 1e-4
        return probabilities


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
    fusion_weights: np.ndarray | None = None,
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
    weights = FUSION_WEIGHTS if fusion_weights is None else fusion_weights
    (output_dir / "fusion_v1.json").write_text(
        json.dumps(
            {
                "selected": "CWF",
                "classes": CLASSES,
                "w_xgb": np.asarray(weights).tolist(),
            }
        ),
        encoding="utf-8",
    )
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


class CaseRuntimeTests(unittest.TestCase):
    def _load_fake_runtime(self, root: Path, metadata_path: Path):
        checkpoint = root / "outputs" / "vit_final_v1.pt"
        torch.save(FakeViT().state_dict(), checkpoint)
        with patch(
            "experiments.phase_f.image_eval.models.vit_b_16",
            side_effect=lambda weights=None: FakeViT(),
        ):
            return load_case_runtime(
                root,
                metadata_path,
                device=torch.device("cpu"),
            )

    def test_loads_strict_vit_checkpoint_and_saved_fusion_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(root)

            runtime = self._load_fake_runtime(root, metadata_path)

        self.assertEqual(runtime.clinical.class_names, CLASSES)
        np.testing.assert_array_equal(runtime.xgb_weights, FUSION_WEIGHTS)
        self.assertEqual(runtime.device, torch.device("cpu"))

    def test_predicts_each_branch_and_targets_gradcam_at_cwf_class(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(root)
            runtime = self._load_fake_runtime(root, metadata_path)
            from experiments.phase_f import image_eval

            original_gradcam = image_eval._gradcam_maps
            observed_targets = []
            observed_inputs = []

            def capture_gradcam(model, normalized, target_classes):
                observed_targets.extend(np.asarray(target_classes).tolist())
                observed_inputs.append(normalized.detach().cpu().clone())
                return original_gradcam(model, normalized, target_classes)

            image = torch.linspace(0.0, 1.0, 224 * 224).reshape(1, 1, 224, 224)
            image = image.repeat(1, 3, 1, 1)
            with patch.object(
                image_eval,
                "_gradcam_maps",
                side_effect=capture_gradcam,
            ):
                first = predict_uploaded_case(image, clinical_inputs(), runtime)
                second = predict_uploaded_case(torch.flip(image, dims=[-1]), clinical_inputs(), runtime)

        self.assertEqual(first.vit_probabilities.shape, (6,))
        self.assertEqual(first.cwf_probabilities.shape, (6,))
        self.assertTrue(np.isfinite(first.vit_probabilities).all())
        self.assertTrue(np.isfinite(first.cwf_probabilities).all())
        self.assertAlmostEqual(float(first.cwf_probabilities.sum()), 1.0, places=6)
        self.assertEqual(
            first.cwf_predicted_class_index,
            int(first.cwf_probabilities.argmax()),
        )
        self.assertNotEqual(
            first.vit_predicted_class_index,
            first.cwf_predicted_class_index,
        )
        self.assertEqual(first.gradcam_map.shape, (14, 14))
        self.assertTrue(np.isfinite(first.gradcam_map).all())
        self.assertEqual(
            observed_targets,
            [
                first.cwf_predicted_class_index,
                second.cwf_predicted_class_index,
            ],
        )
        self.assertEqual(len(observed_inputs), 2)
        self.assertFalse(torch.equal(observed_inputs[0], observed_inputs[1]))
        target = first.cwf_predicted_class_index
        expected_xgb_term = FUSION_WEIGHTS[target] * first.clinical.probabilities[target]
        expected_vit_term = (1.0 - FUSION_WEIGHTS[target]) * first.vit_probabilities[target]
        self.assertAlmostEqual(first.xgb_weighted_term, expected_xgb_term, places=7)
        self.assertAlmostEqual(first.vit_weighted_term, expected_vit_term, places=7)
        self.assertAlmostEqual(
            first.cwf_probabilities[target],
            (expected_xgb_term + expected_vit_term)
            / first.cwf_normalization_constant,
            places=7,
        )

    def test_rejects_malformed_checkpoint_and_cwf_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(
                root,
                fusion_weights=np.asarray([0.5, 0.5]),
            )
            torch.save(FakeViT().state_dict(), root / "outputs" / "vit_final_v1.pt")
            with self.assertRaises(ValueError):
                load_case_runtime(root, metadata_path, device=torch.device("cpu"))

            metadata_path = create_clinical_project(
                root,
                fusion_weights=FUSION_WEIGHTS,
            )
            torch.save(
                {"wrong": torch.tensor(1)},
                root / "outputs" / "vit_final_v1.pt",
            )
            with patch(
                "experiments.phase_f.image_eval.models.vit_b_16",
                side_effect=lambda weights=None: FakeViT(),
            ):
                with self.assertRaises(RuntimeError):
                    load_case_runtime(
                        root,
                        metadata_path,
                        device=torch.device("cpu"),
                    )

    def test_rejects_invalid_clinical_branch_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = create_clinical_project(root)
            runtime = self._load_fake_runtime(root, metadata_path)
            invalid_clinical_runtime = replace(
                runtime.clinical,
                model=InvalidProbabilityClassifier(),
            )
            invalid_runtime = replace(runtime, clinical=invalid_clinical_runtime)

            with self.assertRaises(ValueError):
                predict_uploaded_case(
                    torch.zeros((1, 3, 224, 224)),
                    clinical_inputs(),
                    invalid_runtime,
                )

            invalid_vit_runtime = replace(runtime, vit_model=InvalidLogitViT())
            with self.assertRaises(ValueError):
                predict_uploaded_case(
                    torch.zeros((1, 3, 224, 224)),
                    clinical_inputs(),
                    invalid_vit_runtime,
                )


if __name__ == "__main__":
    unittest.main()
