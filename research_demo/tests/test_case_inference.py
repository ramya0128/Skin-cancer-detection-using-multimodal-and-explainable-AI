from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
from PIL import Image
import torch

from research_demo.inference import (
    ClinicalInputs,
    encode_clinical_inputs,
    fuse_cwf,
    prepare_uploaded_image,
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
            fuse_cwf(xgb, vit, np.asarray([0.75, 0.25, 0.5]))
        with self.assertRaises(ValueError):
            fuse_cwf(xgb, vit, np.asarray([0.75, 0.25, 0.5, 0.5, 0.5, 1.1]))


if __name__ == "__main__":
    unittest.main()
