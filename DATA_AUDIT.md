# PAD-UFES-20 Data Audit

## Dataset and identifiers

- Dataset: PAD-UFES-20, local metadata and image directory are the data sources.
- Total metadata rows: 2,298.
- Unique patients: 1,373.
- Identifier fields:
  - `img_id`: image identifier and primary split key.
  - `patient_id`: patient identifier; used for patient-level splitting and clustered bootstrap resampling.
  - `lesion_id`: lesion identifier; 79 lesion IDs occur across train/test, despite no patient overlap.
- The workspace has no dataset license file. No license term is claimed here. Use of the data remains subject to the terms of the authorized source and the sharing restrictions in `README.md`.

## Frozen split

- Training rows: 1,837.
- Test rows: 461.
- Patients: 1,098 training, 275 test.
- Patient overlap: 0.
- Image-ID overlap: 0.
- Lesion-ID overlap across train/test: 79.
- Split unit: patient-level with five folds recorded in `outputs/split_v1.json`.

| Class | Train count | Test count |
|---|---:|---:|
| ACK | 603 | 127 |
| BCC | 671 | 174 |
| MEL | 45 | 7 |
| NEV | 195 | 49 |
| SCC | 143 | 49 |
| SEK | 180 | 55 |

MEL has only 7 test cases and must not be interpreted alone. OOF counts are shown alongside test results.

## Missingness

The eight model variables used by the tabular branch have no missing values in the metadata:

- `grew`, `itch`, `changed`, `elevation`, `bleed`, `hurt`, `region`, `age`: 0 missing each.

The pipeline still preserves NaN/unknown dummy handling for the encoded tabular matrix, but this local audit found no missing values in these eight source fields.

## Duplicate audit across train/test

- Exact image duplicate check: 0 cross-split exact-byte duplicate test hits.
- Perceptual duplicate check: a fixed 9x8 dHash found one cross-split perceptual-hash collision containing:
  - Training candidates: `PAT_409_2614_24.png`, `PAT_1373_1283_85.png`
  - Test candidate: `PAT_1547_1876_839.png`
- Handling: no image was removed or relabeled. The collision is retained as an audit flag and no model-selection or threshold decision was based on it.
- The 79 shared lesion IDs are retained as identifier overlap findings, not treated as duplicate image evidence.

## Test exposure log

| Notebook | Test data or prediction read | Purpose |
|---|---|---|
| `01_base_outputs.ipynb` | Test labels and test images | Final branch-model reporting and frozen test probabilities only; no test-label tuning |
| `02_fusion.ipynb` | Saved test probabilities | Report-only fusion comparison; branch models already frozen |
| `03_modality_shapley.ipynb` | Test labels and saved branch probabilities | Probability-level modality Shapley reporting |
| `03b_modality_shapley.ipynb` | Test labels, test images, and final branch predictions | Input-level test explanation reporting from completed/final models |
| `04_treeshap.ipynb` | Test labels, tabular rows, and final XGBoost predictions | Final-model clinical-branch explanation and descriptive audit; no model/background selection |
| `05_image_explainers.ipynb` | Test images, labels, frozen CWF targets, and final ViT predictions | Final-model image-branch maps and visual-only reporting; no method ranking or tuning |

**Protocol deviation:** during implementation debugging of the manual ViT attention path used by `05_image_explainers.ipynb`, a one-image smoke test used a frozen test image to compare logits and confirm gradient connectivity. This was an implementation check, not a comparison used to select or rank explainers, but it does not comply with the strict no-test-debugging rule. Consequently, the image-map results must be treated as descriptive/exploratory rather than clean confirmatory evidence. The test set was not used to select a background, share rule, threshold, or model. Do not use it for further method development.

The `unknown` shortcut groups in `04_treeshap.ipynb` are activated encoded `UNK`/`NAN` dummy columns. They should not be read as evidence of nulls in the eight raw metadata variables: this audit found zero raw nulls in those variables, and the nonzero groups are explicit encoded unknown/unseen categories.

## Explanation-run software and provenance notes

- `04_treeshap.ipynb` records Python 3.13.7, NumPy 2.2.5, pandas 2.3.3, scikit-learn 1.6.1, SciPy 1.15.3, SHAP 0.52.0, and XGBoost 3.2.0. The frozen model registry records XGBoost 3.4.1 for the original model environment; the explanation notebook loaded the existing model without retraining and recorded exact prediction reproduction.
- `05_image_explainers.ipynb` records Python 3.11.15, NumPy 2.4.6, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA 12.8, and an RTX 5050 Laptop GPU. Its saved ViT recomputation differs from the frozen base-output probabilities by at most 0.0007751882 and has one argmax disagreement; retain this cross-environment discrepancy in interpretation.
- `image_explainers_v1.npz` was last modified at 15:01 on 2026-11-18, while `05_image_explainers.ipynb` was last modified at 20:54 that day. Although the notebook has saved outputs for all six code cells, filesystem timestamps do not establish that the NPZ was generated from the notebook's latest saved source. Treat exact source-to-artifact correspondence as unverified.

## Hardware and software

- NVIDIA GPU visible: `NVIDIA GeForce RTX 5050 Laptop GPU`.
- The shell's default Python 3.13 environment has CPU-only PyTorch `2.11.0+cpu`.
- The `outfit_env` Jupyter kernel is Python 3.11/PyTorch `2.5.1+cu121` and sees the GPU, but PyTorch 2.5.1's supported CUDA architectures do not include RTX 5050's `sm_120`.
- Therefore GPU execution compatibility must be verified before any new ViT training. Completed OOF work does not need retraining.
- The executed 05 notebook used the separate compatible GPU environment listed above; this audit did not rerun it.

