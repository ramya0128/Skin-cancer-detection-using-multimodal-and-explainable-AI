# Skin Cancer Detection with Multimodal and Explainable AI

This repository contains a patient-level train/test split, image and clinical
branch models, multimodal fusion, and modality-level Shapley analyses.

## Collaborator setup

### 1. Download the prepared artifacts

Ask the project owner to upload the artifact bundle (for example,
`skin-lesion-v2-artifacts.zip`) to a **restricted Drive folder** and share it
with the collaborator's Google account. The actual link is not available yet:

> **Drive link:** _Add the restricted shared-folder URL here before distributing this README._

Sign in to that authorized account, open the shared link, download the ZIP (or
use Drive's folder download), and extract it into the repository root. Confirm
that `outputs/` and `cache/` appear directly under the root—not nested inside
an extra extracted folder.

The Drive bundle should preserve these paths when extracted into the repository
root:

```text
outputs/
  base_outputs_v1.npz
  fusion_v1.npz
  modality_shapley_v1.npz
  split_v1.json
  xgb_final_v1.joblib
  vit_final_v1.pt
  modality_shapley_v2_oof_checkpoint.npz
  modality_shapley_v2.npz        # completed result, if sharing it
cache/
  images_uint8_224.pt
```

The OOF checkpoint contains the completed five-fold input-level results. The
final `modality_shapley_v2.npz` is useful if you only need to inspect saved
results and do not need to rerun the analysis.

### 2. Configure the local dataset

The dataset itself is not included in the artifact list. Obtain it only through
an authorized source. Edit `META` and `IMG_ROOT` in `common.py` to point to the
collaborator's local metadata CSV and image directory. The image directory must
contain the lesion images referenced by the frozen split.

### 3. Set up Python

There is currently no `requirements.txt`. Use a Python/Jupyter environment with
compatible versions of:

- NumPy, pandas, scikit-learn, SciPy, Pillow, joblib, and XGBoost
- PyTorch and torchvision
- Jupyter and the VS Code Jupyter extension (if using VS Code)

For a CPU-only machine, install the CPU build of PyTorch and torchvision using
the official PyTorch installation selector. The project was developed with
Python 3.11; compatible package versions may differ on other Python versions.

## Which notebooks should run?

### If the complete prepared artifact bundle is available

Do **not** rerun `01_base_outputs.ipynb` or `02_fusion.ipynb`; their outputs are
already included. Continue with `03b_modality_shapley.ipynb` only after reading
the CPU limitation below.

### If reproducing from scratch

Run the notebooks in order:

1. `01_base_outputs.ipynb` — creates the frozen split, image cache, branch
   models, and base OOF/test probabilities. This trains ViTs and is expensive
   on CPU.
2. `02_fusion.ipynb` — selects fusion using saved base outputs; it does not
   retrain the branch models.
3. `03b_modality_shapley.ipynb` — runs input-level Shapley analysis and
   checkpoints each completed OOF fold.

## Important: CPU-only execution of 03b

The current `03b_modality_shapley.ipynb` explicitly requires CUDA during setup.
As written, it stops on a CPU-only computer **before it can restore the
completed OOF checkpoint**. Downloading the checkpoint alone does not bypass
that guard.

Before a CPU-only collaborator runs 03b, their coding agent must update the
notebook to allow CPU execution only when a compatible checkpoint shows all
five OOF folds complete, and to fail before training if any fold is missing.
The final test-set ViT inference still runs on CPU and may be slow. Do not remove
the missing-fold safeguard or start the five-fold ViT training on CPU.

After that CPU-aware update is in place, select the CPU Python environment in
VS Code, open `03b_modality_shapley.ipynb`, restart its kernel, and run the
notebook from top to bottom. Confirm the OOF cell reports all five folds as
already complete and skipped. If it reports an incomplete or incompatible
checkpoint, stop; do not bypass the guard.

If the collaborator only needs the existing analysis, use
`outputs/modality_shapley_v2.npz` rather than rerunning the notebook.

## Sharing and privacy

Keep the Drive folder restricted to authorized collaborators. The image cache
contains lesion images; metadata and saved per-case predictions can also be
sensitive. Share these files only when dataset terms, consent, and privacy
requirements permit it. Do not make the folder public by default.
