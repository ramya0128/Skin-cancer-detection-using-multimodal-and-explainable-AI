# Modality Shapley v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible 03b notebook that computes cross-fitted input-level modality Shapley, compares it with saved probability-level Shapley, and writes versioned v2 results without changing v1 artifacts.

**Architecture:** Keep the work in a new notebook, reusing the dataset, split, features, ViT recipe, and CWF definitions from notebooks 01–03. Refit each fold's XGBoost and two-stage ViT transiently to compute OOF counterfactuals, then use the saved final branch models for the test set. Save only explanations and reproducibility metadata, not fold model checkpoints.

**Tech Stack:** Python, Jupyter, NumPy, pandas, scikit-learn, XGBoost, PyTorch/torchvision, joblib, Pillow.

**Spec:** `docs/superpowers/specs/2026-11-15-modality-shapley-v2-design.md`

## Global Constraints

- Use the frozen patient-level split and existing 1,837/461 train/test rows.
- Never use test data for model fitting, background sampling, selection, or tuning.
- Match OOF counterfactual models to the held-out fold that generated each OOF prediction; refit transiently and do not persist fold checkpoints.
- Use B=50 deterministic, class-stratified background rows drawn only from the applicable training partition.
- Preserve `outputs/*_v1.*`; write the result to `outputs/modality_shapley_v2.npz`.
- The ViT uses head-only stage 1 followed by full-model fine-tuning in stage 2; evaluate blurred images through the full model.
- Show per-class sample counts, including the small MEL test count, and omit negative-contribution rates from findings.

## Review Focus

- Misaligned image/metadata/split order: assert image IDs and row counts against `split_v1.json` before any explanation.
- Missing/corrupt cache or model artifact: fail with the exact required path instead of substituting a fallback.
- OOF fold leakage in background selection: assert all sampled background rows are in that fold's training partition and never its validation partition.
- Empty or zero-mass Shapley denominator: use a finite zero-share guard and assert all saved values are finite.
- Rare-class resamples lacking a class: report observed counts and use bootstrap summaries that tolerate absent classes without fabricating a class estimate.

---

### Task 1: Create the 03b input-level Shapley notebook

**Files:**
- Create: `03b_modality_shapley.ipynb`
- Read: `01_base_outputs.ipynb`, `02_fusion.ipynb`, `03_modality_shapley.ipynb`, `common.py`
- Read: `outputs/base_outputs_v1.npz`, `outputs/fusion_v1.npz`, `outputs/modality_shapley_v1.npz`, `outputs/split_v1.json`, `outputs/xgb_final_v1.joblib`, `outputs/vit_final_v1.pt`, `cache/images_uint8_224.pt`

**Interfaces:**
- Inputs: frozen artifacts and `common.py` dataset paths; no changes to their schemas.
- Produces: `outputs/modality_shapley_v2.npz` with `classes`, `prior`, `uniform`, `w_xgb`, `oof_input_phiT`, `oof_input_phiI`, `oof_input_rho`, `oof_input_pred`, `oof_input_f`, `oof_input_base`, corresponding `test_input_*` arrays, `oof_prob_*` and `test_prob_*` comparison arrays, and fixed OOF/test background indices.
- Notebook helpers: `cwf_apply(PT, PI, wT)`, `shapley_two(PT, PI, wT, bT, bI)`, and `input_level_shapley(...)`; each returns per-case signed attributions, share, predicted class, full-input payoff, and absent-both baseline payoff.
- Counterfactual rule: absent T is the mean branch probability over 50 fixed, class-stratified training-background rows; absent I is the ViT probability for a deterministically Gaussian-blurred raw image.

- [ ] **Step 1: Add a notebook setup and artifact-validation section**

  Load the frozen split, base outputs, fusion weights, v1 explanation outputs, metadata, and cached raw images. Reconstruct the one-hot clinical matrix exactly as in notebook 01. Assert ordered IDs, expected row counts, six-class probability shapes/sums, and required paths before continuing.

- [ ] **Step 2: Add deterministic coalition and background helpers**

  Define the four-coalition two-player calculation, deterministic class-stratified B=50 backgrounds, and raw-image blur/inference helpers. Add lightweight synthetic self-checks for the efficiency identity, zero-denominator share handling, deterministic background selection, and no validation/test indices in OOF backgrounds.

- [ ] **Step 3: Add transient OOF branch refitting and input-level evaluations**

  For each fold, recreate XGBoost and the existing two-stage ViT recipe on that fold's training rows. Use its held-out rows for full inputs and blurred images; average its XGBoost predictions over the fold's 50 background rows for absent-tabular coalitions. Use the same fold-specific CWF weight used for those OOF rows. Release the models after processing the fold; do not save checkpoints.

- [ ] **Step 4: Add one-time test evaluations using final models**

  Load the saved final XGBoost and ViT; form a deterministic B=50 background from training rows only. Compute test counterfactuals without fitting or selecting on test data.

- [ ] **Step 5: Retain and compare probability-level Shapley**

  Load the v1 signed probability-level arrays and verify alignment against the saved base outputs. Include those values as the secondary comparison; report Spearman correlation and sign agreement overall and per class between input-level and probability-level shares.

- [ ] **Step 6: Add summaries and save the v2 output**

  Print OOF/test share summaries by true and predicted class with counts, patient-cluster bootstrap CIs, the CWF weight per class, correct-versus-wrong patient-cluster bootstrap comparisons, and error-detection AUROC for share, image attribution, disagreement, and confidence. Fit confidence-only and all-feature logistic error detectors on OOF training explanations and evaluate them once on the patient-independent test set; do not cross-validate over precomputed explanations because upstream branch-model fits may include patients from another CV validation fold. This test evaluation is reporting-only and does not tune or select a model. Omit negative-contribution rates. Save the documented arrays and background indices to `modality_shapley_v2.npz`.

- [ ] **Step 7: Validate notebook structure and saved result schema**

  Validate notebook JSON, compile every code cell without executing training, and run any lightweight self-check cells. If the dataset/GPU environment is available, run the notebook through the v2 save cell and verify the saved arrays' lengths, finite values, classes, and exact v1 artifact hashes remain unchanged.

  Run: `python -c "import json, ast; p='03b_modality_shapley.ipynb'; n=json.load(open(p, encoding='utf-8')); [ast.parse(''.join(c.get('source', []))) for c in n['cells'] if c['cell_type']=='code']; print('notebook JSON and code cells valid')"`
  Expected: prints `notebook JSON and code cells valid`; full execution additionally prints and saves `modality_shapley_v2.npz`.
