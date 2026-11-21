# Phase F Faithfulness and Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved fixed exploratory Phase F evaluation of the frozen image-plus-clinical model, without modifying frozen artifacts or using test results for tuning.

**Architecture:** Keep GPU image evaluation and CPU/tabular TreeSHAP evaluation in separate runnable scripts because the available environments have different packages. Both scripts consume the same frozen inputs and a fixed JSON config, write resumable results only beneath a new `outputs\phase_f_v1` directory, and record versions and hashes. A final summarizer computes patient-cluster uncertainty and report-ready tables/figures.

**Tech Stack:** Python 3.11/PyTorch 2.7.1+cu128/torchvision 0.22.1 for image work; the available Python 3.13 environment with NumPy, pandas, SciPy, SHAP 0.52.0, joblib, and XGBoost 3.2.0 for tabular work; standard-library `unittest` for focused checks.

**Spec:** `docs\superpowers\specs\2026-11-18-controlled-experiments-design.md`

## Global Constraints

- Use the frozen 461-row test split once under the fixed exploratory protocol; do not tune, select, or debug methods on its results.
- Do not overwrite or modify frozen v1 models, splits, notebooks, saved results, or the image cache.
- Keep ViT-branch explanations, XGBoost-branch explanations, and CWF scores distinct.
- Keep targets fixed to the saved per-row target classes throughout interventions.
- Use the saved class-stratified 200-row training background and replace whole clinical variable groups.
- Use a 16x16 image patch, Gaussian-blur radius 32, a centre-Gaussian reference with sigma 3.5 map cells, 5% perturbation increments, and 20 fixed-seed random orders.
- Use horizontal flip and +/-10% brightness stability perturbations; analyze only unchanged-prediction cases and report their denominators.
- Use 2,000 patient-cluster bootstrap resamples and Holm correction within predeclared comparison families.
- Save every new result under `outputs\phase_f_v1`; fail if the directory already exists unless an explicit resume matches its config and input hashes.
- Record environment versions, commands, logs, seeds, and SHA-256 hashes before and after the run.
- Report all real-data test results as exploratory; they are not confirmatory evidence.

## Review Focus

- **Test-array misalignment:** a saved map or probability belongs to a different `img_id`; Task 1 tests exact ordered ID alignment across split, NPZs, and cached images.
- **Intervention target drift:** the predicted class changes after masking and silently changes the score being tracked; Task 2 tests that target indices remain fixed and CWF probabilities remain normalized.
- **Invalid clinical categories:** independently replaced one-hot columns create impossible categories; Task 4 tests that each group is copied from a single background row and that every encoded column belongs to exactly one variable.
- **Stability denominator bias:** transformed cases are selectively omitted without a count; Task 3 tests unchanged-prediction masks and requires explicit per-transform denominators.
- **Randomization leakage or nondeterminism:** a control touches test labels during fitting or changes across seeds; Task 4 tests train-only fitting and deterministic label permutation, and Task 3 tests reproducible randomized ViT weights.

---

## Scope and file map

This is the first sub-plan because Phase F is the highest-priority execution
stage. Phase G, Phase I, Phase H, and remaining project-wide Phase J work are
independent workstreams and will receive separate implementation plans before
their implementation.

| File | Responsibility |
|---|---|
| `experiments\phase_f\config.json` | Fixed seeds, baselines, perturbation schedule, target rules, bootstrap count, and output version. |
| `experiments\phase_f\io.py` | Frozen-artifact loading, ordered-ID assertions, config validation, SHA-256 input snapshots, and safe output-directory/resume checks. |
| `experiments\phase_f\image_eval.py` | ViT loading, image deletion/insertion, random/centre references, perturbation stability, and progressive ViT randomization. |
| `experiments\phase_f\tabular_eval.py` | Feature-group reconstruction, exact background interventions, XGBoost label-randomization control, and separate CWF scoring. |
| `experiments\phase_f\summarize.py` | Patient-cluster bootstrap, Holm-adjusted comparisons, CSV/JSON summaries, figures, and final input-hash verification. |
| `experiments\phase_f\tests\` | Pure/helper tests runnable without fitting the real models or reading the test split. |
| `outputs\phase_f_v1\` | New run config, logs, intermediate checkpoints, final arrays, figures, hashes, environment records, and summaries. Created only by the runner. |
| `ANALYSIS_PLAN.md` | Add a dated note recording the approved one-time exploratory test protocol and linking its run directory after execution. |
| `ARTIFACT_AUDIT.md` | Update Phase F status only after scripts/results are actually validated. |

### Interfaces

- `io.py`
  - `load_frozen_arrays(root: Path) -> dict[str, object]`: loads the required NPZ/JSON arrays and asserts the exact 461 ordered test IDs, class order, shapes, finiteness, and agreement across saved artifacts.
  - `sha256_file(path: Path) -> str`: streams a file and returns its lowercase SHA-256.
  - `prepare_run_dir(output_dir: Path, config_hash: str, input_hashes: dict[str, str], resume: bool) -> None`: refuses accidental overwrite; resume is allowed only when stored config and input hashes match.
- `image_eval.py`
  - `ranked_image_curves(...) -> dict[str, np.ndarray]`: computes fixed-target deletion/insertion curves for saved maps and baselines, including separate ViT and CWF scores.
  - `run_image_stability(...) -> dict[str, np.ndarray]`: stores map similarity and unchanged-prediction masks for flip and brightness transforms.
  - `run_vit_randomization(...) -> dict[str, np.ndarray]`: stores progressive head/block/full-encoder randomization comparisons.
- `tabular_eval.py`
  - `build_feature_groups(model_columns: list[str], variables: list[str]) -> list[np.ndarray]`: maps each encoded column to exactly one of the eight clinical variables.
  - `run_tabular_interventions(...) -> dict[str, np.ndarray]`: computes whole-group background interventions and fixed-target XGBoost/CWF curves.
  - `run_label_randomization_control(...) -> dict[str, np.ndarray]`: fits the fixed-seed label-permuted control on training rows only and returns its held-out predictions and grouped explanation comparison.
- `summarize.py`
  - `cluster_bootstrap(values: np.ndarray, patient_ids: np.ndarray, n_resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]`: returns percentile bounds by resampling patients, not images.
  - `holm_adjust(p_values: np.ndarray) -> np.ndarray`: returns monotone Holm-adjusted p-values without changing the displayed raw estimates.
  - `paired_cluster_signflip_p(values_a: np.ndarray, values_b: np.ndarray, patient_ids: np.ndarray, n_permutations: int, seed: int) -> float`: aggregates paired differences by patient and estimates a two-sided sign-flip p-value using 10,000 fixed-seed permutations.

## Task 1: Freeze and validate inputs; establish run metadata

**Files:**
- Create: `experiments\phase_f\config.json`
- Create: `experiments\phase_f\io.py`
- Create: `experiments\phase_f\tests\test_io.py`
- Read: `outputs\split_v1.json`, `outputs\base_outputs_v1.npz`, `outputs\fusion_v1.npz`, `outputs\image_explainers_v1.npz`, `outputs\treeshap_v1.npz`, `outputs\xgb_final_v1.joblib`, `outputs\vit_final_v1.pt`, `cache\images_uint8_224.pt`

**Interfaces:**
- Consumes: the frozen artifacts listed above and project paths from `common.py`.
- Produces: validated arrays, a run config, and a before-run input-hash snapshot. No model is fit and no model inference is performed in this task.

- [ ] **Step 1: Write failing tests for ordered IDs, required arrays, and overwrite protection**

  In `test_io.py`, use temporary synthetic NPZ/JSON fixtures to assert that:
  - mismatched test row keys raise `ValueError`;
  - missing arrays or wrong shapes raise `ValueError`;
  - an existing nonempty run directory is rejected;
  - resume succeeds only when both config and protected input hashes match.

- [ ] **Step 2: Run the focused tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_f\tests' -p 'test_io.py' -v`

  Expected: tests fail because the loader and output guard are not implemented.

- [ ] **Step 3: Implement config and artifact-loading helpers**

  Set seed `42`, background size `200`, blur radius `32`, patch size/stride `16`, perturbation fraction `0.05`, random-order count `20`, bootstrap count `2000`, fixed-target behavior, and output directory `outputs\phase_f_v1`. Implement the interface signatures above using only standard library and NumPy in `io.py`.

- [ ] **Step 4: Run focused tests and validate real input metadata**

  Run the `test_io.py` command above and require PASS. Then run `io.py --validate-only` with the GPU Python environment; it must report 461 ordered IDs, six classes, eight clinical groups, and matching frozen artifact hashes without loading model weights or computing predictions.

## Task 2: Image deletion/insertion and random-order controls

**Files:**
- Create: `experiments\phase_f\image_eval.py`
- Create: `experiments\phase_f\tests\test_image_curves.py`
- Consume: validated arrays and config from Task 1.

**Interfaces:**
- Consumes: uint8 test images, saved image maps, frozen ViT/CWF targets, saved XGBoost probabilities, class fusion weights, and the final ViT checkpoint.
- Produces: per-row curves and AUCs for all eight saved map arrays (with the identical attention-rollout arrays computed once and referenced under both saved target names), random maps, centre-Gaussian maps, and 20 fixed-seed random orders. Each map is scored in ViT and CWF probability spaces separately.

- [ ] **Step 1: Write failing tests for patch replacement, ranking, and fixed-target CWF scoring**

  In `test_image_curves.py`, assert that deleting all ranked patches yields the Gaussian-blurred image, inserting all patches yields the original image, the 5% schedule covers all 196 patches exactly once, and the CWF score uses the same class index before and after intervention and sums to one.

- [ ] **Step 2: Run the focused image helper tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_f\tests' -p 'test_image_curves.py' -v`

  Expected: tests fail because the image evaluation helpers are not implemented.

- [ ] **Step 3: Implement deterministic image curves**

  Implement `ranked_image_curves(...)`. Use the saved 14x14 maps to rank 16x16 image patches, substitute the matching Gaussian-blurred patch for deletion, and begin from the blurred image for insertion. Use 20 cumulative steps with patch counts `round(step * 196 / 20)`, ending at all 196 patches. Use the fixed saved target class; compute the ViT score and CWF score separately. Build random and centre-Gaussian references with seed `42`; generate 20 independent orders with seeds `42..61`. Use the existing saved image maps, not new explainer selection.

- [ ] **Step 4: Add resumable chunk execution and a timing-only preflight**

  Add a timing-only preflight that measures a fixed small batch but emits no test metric summaries. Save completed row chunks atomically beneath `outputs\phase_f_v1`; resume only after Task 1 confirms matching hashes/config. Include the exact tested row range and elapsed time in the log.

- [ ] **Step 5: Run helper tests and the fixed full image evaluation**

  Run `test_image_curves.py` and require PASS. Then run `image_eval.py` on all 461 test rows, without inspecting partial score values to change the protocol. Require all expected per-row curves to be finite and complete, and log any CUDA/runtime exception explicitly.

## Task 3: Image stability and ViT model-randomization tests

**Files:**
- Modify: `experiments\phase_f\image_eval.py`
- Create: `experiments\phase_f\tests\test_image_sanity.py`
- Consume: Task 1 validated inputs and Task 2 model/map helpers.

**Interfaces:**
- `run_image_stability(...) -> dict[str, np.ndarray]` returns original/transformed predictions, unchanged-prediction masks, per-map Spearman/SSIM values, and denominators for horizontal flip, brightness `0.9`, and brightness `1.1`.
- `run_vit_randomization(...) -> dict[str, np.ndarray]` returns map similarities and model-output changes after head-only, cumulative final-block-plus-head, and full-encoder-plus-head randomization.

- [ ] **Step 1: Write failing tests for deterministic transforms and progressive randomization**

  In `test_image_sanity.py`, assert that flip-and-map-back restores a synthetic symmetric map, brightness factors are applied in raw `[0,1]` space with clipping before normalization, unchanged-prediction masks retain exact denominators, and fixed-seed randomized model copies are identical across repeated runs but differ from the frozen weights.

- [ ] **Step 2: Run focused tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_f\tests' -p 'test_image_sanity.py' -v`

  Expected: tests fail because the perturbation and randomization helpers are not implemented.

- [ ] **Step 3: Implement fixed image transformations and similarity calculations**

  Compute horizontal flip, inverse-map the attribution map before comparison, and compute brightness transforms by multiplying raw images by `0.9` and `1.1` with clipping to `[0,1]`. Compare only cases with unchanged frozen-target predictions and save the mask/count for each method and transform. Use Spearman and SSIM as defined in the design; do not select a transform or threshold from the test results.

- [ ] **Step 4: Implement progressive ViT randomization**

  Deep-copy the frozen architecture, randomize the classifier head, then cumulatively randomize the final encoder block and finally all encoder blocks using fixed seeds. Recompute Grad-CAM, attention rollout, Chefer-style, and occlusion maps for the same fixed target classes; save similarity to the original maps plus the randomized model's probability/output changes. Do not fit on any data.

- [ ] **Step 5: Run focused tests and the fixed full sanity evaluations**

  Run `test_image_sanity.py` and require PASS. Then run both sanity evaluations on all 461 rows; require finite metrics and explicit unchanged-prediction counts. Keep all outputs in the existing matching `outputs\phase_f_v1` run directory.

## Task 4: Grouped clinical interventions and XGBoost randomization control

**Files:**
- Create: `experiments\phase_f\tabular_eval.py`
- Create: `experiments\phase_f\tests\test_tabular_interventions.py`
- Consume: validated arrays/config from Task 1 and saved training background indices.

**Interfaces:**
- Consumes: frozen split and base outputs, metadata, saved XGBoost model/columns, saved TreeSHAP arrays, frozen ViT probabilities, CWF weights, and training-only rows.
- Produces: grouped feature intervention curves for eight variables, XGBoost/CWF scores, the label-permuted control's held-out predictions and TreeSHAP comparison, and environment/input hashes.

- [ ] **Step 1: Write failing tests for one-hot grouping, whole-row replacement, and train-only label permutation**

  In `test_tabular_interventions.py`, assert that each of the 40 encoded columns maps to exactly one of `grew`, `itch`, `changed`, `elevation`, `bleed`, `hurt`, `region`, or `age`; every intervention copies all columns in a variable group from the same background row; and the control's permuted training labels are deterministic and do not include test labels.

- [ ] **Step 2: Run focused tabular tests and verify failure**

  Run: `& 'C:\Users\Ramya Oli\.copilot\session-state\9f3be920-1523-47bb-bc4a-f28bd35ffddc\files\py313-shap52\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_f\tests' -p 'test_tabular_interventions.py' -v`

  Expected: tests fail because feature grouping and intervention helpers are not implemented.

- [ ] **Step 3: Implement exact grouped feature interventions**

  Reconstruct `X_train`/`X_test` using the frozen split and the 04 notebook's dummy-column order. Use the saved 200 training-background indices. Evaluate each of the 256 group coalitions against all 200 background rows in bounded batches; for each test row, groups present in the coalition come from that row and absent groups come together from one background row. Average predicted probabilities over background rows. Use the saved TreeSHAP attribution for the fixed CWF target to order groups. Derive deletion/insertion curves and 20-seed random-order curves from the 256 coalition values, reporting XGBoost and recomputed CWF scores separately.

- [ ] **Step 4: Implement the label-permuted XGBoost control**

  Create the control from the frozen training matrix using the 01 notebook recipe `XGBClassifier(objective='multi:softprob', num_class=6, eval_metric='mlogloss', random_state=42)` and `np.random.default_rng(42).permutation(y_train)`. Fit only the permuted training labels. Save the control model to the new run directory, generate held-out XGBoost and recomputed CWF predictions once, and use the same 256-coalition, 200-row interventional background calculation to compare grouped probability-Shapley rankings against the frozen model. Record XGBoost/SHAP versions and clearly label the control as trained in the available XGBoost 3.2.0 environment.

- [ ] **Step 5: Run focused tests and execute the tabular evaluation**

  Run `test_tabular_interventions.py` and require PASS. Run `tabular_eval.py` with the Python 3.13 SHAP environment; require exact row/column alignment, finite probability curves, all eight groups present, deterministic random orders, and no training/test ID overlap in fitted data.

## Task 5: Statistical summaries, report artifacts, and frozen-input verification

**Files:**
- Create: `experiments\phase_f\summarize.py`
- Create: `experiments\phase_f\tests\test_summary_stats.py`
- Modify after validated execution: `ANALYSIS_PLAN.md`
- Modify after validated execution: `ARTIFACT_AUDIT.md`
- Output: `outputs\phase_f_v1\`

**Interfaces:**
- Consumes: completed image and tabular result files, patient IDs aligned to frozen test rows, run config, and the before-run hash snapshot.
- Produces: pooled/per-class CSVs, 2,000 patient-cluster bootstrap intervals, Holm-adjusted p-values, figures, `summary.json`, `environment.json`, command logs, final input hashes, and a completion/status record.

- [ ] **Step 1: Write failing tests for patient-cluster bootstrap and Holm correction**

  In `test_summary_stats.py`, use tiny data with repeated rows per patient to assert that bootstrap resampling samples patient clusters, not rows, returns deterministic bounds for a fixed seed, sign-flip p-values use patient-aggregated paired differences and are deterministic, and Holm-adjusted p-values are monotone and no smaller than raw p-values.

- [ ] **Step 2: Run focused tests and verify failure**

  Run: `& 'C:\Users\Ramya Oli\.copilot\session-state\9f3be920-1523-47bb-bc4a-f28bd35ffddc\files\py313-shap52\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_f\tests' -p 'test_summary_stats.py' -v`

  Expected: tests fail because the reporting helpers are not implemented.

- [ ] **Step 3: Implement report summaries and outputs**

  Implement the declared helper signatures. For paired method-versus-control comparisons, average row-level AUC differences within patient, compute two-sided sign-flip p-values from 10,000 fixed-seed permutations, and apply Holm correction within separate predeclared image deletion/insertion and tabular deletion/insertion target families. Report each method, target, class count, denominator, AUC, and patient-cluster bootstrap interval without selecting a winner. Write JSON/CSV/PNG outputs and a concise report stating the test-exposure limitation and branch-versus-fusion distinction.

- [ ] **Step 4: Validate all outputs and frozen inputs**

  Run all focused tests in both environments. Validate row counts, class counts, finite values, required report files, and version/config/log presence. Recompute SHA-256 for all protected frozen inputs and require equality with the before-run snapshot. The final output directory must contain no modifications to frozen files.

- [ ] **Step 5: Update phase status from evidence**

  Update `ANALYSIS_PLAN.md` to record the approved exploratory protocol and exact run path. Update the Phase F row in `ARTIFACT_AUDIT.md` to distinguish implemented, executed, validated, failed, and pending components. Do not mark Phase F complete if any required experiment or validation failed or is absent.

## Execution order and handoff

Run Tasks 1–5 in order. The image and tabular tasks may execute in separate
environments but must use the same frozen input snapshot, config, target rule,
and output version. If an experiment fails, preserve its logs and partial
outputs; fix only the failure, rerun only the affected versioned component,
and never replace a frozen v1 artifact.

After this Phase F plan is approved and executed, prepare separate plans for
Phase G, Phase I, Phase H, and remaining Phase J work before implementing those
independent workstreams.
