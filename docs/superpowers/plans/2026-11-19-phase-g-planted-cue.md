# Phase G Planted-Cue Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and run only the approved balanced 0.85/0.85 synthetic planted-cue pilot with reproducible data, controlled models, explanations, statistics, and protected outputs.

**Architecture:** Keep the experiment in `experiments\phase_g\` with separate generator, model/training, explanation, evaluation, and run/IO modules. Train three independently initialized models on the same synthetic patients, calibrate only on a held-out validation-calibration subset, then compare calibrated NLL on identical untouched test patients. Store results in a new synthetic-only versioned output directory.

**Tech Stack:** Python 3.11 GPU environment; PyTorch/torchvision, NumPy, Pillow, SciPy, scikit-image, and standard-library `unittest`. Reuse installed packages where compatible; add a Phase G requirements file only if preflight confirms a needed dependency is absent.

**Spec:** `docs\superpowers\specs\2026-11-19-phase-g-planted-cue-design.md` and `docs\superpowers\specs\2026-11-19-phase-g-pilot-config-proposal.yaml`

## Global Constraints

- Generate 12,000 synthetic patients, equally divided across six synthetic labels: 8,400 train, 1,800 development, and 1,800 test.
- Split patient IDs before rendering; keep every derived input and intervention in its parent patient's partition.
- In the sole initially authorized condition, each cue has reliability 0.85, with incorrect values sampled uniformly among the other five labels and cue errors independent conditional on label.
- Train scratch-initialized image-only, clinical-only, and joint models on identical training patient IDs; use no PAD inputs, pretrained checkpoints, or auxiliary modality losses.
- Implement generator semantics for all six predeclared conditions, but enable execution only for the balanced pilot; tests of non-pilot semantics use deterministic unit fixtures, not additional runs.
- Split development equally into 900 model-selection patients and 900 calibration patients, stratified 150 per class in each subset; no test labels for checkpoint selection or temperature scaling.
- Primary effects are `NLL(image_only)-NLL(joint)` and `NLL(clinical_only)-NLL(joint)` on the same unmodified test patients; each Holm-adjusted one-sided 95% lower bound must exceed 0.10 nats per patient.
- Joint-model accuracy must be at least 0.80. Original 0.05 cue-mask accuracy drops are diagnostics only; masking NLL/probability changes are intervention sensitivity, not natural-distribution incremental NLL.
- Use 10,000 paired, class-stratified bootstrap replicates for primary NLL bounds; refit only validation temperature scalars within each replicate, not model weights.
- Use 200 class-stratified synthetic training patients as the donor background for each exact two-player Shapley game; verify efficiency within `1e-5`.
- Keep cue reliance, explanation localization, explanation faithfulness, stability, and Shapley attribution as distinct outcomes and label their targets.
- Execute only the balanced condition. Other condition definitions remain unauthorized.
- Do not use any real-data test split, tune on synthetic test labels, overwrite frozen Phase F/v1 artifacts, or reuse an existing output directory.
- Hash protected real inputs before and after the run. Save the frozen config, code/environment versions, seeds, logs, model checkpoints, predictions, metrics, and output hashes under `outputs\synthetic_phase_g_v1\pilot_balanced_085_085_v1`.

## Review Focus

- **Patient leakage or split-dependent rendering:** test disjoint patient IDs, class counts, deterministic sample-by-ID rendering, and parent-partition inheritance in Task 1.
- **Cue/mask construction errors:** test exact marker/lesion masks, minimum 16-pixel separation, class-independent lesion parameters, and cue-noise semantics in Task 1.
- **Test-label leakage through model selection or calibration:** test disjoint selection/calibration/test IDs, validation-only temperature fitting, and identical test IDs across models in Task 2.
- **NLL/threshold or multiplicity mistakes:** test analytic oracle values, paired contrast signs, centered bootstrap-tail calculations, Holm step-down/inversion, and threshold comparisons in Task 2.
- **Attribution target or construct confusion:** test fixed target classes and separately typed localization, faithfulness, and intervention-sensitivity results in Task 3.

---

## Scope and File Map

| File | Responsibility |
|---|---|
| `experiments\phase_g\config.json` | Executable frozen copy of the approved v2 proposal; contains only the balanced run authorization. |
| `experiments\phase_g\generate.py` | Deterministic labels, patient splits, render streams, images, cues, exact masks, and generation metadata. |
| `experiments\phase_g\model.py` | Image-only, clinical-only, and joint architectures; train/validation loops; temperature scaling. |
| `experiments\phase_g\explain.py` | Image-only and joint image maps, clinical categorical attribution, and the two exact joint Shapley games. |
| `experiments\phase_g\evaluate.py` | Natural-distribution NLL contrasts, paired bootstrap/Holm, cue-mask diagnostics, localization, faithfulness, and stability metrics. |
| `experiments\phase_g\io.py` | Config/schema validation, safe run directory, atomic saves, hashes, environment metadata, and protected-input checks. |
| `experiments\phase_g\run.py` | Ordered CLI orchestration, preflight, train/validation, one held-out evaluation, and final manifest. |
| `experiments\phase_g\tests\` | Synthetic-only unit tests using standard-library `unittest`; no tests fit on real or held-out pilot data. |
| `outputs\synthetic_phase_g_v1\pilot_balanced_085_085_v1\` | Created by the runner only after preflight; contains all pilot artifacts and results. |
| `ANALYSIS_PLAN.md` | Add a dated result/status note only after execution and validation. |
| `ARTIFACT_AUDIT.md` | Update Phase G status only after artifacts have been verified. |

### Interfaces

- `SyntheticPatient` is a frozen record with `patient_id: str`, `label: int`,
  `partition: str`, `image: np.ndarray` (uint8 `[224,224,3]`),
  `image_cue: int`,
  `clinical_value: int`, `background_canvas: np.ndarray`
  (uint8 `[224,224,3]`), `lesion_mask: np.ndarray` (bool `[224,224]`),
  `marker_mask: np.ndarray` (bool `[224,224]`), and
  `render_metadata: dict[str, object]`.
- `generate.py`
  - `make_patient_splits(config: dict[str, object]) -> tuple[dict[str, list[str]], dict[str, int]]`: returns disjoint class-stratified patient-ID lists and a separate opaque-ID-to-label mapping for `train`, `validation_model_selection`, `validation_calibration`, and `test`.
  - `render_patient(patient_id: str, label: int, partition: str, config: dict[str, object]) -> SyntheticPatient`: returns one image, clinical value, background canvas, exact masks, and stream metadata.
  - `generate_split(patient_ids: list[str], labels: dict[str, int], partition: str, config: dict[str, object]) -> list[SyntheticPatient]`: renders a requested split without consulting other partitions.
  - `make_conflicting_assignments(patient_ids: list[str], rng: numpy.random.Generator) -> dict[str, str | None]`: assigns exactly half of an even-sized ID set to conflict, with the two correct-cue orientations equally represented; other IDs map to `None`.
  - `generate_cues(label: int, condition: str, rngs: dict[str, numpy.random.Generator], conflict_orientation: str | None = None) -> tuple[int, int]`: returns image and clinical cue codes for any configured condition; conflicting samples require `"image"` or `"clinical"` orientation, and run-time config validation still authorizes only `balanced`.
- `model.py`
  - `TrainingResult` contains `model: torch.nn.Module`, `selected_epoch: int`, `history: list[dict[str, float]]`, and `checkpoint_path: Path`.
  - `build_model(kind: str, config: dict[str, object]) -> torch.nn.Module`: accepts `image_only`, `clinical_only`, or `joint`.
  - `train_model(kind: str, train_data: list[SyntheticPatient], selection_data: list[SyntheticPatient], config: dict[str, object]) -> TrainingResult`: trains only on train IDs and selects by the model-selection subset.
  - `fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float`: fits one positive scalar on calibration labels only.
  - `predict_probabilities(model: torch.nn.Module, data: list[SyntheticPatient], temperature: float) -> np.ndarray`: returns normalized six-class calibrated probabilities.
- `evaluate.py`
  - `paired_nll_gains(probabilities: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, np.ndarray]`: returns paired per-patient `image_only - joint` and `clinical_only - joint` NLL gains.
  - `primary_nll_inference(probabilities: dict[str, np.ndarray], test_logits: dict[str, np.ndarray], test_labels: np.ndarray, calibration_logits: dict[str, np.ndarray], calibration_labels: np.ndarray, test_classes: np.ndarray, replicates: int, seed: int, threshold: float) -> dict[str, object]`: performs nested paired class-stratified calibration/test bootstrap and Holm-inverted lower bounds. It uses `probabilities` for the frozen-calibration point estimates and applies each replicate's refitted temperature to `test_logits`.
  - `cue_mask_sensitivity(joint_model: torch.nn.Module, patients: list[SyntheticPatient], config: dict[str, object]) -> dict[str, np.ndarray]`: averages 20 replacement draws per patient; results are explicitly marked diagnostic.
  - `build_localization_controls(masks: dict[str, np.ndarray], seed: int, repeats: int = 20) -> dict[str, np.ndarray]`: returns fractional patch-grid masks for area-matched marker/lesion regions, random maps, and a center-Gaussian control.
  - `localization_metrics(maps: np.ndarray, masks: dict[str, np.ndarray], controls: dict[str, np.ndarray], patient_ids: np.ndarray) -> dict[str, np.ndarray]`: returns marker/lesion/background mass, area enrichment, pointing-game, and region-control results.
  - `deletion_insertion_metrics(model: torch.nn.Module, model_kind: str, images: torch.Tensor, categories: np.ndarray | None, background_canvases: torch.Tensor, maps: dict[str, np.ndarray], targets: np.ndarray, config: dict[str, object]) -> dict[str, np.ndarray]`: returns calibrated target-specific fixed-score curves/AUCs and random/center controls for image-only or joint-fusion scores.
  - `stability_metrics(model: torch.nn.Module, model_kind: str, images: torch.Tensor, categories: np.ndarray | None, maps: dict[str, np.ndarray], targets: np.ndarray, config: dict[str, object]) -> dict[str, np.ndarray]`: returns calibrated transformed predictions, map similarities, and unchanged-prediction denominators.
  - `bootstrap_mean_interval(values: np.ndarray, replicates: int = 2000, seed: int = 20261119) -> tuple[float, float]`: returns a two-sided patient-level percentile interval.
  - `paired_signflip_p(values_a: np.ndarray, values_b: np.ndarray, draws: int = 10000, seed: int = 20261119) -> float`: returns a two-sided patient-level sign-flip p-value with add-one correction.
- `io.py`
  - `validate_config(config: dict[str, object]) -> None`: rejects unauthorized conditions, changed thresholds, or incompatible split counts.
  - `prepare_run_dir(output_dir: Path, config_hash: str, source_hashes: dict[str, str], resume: bool) -> None`: refuses overwrite; resume requires an exact config/source identity match.
  - `hash_protected_inputs(root: Path) -> dict[str, str]`: hashes the protected paths listed in `experiments\phase_f\io.py` without loading their contents.
- `run.py`
  - CLI `--validate-config` validates the frozen balanced config and output-path safety without rendering data, reading test labels, or writing results.
  - CLI `--preflight-only` measures a small synthetic GPU batch and validates storage without evaluating held-out metrics.
  - CLI `--run` executes the frozen balanced config; no test labels are loaded until all checkpoints and calibrators are frozen.
- `explain.py`
  - `compute_image_maps(model: torch.nn.Module, model_kind: str, images: torch.Tensor, targets: np.ndarray, categories: np.ndarray | None, methods: tuple[str, ...], batch_size: int = 16, temperature: float = 1.0) -> dict[str, np.ndarray]`: returns 14x14 maps for only the specified model and fixed targets; joint maps hold clinical categories fixed.
  - `compute_clinical_attributions(model: torch.nn.Module, clinical_values: np.ndarray, targets: np.ndarray, reference: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]`: computes the exact categorical attribution against the uniform six-category reference.
  - `compute_joint_shapley(model: torch.nn.Module, patients: list[SyntheticPatient], targets: np.ndarray, donors: list[SyntheticPatient], game: str, temperature: float = 1.0) -> dict[str, np.ndarray]`: accepts `game="modality"` or `"cue"` and returns exact two-player values checked for efficiency.

## Task 1: Config, deterministic generator, and protected IO

**Files:**
- Create: `experiments\phase_g\config.json`
- Create: `experiments\phase_g\generate.py`
- Create: `experiments\phase_g\io.py`
- Create: `experiments\phase_g\tests\test_generation.py`
- Create: `experiments\phase_g\tests\test_io.py`

**Interfaces:**
- Consumes: the v2 design/config proposal and only synthetic labels/streams.
- Produces: `SyntheticPatient` records, four disjoint patient-ID partitions, an independent patient-ID-to-label mapping, exact masks, and guarded output/hash helpers for later tasks.

- [ ] **Step 1: Write failing tests for split counts, deterministic generation, and renderer invariants**

  Assert 1,400 train, 150 model-selection, 150 calibration, and 300 test IDs per class; all four ID sets are disjoint; rendering the same patient ID twice is byte-identical; cue values are in `[0,5]`; masks are binary and in bounds; the marker is at an allowed corner; marker/lesion separation is at least 16 pixels; and no lesion/background rendering parameter depends on class.

  Also test each condition's specified generator semantics with deterministic RNG fixtures: balanced `.85/.85` independent errors; image-dominant `.95/.60`; clinical-dominant `.60/.95`; redundant shared `.85` noisy code; conflicting 50% one-correct/one-decoy cases with balanced orientation and `.85` independent noise in the remainder; and synergistic `U`, `(Y-U) mod 6` shares with independent `.15` corruption. These tests must not launch training or produce pilot results for those conditions.

- [ ] **Step 2: Run the focused tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_generation.py' -v`

  Expected: FAIL because the generator and typed patient record do not exist.

- [ ] **Step 3: Write failing tests for non-overwrite and protected-input hash stability**

  Assert a pre-existing nonempty run directory is rejected, resume requires matching config/source hashes, and hash collection reads only the protected file bytes without loading model/data objects.

- [ ] **Step 4: Run the focused IO tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_io.py' -v`

  Expected: FAIL because Phase G run guards are not implemented.

- [ ] **Step 5: Implement deterministic configuration, split, rendering, and IO interfaces**

  Use the proposal's 12,000 balanced labels and fixed streams; generate 224x224 RGB images, background canvases, irregular lesion masks, the six equal-area/equal-luminance glyphs, cue values, and 20 reproducible replacement values per test patient. Implement all six condition-generation semantics, but make config validation reject execution unless the condition is `balanced`. Keep balanced cue-noise streams conditionally independent and assign patients before rendering.

- [ ] **Step 6: Run focused tests and a train-only generation audit**

  Run both `unittest` commands above. Require PASS, exact partition counts, deterministic hashes for repeated generation, valid masks, and empirical training-only cue reliability within five binomial standard deviations of 0.85, separately for image and clinical cues (`5 * sqrt(8400 * 0.85 * 0.15)` correct-count deviation). Do not inspect test outcomes.

- [ ] **Step 7: Commit the generator deliverable**

  ```text
  feat: add deterministic Phase G synthetic generator

  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

## Task 2: Controlled models, validation-only calibration, and primary NLL criterion

**Files:**
- Create: `experiments\phase_g\model.py`
- Create: `experiments\phase_g\evaluate.py`
- Create: `experiments\phase_g\tests\test_model.py`
- Create: `experiments\phase_g\tests\test_primary_statistics.py`

**Interfaces:**
- Consumes: the four generated partitions from Task 1.
- Produces: three independently trained model checkpoints, three frozen temperatures, calibrated test probabilities, paired per-patient NLL gains, and the primary gate result.

- [ ] **Step 1: Write failing model tests**

  Assert each model returns finite logits of shape `(batch, 6)`; the image-only and joint models use the same ViT dimensions; clinical input is categorical rather than scalar; and no auxiliary modality loss is present in joint training.

- [ ] **Step 2: Run model tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_model.py' -v`

  Expected: FAIL because Phase G model constructors do not exist.

- [ ] **Step 3: Write failing calibration and analytic-oracle tests**

  Assert temperature fitting accepts only calibration logits/labels, returns a finite positive scalar, and never requires test labels. Add analytic tests for Bayes accuracy `0.85`, exact joint NLL `0.302653`, one-cue NLL `0.664125`, and incremental ideal NLL `0.361472` nats.

- [ ] **Step 4: Run primary-statistics tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_primary_statistics.py' -v`

  Expected: FAIL because calibration, paired contrasts, and inference helpers are not implemented.

- [ ] **Step 5: Implement model training and validation/calibration separation**

  Train image-only, clinical-only, and joint models on the identical 8,400 train IDs. The image encoder is ViT input 224/patch 16/four blocks/hidden 192/three heads/MLP 384; the clinical model uses a categorical embedding of 16, hidden layer 32 with GELU, and six-class output; the joint head concatenates image and 16-wide clinical embeddings into hidden width 128 with GELU/dropout 0.1. Use six-class cross-entropy, AdamW (`lr=3e-4`, `weight_decay=1e-4`), batch size 64, at most 50 epochs, and patience 8 for all. Use the same 900 model-selection IDs; do not use the 900 calibration IDs for checkpoint selection. Fit a separate temperature per frozen model on the shared 900 calibration IDs only. Save training histories, selected epochs, temperatures, and partition hashes.

- [ ] **Step 6: Implement paired NLL contrasts and Holm-inverted bootstrap bounds**

  Compute per-patient `NLL(image_only)-NLL(joint)` and `NLL(clinical_only)-NLL(joint)` on identical, unmodified test IDs. Retain each model's raw calibration and test logits. For each of 10,000 replicates, resample 150 calibration patients per class and 300 test patients per class, with common sampled IDs across all three models and both contrasts; refit only the three temperatures, not model weights, then apply those replicate temperatures to the sampled test logits. For each candidate threshold `c`, calculate `p_j(c) = (1 + count(Delta_j* - Delta_hat_j >= Delta_hat_j - c)) / 10001`. Apply the specified two-test Holm adjustment (`p_adj,(1)=min(1,2*p_(1))`; `p_adj,(2)=min(1,max(2*p_(1),p_(2)))`) and invert these adjusted tests over `c` to obtain simultaneous one-sided lower bounds. Pass only if both bounds exceed `0.10` nats and joint accuracy is at least `0.80`.

- [ ] **Step 7: Run focused tests and synthetic-only validation**

  Run `test_model.py` and `test_primary_statistics.py`; require PASS. Validate NLL contrast sign conventions, centered-tail arithmetic, Holm monotonicity and inversion at boundary cases, bootstrap pairing, exact class resample counts, temperature positivity, and test-independent calibration. Compare analytic Bayes values to the oracle helpers within numerical tolerance. Save calibrated and uncalibrated NLLs and raw/adjusted p-values at `c=0.10`.

- [ ] **Step 8: Commit the model/statistics deliverable**

  ```text
  feat: add calibrated Phase G models and NLL gate

  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

## Task 3: Cue intervention sensitivity and explanation outcomes

**Files:**
- Create: `experiments\phase_g\explain.py`
- Modify: `experiments\phase_g\evaluate.py`
- Create: `experiments\phase_g\tests\test_explain.py`
- Create: `experiments\phase_g\tests\test_evaluation.py`

**Interfaces:**
- Consumes: frozen checkpoints/temperatures from Task 2 and exact masks/canvases from Task 1.
- Produces: image-only and joint-target maps, clinical and Shapley attributions, cue-mask diagnostics, localization results, faithfulness curves, and stability summaries as separate outputs.

- [x] **Step 1: Write failing tests for explainer targets and fixed-class handling**

  Use a small deterministic multimodal test model to assert output map shape `(N,14,14)`, separate image-only and joint-fusion score targets, fixed true-class indices through interventions, four-coalition Shapley efficiency within `1e-5`, and correct clinical categorical reference behavior.

- [x] **Step 2: Run explainer tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_explain.py' -v`

  Expected: FAIL because Phase G explainer adapters are not implemented.

- [x] **Step 3: Write failing tests that distinguish localization from faithfulness**

  Assert localization reports marker/lesion/background mass and area-normalized enrichment without consuming perturbation scores; faithfulness reports fixed-score deletion/insertion curves against controls without being labeled localization; and cue-mask sensitivity is stored in a separate result group.

- [x] **Step 4: Run evaluation tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_evaluation.py' -v`

  Expected: FAIL because Phase G metric functions are not implemented.

- [x] **Step 5: Implement explainers and intervention/localization metrics**

  Implement Grad-CAM, Chefer, class-independent attention rollout, and occlusion separately for image-only and joint models; exact categorical clinical attribution; whole-modality and cue-level two-player Shapley using the same 200 class-stratified training donors (33 or 34 per class); 20-draw marginal cue replacements; fractional mask-to-14x14 projection; 20 area-matched localization controls per patient, 20-seed random-map and center-Gaussian controls; and the planned stability transforms. Include literal cue-absence sensitivity as a separately labeled out-of-support diagnostic, never as a gate.

- [x] **Step 6: Implement faithfulness curves with prespecified controls**

  Evaluate deletion/insertion at 5% patch steps using the saved background-only canvas and Gaussian blur sensitivity baseline. Compare target-specific Grad-CAM/Chefer/occlusion rankings with 20 random orders and center-Gaussian rankings. Keep image-only score and joint fusion score results separate; describe attention rollout as a class-independent control only.

- [x] **Step 7: Run focused tests and require PASS**

  Run `test_explain.py` and `test_evaluation.py`; require PASS for target invariance, mask projection, exact Shapley efficiency, intervention-only output changes, faithfulness endpoints, control reproducibility, and distinct output schemas. Validate 2,000 patient-level bootstrap replicates for descriptive intervals and 10,000 patient-level sign flips with add-one correction for prespecified paired continuous contrasts. Apply Holm within: the two cue-mask comparisons; separate four-method marker and lesion localization families for each of image-only and joint targets; and separate six-contrast families for image-only deletion, image-only insertion, joint-fusion deletion, and joint-fusion insertion (three target-specific methods against random-order and center-Gaussian controls). Stability is descriptive with intervals and no hypothesis tests. Do not treat replacement draws or patches as independent patients. **Verified:** all 35 Phase G tests pass, including the default 2,000 bootstrap and 10,000 sign-flip implementations; unit-test fixtures are isolated synthetic training-partition patients, not the held-out pilot.

- [ ] **Step 8: Commit the explanation/evaluation deliverable**

  ```text
  feat: add Phase G attribution and evaluation metrics

  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

## Task 4: Safe orchestration, GPU preflight, and reproducibility manifest

**Files:**
- Create: `experiments\phase_g\run.py`
- Modify: `experiments\phase_g\io.py`
- Create: `experiments\phase_g\tests\test_run.py`

**Interfaces:**
- Consumes: validated config, generator, models, explainers, and evaluation interfaces from Tasks 1-3.
- Produces: ordered CLI stages, run identity, resumable synthetic-only artifacts, environment/hardware records, logs, and before/after hashes for protected inputs.

- [ ] **Step 1: Write failing tests for stage ordering and output safety**

  Assert `--preflight-only` never reads test labels or writes result metrics; `--run` refuses an existing output directory; resume requires identical config/source hashes; and test evaluation cannot start before all checkpoints and calibrators are frozen.

- [ ] **Step 2: Run run-guard tests and verify failure**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -p 'test_run.py' -v`

  Expected: FAIL because the Phase G runner is not implemented.

- [ ] **Step 3: Implement environment capture, hash guards, atomic saves, and CLI ordering**

  Record Python, PyTorch, torchvision, NumPy, SciPy, Pillow, scikit-image, CUDA, device, and driver versions. Hash protected Phase F/v1 paths before the run. Make chunk/checkpoint writes atomic and record completed stage identities. Ensure the final held-out evaluation is a single fixed pass with no result-driven changes.

- [ ] **Step 4: Run preflight on the selected GPU interpreter**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' 'D:\skin_project\experiments\phase_g\run.py' --preflight-only`

  Expected: reports the selected GPU/CUDA and a small synthetic timing/memory check, confirms the new output path is unused, and emits no held-out metrics or test predictions.

- [ ] **Step 5: Run all Phase G unit tests**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -v`

  Expected: all tests PASS before the full pilot is launched.

- [ ] **Step 6: Commit the runner deliverable**

  ```text
  feat: add guarded Phase G pilot runner

  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

## Task 5: Execute only the balanced pilot and validate/report artifacts

**Files:**
- Consume: all `experiments\phase_g\` code and frozen config.
- Create: `outputs\synthetic_phase_g_v1\pilot_balanced_085_085_v1\**`
- Modify after verification: `ANALYSIS_PLAN.md`
- Modify after verification: `ARTIFACT_AUDIT.md`

**Interfaces:**
- Consumes: verified test suite, exact config/source hashes, and protected-input hash snapshot.
- Produces: one balanced pilot run, one primary gate report, separated diagnostic/explanation outputs, and a validated manifest.

- [ ] **Step 1: Run the balanced-only config validation**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' 'D:\skin_project\experiments\phase_g\run.py' --validate-config`

  Expected: confirms only `balanced` is authorized, exactly 0.85/0.85 cue reliability, 12,000 patient counts, NLL threshold 0.10, Holm family size 2, and non-overwrite output behavior.

- [ ] **Step 2: Execute the one approved synthetic pilot**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' 'D:\skin_project\experiments\phase_g\run.py' --run`

  Expected: train and validate only on synthetic train/development IDs; fit temperatures only on the calibration subset; then evaluate the same untouched 1,800 synthetic test patients once. Save all intermediate/final artifacts, logs, and hashes. Do not inspect partial test metrics to alter code, thresholds, or methods.

- [ ] **Step 3: Validate output completeness and protected artifacts**

  Require all 1,800 test IDs exactly once in model predictions and metric rows; finite normalized probability arrays; all model/temperature/calibration metadata; both NLL gains, raw and adjusted p-values, and Holm-inverted lower bounds; cue-mask diagnostics; explanation arrays, figures, outputs, and denominators; run configuration, versions, seeds, logs, output hashes; and unchanged hashes for every protected Phase F/v1 input.

- [ ] **Step 4: Re-run the full focused Phase G test suite**

  Run: `& 'D:\vit_gpu_env\Scripts\python.exe' -m unittest discover -s 'D:\skin_project\experiments\phase_g\tests' -v`

  Expected: all tests PASS after execution; report failures explicitly and do not mark the pilot validated if any required check fails.

- [ ] **Step 5: Write the factual result/status note**

  Update `ANALYSIS_PLAN.md` and `ARTIFACT_AUDIT.md` with the exact run ID, GPU/environment, artifact paths/hashes, accuracy floor result, two NLL effects and adjusted lower bounds, mask diagnostic outcomes, explanation findings, limitations, and whether expansion criteria passed. Explicitly state that a single pilot is not completion of Phase G.

- [ ] **Step 6: Commit only validated Phase G source/report changes**

  ```text
  feat: run and report balanced Phase G pilot

  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

## Handoff Condition

This plan does not authorize implementation until reviewed. After approval, choose one execution method: subagent-driven task implementation with review gates, or native execution in this session. The pilot itself remains limited to the balanced condition; no other condition may run without separate approval.
