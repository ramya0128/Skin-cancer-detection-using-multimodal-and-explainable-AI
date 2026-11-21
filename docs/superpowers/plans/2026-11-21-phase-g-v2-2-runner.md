# Phase G v2.2 Condition Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, resumable v2.2 runner for the five remaining Phase G conditions, with test access and execution disabled until explicit protocol and condition authorization is provided.

**Architecture:** Add condition-aware data generation and an isolated runner under `experiments/phase_g_v2_2/`, reusing the existing model and analysis functions without editing historical Phase G sources. Separate configuration validation, authorization, GPU/training, and gated evaluation stages; keep all outputs condition-local and hash-verified.

**Tech Stack:** Python, NumPy, PyTorch, existing Phase G generator/model/evaluation/explanation modules, unittest.

**Spec:** [2026-11-21-phase-g-v2-2-runner-design.md](../specs/2026-11-21-phase-g-v2-2-runner-design.md)

## Global Constraints

- Support exactly `image_dominant`, `clinical_dominant`, `redundant`, `conflicting`, and `synergistic`; do not rerun `balanced`.
- Preserve 12,000 patients with 1,400 train, 150 model-selection, 150 calibration, and 300 test patients per class; split before rendering.
- Preserve the configured architectures, seeds, AdamW settings, batch size 64, 50-epoch maximum, and patience 8.
- Require `cuda:0` for training and GPU-compatible inference; never silently fall back to CPU or use mixed precision.
- Keep checked-in execution disabled, the condition allowlist empty, and test access forbidden.
- Do not modify real-data models or historical Phase G v1/v2/v2.1 sources or outputs.
- Do not execute a full condition or train full models; only run the authorized, non-evidentiary batch-64 timing preflight.

## Review Focus

1. Conflicting assignments must cover the full patient-ID set at exact 25%/25%/50% and be passed into actual rendering — test rendered patient records and proportions in Task 1.
2. Synergistic corruption must retain the specified share with probability 0.85 and replace it with one of the other five values — test cue generation and joint distribution in Task 1.
3. Authorization must bind protocol, cohort, decision policy, and exact condition allowlist without allowing test access by default — test missing/mismatched authorization and denied test-stage access in Task 2.
4. An interrupted or mismatched run must not overwrite checkpoints or reuse stale stages — test stage hashes, resume identity, and output isolation in Task 3.
5. CUDA failure or batch-64 OOM must stop visibly without CPU fallback or batch-size substitution — test device guards in Task 3 and record actual preflight outcome in Task 5.

---

### Task 1: Condition Configuration and Dataset Generation

**Files:**
- Modify: `experiments/phase_g_v2_2/config.json`
- Create: `experiments/phase_g_v2_2/data.py`
- Test: `experiments/phase_g_v2_2/tests/test_data.py`

**Interfaces:**
- Consumes: `experiments/phase_g/generate.py` split and patient types; `experiments/phase_g_v2_2/generator.py` and `render.py` conflict helpers.
- Produces:
  - `condition_config(config: dict, condition: str) -> dict`
  - `make_condition_assignments(patient_ids: list[str], condition: str, seed: int) -> dict[str, str]`
  - `generate_condition_split(patient_ids: list[str], labels: dict[str, int], partition: str, config: dict, condition: str, assignments: dict[str, str] | None = None) -> list[SyntheticPatient]`
  - `audit_condition_cues(patients: Sequence[SyntheticPatient], condition: str) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests** for the five condition configurations and their cue semantics: independent image-dominant (0.95/0.60), clinical-dominant (0.60/0.95), identical redundant cues at shared 0.85 reliability, conflicting 25% image-correct/clinical-decoy + 25% reverse + 50% independent 0.85/0.85, and synergistic U/V share retention at 0.85 with corruption among other five shares. Include rendered conflicting records, exact assignment proportions, deterministic split IDs, and partition disjointness.
- [ ] **Step 2: Run the new tests and confirm expected failures** with `python -m unittest experiments.phase_g_v2_2.tests.test_data -v`.
- [ ] **Step 3: Implement condition configuration, deterministic assignments, and condition-specific split rendering** in `data.py`. Use the conflict renderer for conflicting patients and the existing generator cue behavior for the other conditions without editing historical sources.
- [ ] **Step 4: Re-run the focused tests** and confirm the rendered-cue audits and split checks pass.

### Task 2: Hash-Bound Authorization and Output Isolation

**Files:**
- Modify: `experiments/phase_g_v2_2/execution.py`
- Modify: `experiments/phase_g_v2_2/config.json`
- Test: `experiments/phase_g_v2_2/tests/test_execution.py`

**Interfaces:**
- Consumes: Task 1 supported-condition names and output-directory mapping.
- Produces:
  - `validate_static_config(config: dict[str, Any]) -> dict[str, Any]`
  - `validate_run_authorization(config: dict[str, Any], authorization: dict[str, Any], requested_condition: str, root: Path) -> dict[str, Any]`
  - `condition_output_directory(root: Path, condition: str) -> Path`

- [ ] **Step 1: Write failing tests** proving the checked-in defaults deny execution/test access; missing authorization, draft protocol, mismatched config/protocol/cohort hashes, missing threshold or multiplicity policy, unknown/unauthorized conditions, duplicate allowlist entries, and output paths outside the v2.2 root all fail closed.
- [ ] **Step 2: Run the focused tests and confirm expected authorization failures** with `python -m unittest experiments.phase_g_v2_2.tests.test_execution -v`.
- [ ] **Step 3: Implement static validation and external hash-bound authorization validation.** Require explicit approved protocol hash, frozen cohort membership hash, decision policy, exact condition allowlist, and execution enablement; do not infer missing values or enable the checked-in config.
- [ ] **Step 4: Re-run authorization tests** and confirm allowed test access is possible only when every required field matches and the requested condition is explicitly authorized.

### Task 3: CUDA Training, Calibration, and Safe Resume

**Files:**
- Create: `experiments/phase_g_v2_2/pipeline.py`
- Modify: `experiments/phase_g_v2_2/runner.py`
- Test: `experiments/phase_g_v2_2/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–2 data and authorization APIs; `experiments/phase_g/model.py` model/training/calibration APIs.
- Produces:
  - `require_cuda(device_name: str = "cuda:0") -> torch.device`
  - `train_condition_models(output_dir: Path, config: dict[str, Any], train_data: list[SyntheticPatient], selection_data: list[SyntheticPatient], resume: bool) -> dict[str, nn.Module]`
  - `calibrate_condition_models(output_dir: Path, models: dict[str, nn.Module], calibration_data: list[SyntheticPatient], config: dict[str, Any]) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]`
  - `run_training_preflight(config: dict[str, Any], root: Path) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests** for CUDA-unavailable/device mismatch refusal, condition-local checkpoint paths, frozen model-selection-only checkpoint choice, calibration-data-only temperature fitting, resume hash mismatch refusal, and recovery from an incomplete stage without overwriting validated artifacts.
- [ ] **Step 2: Run the focused tests and confirm expected failures** with `python -m unittest experiments.phase_g_v2_2.tests.test_pipeline -v`.
- [ ] **Step 3: Implement CUDA enforcement and training/calibration orchestration** using the approved model builder/trainer and exact configured batch size, seeds, optimizer, epoch maximum, and patience. Add atomic checkpoint/stage writes and hash-verified resume.
- [ ] **Step 4: Re-run the focused tests** and confirm successful stage recovery remains condition-isolated.
- [ ] **Step 5: Run only the non-evidentiary batch-64 training-loop preflight** on `cuda:0` for image-only, clinical-only, and joint models. Record GPU/device, PyTorch/CUDA versions, available VRAM, peak VRAM, per-step throughput, estimated epoch time, available system RAM, and free disk; fail and report if batch 64 OOMs. Keep disposable artifacts outside condition result directories.

### Task 4: Authorized Test, Predictive, and Explanation Stages

**Files:**
- Modify: `experiments/phase_g_v2_2/pipeline.py`
- Modify: `experiments/phase_g_v2_2/runner.py`
- Test: `experiments/phase_g_v2_2/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3 frozen checkpoints and calibrators; Task 2 validated authorization; existing Phase G predictive/explanation APIs and v2.1 localization/SLIC utilities.
- Produces:
  - `run_authorized_test_stage(output_dir: Path, condition: str, config: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]`
  - A condition-local complete report and artifact manifest only after every required stage passes.

- [ ] **Step 1: Write failing tests** that attempt test generation/evaluation before authorization, before all three checkpoints and calibrators are frozen, with a cohort-hash mismatch, and with an incomplete explanation stage; assert no test data is generated or loaded and no completion manifest is written.
- [ ] **Step 2: Run the focused tests and confirm the test stage remains blocked** with `python -m unittest experiments.phase_g_v2_2.tests.test_pipeline -v`.
- [ ] **Step 3: Implement the authorized test-stage transition.** Generate/evaluate test inputs only after validated authorization and frozen models/calibrators; use the approved predictive and explanation analyses, v2.1 availability-aware translated-control accounting, and configured SLIC metadata. Keep test outputs and manifests in the condition directory.
- [ ] **Step 4: Re-run the focused tests** and confirm denied paths remain side-effect-free and completion is emitted only after all required stages.

### Task 5: CLI, Runtime Report, Documentation, and Regression Verification

**Files:**
- Modify: `experiments/phase_g_v2_2/runner.py`
- Modify: `experiments/phase_g_v2_2/README.md`
- Modify: `experiments/phase_g_v2_2/CHANGELOG.md`
- Modify: `outputs/synthetic_phase_g_v2_2/TEST_RESULTS.json`
- Modify: `outputs/synthetic_phase_g_v2_2/REPRODUCIBILITY_MANIFEST.json`
- Create: `outputs/synthetic_phase_g_v2_2/RUNNER_IMPLEMENTATION_REPORT.md`
- Test: `experiments/phase_g_v2_2/tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 1–4 APIs.
- Produces CLI modes for validation, non-evidentiary preflight, and condition execution requiring an explicit authorization record; commands for each of the five conditions; final test and preflight reports.

- [ ] **Step 1: Write failing CLI tests** for all five condition commands, invalid command/condition rejection, no-run default, required authorization-file handling, and refusal to reuse or overwrite historical output paths.
- [ ] **Step 2: Run the focused CLI tests and confirm expected failures** with `python -m unittest experiments.phase_g_v2_2.tests.test_runner -v`.
- [ ] **Step 3: Implement CLI modes and documentation.** Keep execution disabled in checked-in config; use `python -m experiments.phase_g_v2_2.runner --preflight-only` for timing and `python -m experiments.phase_g_v2_2.runner --condition <condition> --authorization-file <path> --run [--resume]` for an explicitly authorized run; document these commands for each condition, authorization record fields, per-condition paths, resume behavior, runtime estimates, VRAM, and outstanding protocol decisions.
- [ ] **Step 4: Run all requested regression suites**:
  - `python -m unittest discover -s experiments/phase_g/tests -v`
  - `python -m unittest discover -s experiments/phase_g_v2/tests -v`
  - `python -m unittest discover -s experiments/phase_g_v2_1/tests -v`
  - `python -m unittest discover -s experiments/phase_g_v2_2/tests -v`
- [ ] **Step 5: Recompute protected historical hashes** and verify no source/output from v1, v2, or v2.1 changed; verify checked-in v2.2 execution remains disabled and no condition result directory was created.
- [ ] **Step 6: Produce the implementation report** with exact files changed, model/training configuration, CUDA and batch-64 measurements, per-model/per-condition/five-condition runtime and disk estimates, regression results, protected hashes, exact future commands, and unresolved approvals. Do not start a full condition or train full models.

