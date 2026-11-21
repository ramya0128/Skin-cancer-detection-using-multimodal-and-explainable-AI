# Phase G v2.2 Condition Runner Design

## Intent

Implement a separate, resumable v2.2 runner for the five remaining synthetic
conditions. It should let a researcher reproduce the approved model-training,
calibration, predictive, and explanation analyses without touching historical
artifacts, while making accidental or premature execution impossible.

## Goals

- Support `image_dominant`, `clinical_dominant`, `redundant`, `conflicting`,
  and `synergistic`; do not rerun the balanced pilot.
- Reuse the verified Phase G model architectures, training configuration,
  Phase G evaluation/explanation functions, and v2.1 correction utilities
  without editing those historical sources.
- Split patient IDs before rendering, keep all partitions patient-disjoint,
  and keep test labels/data inaccessible until the explicit execution and
  prospective-evaluation gates pass.
- Use `cuda:0` for training and GPU-compatible inference; fail rather than
  falling back to CPU. Do not change the configured batch size or enable
  mixed precision.
- Store condition-specific configurations, checkpoints, logs, intermediate
  results, and manifests in unique v2.2 output directories. Support safe
  interruption/resume through verified stage artifacts.
- Provide non-evidentiary timing preflight and focused tests, but do not run a
  full condition as part of implementation.

## Non-goals and anti-goals

- Do not alter, overwrite, or retrain original real-data models or historical
  Phase G v1/v2/v2.1 outputs.
- Do not change model architectures, seeds, splits, metrics, condition
  semantics, thresholds, or multiplicity rules.
- Do not grant execution or test access by default.
- Do not describe a software test or preflight as scientific evidence.
- Do not claim completion until training, calibration, authorized test
  evaluation, explanation analyses, and integrity checks all finish.

## Approaches

1. **Complete gated v2.1 pipeline — selected.** Build a condition-aware v2.2
   orchestrator that reuses the established model and analysis functions,
   stores its own versioned outputs, and refuses test/explanation stages until
   the prospective protocol and condition authorization are approved. This
   best matches the requested fully reproducible condition runs while
   respecting current restrictions.
2. **Training/calibration only.** Smaller implementation, but leaves no
   complete route to approved test and explanation results; rejected because
   it would require a second runner before any condition can be scientifically
   completed.
3. **Adapt the existing pilot runner in place.** Reuses the most orchestration
   code, but its validation is balanced-pilot-specific and modifying it risks
   historical behavior; rejected in favor of a separate v2.2 execution path.

## Current-state findings and constraints

- The present v2.2 `runner.py` validates requests only; it does not generate
  all conditions or train models.
- The v2.2 renderer is specific to conflicting records. The existing generator
  contains other condition cue logic, but the historical renderer reads
  `pilot_condition` and is not a safe condition-specific execution path as-is.
- The model module exposes the three approved architectures and `train_model`,
  `fit_temperature`, and `predict_logits`. Its trainer falls back to CPU if
  CUDA is unavailable, so the v2.2 runner must enforce CUDA before calling it
  and verify model and batch devices.
- Existing Phase G orchestration contains training, calibration, test
  prediction, explanation, statistics, resume, and manifest behavior, but it
  is balanced-pilot-specific and must not be changed for v2.2.
- The v2.1 prospective protocol is marked not approved for execution. It also
  says a fresh untouched evaluation subset must be declared and leaves the
  predictive threshold and multiplicity-adjusted interpretation unresolved.
  Therefore implementation must remain fail-closed for test access until the
  researcher approves the protocol, evaluation cohort, thresholds, and
  condition-specific execution authorization. No values will be inferred.

## Proposed design

- Extend only `experiments/phase_g_v2_2/` with condition-specific configuration
  validation, generation adapters, the executable orchestration, and tests.
- Bind each supported condition to its fixed reliability/corruption semantics.
  Use the v2.2 conflicting assignment/render path for conflicting records and
  verify the produced records; use the existing generator logic for the other
  conditions without editing its source.
- Reuse the existing patient split generator and frozen per-class counts:
  1,400 train, 150 model-selection, 150 calibration, and 300 test patients per
  class. Persist split IDs and verify disjointness before rendering. Keep test
  record generation/access behind the separate gate.
- Train image-only, clinical-only, and joint models with the configured
  architectures, AdamW settings, batch size 64, epoch limit, patience, and
  seeds. Save condition-local checkpoints and selection histories atomically.
- Fit scalar temperatures on calibration data only, persist calibration
  logits and frozen temperatures, then verify all checkpoints and calibrators
  before any test-stage transition.
- Once separately authorized, reuse the approved predictive and explainability
  analysis implementations, including v2.1 availability-aware translated
  controls and SLIC metadata. Keep test outputs, condition reports, and
  manifests condition-local.
- Make stages idempotent and resumable: match the frozen config and source
  hashes, verify stage output hashes, refuse conflicting or incomplete
  checkpoints, and write final completion status only after all required
  stages pass.
- Define a hash-bound authorization record for any future execution. It must
  explicitly identify the approved protocol revision/hash, frozen evaluation
  cohort and membership hash, predictive threshold and multiplicity policy,
  exact condition allowlist, and execution enablement. Validate that each
  field is present and consistent; never infer or default an approval value.
  Until a valid record is supplied, test access and condition execution remain
  blocked.
- Add a separate, explicitly non-evidentiary timing-preflight mode that runs
  the actual training loop with batch size 64 on disposable synthetic data,
  reports peak CUDA memory/throughput/epoch timing, and writes outside all
  condition result directories. It must not relax scientific configuration or
  leave condition checkpoints.
- Keep the checked-in execution configuration disabled, with an empty
  allowlist and test access forbidden. Implement future authorization checks
  so arbitrary condition names, protocol states, or output paths cannot bypass
  the gate.

## Validation and acceptance

- Test all five condition configurations and cue semantics, including actual
  rendered records, deterministic partition separation, and empirical
  reliability/distribution checks.
- Test refusal of CPU fallback, test data before authorization, mismatched
  resume hashes, unsafe output paths, and historical-output overwrite
  attempts.
- Test a successful disposable interrupted-run/resume flow without executing
  a scientific condition.
- Run all v2.2, v2.1, Phase G v1, and Phase G v2 regression suites.
- Measure the real configured batch-size preflight and report exact timing,
  peak memory, estimated per-model/per-condition/total runtime, and disk
  requirements. Do not start full model training or any condition.
- Recheck all protected historical hashes and confirm the v2.2 execution
  configuration remains disabled.

## Remaining researcher decisions

The runner implementation can enforce these gates, but must not resolve them:

1. Approve the prospective protocol for execution.
2. Declare the untouched evaluation cohort/partition before any remaining
   condition result is viewed.
3. Approve the predictive threshold and multiplicity-adjusted interpretation.
4. Authorize the exact conditions to run and enable execution in a separate,
   reviewed authorization step.

