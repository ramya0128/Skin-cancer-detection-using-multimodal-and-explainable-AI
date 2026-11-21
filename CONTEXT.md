# Project Context and Handoff

**Updated:** 2026-11-21

## Project in one paragraph

This repository studies skin-lesion classification using image and clinical
features, explanation methods, and a separate controlled synthetic experiment.
It contains a real-data PAD-UFES-20 notebook pipeline and a synthetic Phase G
pipeline. Keep their datasets, model results, explanation targets, and
scientific claims separate. The repository-wide component map is in
[ARCHITECTURE.md](./ARCHITECTURE.md).

## Current status

| Workstream | Status | Important boundary |
|---|---|---|
| Real-data base models and fusion | Frozen artifacts exist | Do not overwrite saved checkpoints, split, or base outputs |
| Modality Shapley v1/v2 | Saved; all five OOF folds were marked complete | Existing final/test results are not permission to rerun or retune |
| Notebook 04/05 explanations | Saved outputs audited | Branch explanations are not explanations of the fused function |
| Phase F | Executed under a fixed exploratory authorization | Test results remain exploratory due the documented test-image debugging deviation |
| Phase G v1 | Balanced synthetic pilot artifacts exist | Historical pilot is not a fresh holdout |
| Phase G v2/v2.1 | Validation-only implementations | Use frozen models; no remaining-condition authorization |
| Phase G v2.2 | Engineering foundation exists; prospective protocol is draft and blocked | No condition experiment, fresh cohort, or held-out evaluation is authorized |

## Real-data pipeline and saved artifacts

The frozen PAD-UFES-20 split contains 1,837 training rows and 461 test rows.
It is patient-disjoint (1,098 train patients, 275 test patients), but has 79
lesion IDs across the partitions and one flagged perceptual-hash collision.
MEL has seven frozen test rows. The test set has already been used for final
reporting and exploratory analyses, so it is not a clean confirmatory holdout.
See [DATA_AUDIT.md](./DATA_AUDIT.md).

The notebook flow is:

1. `01_base_outputs.ipynb`: frozen split, image cache, image and clinical
   branches, OOF/test probabilities, and final checkpoints.
2. `02_fusion.ipynb`: fusion over saved branch outputs; it does not retrain
   those branch models.
3. `03_modality_shapley.ipynb` and `03b_modality_shapley.ipynb`:
   probability-level and input-level modality Shapley. The saved v2 OOF
   checkpoint was marked complete for all five folds.
4. `04_treeshap.ipynb`: explanations of the clinical XGBoost branch.
5. `05_image_explainers.ipynb`: image maps computed from the ViT branch. A
   CWF-selected class used as a target does not make the map a CWF explanation.

Frozen outputs and caveats are catalogued in
[ARTIFACT_AUDIT.md](./ARTIFACT_AUDIT.md) and
[NOTEBOOK_04_05_AUDIT.md](./NOTEBOOK_04_05_AUDIT.md). Phase F results are under
[outputs/phase_f_v1/](./outputs/phase_f_v1/) and remain exploratory. Further
work on the frozen test set must not be used for method development or
confirmatory claims; confirmatory work requires an untouched external/holdout
cohort.

## Phase G synthetic work

### Existing versions

- `experiments/phase_g/` holds the original patient renderer, split logic,
  model code, and balanced-pilot runner. The historical balanced pilot is
  separate from the remaining v2.2 conditions.
- `experiments/phase_g_v2/` is validation-only and uses frozen v1 models.
- `experiments/phase_g_v2_1/` contains the availability-aware localization and
  SLIC corrections.
- `experiments/phase_g_v2_2/` contains condition generation, conflicting
  assignment/rendering, a lazy dataset, cohort-manifest validation, explanation
  and faithfulness wrappers, statistical utilities, and a fail-closed runner.

The generator defines the five remaining cue conditions: image-dominant
(0.95/0.60), clinical-dominant (0.60/0.95), redundant shared noisy cue (0.85),
conflicting (25% image-correct/clinical-decoy, 25% clinical-correct/image-decoy,
50% independent 0.85/0.85), and synergistic complementary six-class shares
with independent 0.85 share retention. Images use a label-independent
synthetic lesion and background, an exact lesion mask, and an exterior marker
for the image cue; the clinical cue is a separate category.

### Recent image-tampering change — not runtime-verified

The shared renderer now accepts and forwards conflicting orientation into cue
generation; conflicting splits require a complete orientation mapping. The
v2.2 wrapper delegates to that shared renderer. The v2.2 condition API and
conflicting renderer reject test-partition rendering. Generated-patient tests
were added for cue distributions, marker encoding, mask invariance,
determinism, split orientation, and test-generation refusal.

These tests have **not** run. The focused Python test command could not start
because the sandbox could not reach its local proxy. Pylance reported no
diagnostics in the changed files, which is not runtime test evidence. Details
are in
[IMAGE_TAMPERING_IMPLEMENTATION_REPORT.md](./outputs/synthetic_phase_g_v2_2/IMAGE_TAMPERING_IMPLEMENTATION_REPORT.md).

### v2.2 authorization and blockers

`experiments/phase_g_v2_2/config.json` currently says:

- `protocol_status: draft_unapproved`
- `execution_enabled: false`
- `authorized_conditions: []`
- `test_data_access: forbidden`

The runner fails closed. Protocol compatibility currently blocks condition
execution because the checked-in analysis config does not match the required
Grad-CAM variants, SLIC faithfulness settings/baselines, and primary-statistics
methodology. The test-stage entry point also explicitly stops before test
generation. The existing disposable GPU preflight trained each model for one
epoch only; it was not a condition run. Its memory check found that the full
eager patient arrays exceed the observed available-RAM floor. See the
[v2.2 README](./experiments/phase_g_v2_2/README.md),
[protocol review](./docs/phase_g_v2_2/PHASE_G_V2_2_PROTOCOL_REVIEW.md),
[statistical plan](./docs/phase_g_v2_2/PHASE_G_V2_2_STATISTICAL_PLAN.md), and
[decision summary](./docs/phase_g_v2_2/PHASE_G_V2_2_DECISION_SUMMARY.md).

Do not train the 15 condition models, generate the fresh evaluation cohort,
render test records, run held-out inference, or change authorization settings
until the investigator has approved the protocol and supplied explicit
hash-bound authorization.

## Explanation status and interpretation

- Real-data TreeSHAP describes the clinical XGBoost branch.
- Real-data pixel maps describe the ViT image branch.
- Modality Shapley describes the specified two-player CWF function.
- Phase F evaluates faithfulness/stability and interventions, but its real-data
  test-set results are exploratory.
- v2.2 contains wrappers for four Grad-CAM variants and SLIC faithfulness,
  but they are not yet connected to a completed condition evaluation. No v2.2
  explanation results exist.

Never report these targets as interchangeable. See
[NOTEBOOK_04_05_AUDIT.md](./NOTEBOOK_04_05_AUDIT.md).

## Setup, privacy, and reproducibility

- The dataset is not bundled. Configure the local metadata/image paths in
  `common.py` only for an authorized dataset copy; do not copy machine-specific
  paths into project documentation.
- Dataset licensing terms are not verified in this workspace. Follow the
  authorized source's terms and the privacy instructions in
  [README.md](./README.md).
- No root `requirements.txt` or `pyproject.toml` pins the environment.
  Notebook environments have differed; record actual library, CUDA, and GPU
  versions for each new run.
- Do not overwrite `outputs/` or `cache/` artifacts. Check their manifests and
  phase-specific safeguards first.

## Safe next steps

1. Enable local network access for the sandbox, then run the focused
   generator tests:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest experiments.phase_g_v2_2.tests.test_data experiments.phase_g_v2_2.tests.test_conflicting_generator
   ```

2. Resolve and approve the v2.2 protocol, statistical decisions, cohort
   manifest, and explanation/faithfulness integration before any experimental
   execution.
3. Keep the existing real-data test set exploratory and untouched for any
   future confirmatory work.

## Read-first index

- [Architecture](./ARCHITECTURE.md)
- [Project setup](./README.md)
- [Analysis plan](./ANALYSIS_PLAN.md)
- [Data audit](./DATA_AUDIT.md)
- [Saved artifact audit](./ARTIFACT_AUDIT.md)
- [Notebook 04/05 audit](./NOTEBOOK_04_05_AUDIT.md)
- [Phase G v2.2 implementation plan](./docs/phase_g_v2_2/PHASE_G_V2_2_IMPLEMENTATION_PLAN.md)
- [Phase G v2.2 implementation report](./outputs/synthetic_phase_g_v2_2/ENGINEERING_IMPLEMENTATION_REPORT.md)
- [Phase G v2.2 verification report](./outputs/synthetic_phase_g_v2_2/VERIFICATION_REPORT.md)
- [Image-tampering implementation report](./outputs/synthetic_phase_g_v2_2/IMAGE_TAMPERING_IMPLEMENTATION_REPORT.md)
