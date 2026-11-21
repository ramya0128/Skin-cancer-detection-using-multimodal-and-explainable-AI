# Project Architecture

## Purpose and scope

This repository contains research code and saved results for skin-lesion
classification and explanation. It has two distinct experimental tracks:

1. **Real-data track:** PAD-UFES-20 image and clinical/tabular classification,
   late fusion, and explanation analysis.
2. **Synthetic track:** controlled Phase G data with planted image and
   clinical cues, intended to test model behavior and explanations under known
   cue relationships.

The tracks use different data, model configurations, and scientific claims.
Synthetic results must not be presented as clinical performance, and
real-data branch explanations must not be presented as explanations of the
fused model.

## System overview

```text
REAL-DATA TRACK
Authorized PAD-UFES-20 files (local, not bundled)
        |
        +--> common.py / notebooks: metadata, images, transforms, frozen
        |    patient-level split
        |
        +--> image branch (ViT) -----------+
        +--> clinical branch (XGBoost) ----+--> saved branch outputs
                                             |
                                             +--> CWF late fusion
                                             |
                                             +--> explanation analyses
                                                  - branch-level image maps
                                                  - branch-level TreeSHAP
                                                  - CWF-level modality Shapley
                                                  - exploratory faithfulness

SYNTHETIC PHASE G TRACK
Frozen condition definition + deterministic patient IDs/seed
        |
        +--> label-independent lesion and background rendering
        +--> condition-specific image cue --> exterior image marker
        +--> condition-specific clinical cue --> categorical record
        |
        +--> image-only ViT
        +--> clinical-only embedding/MLP
        +--> joint ViT + clinical fusion
        |
        +--> calibration and (only after approval) evaluation/explanations
```

## Repository map

### Real-data entry points

- `01_base_outputs.ipynb` builds the frozen split and base image/clinical
  outputs. It includes ViT training and can be expensive.
- `02_fusion.ipynb` compares fusion methods using saved base outputs; it does
  not retrain the branch models.
- `03_modality_shapley.ipynb` computes probability-level modality Shapley.
- `03b_modality_shapley.ipynb` computes input-level modality Shapley with
  five-fold OOF checkpointing.
- `04_treeshap.ipynb` explains the clinical XGBoost branch.
- `05_image_explainers.ipynb` computes image-branch maps.
- `common.py` provides shared dataset and image-transform code. Its metadata
  and image-root settings must point to an authorized local dataset copy;
  literal machine paths are intentionally not documented here.
- `experiments/phase_f/` contains the fixed exploratory faithfulness,
  stability, intervention, and summary tooling.

### Synthetic experiment packages

- `experiments/phase_g/` contains the original synthetic patient generator,
  split logic, models, evaluation, and balanced-pilot runner.
- `experiments/phase_g_v2/` performs validation-only explanation-map and
  faithfulness checks using frozen v1 models.
- `experiments/phase_g_v2_1/` contains corrective SLIC and availability-aware
  localization utilities.
- `experiments/phase_g_v2_2/` contains the remaining-condition data API,
  conflicting-condition assignment wrapper, lazy dataset, cohort validation,
  explanation/faithfulness adapters, statistical utilities, and fail-closed
  runner.
- `docs/phase_g_v2_2/` contains the protocol review, statistical plan,
  implementation plan, and decision summary.

### Research records and artifacts

- Root audit documents (`DATA_AUDIT.md`, `ARTIFACT_AUDIT.md`,
  `NOTEBOOK_04_05_AUDIT.md`, and `ANALYSIS_PLAN.md`) describe frozen data,
  saved evidence, interpretation limits, and authorized analysis.
- `outputs/` contains frozen model/results and versioned analysis outputs.
  Treat saved artifacts as inputs unless a specific run is authorized.
- `cache/` contains cached image tensors.
- `ara/` is a separate Agent-Native Research Artifact with claims, concepts,
  evidence pointers, and research trace. Its `logic/solution/architecture.md`
  is ARA-specific, not the repository-wide architecture.

## Real-data architecture

### Data and split

The real-data work uses PAD-UFES-20 metadata and images obtained from an
authorized local source. The data are not included in the documented artifact
bundle. The frozen split has 1,837 training rows and 461 test rows across six
classes, grouped by patient: 1,098 training patients and 275 test patients
with no patient overlap. There are 79 lesion IDs shared across the partitions
and one flagged perceptual-hash collision; neither was silently removed.
MEL has only seven frozen test rows.

The frozen test set has already been used for final reporting and exploratory
explanation analyses. A test-image implementation smoke check is also
documented. Accordingly, those test results are descriptive/exploratory, not
clean confirmatory evidence. Confirmatory claims require an untouched
external/holdout cohort.

### Models and explanation targets

- **Image branch:** ViT classifier operating on lesion images.
- **Clinical branch:** XGBoost classifier on encoded clinical/tabular
  features.
- **Fusion:** CWF late fusion over saved branch predictions.
- **Branch explanations:** image maps explain the ViT branch; TreeSHAP explains
  the XGBoost branch. A CWF-selected class used as an image-map target does not
  make that map an explanation of the fused function.
- **Fusion explanation:** modality Shapley explains the specified two-player
  CWF prediction game, with its documented background and interventions.
  It is distinct from branch-level pixel or clinical-feature attribution.

### Saved results

The root `outputs/` includes the frozen split, base predictions, final XGBoost
and ViT checkpoints, fusion results, modality-Shapley results and OOF
checkpoint, TreeSHAP and image-explainer arrays, and audit derivatives.
`outputs/phase_f_v1/` contains a later authorized exploratory analysis. Its
results are preserved and should be interpreted using `ANALYSIS_PLAN.md`,
`DATA_AUDIT.md`, `ARTIFACT_AUDIT.md`, and `NOTEBOOK_04_05_AUDIT.md`.

The source dataset license is not verified in this workspace. Follow the
authorized source's terms and the privacy restrictions in `README.md`.

## Synthetic Phase G architecture

### Patient generation

The renderer creates 224x224 RGB images with a synthetic lesion, an exact
binary lesion mask, a textured irrelevant background, and a 16x16 external
glyph that encodes the image cue. The clinical cue is a separate six-category
value. Lesion geometry and texture are class-independent and use patient-ID
random streams. Patient splitting is performed before rendering.

The remaining condition definitions are:

| Condition | Image cue | Clinical cue |
|---|---|---|
| `image_dominant` | 0.95 reliable | 0.60 reliable |
| `clinical_dominant` | 0.60 reliable | 0.95 reliable |
| `redundant` | Same shared noisy class | Same shared noisy class (0.85 reliability) |
| `conflicting` | 25% image-correct/clinical-decoy; 50% independent 0.85 cue pairs | 25% clinical-correct/image-decoy; 50% independent 0.85 cue pairs |
| `synergistic` | Corrupted share `U` | Independently corrupted share `(Y-U) mod 6` |

Wrong class cues are drawn uniformly from the other five categories. In the
synergistic condition, each share is corrupted independently with probability
0.15; the final shares are individually uninformative about the label, while
modulo-six decoding has expected accuracy about 0.727.

The conflicting orientation is now forwarded through `render_patient` to cue
generation. The v2.2 wrapper delegates to that shared renderer and requires a
complete patient-to-orientation mapping. Focused generated-patient tests were
added, but those tests have not run in the current sandbox.

### Models and lifecycle

The synthetic Phase G model set has three independently trained model types:

1. image-only ViT;
2. clinical-only categorical embedding/MLP;
3. joint image-plus-clinical fusion model.

The intended pipeline separates training, model-selection, calibration, and
evaluation partitions. Calibration uses a separate calibration partition.
Future authorized condition outputs are isolated under
`outputs/synthetic_phase_g_v2_2/<condition>/`.

### Version and execution status

- **v1:** balanced 0.85/0.85 pilot artifacts are preserved.
- **v2:** validation-only re-evaluation using frozen models; no model
  retraining or test access.
- **v2.1:** correction to SLIC/availability handling; it does not authorize
  remaining conditions.
- **v2.2:** condition-generation and engineering components exist, but the
  prospective protocol is `draft_unapproved`. The checked-in config has
  `execution_enabled=false`, `authorized_conditions=[]`, and
  `test_data_access=forbidden`.

The v2.2 runner also blocks execution because the approved analysis
specification and current configuration/runner do not yet agree on the
explanation method IDs, faithfulness settings, and primary statistical gate.
The host-RAM preflight also found that a full eager dataset exceeds the
available-memory floor. No remaining-condition experiment or fresh evaluation
cohort is represented as run.

## Explainability architecture and status

### Real-data analyses

- `04_treeshap.ipynb`: grouped clinical-branch TreeSHAP and exact grouped
  probability Shapley over the configured eight clinical variables.
- `05_image_explainers.ipynb`: ViT-branch Grad-CAM, attention rollout,
  Chefer-style rollout, and occlusion maps.
- `03_modality_shapley.ipynb` and `03b_modality_shapley.ipynb`: CWF-level
  modality Shapley at probability and input levels.
- Phase F evaluates faithfulness/stability and interventions using the frozen
  real-data test set under an exploratory-only interpretation.

These are different explanation targets and should be reported separately.
Saved artifacts and limitations are detailed in
`NOTEBOOK_04_05_AUDIT.md` and `ARTIFACT_AUDIT.md`.

### Synthetic v2.2 explanations

`experiments/phase_g_v2_2/explain.py` wraps four Grad-CAM variants from the v2
implementation. `faithfulness.py` wraps SLIC faithfulness with 128 regions,
21 perturbation fractions, 20 random orders, center-first control, and
renderer-background/mid-gray baselines. These are engineering components, not
completed v2.2 evaluation results; the v2.2 condition runner does not yet
produce end-to-end explanation reports.

## Reproducibility and safety boundaries

- Use the frozen split and saved checkpoints as immutable historical inputs.
- Check provenance, configuration and source hashes before interpreting saved
  results.
- Do not use the frozen real-data test set for further method development.
- Do not generate the fresh v2.2 evaluation cohort, condition test records,
  run held-out inference, or train full condition models until the protocol is
  approved and explicit authorization is issued.
- v2.2 condition generation rejects the `test` partition; the lazy development
  dataset also rejects test access.
- The repository root has no pinned `requirements.txt` or `pyproject.toml`.
  Notebook environments have differed; record the actual Python, PyTorch,
  torchvision, CUDA, and other relevant versions for each new run.

## Authoritative references

- [Project setup and artifact guide](./README.md)
- [Current project handoff](./CONTEXT.md)
- [Data audit](./DATA_AUDIT.md)
- [Saved artifact audit](./ARTIFACT_AUDIT.md)
- [Notebook 04/05 audit](./NOTEBOOK_04_05_AUDIT.md)
- [Analysis plan](./ANALYSIS_PLAN.md)
- [Phase G v2.2 protocol review](./docs/phase_g_v2_2/PHASE_G_V2_2_PROTOCOL_REVIEW.md)
- [Phase G v2.2 statistical plan](./docs/phase_g_v2_2/PHASE_G_V2_2_STATISTICAL_PLAN.md)
- [Phase G v2.2 decision summary](./docs/phase_g_v2_2/PHASE_G_V2_2_DECISION_SUMMARY.md)
- [Phase G v2.2 implementation plan](./docs/phase_g_v2_2/PHASE_G_V2_2_IMPLEMENTATION_PLAN.md)
