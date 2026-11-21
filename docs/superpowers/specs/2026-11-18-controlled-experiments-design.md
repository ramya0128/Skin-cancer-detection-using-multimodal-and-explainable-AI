# Controlled Explanation Experiments Design

## Intent

Advance the project from saved-artifact auditing to controlled execution of
Phase F (faithfulness and stability), a Phase G planted-cue pilot, Phase I
literature comparison, Phase H local second-dataset feasibility, and Phase J
reproducibility. The goal is to obtain measured evidence about the existing
image-plus-clinical fusion system and to test whether an independently trained
multimodal model uses known synthetic cues.

The user approved one fixed exploratory evaluation of the already-exposed
frozen test split and a procedural synthetic-image pilot. These choices do not
make the existing test results confirmatory.

## Goals

- Evaluate the saved image explanations with fixed deletion/insertion,
  random-order, perturbation-stability, and ViT randomization protocols.
- Evaluate grouped clinical explanations with feature interventions and an
  XGBoost label-randomization control.
- Keep ViT-branch explanations, XGBoost-branch explanations, and CWF fusion
  scores separate in both implementation and reporting.
- Build a deterministic synthetic benchmark generator with exact lesion masks,
  a separate categorical clinical cue, independently configurable cue
  reliability, and balanced, dominant, redundant, conflicting, and synergistic
  conditions. Execute only one balanced-reliability pilot initially.
- Verify on held-out synthetic patients that the pilot model learned both
  planted cue streams before authorizing expansion to other conditions.
- Resolve and compare O-Shap, PRISM, and the specified Jin papers against the
  proposed work, distinguishing verified source content from unresolved details.
- Check only locally accessible second-dataset files and report actual
  compatibility; do not download data to fill gaps.
- Save runnable scripts, fixed configurations, environment/version records,
  logs, result hashes, and validation evidence in versioned locations.

## Non-goals and anti-goals

- Do not overwrite or modify frozen v1 models, splits, notebooks, or outputs.
- Do not train or tune on the real-data test set, select explainers or
  thresholds from test results, or use test data for debugging.
- Do not present the exploratory test-set results as confirmatory evidence.
- Do not mix synthetic and PAD-UFES-20 observations or use synthetic pilot
  results as evidence of clinical validity.
- Do not claim the planted-cue benchmark is complete after the pilot.
- Do not make broad novelty claims such as first multimodal XAI benchmark,
  first synthetic explanation benchmark, or first multimodal PAD-UFES model.
- Do not treat attention maps or branch-specific image maps as direct
  explanations of the end-to-end CWF function.

## Approaches considered

### Phase F test-set scope

1. **One fixed test evaluation (chosen):** use the existing 461-row test split
   once, with this protocol fixed before results are inspected; do not tune or
   select methods. Report all results as exploratory because test artifacts
   were previously observed and one test image was used in implementation
   debugging.
2. **Keep the test split sealed:** avoids further use of the exposed split, but
   the required OOF image maps and fold-specific image-model weights do not
   exist, so the requested image faithfulness and stability experiments cannot
   be run on an independent real-data split.

### Synthetic-image construction

1. **Procedural lesion canvases with exact masks (chosen):** guarantees the
   image marker lies outside the generated lesion and keeps all generated data
   independent from PAD-UFES-20. The trade-off is lower photographic realism.
2. **PAD training images with automatic estimated masks:** more realistic
   source images, but no lesion masks are present; segmentation error would
   weaken the known-cue ground truth.
3. **Human-annotated PAD masks:** potentially stronger lesion boundaries, but
   is not an automated pilot and requires a separate annotation effort.

## Phase F: faithfulness and stability protocol

### Frozen data and targets

- Read the existing frozen split, final branch checkpoints, CWF weights, saved
  probabilities, image maps, grouped TreeSHAP values, and the existing
  class-stratified 200-row training background.
- Use the 461 frozen test rows only for this one predeclared exploratory
  reporting pass. Never use outcomes to change code, parameters, methods,
  baselines, or thresholds.
- Keep the ViT target class and CWF target class fixed to the saved targets for
  each row through every intervention. Do not silently switch target after a
  perturbation.
- For image interventions, report ViT probability and CWF probability
  separately; CWF is recomputed from the intervened ViT probability and the
  unchanged XGBoost probability using the frozen fusion weights.
- For clinical interventions, report XGBoost and CWF probabilities separately;
  CWF is recomputed from the intervened XGBoost probability and unchanged ViT
  probability using the frozen fusion weights.

### Image faithfulness

- Use the existing 14x14 maps, Gaussian-blur replacement baseline (radius 32),
  and 5% ranked-pixel increments through full deletion/insertion.
- Compare each saved map method with a seed-controlled random-map reference and
  a fixed centre-Gaussian reference. Average 20 fixed-seed random feature
  orders as the random-order control.
- Evaluate both ViT-branch and CWF target probabilities. Label maps by their
  source/target honestly: a ViT map for a CWF-selected target is not an
  end-to-end CWF pixel attribution.
- Save per-row curves and AUCs, aggregate summaries, class counts, and
  patient-cluster uncertainty.

### Image stability and model randomization

- Evaluate horizontal flip (map back before comparison) and +/-10% brightness
  perturbations. Report map Spearman and SSIM only for cases whose relevant
  prediction remains unchanged; report the unchanged-prediction denominator.
- Randomize the ViT classifier head, then progressively the final encoder
  block and the full encoder, with fixed seeds and the original architecture.
  Compare the resulting explanation maps with the frozen-model maps using
  Spearman and SSIM and report prediction/output changes.
- Keep transformations, randomization schedule, and all similarity measures
  fixed before evaluating the test set. If runtime requires batching, time a
  small fixed batch without using its metric values to alter the protocol.

### Clinical feature intervention and randomization

- Reconstruct the grouped encoded feature matrix in the saved XGBoost column
  order. Replace all columns belonging to a clinical variable together using
  the saved training-background rows; do not independently alter dummy
  columns in a way that creates invalid categories.
- Order the eight groups by the saved TreeSHAP contribution for the fixed
  target. Evaluate deletion/insertion and 20 fixed-seed random group orders,
  separately for XGBoost and CWF outputs.
- Fit one separate XGBoost control on training rows with labels permuted using
  a fixed seed and the frozen model configuration. Do not alter the saved
  checkpoint. Compare the control's held-out behavior and grouped
  explanations with the original model as a randomization sanity check.

### Statistical reporting

- Report per-class and pooled results, patient-cluster bootstrap intervals
  (2,000 fixed-seed resamples), and Holm-adjusted comparisons within
  predeclared metric families, following `ANALYSIS_PLAN.md`.
- Show denominators and missing/unchanged-prediction counts. Do not select or
  rank a method by a single favorable metric.
- Treat all real test-set Phase F results as descriptive/exploratory. They
  cannot establish confirmatory performance; an untouched external or holdout
  set remains necessary.

## Phase G: synthetic planted-cue benchmark

- Generate images from scratch with explicit lesion masks and a marker placed
  outside the lesion by construction. Generate a separate categorical clinical
  cue; neither source contains real PAD-UFES-20 data.
- Use deterministic generation and patient IDs with patient-level
  train/validation/test separation. Keep generated arrays, checkpoints,
  predictions, and reports under a separate synthetic run directory.
- Implement configurable conditions for balanced reliability,
  image-dominant, clinical-dominant, redundant, conflicting, and synergistic
  cues. Define each condition's label-generation and reliability semantics in
  the config; do not conflate contradictory cues with synergy.
- Train a new small multimodal model only on synthetic training patients.
  Choose checkpoints and training settings from synthetic validation patients
  only. The synthetic test patients are held for the predeclared pilot report.
- Run only the balanced-reliability pilot initially, with image and clinical
  cue reliability both set to 0.85. The generated lesion appearance is
  independent of class, so the class signal comes from the planted cues.
- Predeclared pilot gate: full held-out accuracy must be at least 0.80 and
  independently masking either modality's planted cue must reduce held-out
  accuracy by at least 0.05. Report the per-modality cue intervention effects,
  cue localization against the exact image marker mask, and held-out class
  counts. If the gate fails, diagnose only on synthetic training/validation
  data; do not expand conditions until corrected and revalidated.
- A passed pilot authorizes planning/approval for later condition runs; it does
  not mean the full benchmark is executed or complete.

## Phase I: literature and novelty comparison

- Update the literature audit with a source-checked comparison table containing
  each work's contribution, dataset/model setting, explanation method,
  evaluation methodology, what this project adds, and overlap/limitations.
- Resolve the intended O-Shap, PRISM, Jin AAAI-22, and Jin MedIA-23 citations
  against the project notes and primary sources. If an exact identity or
  methodological detail cannot be verified, mark it unresolved rather than
  substituting a similar-title paper.
- Explicitly distinguish Jin et al.'s multi-channel MRI setting and MSFI from
  image-plus-tabular clinical fusion and any proposed modality-level Shapley
  metric. Treat the synthetic known-cue work in related literature as prior art.
- Limit the novelty conclusion to a candidate empirical contribution; do not
  assert novelty until the complete comparison is supported by primary-source
  evidence.

## Phase H: second-dataset feasibility

- Inspect only actual accessible candidate files. Record available identifiers,
  image modalities, metadata, labels, licensing evidence, and compatibility
  with PAD-UFES-20.
- Do not unpickle files of unknown provenance or infer fields/license from a
  dataset description alone.
- If no compatible, authorized files are accessible, document the blocker and
  stop searching rather than downloading or spending excessive time.

## Phase J: reproducibility and artifact safety

- Put runnable source under a dedicated experiments directory and results in
  new versioned subdirectories of `outputs/`; never write to existing v1 paths.
- Save fixed configs/seeds, dependency and hardware versions, full command
  logs, output schemas, SHA-256 hashes, and validation summaries.
- Hash protected inputs before and after each run and verify that the frozen
  split, checkpoints, notebooks, saved v1 results, and image cache remain
  unchanged.
- Keep image/GPU and clinical/TreeSHAP execution environments separate when
  required by their installed packages. Do not install dependencies unless an
  actual validation command demonstrates they are missing.
- Tests must cover synthetic cue placement, deterministic generation,
  patient-level split disjointness, coalition/feature grouping, and saved
  output schemas. A test passing is not evidence that an experiment executed.

## Deliverables and completion labels

- Versioned Phase F results, configurations, figures, logs, and input/output
  hashes.
- A synthetic benchmark generator and one executed balanced-reliability pilot,
  with explicit pass/fail gate and separate synthetic results.
- An updated, source-grounded literature novelty comparison table.
- A local second-dataset feasibility finding, including a blocker if no
  compatible files exist.
- Reproducible scripts/configurations and a run manifest.
- A final status report that distinguishes **implemented**, **executed**,
  **validated**, **failed**, and **pending**. No work is described as completed
  unless its saved outputs and validation evidence exist.

## Approval boundary

This design records the user's selected test-set and synthetic-image choices.
No experiment script, model training, inference, or dependency installation
will begin until the user approves this design. After approval, create an
implementation plan and execute the phases in order, preserving the stated
pilot gate and artifact safeguards.
