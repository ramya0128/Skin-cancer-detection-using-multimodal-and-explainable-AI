# Phase G v2 validation-only re-evaluation

Phase G v2 is a separate, validation-only re-evaluation of the balanced
synthetic pilot. It addresses the Phase G v1 explanation-map validity,
lesion-control geometry, and perturbation-independence concerns without
changing v1 source files or output artifacts.

## Data and execution boundary

- The runner reads the frozen v1 run configuration, model-selection patient IDs
  and labels, validation-calibration temperatures, and image-only/joint frozen
  checkpoints.
- It regenerates only the 900 `validation_model_selection` patients. It never
  loads test images, test labels, test predictions, or the test input archive.
- It does not train or fine-tune models. Clinical-only maps are excluded because
  they do not have image-space explanations.
- CUDA `cuda:0` is mandatory for model inference; the run aborts rather than
  falling back to CPU. CPU work is limited to rendering, SLIC segmentation,
  geometry, and summary calculations.
- A SHA-256 baseline covering v1 source and artifacts is checked before and
  after each run. Each run writes to a new timestamped output directory and
  refuses to overwrite an existing directory.

## Explanation-map candidates and validity

The runner computes four Grad-CAM variants for image-only and joint models:

1. Final ViT block, gradient averaged across patch tokens (`gradcam_v1_last_global`).
2. Final block, token-local activation-gradient product
   (`gradcam_v2_last_token_local`).
3. Penultimate block, gradient averaged across patch tokens
   (`gradcam_v2_penultimate_global`).
4. Penultimate block, token-local activation-gradient product
   (`gradcam_v2_penultimate_token_local`).

Chefer, attention rollout, and occlusion maps are also recomputed on the same
validation patients. Maps are never silently treated as valid because they are
all zero. Grad-CAM reports non-finite raw values, no positive support, or
constant positive maps as explicit invalid reasons. Other map methods report
non-finite, negative, or zero post-normalization maps as invalid. Invalid maps
are excluded from localization and faithfulness summaries; per-patient validity
and reasons remain in the saved arrays.

Localization divides each valid nonnegative map by its total mass and reports
the lesion, marker, and complementary background shares. These three regions
are a disjoint, exhaustive partition and must sum to one per valid patient.
Area enrichment is each region's attribution share divided by its image-area
fraction. Pointing-game values are omitted for invalid maps.

## Shape-preserving localization controls

For each validation patient, 20 control masks are exact integer translations of
the lesion mask. Candidate offsets are enumerated by FFT-based overlap counts;
controls must remain in-frame and must not overlap the original lesion or
marker. Thus each control preserves the lesion's exact shape and area. An
unavailable control is reported as unavailable; there is no random-pixel
fallback. The runner processes controls in bounded batches and saves selected
offsets, candidate-placement counts, availability, and each map's mass inside
the translated controls. Bootstrap resampling is over patients, not control
repeats.

## Independent perturbation evaluation

On a fixed, class-stratified subset of 72 model-selection patients (12 per
synthetic class), faithfulness is evaluated by replacing/restoring connected
SLIC superpixels, not 16x16 patches. The 128-region segmentation is shared
across methods for a patient. Curves use 21 affected-area points and two
replacement baselines: the patient's renderer-generated background canvas and
constant mid-gray. For each map and baseline, deletion/insertion curves are
compared with 20 patient-specific random superpixel orders and a center-first
order. The target is the patient's fixed synthetic label; the joint model's
observed clinical category is held fixed.

Per-patient AUCs and curves are retained. The report summarizes mean AUCs and
paired method-minus-random and method-minus-center differences with percentile
95% patient-bootstrap intervals (2,000 replicates). These are descriptive
validation-set diagnostics, not confirmatory p-values or clinical evidence.
A one-patient CUDA timing/memory benchmark is completed before the full
independent evaluation.

## Marker intervention sensitivity

Marker interventions are reported separately from explanation faithfulness.
For each of the same 72 patients, the rendered glyph is replaced by a different
valid renderer glyph or restored to that patient's exact background canvas.
The joint model receives the unchanged clinical category. Probability changes
are saved per patient; these interventions do not establish attribution
faithfulness.

## Outputs and interpretation

Each run directory contains environment and selection records, map arrays and
diagnostics, valid-only localization and geometry results, independent
superpixel curves/AUCs/controls, marker-intervention results, a validation
summary, v1 preservation evidence, and a SHA-256 run manifest. Calibration
temperatures are consumed from the already-frozen
`validation_calibration` fit; they are not refit on model-selection patients.

This is a methodological re-evaluation on validation data. It does not alter
the v1 held-out test analysis, authorize other synthetic conditions, or
establish real-world skin-cancer performance.
