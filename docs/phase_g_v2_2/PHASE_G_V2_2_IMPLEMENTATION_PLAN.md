# Phase G v2.2 implementation plan

**Status:** Concrete engineering plan; no scientific execution authorization
is included.

## 1. Package boundaries

Keep all work in `experiments/phase_g_v2_2/` and v2.2 output directories.
Do not modify:

- real-data ViT/XGBoost/CWF checkpoints;
- Phase G v1 or v2 historical files;
- historical v1/v2/v2.1 outputs;
- the checked-in execution and test-access defaults.

Existing v2 explanations may be reused rather than rewritten if their
interfaces are compatible. Historical sources remain read-only.

## 2. Explanation pipeline

### 2.1 Grad-CAM variants

The existing v2 implementation already exposes the four required variants:

| Protocol method ID | Existing implementation | Definition |
|---|---|---|
| `gradcam_v1_last_global` | `compute_gradcam(..., layer_index=-1, weighting="global")` | final ViT block, gradient mean over patch channels |
| `gradcam_v2_last_token_local` | `compute_gradcam(..., layer_index=-1, weighting="token_local")` | final ViT block, per-patch token-channel weighting |
| `gradcam_v2_penultimate_global` | `compute_gradcam(..., layer_index=-2, weighting="global")` | penultimate block, gradient mean |
| `gradcam_v2_penultimate_token_local` | `compute_gradcam(..., layer_index=-2, weighting="token_local")` | penultimate block, per-patch weighting |

These variants are technically compatible because the ViT is a torchvision
`VisionTransformer` with encoder blocks that expose `ln_1`, patch-token
activations and gradients.

Implementation requirements:

1. Use float32 input normalized to [0,1], 224x224, NCHW.
2. Use the true target class as the Grad-CAM target.
3. For joint models, pass the frozen observed clinical cue to the forward pass.
4. For image-only models, pass no clinical category.
5. Place models, images and targets on `cuda:0`.
6. Execute all four methods in identical patient batches and freeze
   `deterministic_algorithms=true`.
7. Reuse the existing `MapBatch` validity rules.
8. Store raw maps, valid masks, invalid reasons and gradient/activation
   diagnostics.
9. Validate map shape `(N,14,14)`, finite values, nonnegative values and
   positive support.

The existing v2 methods are the recommended implementation base.

### 2.2 Additional explanation methods

Keep Chefer, attention rollout and occlusion as separate prespecified
secondary explanation methods. Their names are already implemented:

- `chefer`;
- `attention_rollout`;
- `occlusion`.

Do not merge Chefer/rollout/occlusion with the Grad-CAM family.

### 2.3 Reproducibility

Store:

- method ID;
- target class;
- model/checkpoint SHA-256;
- temperature;
- input patient IDs;
- device;
- torch/torchvision/scikit-image versions;
- batch size;
- validity counts;
- map and invalid-reason arrays.

Repeat a deterministic smoke batch as a post-implementation test: same seed,
same checkpoint and same input must produce bitwise or strictly equal maps
within recorded numeric tolerance.

## 3. Faithfulness pipeline

The existing v2 independent superpixel implementation is the best fit and
already implements:

- SLIC 128-region connected segmentation;
- 21 affected-area fractions;
- 20 random patient-specific region orders;
- center-first control;
- patient-level curves and AUC;
- paired patient bootstrap summaries;
- independent separation from marker intervention.

Reuse it, but update the v2.2 wrapper to preserve exact protocol baselines.

### 3.1 Superpixel settings

Use:

- `n_segments=128`;
- `compactness=10.0`;
- `sigma=0`;
- `spacing=None`;
- `convert2lab=False`;
- `channel_axis=-1`;
- `enforce_connectivity=True`;
- `min_size_factor=0.5`;
- `max_size_factor=3`;
- `slic_zero=False`;
- `start_label=0`;
- fixed seed `20261144`;
- scikit-image version in every run manifest.

### 3.2 Baselines

Implement exactly two named baselines:

1. `renderer_background`: the patient-specific
   `background_canvas` image;
2. `mid_gray_128`: a 224x224 RGB array filled with 128.

Do not use Gaussian blur as a replacement baseline in the primary
faithfulness comparison. It may be retained only as an exploratory diagnostic.

### 3.3 Perturbation procedure

For each valid attribution map:

1. Segment the patient image into 128 connected superpixels.
2. Rank superpixels by attribution sum.
3. Generate 21 affected-area fractions from 0 through 1.
4. Deletion: replace ranked superpixels with each baseline.
5. Insertion: replace non-ranked pixels with the baseline and restore ranked
   superpixels.
6. For random controls, perform 20 seeded random region permutations.
7. Generate center-first order from superpixel centroid distance to image
   center.
8. Retain curves for each patient, baseline, map method and direction.

### 3.4 Metrics

The patient-level statistic is trapezoidal AUC of target-class probability
against affected pixel fraction.

For deletion:

- lower AUC indicates faster target probability loss;
- compare method AUC against random and center controls.

For insertion:

- higher AUC indicates faster target probability recovery;
- compare method AUC against random and center controls.

Report:

- method mean AUC;
- random-order mean AUC;
- center-control AUC;
- patient-paired method-minus-random difference;
- patient-paired method-minus-center difference;
- 95% patient bootstrap intervals;
- valid patient count and invalid map count.

Random orders are averaged within patient before the patient bootstrap.

### 3.5 Computational cost

Run faithfulness in patient chunks of 8 or 32, inference batch 64,
GPU-only on `cuda:0`, and write intermediate NPZ files atomically.
Cache the SLIC label array per patient. Cache model logits only for identical
region/baseline variants; do not cache them across unrelated conditions.
Keep 20 random orders and 21 steps as protocol constants.

## 4. Independent localization and geometry

Use the existing v2/v2.1 localization correction:

- maps are valid only after upstream validity and local nonfinite/negative/
  zero-map checks;
- lesion, marker and background shares sum to one;
- unavailable translated controls remain unavailable;
- valid zero-mass controls remain finite zero observations;
- no random-pixel substitution fills unavailable geometry;
- patient means and aggregate denominators are availability-aware;
- random-pixel area-matched masks remain an exploratory null.

## 5. Evaluation cohort

Implement a separate cohort-manifest gate, not a selection inside existing
v1/v2 outputs.

Recommended manifest fields:

```text
schema_version
cohort_name
master_seed
patient_ids
labels_by_patient
condition_independent_geometry
partition
class_counts
development_overlap_sha256
cohort_membership_sha256
protocol_sha256
config_sha256
authorization_timestamp
approver
test_labels_used=false
```

Before test access, verify:

1. 300 IDs per class.
2. 1,800 unique IDs.
3. no overlap with train, model-selection or calibration IDs;
4. no overlap with the old balanced-pilot test IDs;
5. no overlap with v2 72-patient validation IDs;
6. all IDs and labels hash against the manifest;
7. three frozen checkpoints and calibrators exist;
8. protocol and cohort authorization hashes match.

A test-stage run must refuse if any check fails.

## 6. Memory optimization

The least disruptive solution is a v2.2-only streaming/lazy dataset.

Do not modify `_PatientDataset` in historical `phase_g/model.py`.
Instead add v2.2 classes that expose a deterministic item interface:

```text
SyntheticPatientIndexDataset
- partition
- condition
- patient IDs
- labels
- config
- conflict assignments
- bounded LRU patient cache
- __len__
- __getitem__
```

For `__getitem__`, call existing deterministic `generate_condition_split`
for one patient. Batch DataLoader workers then fetch only the batch's
required patients. The bounded cache should hold at most 64-128 patients,
configurable but frozen for an execution.

For complete control over worker process memory:

- use `num_workers=0`;
- keep generation on the orchestrator process;
- batch up to 64;
- stream one batch at a time;
- free lists/tensors after each batch;
- persist a deterministic patient cache only if disk space permits.

Do not keep all 12,000 rendered images in a list. Do not pre-render test
patients before the frozen authorization gate.

Memory-mapped `.npy` or `.npz` caches are alternatives, but add a derived
data artifact and require manifesting, cache invalidation and cleanup.
Streaming generation is smaller and preserves the existing deterministic
patient renderer.

## 7. Implementation sequence

### Phase A — protocol-only changes

1. Add the approved protocol fields and exact test cohort size.
2. Replace the legacy family with ten primary contrasts.
3. Freeze the practical threshold or mark it null pending approval.
4. Freeze explanation method IDs.
5. Update the versioned execution authorization schema.

No patient data or test access is enabled in this phase.

### Phase B — testable explanation improvements

1. Add a v2.2 wrapper around `experiments.phase_g_v2.cam.compute_gradcam`.
2. Map the four protocol IDs explicitly.
3. Add deterministic map smoke tests and invalid-map tests.
4. Add checkpoint, device and input-shape tests.
5. Run unit tests only.

### Phase C — faithfulness integration

1. Wrap `independent_superpixel_faithfulness`.
2. Add exact protocol names for renderer-background and mid-gray.
3. Add SLIC metadata to manifests.
4. Add invalid-map denominator tests.
5. Add 128-SLIC and 20-order deterministic tests.
6. Compare against the existing v2 implementation.

### Phase D — lazy data and cohort gate

1. Add lazy v2.2 dataset generation.
2. Add bounded in-memory cache.
3. Add cohort manifest generation/validation APIs.
4. Add test for 300 IDs/class and no overlap.
5. Add test that test generation is impossible before authorization.

### Phase E — statistical integration

1. Replace the legacy fixed two-contrast primary family with ten.
2. Add 10,000 paired bootstrap support.
3. Refit calibration temperatures inside bootstrap replicates.
4. Add Holm-adjusted one-sided bounds.
5. Separate primary, secondary and exploratory result structures.

### Phase F — regression and execution lock

1. Run v1, v2, v2.1 and v2.2 suites.
2. Recheck all protected hashes.
3. Verify no condition directories or test archives were created.
4. Refresh reports and manifests.
5. Keep `execution_enabled=false` until explicit authorization.

## 8. Automated acceptance tests

Before any experiment authorization:

- exact four Grad-CAM variants reproduce and have valid method IDs;
- joint Grad-CAM receives clinical categories;
- image-only Grad-CAM does not consume categories;
- SLIC settings and seed reproduce identical labels;
- superpixel count and compactness are manifest fields;
- both protocol baselines are present;
- invalid maps are excluded and counted;
- unavailable geometry is not converted to a random mask;
- valid zero controls remain finite zero observations;
- primary family contains exactly ten contrasts;
- bootstrap refits only calibration logits;
- test labels cannot be loaded before frozen models/calibrators;
- cohort contains 300 unique IDs/class and no development overlap;
- historical artifacts remain unchanged;
- lazy rendering is deterministic and memory-bounded;
- no full condition or test record is generated.

## 9. Statistical implementation additions

### 9.1 Primary bootstrap

Implement a deterministic bootstrap API accepting test logits, test labels,
calibration logits, calibration labels, condition IDs, 10,000 replicates and
fixed test/calibration seeds.

For each replicate:

1. Independently resample calibration patients within class.
2. Refit all three scalar temperatures.
3. Resample held-out patients within class.
4. Use the same test row index for image-only, clinical-only and joint.
5. Recompute both modality-versus-joint contrasts.
6. Recompute all ten primary contrasts.

Never refit temperature on test labels. Save raw bootstrap draws and
temperature-refit sensitivity diagnostics.

### 9.2 Adjusted inference

Implement:

- centered-bootstrap one-sided p-values;
- Holm-adjusted p-values across exactly ten contrasts;
- bootstrap-inversion one-sided adjusted lower bounds;
- marginal percentile intervals separately;
- no labeling of percentile intervals as simultaneous.

Implement three success functions:

- contrast: adjusted lower bound > zero;
- condition: both modality contrasts pass;
- experiment: all five condition rules pass.

### 9.3 Power and cohort validation

Add an isolated read-only simulation module. It must not generate or inspect
experiment patients and must not write condition outputs.

The module should accept:

- paired mean NLL effect;
- paired NLL SD;
- per-class n;
- number of conditions;
- two contrasts per condition;
- target power;
- practical threshold;
- correlation between contrast statistics;
- seed.

It should return contrast-level, Holm-family, condition-level and
complete-experiment power under the documented assumptions.

## FINAL INVESTIGATOR DECISIONS REQUIRED

1. Cohort size and seed.
2. Formal threshold versus descriptive reference.
3. Ten-contrast global Holm family.
4. Both-contrast versus one-contrast condition rule.
5. Inclusion of calibration-temperature resampling in primary bootstrap.
