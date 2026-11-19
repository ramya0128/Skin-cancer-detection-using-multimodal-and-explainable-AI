# Phase G planted-cue benchmark design proposal v2

**Status: APPROVED FOR IMPLEMENTATION AND THE BALANCED PILOT ONLY**

This revision retains the balanced 0.85/0.85 pilot but changes how cue
learning is evaluated, as approved for design revision on 2026-11-19. It
supersedes the v1 proposal below without changing the controlling
specification. The original cue-masking accuracy-drop thresholds remain
reported diagnostics; the new primary incremental-prediction criterion uses
natural-distribution joint-versus-unimodal NLL comparisons. It also adds
formal sample-size, calibration, and multiplicity procedures. No Phase G
pilot data, trained model, or pilot result has been generated. The design and
its companion YAML were approved for implementation and execution of the
balanced pilot only; the other five conditions remain unauthorized.

## Intent and scope

Test whether explanation methods for a controlled image-plus-clinical
classifier locate and faithfully represent known image and clinical cues, and
whether the model's predictions depend on those cues. The pilot is synthetic
only. Its results cannot establish clinical validity or explain PAD-UFES-20
lesions.

The pilot answers a narrower question than “do explanations identify
meaningful lesion regions?” In the approved pilot, lesion appearance is
independent of class. Therefore a faithful explanation should not systematically
assign class evidence to lesion morphology; the planted predictive signals are
the exterior image marker and the clinical category. The pilot can test
localization to those signals and resistance to non-predictive lesion and
background regions. It cannot establish that real lesion morphology is
meaningful or that an explanation recognizes cancer pathology. A
lesion-predictive condition would be a separate future experiment requiring
its own approved protocol.

## Existing state and approval boundary

The controlling specification is
`docs/superpowers/specs/2026-11-18-controlled-experiments-design.md`.
It selects procedural lesion canvases, exact masks, patient-level splits, and
one initial balanced pilot with both cue reliabilities at 0.85. Its fixed gate
is:

1. Full held-out accuracy is at least 0.80.
2. Independently masking the image cue reduces held-out accuracy by at least
   0.05.
3. Independently masking the clinical cue reduces held-out accuracy by at
   least 0.05.

This v2 proposal preserves the full-accuracy threshold as a minimum model
performance condition. It retains both original mask accuracy-drop
thresholds in the report, but they are **diagnostics, not mandatory
cue-learning or expansion gates**. The primary cue-learning criterion is
specified below as the multiplicity-adjusted lower confidence bound for
each natural-distribution NLL gain exceeding 0.10 nats per patient. This is a
documented methodological departure from the controlling specification;
that specification is not edited by this proposal.

**Rationale for separating the original accuracy diagnostic:** with uniform
six-class labels and two
conditionally independent, symmetric 0.85-reliable cues, the Bayes-optimal
classifier has 0.85 accuracy with both cues and with either cue alone. When
the cues agree, it predicts their shared value; when they disagree, the two
indicated classes have equal posterior probability. Thus the population
Bayes-optimal accuracy is 0.85 with both cues and with either cue alone;
accuracy alone cannot measure their incremental probability information.
Moreover, the frozen joint model's accuracy after masking depends on its
tie-breaking behavior under the intervention distribution. The original
mask thresholds therefore remain useful intervention-sensitivity
descriptions, but are not evidence by themselves that each cue contributes
unique 0-1 accuracy.

### Approved v2 departure: primary NLL estimand

For each held-out synthetic patient, evaluate the calibrated joint and
unimodal model probabilities on the **same unmodified test observation**.
Define:

- `Delta_image = NLL(image_only) - NLL(joint)`;
- `Delta_clinical = NLL(clinical_only) - NLL(joint)`.

Positive values mean the joint model improves held-out probabilistic
prediction. Each contrast must have a multiplicity-adjusted one-sided 95%
lower confidence bound **greater than 0.10 nats per patient**. Both must pass.
The 0.10-nat threshold is a practical minimum, not a significance-only
threshold. Exact estimation, bootstrap, and Holm procedures are in
Statistical analysis.

This is a deliberate departure from the original three-part gate: full
accuracy `>= 0.80` remains a performance floor; the original two
`>= 0.05` mask accuracy drops are retained as diagnostics; the NLL contrasts
become the primary evidence for incremental predictive contribution.
Masking-based NLL changes are also reported separately as intervention
sensitivity and are not substituted for the natural-distribution contrasts.

For the specified six-class symmetric channel, let `q=0.85` and
`r=(1-q)/5=0.03`. With both cues, the probability of agreement is
`q^2+5r^2=0.727`; the posterior probability of the shared class is
`q^2/(q^2+5r^2)=0.993810`. On disagreement, the posterior for either
reported class is `qr/(2qr+4r^2)=0.467033`. The resulting Bayes accuracy is
exactly `0.85`, while exact enumeration gives Bayes NLL as the posterior entropy
averaged over agreement and disagreement, `0.302653` nats. With either cue
alone, Bayes accuracy is `0.85` and NLL is
`-q ln(q)-5r ln(r)=0.664125` nats. Thus the ideal natural-distribution NLL
gain is `0.361472` nats for either cue. These are generator-level reference
values, not expected measured results for a finite trained network.

At the initial design review, repository inspection found no Phase G
generator, tests, dataset, checkpoint, run configuration, or
`outputs/phase_g*` directory. Phase F artifacts and frozen v1 inputs are
outside this experiment and must remain unchanged. As of the current
implementation update, deterministic generation/IO and the controlled model
and primary-statistics code are implemented and covered by 25 passing Phase G
unit tests; no synthetic pilot data, trained model, or held-out result exists
yet.

## Goals and non-goals

### Goals

- Generate six balanced synthetic labels matching the PAD-UFES-20 class
  cardinality, without using PAD images, metadata, or trained checkpoints.
- Make lesion, exterior-marker, and background masks exact by construction.
- Make the image and clinical cues independently 0.85 reliable in the pilot.
- Prevent patient, template, or derived-view leakage across train, validation,
  and test partitions.
- Evaluate cue reliance, explanation localization, explanation faithfulness,
  explanation stability, clinical attribution, and fusion attribution as
  distinct outcomes.
- Implement configurable definitions for all six conditions in the prior
  specification, but initially execute only the balanced 0.85/0.85 condition.
- Save deterministic configs, model and method versions, logs, schemas, and
  hashes in a synthetic-only, versioned output tree.

### Non-goals

- No Phase F rerun, model replacement, or use of the real test split.
- No claim that a synthetic cue benchmark validates real-world medical
  reasoning or clinical performance.
- No claim that this pilot establishes lesion-related explanation quality.
- No expansion to image-dominant, clinical-dominant, redundant, conflicting,
  or synergistic conditions without a separate decision after the pilot.
- No tuning, explainer selection, or threshold selection on synthetic test
  patients.

## Unit of generation, sample size, and split

**Proposed sample:** 12,000 synthetic patients, one paired image and one
categorical clinical observation per patient, with six equally frequent
classes. The 1,800-patient test count is chosen by a prespecified precision
target and checked against the NLL-gate power calculation below; it is not
treated as an inherited fixed number. Split patients once, before rendering
or generating any variants:

| Partition | Patients | Per class |
|---|---:|---:|
| Train | 8,400 (70%) | 1,400 |
| Validation | 1,800 (15%) | 300 |
| Test | 1,800 (15%) | 300 |

The test set has 300 independent patients per class. Its size is justified by
two prespecified requirements:

1. **Overall accuracy precision:** target a two-sided 95% margin of error no
   larger than 2 percentage points near the gate value `p=0.80`. The usual
   binomial planning approximation gives
   `n = 1.96^2 * 0.80 * 0.20 / 0.02^2 = 1,537`. Rounding up to an equal
   six-class allocation of 300 per class gives `n_test=1,800`; the pooled
   margin is approximately 1.85 percentage points at `p=0.80` and 1.66
   points at `p=0.85`. At 300 per class, class-specific margins are about
   4.5 points at `p=0.80` and 4.0 points at `p=0.85`; these class estimates
   are descriptive, not separately powered confirmatory tests.
2. **NLL-gate precision/power:** under the exact generator, the oracle
   joint-versus-single-cue paired NLL gain has exact expected value
   `0.361471705` nats and theoretical paired SD `0.972498137` nats per
   patient. For conservative planning, use Bonferroni's per-comparison
   one-sided `alpha=0.025` as an upper bound on the Holm threshold for the
   two cue contrasts. To have 80% power for the lower bound to exceed the
   practical floor `0.10`, the normal-approximation requirement is
   `n = ((z_0.975 + z_0.80) * 0.972498137 / (0.361471705 - 0.10))^2`,
   `108.5765`, rounded up to 109 patients. The proposed 1,800 therefore
   substantially exceeds the oracle-based NLL requirement. At 1,800, the
   oracle paired standard error is `0.022922` nats; its conservative expected
   lower bound is `0.316545` nats, above the 0.10 floor. Under this
   approximation, the design has at least 80% power when the true gain is
   about `0.164218` nats or
   larger. This calculation is conditional on the specified generator and
   oracle variance; fitted-model variance may differ, so it is not a
   guarantee.

   For reproducibility, the oracle paired log-loss-gain distribution for
   either contrast is: both cues correct (probability `0.7225`, gain
   `0.1563`); only the unimodal cue correct (`0.1275`, gain `-0.5988`);
   only the added cue correct (`0.1275`, gain `2.7452`); both wrong but
   reporting the same wrong class (`0.0045`, gain `-3.1872`); and both
   wrong with distinct reports (`0.018`, gain `-0.5988`). Its mean is
   `0.361471705` and its standard deviation is `0.972498137`. These
   calculations use the exact six-class channel and Bayes posterior
   probabilities, not simulated pilot results.

The test size is thus driven primarily by the pooled accuracy precision goal
and class-balanced reporting, while also providing ample NLL precision for an
oracle-scale effect. It does not guarantee power for arbitrarily small
effects, model-seed variation, or every explanation endpoint. A single fixed
training seed does not estimate model-seed variability and must be stated as
a limitation.

There is exactly one base observation per patient. Any transformed images,
cue replacements, or repeated explainer baselines inherit that patient's ID
and partition. No patient-specific base canvas, texture seed, or derived
variant may occur in multiple partitions. Per-patient random streams are
derived from the master seed, patient ID, and named random stream, not
generation order. Create the balanced label list first, use a dedicated
master-seed split stream to shuffle IDs within each class into the fixed
70/15/15 counts, then assign opaque patient IDs with a separate seeded
permutation. Neither patient ID nor its rendering seed is derived from the
class label.

Six synthetic class indices are used only to preserve the current task's
six-class structure. They do not represent actual PAD patients or
observations. Class identity is independent of all lesion and background
rendering variables.

### Rationale for proposed defaults

| Default | Rationale and limitation |
|---|---|
| Master seed `20261119` | A fixed, date-stamped seed is distinct from Phase F's seed 42 and makes the synthetic run reproducible. It does not estimate seed-to-seed variability. |
| Six equally frequent labels | Preserves the project's six-class task shape without using real patient records; it avoids rare-class noise in the initial gate. |
| 12,000 patients; 70/15/15 split | 1,800 test patients provide about ±1.85 percentage-point pooled accuracy precision at 0.80, 300/class for descriptive class estimates, and high precision for the 0.10-nat NLL floor under the oracle variance. |
| 224-pixel canvas; 16-pixel patches | Matches the existing ViT input and creates a 14 x 14 map so mask overlap and deletion steps can be evaluated at the current method resolution. |
| Central lesion and corner marker | The proposed lesion bounds leave at least a 16-pixel gap to every allowed marker patch, while randomizing marker corner prevents location from encoding class. Geometry is synthetic, not a medical segmentation prior. |
| Shared texture/noise distributions | Add visual variation without carrying class information. The exact noise levels are engineering defaults for a readable procedural image and have no dermatological calibration. |
| Equal-area binary glyphs | Fixed equal-count, equal-luminance glyphs reduce color/area confounding; the glyph identity is the intended image cue. |
| 0.85 cue reliability; wrong class uniform among the other five | Preserves the approved pilot reliability and gives every incorrect class the same probability. Image and clinical noise are conditionally independent. |
| 20 marginal-replacement draws per patient | Estimates intervention sensitivity to neutralizing the cue-label association without treating repeated draws as independent patients. Mask accuracy changes remain diagnostics, not the primary NLL criterion. |
| Three independent model functions | Image-only, clinical-only, and joint results can be reported separately. A compact scratch ViT supports the project's spatial methods; a learned categorical encoder avoids ordinality; a joint MLP is necessary for later synergy. |
| AdamW and fixed training schedule | A simple fixed starting recipe avoids test-driven tuning. Validation NLL/patience selects checkpoints; these hyperparameters remain proposed, not empirically optimized. |
| 200 training donors and 20 random orders | 200 donors provide 33/34 examples per class for interventional values and match Phase F's background size; 20 orders match the existing random-order control. Neither is asserted to be uniquely optimal. |
| 1,800 held-out patients | Meets the pooled ±2-point accuracy precision target, gives 300 patients per class, and substantially exceeds the oracle-based NLL sample-size requirement. |
| 10,000 NLL bootstrap replicates | Stabilizes the one-sided tail probabilities and Holm-inverted lower bounds; bootstrap sampling is paired and stratified by class. Other descriptive intervals use 2,000 patient bootstrap replicates. |
| 10,000 sign flips | Reuses the Phase F uncertainty practice for paired secondary continuous contrasts; add-one p-value resolution is approximately `1/10,001`. |
| True class as primary attribution target | Keeps the target fixed across interventions and includes model errors, avoiding an explanation evaluation restricted to correctly classified cases. Predicted-class targeting is only a sensitivity analysis. |

## Proposed image and cue generator

### Canvas and lesion

- Render RGB uint8 images at 224 x 224 pixels. This retains the project's
  existing image scale and supports a 14 x 14 map at patch size 16.
- First generate a patient-specific background-only canvas: RGB base level
  224, plus independent per-channel Gaussian noise sampled on a 16 x 16 grid
  with standard deviation 3 and bicubic-upsampled to 224 x 224. Clip to
  `[0, 255]`. Save this exact canvas for controlled insertion/deletion.
- Draw a stochastic, irregular elliptical lesion around the canvas center.
  Center coordinates are independently uniform in `[96, 128]`; horizontal
  and vertical radii are independently uniform in `[36, 52]` pixels. Make a
  64-point ellipse polygon and perturb its radial scale by
  `1 + sum(a_k * sin(k*theta + phase_k))`, for `k=2..5`, with each `a_k`
  sampled uniformly from `[-0.02, 0.02]` and each phase uniformly from
  `[0, 2*pi]`. Rasterize this polygon to make the exact binary lesion mask.
- Sample each lesion's base RGB color uniformly and independently per
  channel from `[100, 190]`. Add class-independent per-channel texture noise
  sampled on a 16 x 16 grid with standard deviation 5 and bicubic-upsampled.
  Composite the lesion over the saved background-only canvas.
- Draw lesion color, texture, boundary perturbation, and background texture
  from class-independent distributions. Do not encode class in lesion shape,
  color, area, position, texture, or background. Patient IDs and rendering
  random streams must be assigned independently of labels.
- Save an exact binary lesion mask from the rasterization operation, before
  any image normalization. The mask is ground truth, not a segmentation
  prediction.

The central, variable-size lesion creates a nontrivial region for localization
without adding label information. The exact coordinate and radius ranges are
proposal defaults and require approval.

### Exterior image cue

- Place a 16 x 16 pixel marker patch at one of the four image corners,
  selected uniformly and independently of class and lesion appearance.
- Use six fixed binary glyphs, one per class, with equal foreground pixel
  counts and equal luminance. The six 4 x 4 binary codewords are two
  foreground cells in disjoint positions:
  `1100/0000/0000/0000`, `0011/0000/0000/0000`,
  `0000/1100/0000/0000`, `0000/0011/0000/0000`,
  `0000/0000/1100/0000`, and `0000/0000/0011/0000`.
  Expand each code cell to 4 x 4 pixels. Render foreground black and
  background white, so glyphs have equal area and mean luminance. This
  avoids color and area acting as unintended class channels.
- The cue mask is the exact 16 x 16 patch. The lesion-mask check must verify
  a minimum 16-pixel gap between cue and lesion masks for every generated
  sample.
- The observed image cue equals the class with probability 0.85. With
  probability 0.15, choose uniformly among the other five cue classes.

Randomizing marker location prevents location from encoding class; equal-area,
equal-luminance glyphs isolate glyph identity as the signal. Requiring a
geometric gap makes the “outside lesion” property mechanically testable.

### Clinical cue

- Generate one categorical value in `{0, 1, 2, 3, 4, 5}`. Do not treat the
  value as ordinal; encode it categorically.
- The observed category equals the class with probability 0.85. With
  probability 0.15, choose uniformly among the other five categories.
- Image-cue and clinical-cue noise draws are independent conditional on the
  class. The cue values are not copied from image-rendering metadata.

Uniform error among the other classes makes the nominal reliability
unambiguous for a six-class task and avoids favoring any wrong class.

### Pilot-only counterfactual cue masking

The gate's mask operator is proposed as **label-independent marginal
replacement**, not blacking out pixels or inserting an unseen sentinel:

- Image: keep the same image, marker position, mask, lesion, and background;
  replace only the glyph with one drawn uniformly from the six glyphs,
  independently of the true label and original glyph.
- Clinical: replace only the categorical value with an independent uniform
  draw from the six categories.
- Use 20 deterministic replacement draws per held-out patient and average
  each patient's masked outcome before computing the gate and statistics.

This keeps both intervention values on the generator's support and isolates
the cue-label association rather than an artificial black patch or an
untrained “missing” value. Averaging repeated draws reduces Monte Carlo noise;
the patient, not each replacement draw, remains the statistical unit. It is
an operational definition of masking that needs explicit approval because
the original specification did not define the operator.

The replacement is on the finite cue-value support, but it does not preserve
the natural joint distribution: cue agreement falls from `0.727` to `1/6`
when either cue is independently replaced. Consequently, the frozen model's
masked accuracy, probability, or NLL can reflect distribution-shift
sensitivity and tie-breaking as well as reliance on the removed cue. These
are reported as intervention-sensitivity diagnostics, not as the primary
natural-distribution incremental-NLL estimand.

Also report a literal cue-absence sensitivity analysis (restore the
pre-marker background pixels; use a clinical `MASKED` token) separately and
label it as an out-of-support intervention unless training contains masked
examples. In the implementation, reserved category index 6 maps
deterministically to a zero clinical feature vector; it is never trained or
calibrated. This analysis does not replace the primary gate and must not be
used to pass or fail it.

## Condition semantics

Only the balanced condition is authorized for the initial execution. The
following are proposed definitions for future configuration; their values are
not authorization to run them.

| Condition | Proposed generation semantics | Why it is distinct |
|---|---|---|
| Balanced pilot | Independent image and clinical cues, each 0.85 accurate. | Tests two comparably reliable, separately available signals. |
| Image-dominant | Independent cue errors; image reliability 0.95, clinical 0.60. | One modality is deliberately more informative. |
| Clinical-dominant | Independent cue errors; image reliability 0.60, clinical 0.95. | Reverse dominance control. |
| Redundant | Draw one noisy class code at 0.85 reliability and encode that same code in both modalities. | Errors are perfectly correlated; the second modality duplicates rather than independently adds evidence. |
| Conflicting | In a predeclared 50% subset, one cue is correct and the other is a uniformly selected decoy, with orientation balanced; in the remaining 50%, both cues are independently noisy at 0.85. | Exactly one cue is correct per conflict patient; the marginal accuracy is 0.675 for each cue and will be reported rather than mislabeled as 0.85. |
| Synergistic | Draw image share `U` uniformly from six values and clinical share `V = (Y - U) mod 6`; independently corrupt either share with probability 0.15 by uniform replacement with another value. | Each share alone is independent of the label; the pair encodes it. “0.85 reliability” means share-retention probability, not individual label accuracy. This is explicitly not the balanced pilot's reliability semantics. |

These are reproducible proposals, not silently finalized methodology. They
must remain in configuration as `not_authorized_to_run` until separately
approved. In particular, the synergy condition requires a joint model capable
of interactions; a probability-average late-fusion model cannot recover a
label that neither unimodal probability vector contains.

## Proposed multimodal model and explainer compatibility

### Model

Train three new, scratch-initialized functions on the same synthetic training
patients:

- Image encoder: torchvision-compatible ViT with input 224, patch size 16,
  four encoder blocks, hidden width 192, three attention heads, and MLP width
  384. No ImageNet or PAD pretraining.
- **Image-only model:** the ViT and a six-class linear head, trained from the
  image alone.
- **Clinical-only model:** six-category embedding (width 16), a 32-unit GELU
  hidden layer, and a six-class head, trained from the category alone. Never
  feed the category as a scalar.
- **Joint multimodal model:** a separately initialized ViT plus clinical
  embedding (width 16); concatenate the two representations and pass them
  through a two-layer MLP (hidden width 128, GELU, dropout 0.1) to six logits.
  The joint head is needed for the proposed synergistic condition. Do not add
  unimodal auxiliary losses to this model: the synergistic condition has no
  label information in either cue alone, and auxiliary supervision would
  impose a conflicting objective.
- Proposed optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`,
  batch size 64, maximum 50 epochs. Early stopping uses validation fused
  negative log-likelihood (or unimodal validation NLL) with patience 8;
  checkpoint selection is validation only. Use separate deterministic model
  initialization seeds derived from the master seed.

Use the exact same synthetic training patient IDs for all three models.
Apply the same fixed loss type (six-class cross-entropy), optimizer settings,
maximum epochs, batch size, early-stopping patience, and validation-based
checkpoint-selection rule. Image-only and joint models use the same ViT
backbone specification; the clinical-only model uses a categorical encoder
appropriate to its input. Do not add modality-specific auxiliary losses or
choose different hyperparameters by test performance. The models are
independently initialized and trained; comparisons are paired at the patient
level but remain conditional on this model recipe and seed.

Partition the 1,800-patient development set before training into a
900-patient model-selection subset and a 900-patient calibration subset,
stratified to 150 patients per class in each. All models use identical
subsets. Select checkpoints and early-stop using only the model-selection
subset. After checkpoint selection, fit one positive scalar temperature per
model by minimizing multiclass NLL on the calibration subset; freeze the
temperatures before final test evaluation. Use calibrated probabilities for
the primary NLL contrasts, and report uncalibrated NLL as a sensitivity
result. Temperature scaling uses no held-out test labels and does not change
the predicted class except for exact ties.

Separate image-only, clinical-only, and joint functions answer the
independence question directly; these models share no trained weights. The
small ViT is selected to support the project's spatial explainers, while the
joint interaction head can represent later synergy. Three models add training
cost but avoid using auxiliary losses to create apparent branch predictive
power. They still differ from the frozen PAD CWF system. Thus, the pilot
cannot directly validate the frozen CWF model or generalize its results to
PAD-UFES-20.

### Explanation targets (must be reported separately)

1. **Image-only explanation:** Grad-CAM and Chefer target the image-only
   model's fixed class score; occlusion recomputes its output. Attention
   rollout is a class-independent control, not a target-specific attribution.
2. **Clinical-only explanation:** exact categorical intervention/SHAP over
   the six category values for the clinical-only model. A category is one
   variable, not six ordinal values. Use a uniform six-category reference
   distribution and report the observed category's signed fixed-class
   probability difference from that reference, plus its absolute magnitude.
3. **Joint image-input explanation:** Grad-CAM and Chefer gradients target
   the joint fusion class score through the joint image encoder; occlusion
   recomputes the joint output. This map is a pixel attribution for the joint
   model, not an image-only model explanation.
4. **Fusion/modality explanation:** exact two-player interventional Shapley
   values for `{entire image, clinical category}` over the joint fused
   probability, using train-only donor patients and a fixed class across
   coalitions. Separately compute joint-model cue-level Shapley for
   `{image marker, clinical cue}`. Check efficiency for both games.

Image-only maps, clinical-only category attribution, joint-model image maps,
whole-modality Shapley, and cue-level Shapley answer different questions. Do
not call one the other. The exact modality game explains the joint model
under its specified donor-background intervention, not an intrinsic or
data-independent "importance" of a modality.

Apply each model's frozen validation-only temperature to all probability-valued
explanation and intervention outcomes (clinical category values, occlusion
scores, Shapley coalition values, cue sensitivity, deletion/insertion scores,
and stability probabilities). Grad-CAM and Chefer target the fixed-class
logit; temperature is not fitted or changed using held-out patients.

## Evaluation: separate constructs and measures

### A. Cue reliance (model behavior)

Evaluate the joint, image-only, and clinical-only models on the same
unmodified held-out synthetic patients. The primary incremental-prediction
outcomes are the two natural-distribution NLL contrasts defined above. Report
full accuracy and the original `>=0.05` accuracy-drop thresholds as separate
diagnostics.

Separately, apply each primary marginal-replacement mask to the joint model.
For each patient, average the 20 deterministic replacement outcomes before
calculating paired accuracy, fixed-class probability, and NLL changes. Report
these only as intervention sensitivity: marginal replacement changes the
joint cue distribution, and the results are not the natural-distribution NLL
contrasts or the primary cue-learning criterion. Report per-class counts and
results. Cue reliance/intervention sensitivity says whether this model's
output changes under cue neutralization; it does not establish that an
explanation map is localized or faithful.

### B. Explanation localization and attribution distribution

For each applicable image method, separately for image-only and joint-target
maps, compare normalized positive-map mass in:

- exact exterior marker mask;
- exact lesion mask;
- remaining background mask.

Report region mass share, region area fraction, and area-normalized
enrichment. Also report the top-pixel/pointing-game hit for marker and lesion.
Downsample binary masks to the 14 x 14 patch map by fractional pixel
intersection, not nearest-neighbor thresholding. Compare marker and lesion
localization with 20 area-matched random regions per patient, averaged within
patient: marker controls are uniformly sampled 16 x 16 patches that avoid the
lesion and true marker; lesion controls are valid random translations of the
exact lesion mask that avoid the true lesion and marker. Also report
20-seed random-map and center-Gaussian controls. The cue mask and lesion mask
are non-overlapping by construction.

For clinical methods, report whether the attribution goes to the observed
clinical category and how its sign/magnitude changes under category
replacement. For fusion, report image-cue and clinical-cue Shapley
contributions independently.

Localization asks where a map places attribution; high marker localization
does not by itself prove the map is faithful.

### C. Explanation faithfulness (attribution predicts model response)

- Use the true class as the primary fixed target for every held-out patient,
  including model errors. Report predicted-class targeting only as a
  separately labeled sensitivity analysis.
- Evaluate deletion and insertion in 5% patch steps. Deletion starts from
  the original image and replaces ranked 16 x 16 patches with the
  same-coordinate pixels from that patient's saved background-only canvas.
  Insertion starts from that canvas and restores patches from the original in
  ranked order. This baseline has no lesion or cue and may be outside the
  model's training support; also report Gaussian-blur-baseline curves as a
  prespecified sensitivity analysis. Evaluate image-only maps against the
  image-only score and joint-target maps against the joint fusion score.
- Compare each target-specific map ranking (Grad-CAM, Chefer, and occlusion)
  to 20 seeded random orders and center-Gaussian ranking. Use lower deletion
  and higher insertion AUC as directional evidence, not as a standalone
  method ranking. Compute attention-rollout curves as a descriptive
  class-independent control, not as a target-specific faithfulness test.
  Evaluate the same map methods separately for the image-only and joint
  models.
- For the clinical category and exact cue-level Shapley, neutralize/restore
  the category and image marker under the same intervention semantics.

Faithfulness asks whether ranked evidence interventions change the specified
model score as predicted. It is distinct from model cue reliance: an
intervention can show reliance even when an explanation fails to identify the
cue.

### D. Explanation stability

Recompute explanations after horizontal flip (transform masks and maps back)
and brightness factors 0.9/1.1. Report prediction-change counts, then
Spearman and SSIM on the unchanged-prediction subset and valid denominators.
These are stability measures, not localization or faithfulness measures.

### E. Fusion and synergy attribution

Compute both of these prespecified games on the fused fixed-class probability:

- **Whole-modality game:** players are `{entire image, clinical category}`.
  For each of 200 class-stratified training-patient donors, evaluate the four
  coalitions by substituting the donor's full image and/or category wherever
  a player is absent. Use the same donor row for both modalities to preserve
  their joint background distribution.
- **Cue-only game:** players are `{image marker, clinical cue}`. Retain the
  test patient's non-cue image pixels; substitute only donor marker glyphs
  and/or donor clinical categories.

Keep the target class fixed across coalitions and verify
`sum(phi) = v(both present) - v(neither present)` within `1e-5` for each game.
The modality game is sensitive to its full-image donor distribution; the
cue-only game isolates the known planted information. Neither is an intrinsic
or data-independent modality importance.

## Statistical analysis

- Statistical unit: synthetic patient. All three models are evaluated on the
  same 1,800 held-out patients. For every comparison, retain each patient's
  paired predictions; do not compare unpaired aggregate metrics.
- **Primary NLL contrasts:** define the per-patient effects
  `d_image = NLL(image_only) - NLL(joint)` and
  `d_clinical = NLL(clinical_only) - NLL(joint)` from validation-calibrated
  probabilities on unmodified test inputs. Estimate each contrast as the
  mean of its 1,800 paired patient differences.
- **Paired, class-stratified bootstrap:** use 10,000 replicates. In each
  replicate, sample 150 calibration patients with replacement within each
  class from the calibration subset and 300 test patients with replacement
  within each class from the test set. Use the same sampled calibration IDs
  for all three model calibrators and the same sampled test IDs for all three
  model predictions and both NLL contrasts. Refit each model's scalar
  temperature on that replicate's calibration sample, then recompute both
  paired NLL contrasts on the paired test sample. The bootstrap is
  conditional on the trained model weights and selected checkpoints; it
  propagates calibration-set and test-patient sampling uncertainty, but not
  training-seed or generator-seed variability.
- **Holm-adjusted lower bounds:** for each candidate threshold `c`, test the
  two one-sided hypotheses `H0_j: Delta_j <= c` against `Delta_j > c`.
  For each contrast, calculate the centered-bootstrap one-sided p-value
  `p_j(c) = (1 + count(Delta_j* - Delta_hat_j >= Delta_hat_j - c)) / 10001`.
  Sort the two p-values, `p_(1) <= p_(2)`, and apply Holm at familywise
  `alpha=0.05`: adjusted values are
  `p_adj,(1)=min(1, 2*p_(1))` and
  `p_adj,(2)=min(1, max(2*p_(1), p_(2)))`. Invert these same adjusted tests
  over `c` to obtain simultaneous one-sided lower confidence bounds for
  `Delta_image` and `Delta_clinical`. The primary criterion passes only if
  **both adjusted lower bounds exceed 0.10 nats per patient**. Keep the
  bootstrap replicates paired across the two contrasts; Holm, rather than
  separate unadjusted intervals, controls the two-contrast family.
- Report the paired-bootstrap point estimates, adjusted lower bounds, raw
  and Holm-adjusted p-values at `c=0.10`, and calibrated and uncalibrated
  NLLs. Use 10,000 replicates so tail resolution is adequate for the
  one-sided familywise test. Calibration and bootstrap procedures are
  frozen before test labels are accessed.
- For the original cue-mask accuracy diagnostics, average the 20 randomized
  replacement outcomes within patient first. Report paired risk differences
  and patient-bootstrap intervals. Do not treat replacement draws as
  independent or apply McNemar to the averaged outcome. Report masked
  probability/NLL changes separately as intervention sensitivity.
- For other descriptive metrics, report patient-bootstrap percentile
  intervals (2,000 replicates, fixed seed). For prespecified paired
  continuous method contrasts, report mean paired effects, intervals, and
  two-sided patient-level sign-flip p-values (10,000 draws, add-one
  correction).
- Apply Holm correction within predeclared families: the two cue-mask
  comparisons; four image-method marker-localization comparisons and four
  lesion-localization comparisons for each of the image-only and joint
  targets, with each region compared against its area-matched random regions;
  and faithfulness comparisons. The two primary NLL contrasts form their own
  Holm family, as specified above. For faithfulness, use one family per model
  target and direction: image-only deletion/insertion and joint-fusion
  deletion/insertion each contain 6 contrasts (three target-specific methods
  against random-order and center-Gaussian baselines). Attention rollout is
  not a target-specific faithfulness method. Stability estimates are
  descriptive with intervals; no stability hypothesis tests are used in the
  pilot. Class-stratified estimates are descriptive with intervals; do not
  choose a method from a favorable class.
- The full-accuracy floor and NLL incremental-contribution criterion are the
  v2 decision criteria. The original mask accuracy-drop thresholds remain
  diagnostics and cannot override either criterion.
- Report effect sizes in natural units: accuracy percentage-point risk
  difference; natural-distribution NLL gain in nats per patient; intervention
  probability/NLL change; marker/lesion attribution mass difference;
  enrichment ratio; and paired map similarity. Include class and patient
  denominators.
- A single deterministic pilot supports inference conditional on this
  generated sample and training seed only. It does not quantify variation
  across render seeds, model seeds, or real-world populations.

The family definitions and ordering must be frozen in the run config before
the held-out results are inspected.

## Failure and stop rules

- Validate renderer invariants, mask geometry, class balance, cue reliability,
  split disjointness, and model input/output schemas on synthetic training
  data before any pilot training.
- If training or validation fails, debug only on synthetic train/validation
  patients. Do not inspect held-out results during debugging.
- Run the held-out pilot once. Evaluate the full-accuracy floor and both
  multiplicity-adjusted NLL lower bounds as the v2 criteria; report the
  original mask accuracy drops as diagnostics regardless of outcome. A pass
  permits a separate plan for later conditions; it does not complete Phase G.
- If either v2 criterion fails, save the result and stop. Diagnose using only
  train/validation evidence, propose any correction as a new version, and
  obtain approval before rerunning or expanding. A mask-accuracy diagnostic
  below 0.05 alone is not a v2 failure.
- Never overwrite a prior config, run, checkpoint, or result directory.

## Proposed directory structure

```text
docs/superpowers/specs/
  2026-11-19-phase-g-planted-cue-design.md       # this proposal
  2026-11-19-phase-g-pilot-config-proposal.yaml  # non-executable proposal

experiments/phase_g/                              # create only after approval
  config.json
  generate.py
  model.py
  explain.py
  evaluate.py
  tests/
    test_generation.py
    test_splits.py
    test_model_explainers.py
    test_metrics.py

outputs/synthetic_phase_g_v1/                    # separate from real-data outputs
  pilot_balanced_085_085_v1/
    config.json
    environment.json
    dataset_manifest.json
    masks/
    checkpoints/
    predictions/
    explanations/
    metrics/
    figures/
    execution.log
    manifest.json
```

The output tree is proposed only. It must not be created until design approval.
No Phase F output path is reused.

## Proposed testing strategy (before execution)

1. **Renderer invariants:** image size/dtype; exact binary lesion and cue
   masks; no overlap and at least 16-pixel gap; cue patch at allowed corner;
   mask area and glyph equality; all masks within image bounds.
2. **Statistical generation semantics:** deterministic sample-by-ID output;
   exact balanced label counts; empirical cue reliability validated on a
   sufficiently large training-only generation; independent image/clinical
   noise streams; lesion/background rendering variables independent of
   class by construction and checked by a train-only audit.
3. **Leakage and model comparison:** patient-ID sets disjoint; all generated
   variants retain their parent partition; no duplicated base seeds across
   partitions; all models train and validate on identical patient partitions;
   all primary comparisons use the same test patients; test labels cannot
   feed training, checkpoint selection, or temperature fitting.
4. **Calibration:** temperature is positive, fit per model only on the
   disjoint validation-calibration subset, frozen before test access, and
   applied consistently to primary NLL comparisons. Verify calibration
   outputs are finite and normalized; save fitted temperatures and validation
   partition hashes.
5. **Model/explainer compatibility:** output shapes and probability
   normalization; deterministic inference; ViT map dimensions; target-specific
   fusion gradients; attention-rollout labeled non-target-specific; clinical
   categorical attribution does not impose ordinal effects.
6. **Intervention/faithfulness metrics:** neutral replacement changes only
   the chosen cue; mask draw averaging groups by patient; deletion/insertion
   endpoints; exact two-cue Shapley efficiency on an additive toy function and
   efficiency for saved pilot rows.
7. **Saved outputs:** schema, finite values, class/patient IDs, complete
   coverage, config/source/environment hashes, and before/after hashes for
   protected real-data inputs.

Passing these tests validates code and artifacts only; it is not evidence that
the pilot gate passed.

## Known limitations and interpretation

- Exact masks provide synthetic spatial ground truth, but the renderer is not
  photorealistic and may be easier than dermatology images.
- The balanced pilot's lesion is deliberately non-predictive. It cannot
  establish that methods identify biologically meaningful lesion features.
- Cue masking by marginal replacement measures model response to loss of
  cue-label information. It is not proof of causal behavior outside the
  synthetic generator.
- The model differs from the frozen PAD CWF architecture; no direct
  generalization claim is supported.
- The model uses one training seed; bootstrap intervals do not capture
  generator-seed or training-seed variability.
- The random-replacement gate, cue localization, map faithfulness,
  clinical attribution, and fusion Shapley answer different questions and
  must not be conflated.

## Approval and revision record

### Revision record

This v2 design incorporates the user's 2026-11-19 approvals to:

- retain the original 0.05 cue-mask accuracy drops as diagnostics rather than
  mandatory cue-learning gates;
- use natural-distribution joint-versus-unimodal NLL as the primary
  incremental-prediction measure;
- require each Holm-adjusted one-sided 95% lower confidence bound to exceed
  `0.10` nats per patient;
- justify the held-out sample size by accuracy precision and NLL power;
- control training, validation, and calibration across the three models;
- keep cue-mask intervention sensitivity, explanation localization, and
  explanation faithfulness as distinct analyses.

This revision does not alter the balanced 0.85/0.85 cue-generation design.
It documents a specific departure from the original accuracy-only
three-part gate; the controlling specification remains untouched.

The user approved this v2 design, its companion configuration, and the
implementation plan on 2026-11-19, authorizing only the balanced pilot. The
exact-enumeration correction to the oracle NLL and power values is a numerical
correction; it does not change the design, sample-size rationale, or decision
criteria.
