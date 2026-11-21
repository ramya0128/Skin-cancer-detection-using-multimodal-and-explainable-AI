# Phase G v2.2 prospective protocol review

**Date:** 2026-11-21  
**Document status:** Recommended methodology for investigator review; not approved.  
**Execution status:** No condition is authorized and no remaining-condition data or test inference may run.

## 1. Existing protocol assessment

The existing prospective protocol has the right partition structure and core
endpoint: calibrated multiclass NLL is a scientifically sensible primary
outcome, and paired patient-level inference is appropriate for the six-class
synthetic task.

The protocol must be changed in these areas:

1. The 12,000-patient split has no unused patients available for a genuinely
   fresh holdout. All 2,000 patients per class are already assigned to train,
   model selection, calibration or the existing test partition.
2. The existing balanced-pilot test partition and the v2 72-patient subset
   cannot be described as fresh remaining-condition holdouts.
3. The legacy 0.80 accuracy floor and fixed two-contrast gate are not
   scientifically justified primary criteria. They can remain exploratory
   summaries, not success criteria.
4. The protocol's exact explainability family names and the available
   implementation are not reconciled.
5. The execution gate requires a protocol, cohort, and threshold decision,
   and it does so correctly. It should remain fail-closed.

## 2. Recommended primary outcome

The primary outcome is **patient-level calibrated multiclass negative log
likelihood (NLL)** on a frozen, new untouched cohort.

For patient i, with calibrated joint probability p_ij(K) for true class Y_i:

```
NLL_i(model) = -log(p_ij(Y_i))
```

For each condition c, define two incremental effects:

```
Delta_image,c = mean(NLL_image-only,c - NLL_joint,c)
Delta_clinical,c = mean(NLL_clinical-only,c - NLL_joint,c)
```

Positive values indicate lower joint NLL and therefore better joint calibration.
The estimand is the mean paired difference in held-out multiclass NLL for
calibrated image-only versus joint and clinical-only versus joint models.

The two monomodal-vs-joint comparisons are paired by patient and must retain
the same split, patient rendering seed, frozen models and temperature
scaling across the comparisons.

## 3. Primary hypothesis

For condition c and modality m in {image, clinical}:

```
H0,c,m: Delta_c,m <= 0
H1,c,m: Delta_c,m > 0
```

The primary result is a one-sided test with the 95% bootstrap lower confidence
bound as the decision statistic. Temperature scaling is fitted inside every
bootstrap replicate on the frozen calibration partition, never on test labels.

## 4. Primary metric and success criterion

The primary metric is mean calibrated multiclass NLL difference. Accuracy is
reported in every condition but is secondary.

Recommended primary success rule:

- Confirm the lower bound of the patient-bootstrap 95% interval for
  `Delta_c,m` is above zero.
- Simultaneously require Holm-adjusted one-sided p < 0.05 across the
  ten primary contrasts (five conditions by two modality contrasts).
- Report, but do not use to determine the primary success, a minimum
  practically important improvement threshold.

A recommended default practical threshold is **0.10 nat** for an
incremental contribution, subject to investigator approval. This threshold
would require a lower confidence bound above +0.10 nat. The threshold should
be frozen and documented before test access. It should not be inherited by
the current protocol merely because legacy code contains it.

The threshold is a practical superiority boundary, not a statistical null.
If formally adopted, success at a contrast requires the multiplicity-adjusted
lower bound to exceed +0.10. If not adopted, 0.10 is a descriptive reference
only and inferential success remains a lower bound above zero.

To avoid interpretive drift, the protocol should distinguish:

- **Inferential success:** adjusted lower confidence bound > 0.
- **Practical success:** adjusted lower confidence bound > approved threshold.
- **Accuracy success:** descriptive or exploratory only.

## 5. Secondary endpoints

Secondary endpoints are:

1. Macro and per-class held-out accuracy.
2. Calibrated NLL for each of the three models.
3. Uncalibrated NLL sensitivity.
4. Cue-intervention target probability and NLL changes.
5. Four prespecified Grad-CAM localization variants.
6. Chefer, attention rollout and occlusion localization, separately.
7. Independent SLIC-superpixel deletion/insertion faithfulness for every
   valid explanation map.
8. Shape-preserving translated lesion-control mass.
9. Marker and lesion localization, area enrichment and pointing game.
10. Cue replacement and marker intervention sensitivity.
11. Stability, as a descriptive robustness endpoint.

Secondary hypotheses receive explicit family-level Holm correction. Localization,
faithfulness and cue intervention families are not pooled with the primary
NLL family.

## 5A. Three-level success interpretation

- **Contrast-level:** the relevant Holm-adjusted one-sided lower bound exceeds
  zero.
- **Condition-level:** both image-only-versus-joint and
  clinical-only-versus-joint contrasts exceed zero.
- **Experiment-level:** all five conditions, some conditions, or no conditions
  meet the condition rule.

Requiring both contrasts is stricter than reporting either contrast
individually and should be a prespecified policy choice. Failure to reject a
null is not evidence of equivalence.

## 6. Legacy accuracy criterion

The 0.80 joint accuracy floor is removed as a primary or co-primary success
criterion. It conflates a classifier-accuracy target with an NLL contribution
research question and can be satisfied while calibration is poor.

Accuracy may remain a:

- descriptive endpoint;
- condition-level feasibility check;
- exploratory sensitivity analysis.

It should not determine whether a condition "passes."

## 7. Statistical testing

Use 10,000 paired stratified bootstrap replicates for the primary NLL
contrasts, stratified by synthetic class. Each replicate resamples whole
patients within class, refits the calibration temperatures on resampled
calibration logits, recomputes the test probabilities, and recomputes all
ten contrasts.

The bootstrap procedure is defined as follows:

1. Freeze the test patient IDs and labels, three test logits, all 150
   calibration patients per class, and all 900 calibration logits per model.
2. Before unblinding test results, set the bootstrap seed to a fixed value
   and store it in the run manifest.
3. For replicate r:
   - independently resample calibration patient rows with replacement within
     each of the six classes;
   - refit T_image,r, T_clinical,r and T_joint,r using only the resampled
     calibration logits and labels;
   - resample held-out test patients with replacement within each class;
   - use the same resampled test row indices for image-only, clinical-only
     and joint probabilities;
   - compute the patient-level paired NLL gains for both monomodal-versus-joint
     contrasts;
   - preserve pairing of test rows across the three models within the replicate.
4. Recompute all ten contrasts for the five conditions. Patient IDs and
   condition-independent geometry are fixed across conditions only when the
   same five-condition cohort design is used; separate conditions are not
   treated as independent repeats of the same patients.

The primary bootstrap estimate is the mean of the patient-level gains in
each replicate. Calibration-partition resampling is included in the primary
uncertainty estimate because temperature fitting is part of the deployed
analysis. It should not be presented as an independent sensitivity-only
procedure. An additional bootstrap that freezes the fitted temperatures may
be reported as a diagnostic sensitivity analysis.

For contrast g, report:

- the marginal percentile 95% interval;
- the bootstrap estimate and raw one-sided p-value;
- the Holm-adjusted p-value;
- the Holm-adjusted one-sided lower bound.

The marginal percentile interval is not simultaneous. It is a descriptive
interval for contrast g alone. Simultaneous multiplicity-adjusted inference
is obtained by bootstrap inversion of Holm thresholds over the joint
bootstrap distribution, or equivalently by reporting the family-specific
Holm-adjusted lower bounds. Both are subject to the exact same frozen
calibration and patient resampling seed.

Use cluster-style paired resampling only if a patient has repeated variants;
the current design has one row per patient per condition and should remain
one patient-level unit.

The bootstrap seed should be a separate frozen seed. Confidence intervals
should use percentile intervals. One-sided multiplicity-adjusted lower bounds
should be obtained from bootstrap inversion rather than treated as Wald bounds.

## 8. Multiple-comparison correction

The recommended primary family is:

- 5 conditions x 2 modality contrasts = 10 NLL contrasts.

Apply Holm correction within this primary family, at one-sided alpha 0.05.
Report raw, Holm-adjusted p-values and simultaneous 95% lower bounds.

Separately prespecify each secondary family:

- localization map/method/region combinations;
- faithfulness method x direction x baseline;
- cue intervention comparisons.

Do not reuse the primary Holm result as confirmation for secondary endpoints.

## 9. Missing data and failed runs

There is no missing synthetic label or patient ID category. Missingness occurs
through invalid explanation maps or unavailable geometry controls.

Rules:

- Invalid explanation maps are excluded and explicitly counted.
- Patient means for unavailable translated controls are NaN.
- Aggregate denominators include only available finite repeat observations.
- Valid zero-mass controls remain valid observations.
- Failed model, calibration or test-generation stages are protocol failures,
  not missing-at-random observations.
- A condition is incomplete until the checkpoints, calibration, test
  predictions and explanation stages are all complete.
- No imputation or rerun-to-pass is permitted.

## 10. Untouched evaluation cohort

The 12,000-patient master design currently contains no spare patients. A
separate prospective cohort is required. The initial recommendation of 1,800
patients (300 per class) is not adopted automatically; it remains a
candidate until a power/sensitivity decision is approved.

For the 1,800-patient candidate:

- 300 patients per synthetic class;
- 12,000-patient development split + 1,800-cohort = 13,800 patients;
- a distinct master seed and patient-ID derivation stream;
- no overlap with train, model selection or calibration IDs;
- same 1,800 IDs, geometry streams and labels across conditions;
- condition-specific cue and decoy streams generated only after the cohort is
  frozen;
- patient-level class-balanced splits are not used for tuning.

A smaller cohort may be defensible if the predicted paired effect, variance
and practical threshold imply adequate power. A larger cohort may be required
if calibration/model variability raises variance or the 0.10-nat threshold
is formally required.

If preserving exactly 12,000 total patients is mandatory, the investigator
must approve a partition revision that explicitly removes capacity from
existing development partitions. This is not a trivial reallocation: it
changes training/calibration sample sizes and invalidates the previous
configuration hash.

The old pilot test partition and v2 72-patient subset remain historical
diagnostics and must never be renamed as the fresh holdout.

### 10A. Cohort power scenarios

The following normal-approximation scenarios assess contrast-level power
under a Bonferroni-like Holm surrogate (alpha/10). They are scenario
calculations, not Monte Carlo estimates:

| Per-class n | SD | Effect | Approx. contrast power |
|---:|---:|---:|---:|
| 50 | 0.20 | 0.05 | 0.960 |
| 50 | 0.50 | 0.10 | 0.888 |
| 50 | 0.80 | 0.10 | 0.340 |
| 100 | 0.50 | 0.05 | 0.449 |
| 100 | 0.80 | 0.10 | 0.686 |
| 150 | 0.80 | 0.10 | 0.881 |
| 300 | 0.50 | 0.05 | 0.952 |
| 300 | 0.80 | 0.05 | 0.530 |

For a formal 0.10-nat threshold, margin is effect minus 0.10:

| Effect | SD | Per-class n for 80% | Per-class n for 90% |
|---:|---:|---:|---:|
| 0.15 | 0.35 | 96 | 122 |
| 0.15 | 0.50 | 195 | 248 |
| 0.15 | 0.80 | 499 | 635 |
| 0.20 | 0.50 | 49 | 124 |
| 0.20 | 0.80 | 125 | 159 |

At effect exactly 0.10, a threshold of 0.10 is a boundary test and has no
practical superiority power by definition.

**Smallest defensible recommendation:**

- If 0.10 nat remains descriptive and the paired SD is plausible at 0.50 or
  lower, 300 patients per class provides at least 80% contrast-level power
  for effects of 0.10 or greater.
- If 0.10 nat becomes a formal practical threshold, and the relevant
  high-variance scenario is effect 0.15/SD 0.80, use at least 499 patients
  per class for 80% contrast-level power or 635 per class for 90%.
- If effect 0.20/SD 0.80 is the minimum scenario, 125 per class is sufficient
  for 80% and 159 for 90% contrast-level practical-threshold power.

These are one-contrast normal approximations, not condition-level or
five-condition power. Requiring both contrasts and all five conditions
requires multivariate simulation and may require a larger cohort.

The cohort manifest is an independent approval artifact and must contain:

- schema version;
- cohort/seed;
- condition-independent patient IDs;
- label by patient;
- partition and class count;
- overlap hashes against every development partition;
- cohort membership SHA-256;
- authorization timestamp and approver;
- frozen protocol/config SHA-256 values.

Test data must remain inaccessible until all three checkpoints and calibration
temperatures are verified and the cohort authorization hash matches.

## 11. Research decisions requiring explicit approval

The following are not ordinary implementation choices:

1. Approve calibrated NLL as the primary outcome.
2. Approve 10 Holm-corrected primary contrasts.
3. Approve or replace the candidate 0.10-nat practical threshold.
4. Approve 300 test patients per class, or approve a simulation-based revision.
5. Approve a 1,800-patient fresh cohort rather than reclassifying old data.
6. Approve separate condition-level vs global multiplicity interpretation.
7. Approve whether final success requires both modality contrasts or only one
   for a condition.

The recommended defaults are concrete and can be frozen once approved.

## FINAL INVESTIGATOR DECISIONS REQUIRED

1. **Cohort size:** approve 1,800 patients, a smaller size with a simulation
   justification, or a larger size.
2. **Primary threshold:** approve 0.10 nat as a formally required practical
   superiority threshold, retain it as descriptive reference only, or replace
   it with an alternative.
3. **Primary multiplicity:** approve the 10 Holm-corrected contrasts.
4. **Condition success:** approve that both image and clinical contrasts must
   succeed for a condition to pass.
5. **Cohort generation:** approve a distinct fresh 13,800-patient design or a
   revised partition of the existing 12,000-patient design.
6. **Calibration uncertainty:** approve inclusion of calibration resampling in
   the primary bootstrap. A frozen-temperature bootstrap is then sensitivity
   analysis only.

Until these fields are documented and hash-bound, protocol approval and
scientific execution remain blocked.
