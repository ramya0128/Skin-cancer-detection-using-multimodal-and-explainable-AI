# Phase G v2.2 statistical plan

**Status:** Recommended prespecified analysis; not yet approved for execution.

## 1. Unit of analysis

The primary unit is a patient. The synthetic labels are balanced exactly by
class within the test cohort. One primary record per patient contains:

- true class;
- calibrated image-only NLL;
- calibrated clinical-only NLL;
- calibrated joint NLL;
- patient ID;
- condition;
- availability and validity flags.

All patient IDs must be unique within a condition and a fresh cohort. No
image/patch/superpixel row is substituted for the patient as the primary
unit.

## 2. Primary estimands

Let:

- `N_image,i = -log p_image,i(Y_i)`
- `N_clinical,i = -log p_clinical,i(Y_i)`
- `N_joint,i = -log p_joint,i(Y_i)`

Define:

```
D_image,i = N_image,i - N_joint,i
D_clinical,i = N_clinical,i - N_joint,i
```

The estimands are:

```
Delta_image,c = E_c[D_image,i]
Delta_clinical,c = E_c[D_clinical,i]
```

Positive values mean that the joint model has lower NLL than the monomodal
model. A 0.10-nat difference means an average 0.10 nat decrease in the negative
log probability of the true class; it is reported as the candidate minimum
practical effect, not a statistical null.

## 3. Primary hypotheses

For each condition c and modality m:

```
H0,m,c: Delta_m,c <= 0
H1,m,c: Delta_m,c > 0
```

Use:

- alpha = 0.05 one-sided;
- Holm adjustment across 10 contrasts;
- 10,000 bootstrap replicates;
- 95% bootstrap intervals;
- one-sided bootstrap lower bounds.

## 4. Temperature calibration

Calibrate each model with scalar temperature scaling on only the
`validation_calibration` partition:

```
T_hat = argmin_T mean[-log softmax(z/T)[Y]]
```

Freeze T_hat before test access. In every bootstrap replicate:

1. Resample calibration patients within class.
2. Refit all three temperatures.
3. Resample held-out patients within class.
4. Compute calibrated test probabilities.
5. Recompute patient-level NLL differences.

Do not refit on test labels. The bootstrap result is a full patient-and
calibration uncertainty estimate.

## 4A. Bootstrap algorithm

The primary resampling unit is the patient:

1. Set seed 20261123 or a later protocol-approved replacement.
2. Independently resample 150 calibration patients per class for each model.
3. Refit scalar temperatures on these 900 calibration logits per model.
4. Resample held-out patients stratified by class.
5. Apply the three refitted temperatures to the corresponding test logits.
6. Compute D_image and D_clinical for every selected test row.
7. Calculate both contrasts for all five conditions. Because conditions are
   separate experiments, pairing is preserved by patient ID and fixed row
   index inside a condition; it is not assumed that five condition contrasts
   are independent.
8. Save the raw bootstrap vector for all 10 contrasts.

The same test row resample is used for all three models in a replicate.
Therefore the bootstrap captures the paired model difference. The same
calibration resample and test row resample may be reused across conditions
only if the exact fresh patient-ID set is shared; otherwise condition-specific
cohort resamples use the fixed condition seed and record the pairing rule.

For Holm-adjusted inference:

1. For threshold d, obtain the bootstrap-centered one-sided p-value for each
   contrast.
2. Sort the ten p-values and multiply by the remaining family rank, as in
   Holm.
3. Apply Holm cumulative maximum.
4. For a lower confidence bound, find the largest d such that the
   Holm-adjusted p-value for H0: Delta <= d remains <= alpha.

The lower bounds are one-sided family-adjusted bounds obtained by inversion.
Ordinary percentile intervals are marginal and must be labeled as such.
No ordinary percentile interval should be called simultaneous.

## 5. Confidence intervals

Report:

- point estimate;
- percentile 95% two-sided interval;
- one-sided 95% lower bound;
- Holm-adjusted p-value for the one-sided effect > zero;
- sample size and valid-patient count.

For contrast g and threshold d:

```
p_g(d) = (1 + count{B_r < d - centered_shift})/(R+1)
```

The implementation's existing bootstrap-inversion lower-bound logic may be
retained, but the contrast family must be changed from the legacy fixed two
contrasts to the prespecified ten primary contrasts. The fixed accuracy floor
must be removed from `passes_primary_gate`.

## 6. Effect sizes

Report:

- mean NLL gain with confidence interval;
- median paired NLL gain;
- paired NLL gain per 10 patients or per class;
- joint and monomodal accuracy;
- Brier score as a secondary calibration measure;
- standardized mean difference for descriptive context.

Do not substitute accuracy for NLL as the primary effect.

## 7. Multiple-comparison plan

### Primary family

Ten contrasts:

```
{image-only-joint, clinical-only-joint} x 5 conditions
```

Holm correction is applied across this family. The candidate 0.10-nat threshold
is not altered by the correction.

### Secondary families

Secondary comparisons are grouped by analysis and Holm-adjusted within each
family:

- cue intervention: image and clinical replacements;
- localization: model target x region x method;
- faithfulness: model target x direction x baseline x method;
- stability: descriptive only, no confirmatory tests.

No secondary result may be promoted to primary evidence merely because it
crosses 0.05.

## 8. Sample size and precision

The historical 1,800-patient test partition is not a fresh cohort. A proposed
fresh cohort with 300 patients per class would provide:

- 300 independent observations per class;
- balanced representation;
- approximately 1,800 patient-level paired differences;
- substantial precision for mean NLL estimates;
- enough class-stratified bootstrap support for all six labels.

Because no formal power calculation is possible until the trained model
effect distribution is observed, an investigator should perform a
simulation-based power check against the exact Bayes baselines before lock:

1. Generate probability distributions from the exact condition baselines.
2. Treat calibration uncertainty and model misspecification as nuisance
   distributions.
3. Simulate the paired NLL contrast for 300 per-class test patients.
4. Determine power for alpha 0.05, Holm-adjusted across 10 contrasts.
5. Replace or retain n=1,800 only through an approved protocol amendment.

The candidate 1,800-patient cohort is not justified by external clinical
power. It is a prespecified synthetic precision choice.

### 8A. Power assumptions

The following is a transparent normal-approximation scenario analysis, not
a claim about the trained model's unknown finite-sample distribution.
It assumes:

- paired patient-level NLL differences;
- additive mean effect mu;
- standard deviation SD across patient pairs;
- independent contrasts for a worst-practical-size comparison;
- one-sided alpha 0.05;
- Holm adjustment across ten contrasts;
- no dropout or failed patient NLL.

For one contrast, the Bonferroni-adjusted Holm boundary is represented as
z_(1-alpha/10)=2.576. Approximate power is:

```
Power(mu, SD, n) ~= Phi(sqrt(n)*mu/SD - 2.576)
```

This is intentionally conservative relative to Holm's exact distribution
when effects are unequal or the largest p-value is selected first.

| Per-class n | Total n | SD | Effect 0.05 | Effect 0.10 | Effect 0.15 | Effect 0.20 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 300 | 0.20 | 0.960 | 1.000 | 1.000 | 1.000 |
| 50 | 300 | 0.35 | 0.459 | 1.000 | 1.000 | 1.000 |
| 50 | 300 | 0.50 | 0.199 | 0.888 | 1.000 | 1.000 |
| 50 | 300 | 0.80 | 0.068 | 0.340 | 0.673 | 1.000 |
| 100 | 600 | 0.35 | 0.822 | 1.000 | 1.000 | 1.000 |
| 100 | 600 | 0.50 | 0.449 | 1.000 | 1.000 | 1.000 |
| 100 | 600 | 0.80 | 0.111 | 0.686 | 1.000 | 1.000 |
| 150 | 900 | 0.50 | 0.664 | 1.000 | 1.000 | 1.000 |
| 150 | 900 | 0.80 | 0.190 | 0.881 | 1.000 | 1.000 |
| 300 | 1,800 | 0.50 | 0.952 | 1.000 | 1.000 | 1.000 |
| 300 | 1,800 | 0.80 | 0.530 | 1.000 | 1.000 | 1.000 |

The table is contrast-level power. It does not represent condition- or
experiment-level power.

Condition success requires two contrasts. If the contrasts are independent
and equal, approximate condition power is power_i^2. The 0.50/0.10 case with 300 per class has near-unity individual power and
near-unity condition power. The 0.80/0.05 case has approximately 0.530
contrast power and 0.281 condition power. Positive correlation can alter this
calculation, and the multivariate distribution should be simulated before
approval.

### 8B. Practical-threshold sensitivity

For a formally required 0.10-nat threshold, power is calculated for margin
mu - 0.10. This is substantially more demanding than testing zero.

| True effect | Margin over 0.10 | SD | Per-class n for 80% | Per-class n for 90% |
|---:|---:|---:|---:|---:|
| 0.10 | 0.00 | 0.35 | undefined | undefined |
| 0.15 | 0.05 | 0.35 | 96 | 122 |
| 0.15 | 0.05 | 0.50 | 195 | 248 |
| 0.15 | 0.05 | 0.80 | 499 | 635 |
| 0.20 | 0.10 | 0.50 | 49 | 124 |
| 0.20 | 0.10 | 0.80 | 125 | 159 |

These are Holm-corrected contrast-level normal calculations. Exact totals
are 6 times the per-class values. At exactly 0.10, the formal threshold test
has no meaningful superiority power; it is a boundary case.

Five-condition complete success requires all ten contrasts. Under independent
equal effects, the approximate probability is power_i^10. This would be
reasonable only for strong, low-variance effects and would be a very stringent
experiment-level criterion.

## 9. Independence and aggregation

Patient-level independence is assumed by the synthetic generation model.
No lesion pixels, superpixels or explanation patches are independent
replicates in the primary analysis.

Within-patient repeated superpixels are repeated measures within one patient.
Random superpixel orders are averaged within patient before patient bootstrap.
Faithfulness AUC is the patient-level statistic.

## 10. Missingness and failed runs

No test patient may be dropped for label or NLL failure. Explanation validity
is separate from NLL validity:

- report valid/invalid map denominator;
- exclude only invalid map rows;
- never convert invalid map or unavailable control values to zero;
- preserve valid zero attribution as a finite observation;
- report unavailable geometry separately;
- stop a failed condition run rather than reporting partial scientific success.

## 11. Interpretation rules

Conclude a modality contrast is established only if:

1. the calibrated NLL estimate is positive;
2. the Holm-adjusted one-sided p is below 0.05;
3. the one-sided lower bound excludes zero;
4. the practical threshold, if approved, is not contradicted.

Conclude a condition improves multimodal reliability only if its two
primary contrasts pass the above rule. Accuracy alone does not support this
conclusion.

## 11A. Three-level success interpretation

### A. Contrast-level success

Contrast g succeeds if its Holm-adjusted one-sided lower confidence bound
exceeds zero. This is evidence against the contrast-specific null
Delta_g <= 0. It is not evidence that the effect is practically important
unless the approved threshold is also excluded.

### B. Condition-level success

A condition succeeds only if both image-only-versus-joint and
clinical-only-versus-joint contrasts succeed. Either contrast alone is a
secondary descriptive result unless the protocol is amended. Requiring both
contrasts is stringent because it asks for multimodal benefit to both
component views. It is justified when the scientific hypothesis is that both
modalities contribute.

### C. Experiment-level interpretation

- **All five conditions succeed:** all ten primary contrasts meet the contrast
  rule, or all five conditions meet the condition rule. These are equivalent
  when condition success requires both contrasts.
- **Some conditions succeed:** report the exact number and identities, with
  family-adjusted bounds. Do not synthesize a global success claim.
- **No condition succeeds:** report no contrast-level evidence of joint
  improvement. This is not evidence of equivalence or that joint models are
  inferior.
- **Mixed:** report paired condition-specific estimates and uncertainty.
  Failure to reject a null is not evidence of equivalence.

## 12. Legacy removal

Remove from the primary gate:

- joint accuracy floor 0.80;
- fixed contrast family size 2;
- implicit carryover of the 0.10 threshold;
- any all-conditions result based on only one family correction.

Retain the 0.10 threshold only as a preapproved practical-effect rule.

## 13. Practical-threshold recommendation

A 0.10-nat threshold is interpretable as a 9.52% multiplicative reduction
in probability of the true class only in the paired mean-log sense, and it
must not be translated to per-patient odds or clinical risk.

Recommended default:

- retain 0.10 as a **secondary practical-superiority reference**;
- do not require it as a primary success criterion until the investigator
  approves it;
- if it is formally required, require the Holm-adjusted lower bound to
  exceed +0.10;
- use the threshold only for a practical conclusion, never to alter the
  inferential null H0: Delta <= 0.

The threshold is an investigator decision because no validation data or
external clinical precedent was supplied to justify it.

## FINAL INVESTIGATOR DECISIONS REQUIRED

1. Primary threshold: 0.10 nat as formal practical superiority, descriptive
   reference, or replacement.
2. Cohort size: 300 per class remains a candidate, not an adopted fact.
3. Condition rule: require both monomodal contrasts or use contrast-level
   reporting only.
4. Primary family: approve ten global Holm-corrected contrasts.
5. Bootstrap: approve calibration-temperature resampling in the primary
   estimate.
