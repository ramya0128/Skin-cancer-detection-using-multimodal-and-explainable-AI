# Phase G v2.2 decision summary

## Already resolved technically

- Calibrated multiclass NLL is the primary outcome.
- The primary statistical unit is the patient.
- Positive NLL differences mean joint improvement.
- The legacy 0.80 accuracy floor is removed from primary success.
- The primary family is recommended as ten contrasts:
  two modality contrasts across five conditions.
- One-sided Holm correction, 95% patient-bootstrap intervals and bootstrap
  calibration-temperature refitting are recommended.
- The 72-patient v2 subset and balanced-pilot test partition remain
  explicitly non-fresh holdouts.
- The exact four Grad-CAM variants are available through existing v2 code:
  final/penultimate block x global/token-local weighting.
- The existing v2 independent superpixel implementation already has:
  128 SLIC regions, 21 steps, 20 random orders and center-first control.
- Renderer-background and mid-gray baselines can be expressed exactly.
- Lazy deterministic patient rendering is the least disruptive RAM solution.
- All execution remains fail-closed.

## Recommended defaults

1. Candidate fresh test cohort: 1,800 patients, 300 per class; final size
   remains contingent on power assumptions.
2. Primary threshold: 0.10 nat, one-sided 95% lower bound above threshold,
   but only after investigator approval.
3. Primary multiplicity: Holm across ten contrasts.
4. Bootstrap: 10,000 patient resamples, stratified by class, with calibration
  temperature refitting.
5. Secondary multiplicity: separate Holm families for localization,
   faithfulness and cue intervention.
6. Confidence intervals: percentile 95% intervals plus one-sided bounds.
7. Missing explanations: excluded and counted, never replaced with zeros.
8. Memory solution: v2.2-only lazy deterministic rendering with bounded cache.

## Requires explicit investigator decision

1. Approve the fresh cohort size and 13,800-patient total design.
2. Approve the 0.10-nat practical threshold or replace it.
3. Approve whether success requires both monomodal contrasts for a condition
   or permits one contrast at a time.
4. Approve the ten-contrast global primary family versus within-condition
   two-contrast families.
5. Approve the exact fresh cohort seed and manifest format.
6. Approve the condition-specific authority mechanism and number of
   conditions authorized per run.

## Bootstrap finalization

- Use 10,000 patient-level paired bootstrap replicates.
- Stratify test resampling by class.
- Independently resample calibration patients by class.
- Refit three temperatures inside every replicate.
- Use the same test row resample and pairing across all three models.
- Preserve one patient unit, but do not claim cross-condition independence
  merely because the contrasts are calculated in separate conditions.
- Report marginal percentile intervals and Holm-adjusted one-sided bounds
  separately.

## Cohort and power conclusion

The 1,800-patient candidate remains a reasonable starting proposal, but it
is not adopted solely by precedent. At 300 per class:

- contrast power is high for effect 0.10 when SD is 0.20-0.50;
- the SD 0.80 scenario is more sensitive and may not reach 80% for small
  effects;
- a formal 0.10-nat threshold requires effects clearly above 0.10;
- condition success with both contrasts is more stringent and requires
  multivariate power estimation;
- complete five-condition success is the most stringent criterion.

A normal-approximation sensitivity table is in the statistical plan. It is
not an established power estimate until Monte Carlo validation and
model-informed variance assumptions are approved.

## Can be implemented immediately

- Protocol and statistical plan documentation.
- Four-variant Grad-CAM wrapper tests.
- SLIC/fidelity integration tests.
- Availability-aware denominator tests.
- Lazy streaming dataset API.
- Cohort overlap and manifest validation.
- Ten-contrast primary-analysis validation.
- Bootstrap implementation changes.
- Execution authorization hash updates.
- Progress toward a final protocol document.

No implementation step may create condition patients or test records, train
full models, run held-out inference or enable scientific execution.

## FINAL INVESTIGATOR DECISIONS REQUIRED

1. Approve calibrated NLL as primary.
2. Approve 300 per class (1,800 total) or a different cohort size.
3. Approve 0.10 nat as a formal practical threshold, descriptive reference,
   or no threshold.
4. Approve ten primary Holm-corrected contrasts.
5. Approve condition success as both modality contrasts.
6. Approve calibration-temperature resampling inside primary bootstrap.
7. Approve the fresh cohort seed and manifest.
8. Approve the condition authorization scope.

Until these fields are recorded in an approved protocol and hash-bound
authorization record, scientific execution remains blocked.
