# Phase G v2.1 Corrective Implementation Design

## Intent

Correct the translated-control denominator defect found in the Phase G v2
audit, improve reproducibility recording, and create a prospective protocol
for the remaining five conditions. This implementation does not execute or
authorize any remaining condition.

## Goals

- Preserve all Phase G v1 and v2 sources and outputs.
- Build a separate `experiments/phase_g_v2_1` implementation.
- Propagate explicit translation availability through geometry and localization.
- Exclude unavailable translations from translated-control numerators and
  denominators while preserving valid zero-mass controls.
- Record SLIC and environment reproducibility parameters.
- Freeze a prospective protocol for future condition execution.
- Prevent test-row or test-label loading in v2.1 static validation paths.

## Non-goals

- No model retraining.
- No remaining-condition execution.
- No test-image, test-label, test-prediction, or test-logit loading.
- No changes to v1 or v2 source/output files.
- No v2.1 full model inference or held-out evaluation.

## Existing 72-patient subset

The existing 72-patient subset is a validation diagnostic only. It was
sampled after the v2 implementation had begun and must not be retroactively
described as preregistered or untouched. It remains outside the prospective
protocol freeze.

## Availability-aware translation result

The v2.1 localization API will accept:

- `translated_control_masks`
- `translated_control_available`

Each availability entry is aligned as `(N, R)`. A control is valid only when
the availability entry is true and the corresponding mask contains a
genuinely translated lesion. In unavailable slots:

- the result is represented as `NaN`;
- the slot contributes zero to the numerator;
- the slot contributes zero to the denominator;
- an entire unavailable patient has `NaN`;
- an aggregate reports `actual_denominator`.

Valid zero-mass controls remain zero and enter the denominator.

## Reproducibility

SLIC remains at 128 superpixels, compactness 10.0, and 224x224 RGB uint8
input. The v2.1 wrapper records scikit-image version, seed, `channel_axis`,
`convert2lab=False`, `enforce_connectivity=True`, and `start_label=0`.
If repeated calls can differ despite identical settings, verification reports
the limitation rather than claiming determinism.

## Condition authorization

The v2.1 configuration validator will reject any enabled condition unless it
is one of the six exact condition names and appears in an explicit
`authorized_conditions` allowlist. Static validation will prove the
configuration is balanced or that no conditions are enabled. The protocol
freeze is frozen before execution but does not enable execution.

## Prospective protocol

The protocol records unresolved decisions rather than inventing thresholds:

- remaining condition definitions and cue reliabilities;
- master and partition seeds;
- frozen architecture/training procedure;
- exact explainer configurations;
- invalid-map handling;
- localization and translated-control availability denominators;
- SLIC superpixel controls and baselines;
- predictive NLL/calibration analysis;
- paired bootstrap and multiplicity families;
- primary, secondary, and exploratory outcomes;
- condition success criteria requiring approval.

## Approvals

The implementation approach was approved by the user for a separate v2.1
implementation, tests, protocol, and manifests. Execution of conditions
remains prohibited until separate approval.
