# Phase G v2.1 Corrective Implementation

Phase G v2.1 is a separate implementation that fixes availability handling
for shape-preserving translated lesion controls. It does not modify or
overwrite Phase G v1 or v2 artifacts.

## Corrected metric

`translated_control_summary(mass, availability)` accepts `(N, R)` mass and
availability arrays. A slot counts only when both finite and available.
Unavailable slots contribute no numerator and no denominator. An unavailable
patient produces a NaN patient mean. A valid control with zero attribution
remains a valid zero and contributes to the denominator.

## SLIC

The v2.1 wrapper preserves v2 settings:

- 128 superpixels;
- compactness 10.0;
- sigma 0;
- `convert2lab=False`;
- `enforce_connectivity=True`;
- `channel_axis=-1`;
- `start_label=0`;
- seed 20261144.

The wrapper records the installed scikit-image version and verifies identical
segmentations for repeated calls with identical input and settings.

## Condition execution

The v2.1 validator fails closed. It currently permits no remaining-condition
execution: `execution_enabled` must be false and test-data access is forbidden.
Authorization of future conditions requires separate protocol approval.
