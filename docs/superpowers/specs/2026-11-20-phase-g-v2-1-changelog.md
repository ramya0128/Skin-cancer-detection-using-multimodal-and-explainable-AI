# Phase G v2.1 Changes

This version is separate from Phase G v1 and v2 and does not modify their
sources or outputs.

## Fixed

- Translated-control aggregation will receive explicit availability.
- Unavailable repeats/patients will be excluded from control-mass means.
- Valid control masks with zero attribution remain valid zero observations.
- Aggregate denominators and patient availability counts will be explicit.

## Improved

- SLIC deterministic settings and versioned parameter recording.
- Static configuration validation and explicit condition authorization.
- Test-partition isolation checks.
- Pre/post-protection SHA-256 manifests.

## Not changed

- Lesion, marker, and background attribution calculations.
- Existing v1/v2 reports and run outputs.
- Model checkpoints and temperatures.
- The remaining five condition authorization state.
