# Phase G v2.1 Changelog

## Added

- Availability-aware translated-control aggregation.
- Regression coverage for all, partial, and no available translations.
- Valid-zero versus unavailable-zero handling.
- Seeded SLIC wrapper with explicit parameter and version metadata.
- Explicit, fail-closed Phase G v2.1 condition authorization.
- A prospective protocol for the remaining five conditions.

## Fixed

- Translated-control aggregates no longer count unavailable repeats or
  unavailable patients as zero-mass observations.

## Preserved

- All v1 and v2 source and output files remain unchanged.
- No model was retrained.
- No remaining condition was run.
- No test partition was loaded by v2.1 static validation paths.
