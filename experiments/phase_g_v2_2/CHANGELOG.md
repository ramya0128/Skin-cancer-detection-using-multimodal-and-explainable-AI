# Phase G v2.2 Changelog

- Added a separate conflicting-condition generator and rendered-patient wrapper.
- Added 25%/25%/50% deterministic conflicting population assignment.
- Added independent decoy streams for conflicting cue rendering.
- Added exact Bayes accuracy/NLL enumeration tests.
- Added fail-closed remaining-condition execution authorization.
- Added actual image-only, clinical-only, and joint model forward/backward CUDA checks.
- No historical source or output was edited.
- Added CUDA-only v2.2 train/calibration orchestration with condition-specific
  checkpoints, stage hashes, exact batch-64 enforcement, and interrupted-run
  recovery.
- Added an explicitly non-evidentiary batch-64 training preflight and host RAM
  lower-bound check.
- Added CLI dry-run validation, hash-bound authorization requirements, and
  checks preventing preflight outputs from entering condition result folders.
- Tightened authorization so a draft checked-in protocol cannot be overridden
  by an external authorization record.
- Added fail-closed protocol-compatibility gates. Full condition training and
  test evaluation remain disabled until the prospective protocol and analysis
  implementation agree.
