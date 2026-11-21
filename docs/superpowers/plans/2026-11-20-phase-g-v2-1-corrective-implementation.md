# Phase G v2.1 Corrective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superagent-driven development or executing-plans.

**Goal:** Implement and test the availability-aware translated-control correction, SLIC reproducibility metadata, and prospective condition protocol in a separate v2.1 package.

**Architecture:** Add a new package with independent tests and wrappers that adapt v2 logic without editing v1/v2 files. Add static configuration validation and documentation; write no runtime outputs beyond the versioned v2.1 output directory.

**Tech Stack:** Python 3.11, NumPy, skimage, PyTorch metadata-only imports, unittest.

**Spec:** `docs/superpowers/specs/2026-11-20-phase-g-v2-1-corrective-design.md`

## Global Constraints

- No v1/v2 source or output may change.
- No model inference or remaining condition may execute.
- No test rows, labels, predictions, or images may be loaded.
- V2.1 outputs must be separate and reproducibly hashed.
- The existing 72-patient subset remains a validation diagnostic.

## Review Focus

- All-available, partial-available, no-available, and valid-zero controls.
- NaN must not become zero in localization or downstream summaries.
- SLIC parameter and version metadata must be exact.
- Authorization must fail closed.

---

### Task 1: Red tests for translated-control aggregation

**Files:**
- Create: `experiments/phase_g_v2_1/tests/test_availability_localization.py`
- Create: `experiments/phase_g_v2_1/localization.py`

- [ ] Add tests for all available, partial available, no available, valid zero
  mass, aggregate denominator, and downstream NaN handling.
- [ ] Run the test file and verify the availability API fails before implementation.
- [ ] Implement `translated_control_summary(...)` and update its call contract.
- [ ] Run tests and verify green.

### Task 2: Geometry and SLIC reproducibility

**Files:**
- Create: `experiments/phase_g_v2_1/geometry.py`
- Create: `experiments/phase_g_v2_1/slic.py`
- Create: `experiments/phase_g_v2_1/tests/test_slic.py`

- [ ] Add repeated-SLIC and parameter-recording tests.
- [ ] Add geometry-availability tests.
- [ ] Implement wrappers without changing v2 settings.
- [ ] Run tests and verify green.

### Task 3: Prospective protocol and authorization

**Files:**
- Create: `experiments/phase_g_v2_1/config.py`
- Create: `experiments/phase_g_v2_1/tests/test_config.py`
- Create: `docs/phase_g_v2_1/PROSPECTIVE_PROTOCOL.md`

- [ ] Add static tests that balanced-only or no-condition authorization is valid
  and other conditions fail closed.
- [ ] Implement deliberate allowlist and test-partition guards.
- [ ] Freeze the protocol and list unresolved thresholds for approval.
- [ ] Run tests and static validation.

### Task 4: Preservation, manifests, report, and final verification

**Files:**
- Create: `experiments/phase_g_v2_1/CHANGELOG.md`
- Create: `experiments/phase_g_v2_1/README.md`
- Create: `experiments/phase_g_v2_1/tests/test_preservation.py`
- Create: `experiments/phase_g_v2_1/reproducibility.py`

- [ ] Record post-change hashes and verify protected files unchanged.
- [ ] Run unit/regression/existing relevant tests.
- [ ] Run SLIC determinism and read-only test-partition checks.
- [ ] Write implementation report and test results.
- [ ] Do not run model inference or remaining conditions.
