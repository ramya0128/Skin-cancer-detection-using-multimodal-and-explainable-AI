# Whole-Project Architecture and Context Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an accurate whole-project architecture map and refresh the root context handoff for future contributors.

**Architecture:** Add a stable root `ARCHITECTURE.md` describing the real-data and synthetic research tracks, their data/model/explanation flows, artifacts, and boundaries. Update the existing root `CONTEXT.md` with dated project status and links, keeping stable architecture separate from the changing handoff.

**Tech Stack:** Markdown documentation; source of truth is the existing Python, JSON, notebook, and report files.

**Spec:** `docs/superpowers/specs/2026-11-21-project-architecture-context-design.md`

## Global Constraints

- Do not modify models, datasets, notebooks, configurations, saved artifacts, or execution gates.
- Keep PAD-UFES-20 real-data work separate from synthetic Phase G work.
- Label Phase F findings as exploratory; the test-set debugging deviation prevents confirmatory claims.
- Keep Phase G v2.2 marked `draft_unapproved`, with execution disabled and test access forbidden.
- Do not describe planned, unexecuted, or statically inspected work as experimentally validated.
- Do not include machine-specific dataset paths, secrets, or patient-level information.
- Leave `ara/logic/solution/architecture.md` unchanged.

## Review Focus

- Frozen patient split and prior test exposure: state the split and caveat without implying a fresh holdout; verify against `DATA_AUDIT.md`.
- Phase separation: distinguish real-data PAD-UFES-20 models from synthetic Phase G models; verify against configs and phase READMEs.
- Explanation targets: distinguish branch-level maps/TreeSHAP from CWF modality Shapley; verify against `NOTEBOOK_04_05_AUDIT.md`.
- Phase G v2.2 status: distinguish generator code and engineering wrappers from authorized condition runs; verify against config, runner, and status reports.
- Sensitive local setup: document configurable local data paths without reproducing their literal values; verify against `common.py` and `README.md`.

---

### Task 1: Create the repository architecture map

**Files:**
- Create: `ARCHITECTURE.md`

**Interfaces:**
- Consumes: root README, `common.py`, notebook flow, artifact/data audits, experiment package READMEs and configs.
- Produces: stable repository map with links to detailed phase records.

- [ ] **Step 1: Draft the real-data path**

Document local data configuration, patient-level frozen split, image/clinical branch outputs, CWF fusion, explanation notebooks, and saved artifact locations. Link existing authoritative audits rather than copying all result tables.

- [ ] **Step 2: Draft the synthetic Phase G path**

Document v1 balanced-pilot artifacts, v2/v2.1 validation-only work, and v2.2 condition generator/model/explanation/statistics modules. State that v2.2 protocol execution is blocked and no condition results are available.

- [ ] **Step 3: Add component and artifact boundaries**

Include a compact flow diagram; distinguish branch explanations from fusion-level modality explanations; identify test-access, licensing, and reproducibility constraints with links to source records.

- [ ] **Step 4: Review claims and links**

Check each status claim against current configs/reports and verify every relative link resolves to an existing repository path. Do not run experiments for documentation.

### Task 2: Refresh the current project context handoff

**Files:**
- Modify: `CONTEXT.md`

**Interfaces:**
- Consumes: `ARCHITECTURE.md` and the current data, artifact, notebook, Phase F, and Phase G reports.
- Produces: dated status, latest completed work, active blockers, and safe next actions for future agents.

- [ ] **Step 1: Replace stale pipeline/status summary**

Preserve useful collaborator and frozen-artifact guidance, but update the project state to include executed exploratory Phase F work and its test-exposure caveat.

- [ ] **Step 2: Add the Phase G handoff**

Record the current v2.2 condition-generation path, the recent conflict-orientation fix as not runtime-verified, the protocol/runner blockers, disabled execution/test access, and the requirement not to train or generate a held-out cohort without approval.

- [ ] **Step 3: Add a “read first” index**

Link the architecture map and the authoritative data, artifact, notebook, statistical, protocol, decision, and implementation reports. Keep detailed numeric results in their existing reports.

- [ ] **Step 4: Review currency and privacy**

Confirm dates and statuses, remove any contradictory stale instructions, avoid literal local paths or patient identifiers, and verify Markdown links.

