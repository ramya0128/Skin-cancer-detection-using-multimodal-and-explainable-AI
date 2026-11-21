# Whole-project architecture and context documents

## Intent

Give project contributors and future coding agents an accurate repository-wide
map and a concise current handoff for the skin-lesion multimodal/explainability
project. The documents must distinguish implemented code, saved experimental
evidence, exploratory findings, and unapproved or unexecuted work.

## Goals

- Describe the real-data PAD-UFES-20 analysis track and the controlled
  synthetic Phase G track without conflating their data or claims.
- Map the notebooks, shared dataset/model code, experiment packages, important
  outputs, and approval/test-access boundaries.
- Update the existing root `CONTEXT.md` to reflect the latest documented state,
  including executed exploratory Phase F work and blocked Phase G v2.2 work.
- Keep the content navigable with links to authoritative detailed plans and
  audit reports.

## Non-goals

- No model, dataset, notebook, configuration, artifact, or execution-gate
  changes.
- No claim that exploratory frozen-test results are confirmatory.
- No protocol approval, test authorization, or claim that v2.2 conditions or
  explanations have been experimentally validated.
- No replacement of `ara/logic/solution/architecture.md`, which belongs to the
  separate ARA artifact.

## Approaches considered

1. **Root `ARCHITECTURE.md` plus update existing root `CONTEXT.md`** —
   selected. The architecture is stable and the context remains a live
   handoff; existing root documentation conventions are preserved.
2. `docs/ARCHITECTURE.md` plus update root context — valid, but splits the
   repository map from the existing root entry points.
3. One combined document — fewer files, but mixes stable architecture with
   frequently changing status and repeats material already covered by
   `CONTEXT.md`.

## Design

- Create root `ARCHITECTURE.md` with repository layout, data flow, model
  branches/fusion, explanation layers and their actual targets, experiment
  versions, artifact flow, reproducibility, and safety boundaries.
- Update root `CONTEXT.md` as a concise dated handoff: completed evidence,
  current Phase G v2.2 state and blockers, files to read first, verification
  limitations, and explicit do-not-do items.
- Link rather than duplicate detailed statistical, protocol, data-audit,
  notebook-audit, and phase-specific records.
- Do not copy machine-specific dataset paths, credentials, or patient-level
  information.

## Validation

Review both documents against the repository README, data and artifact audits,
analysis plan, phase-specific READMEs/configurations, and current v2.2 runner
and generator. This documentation-only change requires no test run.
