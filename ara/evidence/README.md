# Evidence Index

Evidence references point to existing project artifacts; no raw result files were duplicated.

| Evidence | Location | Provenance / limit |
|---|---|---|
| Frozen and saved output inventory with SHA-256 hashes | [artifact_manifest.json](../artifact_manifest.json) | Existing outputs; the read-only continuation rechecked all 20 output hashes |
| Saved TreeSHAP and image-explainer validation | [ARTIFACT_AUDIT.md](../ARTIFACT_AUDIT.md), [NOTEBOOK_04_05_AUDIT.md](../NOTEBOOK_04_05_AUDIT.md) | Saved notebook results were inspected; no notebook was rerun |
| Protocol and evaluation guardrails | [ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md) | Proposed tests are not evidence that the tests ran |
| Literature and novelty matrix | [LITERATURE_NOVELTY_AUDIT.md](../LITERATURE_NOVELTY_AUDIT.md) | Focused and partial; ambiguous citations and full-text checks remain open |
| Frozen explanation arrays | `outputs/treeshap_v1.npz`, `outputs/image_explainers_v1.npz`, `outputs/modality_shapley_v2.npz` | Existing files; not modified in this session |
