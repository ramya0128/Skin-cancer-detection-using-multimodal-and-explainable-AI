# Claims

## C01: Broad multimodal-explainability novelty is not established
- **Statement**: The project cannot currently claim to be the first multimodal explanation framework or the first multimodal faithfulness benchmark.
- **Status**: weakened
- **Provenance**: ai-suggested
- **Falsification criteria**: A sufficiently scoped literature review would need to show that the identified close works do not cover the claimed contribution.
- **Proof**: [LITERATURE_NOVELTY_AUDIT.md](../LITERATURE_NOVELTY_AUDIT.md)
- **Dependencies**: []
- **Tags**: novelty, related-work

## C02: A lesion-specific planted-cue evaluation may be a narrower contribution
- **Statement**: A controlled evaluation for patient-level image-plus-clinical fusion, with specified exact two-player modality Shapley and planted cues of known reliability/interactions, may provide a defensible distinction from the cited vision-language and text-tabular work.
- **Status**: hypothesis
- **Provenance**: ai-suggested
- **Falsification criteria**: Full-text review finds a substantially equivalent image-plus-clinical planted-cue benchmark, or the planned benchmark fails to establish known cue attribution under its prespecified game.
- **Proof**: pending
- **Dependencies**: [C01]
- **Tags**: planted-cues, modality-shapley, novelty

## C03: Saved-output integrity was preserved during this read-only continuation
- **Statement**: The 20 output-file hashes recorded before the documentation update still match the saved files after it.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: Any output-file SHA-256 differs from its pre-update manifest entry.
- **Proof**: [artifact_manifest.json](../artifact_manifest.json)
- **Dependencies**: []
- **Tags**: provenance, artifact-integrity
