# Problem

The project studies classification from skin-lesion images and clinical/tabular information, with a focus on whether explanation outputs faithfully describe the saved multimodal system.

The current audit separates three explanation targets: clinical-branch TreeSHAP, ViT-branch image maps, and CWF-level modality Shapley. These are not interchangeable. Exact Shapley values are exact only for the chosen game, background, and payoff; they are not clinical or causal ground truth.

## Current gaps

- Saved deletion/insertion, stability, and model-randomization outputs are absent.
- No planted-cue benchmark has been run under the current no-retraining instruction.
- No candidate second-dataset files are present for a license/field audit.
- Recent multimodal explainability and PAD-UFES-20 fusion papers narrow broad novelty claims; full-text comparisons and exact identification of several cited works remain open.

Evidence and limitations are tracked in [ARTIFACT_AUDIT.md](../ARTIFACT_AUDIT.md), [ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md), and [LITERATURE_NOVELTY_AUDIT.md](../LITERATURE_NOVELTY_AUDIT.md).
