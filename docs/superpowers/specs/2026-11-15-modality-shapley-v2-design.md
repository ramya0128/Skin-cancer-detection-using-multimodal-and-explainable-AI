# Modality Shapley v2 Design

## Intent

Create a separate `03b` notebook that makes input-level modality Shapley the primary per-case analysis, while retaining the existing probability-level analysis as a secondary comparison. The purpose is to explain model behavior without changing or overwriting frozen v1 results.

## Scope and constraints

- Read the frozen patient split and existing base outputs; do not change them or train on test labels.
- Keep OOF explanations cross-fitted. For each OOF fold, temporarily refit the matching XGBoost and two-stage ViT on that fold's training partition, calculate absent-modality evaluations, then discard both models. Persistent fold-model checkpoints are not needed.
- For absent tabular input, average XGBoost probabilities over 50 fixed, seeded, class-stratified rows from the applicable training partition.
- For absent image input, apply a deterministic full-image Gaussian blur to the raw 224x224 image and evaluate it with the matching ViT. The ViT is fine-tuned in stage 2, so do not assume a reusable frozen feature extractor.
- Use the final saved branch models and all training rows only for the test explanations. Test results are reporting-only.
- Report modality shares by true and predicted class, class counts, patient-cluster uncertainty, the CWF class weights, and an error-detection preview. Do not report negative-contribution rates as findings.
- Save results to `outputs/modality_shapley_v2.npz`; leave all `_v1` artifacts unchanged.

## Approaches considered

1. **Transient fold refitting (chosen):** refit each OOF fold's branch models, calculate its counterfactuals, and discard the models. This keeps coalition evaluations aligned with the models that produced each OOF row without creating checkpoint artifacts.
2. **Use final models for OOF counterfactuals:** cheaper, but mixes full-input OOF predictions with counterfactuals from models trained on the held-out rows. This can leak held-out labels into the OOF baseline, so it is not suitable for the primary analysis.
3. **Use only saved OOF probabilities:** preserves v1 exactly but cannot evaluate blurred images or marginalize tabular inputs through the branch models; it remains the secondary probability-level analysis.

## Output and validation

The notebook will save signed input-level Shapley values, shares, predictions, coalition payoff/baseline values, probability-level comparison values, the fixed background row indices, class labels, and CWF weights. It will assert data alignment, expected row counts and probability shapes, the two-player efficiency identity, and agreement of the retained probability-level results with v1. Summaries will include OOF and test class counts and flag the small MEL test count.
