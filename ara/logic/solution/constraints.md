# Constraints

- No model retraining, notebook reruns, or new model inference under the current user-selected read-only scope.
- Preserve the frozen v1 outputs and test split.
- Do not use the test set for method development; existing test outputs are already observed and must be treated as exploratory.
- Do not fabricate missing perturbation or planted-cue results.
- Do not infer dataset license or schema from a description; inspect actual authorized files first.
- Keep branch-specific image/clinical explanations distinct from explanations of the fused function.
