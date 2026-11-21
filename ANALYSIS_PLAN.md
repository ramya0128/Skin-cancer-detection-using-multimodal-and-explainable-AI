# Analysis Plan

## Scope and frozen decisions

This plan evaluates explanations of the frozen PAD-UFES-20 late-fusion system. It does not alter the frozen split, branch models, or fusion weights. Because one test image was used during explanation debugging, test-set evaluation is exploratory only; it cannot support confirmatory claims or method selection.

## Primary ground truth

- Payoff: fused probability of the predicted class, using the CWF fused probability.
- Coalition values are computed for `{tabular, image}`.
- Absent tabular: mean XGBoost probability over a fixed 50-row training background.
- Primary tabular background: class-stratified sampling from the applicable training partition.
- Absent image: Gaussian-blurred raw 224x224 image, radius 32, evaluated through the matching ViT.
- OOF: cross-fitted fold models with a fold-training background, five GroupKFold folds.
- Test: final saved branch models and a background sampled only from all frozen training rows.
- Primary per-case result: input-level exact two-player Shapley.
- Probability-level Shapley remains secondary and is not substituted for input-level shares.

The completed OOF checkpoint already contains all five fold results. It must be restored and validated; completed folds must be skipped. No OOF ViT retraining is required for Task 0 or for subsequent work when all five folds are complete.

## Task 0 background result

The class-stratified primary v2 background was compared with a random, non-stratified 50-row background:

| Statistic | OOF | Test |
|---|---:|---:|
| Mean rho_T v2 | 0.598 | 0.584 |
| Mean rho_T random | 0.569 | 0.566 |
| rho Spearman random vs v2 | 0.953 | 0.978 |
| phi_T sign agreement | 95.7% | 95.0% |
| NEV input-v-prob Spearman v2 | 0.445 | 0.228 |
| NEV input-v-prob Spearman random | 0.406 | 0.202 |

The random background did not shrink the NEV input-vs-probability gap. The class-stratified v2 background remains primary. The random version is retained only as a sensitivity analysis.

## Modality-share conversion rule

No unitless heatmap is converted into a modality share on its own.

### Recommended rule

For explainers producing signed or unsigned tabular/feature attributions and image attributions with probability-like units:

`share_T = sum(|attribution_T|) / (sum(|attribution_T|) + sum(|attribution_I|))`

The branch contributions must first be scaled by the CWF tabular weight for the fused predicted class:

- Tabular magnitude: `w_xgb[predicted_class] * sum(|tabular attribution|)`.
- Image magnitude: `(1 - w_xgb[predicted_class]) * sum(|image attribution|)`.

Image maps are fixed to 14x14 and normalized with ReLU followed by per-map min-max normalization. The image sum uses the sum of all 196 normalized map cells.

For unitless maps, including Grad-CAM, attention rollout, and Chefer relevance, the only share comparison uses the frozen stitched conversion above; the map itself is not treated as a standalone modality-share estimate.

### Alternative rule not adopted

Raw signed attribution sums could preserve sign but can cancel and do not provide a stable two-modality fraction. This is not adopted for the primary share metric.

## Explainer metrics

- `share error = |share_explainer - share_exact|`, per case and pooled.
- Tabular-phi sign agreement.
- Spearman agreement of per-case shares.
- Per-class means and per-class counts; MEL test `n=7` remains flagged.
- Patient-cluster bootstrap confidence intervals; 2,000 resamples.
- Deletion/insertion AUC versus 5% steps of removal or insertion.
- Tabular deletion replaces grouped variables with fixed background values in explanation order.
- Image deletion/insertion uses the blur baseline.
- Random deletion/insertion order: mean of 20 fixed-seed runs.
- Randomisation similarity: SSIM and Spearman correlation.
- Stability under horizontal flip and +/-10% brightness, restricted to unchanged-prediction cases.

Random and centre-Gaussian maps are the reference baselines for every localization and faithfulness metric. The centre-Gaussian map is a 14x14 Gaussian centered at the map center and normalized by the same fixed rule.

Deletion/insertion AUC is reported separately for the ViT alone and the fused model. It is not collapsed to a single ranking.

## Map normalisation

- Generate the 14x14 raw map.
- Apply ReLU.
- Apply per-map min-max normalization.
- Keep the normalized map for all image metrics.

Occlusion patch size and stride are fixed before occlusion outputs are viewed. They should be recorded in the notebook's config and saved NPZ metadata.

## Reference baselines

- Random map: seed-controlled random 14x14 map, re-normalized by the fixed rule.
- Centre-Gaussian map: fixed 14x14 symmetric Gaussian, re-normalized by the fixed rule.
- Random deletion order: fixed seeds, 20 independent orderings.

## Selection and multiple-comparison rules

- No test-set tuning.
- OOF data is used for method development only; test data is used once for reporting.
- Primary contrast: input-level share error and sign agreement.
- Exploratory contrasts: rank agreement, faithfulness, randomisation, stability, and error detection.
- Per-class and per-method results are reported without selecting only favorable subsets.
- Confidence intervals use patient clusters.
- Multiple comparisons use Holm correction within a predeclared family, while uncorrected point estimates remain visible.
- No method is ranked by a single metric.

## Limitations

- MEL test count is 7; lead with OOF results.
- SCC recall is low.
- There is no external validation or clinician validation.
- Lesion boxes and concept masks are author annotations until independently validated.
- Lesion masks, concept labels, and lesion boxes do not exist in the base dataset.
- Synthetic planted-cue truth is more definitive than PAD-UFES-20 observational attributions, and will be central to later work.

## Updated master-brief addendum (2026-11-18)

This addendum records the state of the saved 04/05 explanation outputs and gates any new test-set analysis. It does not authorize notebook reruns, model fitting, retraining, or additional test-set experimentation.

### Draft hypotheses and decision rules

These are proposed for review, not a claim that the corresponding experiments ran or an approval to analyze the test set. Any inferential test-set use requires explicit approval after this plan is reviewed.

| ID | Hypothesis | Primary measurement | Planned comparison |
|---|---|---|---|
| H1 | A method intended to estimate modality contribution should agree with the exact input-level two-player CWF Shapley reference. | Per-case absolute modality-share error; tabular contribution sign agreement. | Compare each candidate estimator against the frozen exact input-level reference; report per-class and pooled estimates. |
| H2 | A faithful explanation should cause a larger target-score decrease when its highest-ranked evidence is removed than random and centre-Gaussian rankings. | Deletion AUC over 5% increments through full removal; lower is better for deletion. | Compare each method with random and centre-Gaussian references, separately for ViT and CWF targets. |
| H3 | Explanations should remain similar under small image transformations when the predicted class is unchanged. | Spearman correlation and SSIM of normalized maps after horizontal flip (mapped back) and +/-10% brightness. | Report only unchanged-prediction cases; no threshold or transform tuning on test data. |
| H4 | On a planted-cue benchmark, explanations should assign more attribution mass to the planted cue than to area-matched distractors. | Cue attribution mass, cue localization, and deletion effect under a predeclared cue mask. | Compare against random and centre-Gaussian maps; not testable until a separately approved benchmark exists. |

H1-H4 are candidate questions, not evidence that the planned comparisons ran. The frozen test outputs have already been observed, and one test image was used during image-explainer implementation debugging; any further analysis on that split is exploratory, even after approval. A confirmatory claim requires an untouched external/holdout set and an approved, versioned protocol. H4 is not started and must not be used to justify training under the current no-retraining instruction.

### Results already observed

- `04_treeshap.ipynb` and `05_image_explainers.ipynb` have saved outputs for every code cell. Existing results are audited in `NOTEBOOK_04_05_AUDIT.md`; checksums and timestamps are in `artifact_manifest.json`.
- The existing final XGBoost and ViT checkpoints, split, and v1 base outputs are frozen. No checkpoint or split may be replaced.
- The 04 SHAP background is 200 class-stratified training rows. The modality-Shapley v2 background is separately 50 class-stratified training rows; these are different estimands and must not be conflated.
- The 04 native `pred_contribs` rank check is a raw-margin grouped-attribution comparison; its saved print label incorrectly says “exact probability grouped.” It is only a rank cross-check, not the primary explanation.
- In 05, “fused” names identify the saved CWF-selected target class. The pixel map is still computed from the ViT branch, not from end-to-end CWF probabilities.
- A one-test-image attention implementation smoke check occurred during debugging. This is disclosed as a protocol deviation in `DATA_AUDIT.md`; the test-set image maps are therefore descriptive/exploratory, not clean confirmatory evidence.

### Guardrails for any future confirmatory work

1. Do not use the frozen test set for method development, background selection, debugging, explainer selection, or thresholds.
2. Do not alter or retrain the frozen v1 branch models, split, or fusion weights. Any new control or synthetic model must be a separate, versioned experiment trained only on its designated training data.
3. This fixed Phase F exploratory pass was explicitly authorized and has run. Any further analysis of the frozen test set requires a predeclared protocol and explicit approval, and must remain exploratory. Confirmatory claims require an untouched external/holdout set; the existing test results cannot be made confirmatory retroactively.
4. Report clinical-branch TreeSHAP, ViT-branch image maps, and CWF-level modality Shapley as separate explanation targets. Do not call branch-specific maps explanations of the fused function.
5. Keep real-data evaluation and any planted-cue synthetic benchmark separate. The benchmark is authorized as a controlled new experiment but has not yet been executed.

### Current status

### Phase F execution update (2026-11-19)

The user authorized one fixed exploratory evaluation. Phase F Tasks 1-5 were executed and validated without changing the frozen split or saved branch checkpoints. Results, configs, logs, hashes, per-class summaries, paired comparisons, and figures are under `outputs/phase_f_v1/`; the latest report is `outputs/phase_f_v1/summary_v3/report.md` (the validated `summary_v2` is preserved). The earlier approval-pending status below is superseded by this dated update, but the test-exposure limitation remains.

- Image deletion/insertion and random-order curves cover 461 ordered rows (116 chunks); image stability and ViT randomization cover all rows (696 chunks). These results use ViT-branch maps and report ViT and recomputed CWF scores separately.
- Clinical interventions evaluate all 256 grouped coalitions against the saved 200-row class-stratified training background for each row. Grouped-Shapley efficiency residuals are below `6.7e-16`; the frozen XGBoost test predictions reproduce exactly.
- The seed-42 label-permuted XGBoost 3.2.0 control was fit on the 1,837 training rows only. Its exploratory test accuracy/macro-F1 are `0.236/0.105`; grouped-rank Spearman versus the frozen model is `0.298` for XGBoost and `0.289` for CWF.
- Summary reporting uses 2,000 patient-cluster bootstrap resamples and 10,000 patient-level sign flips across four Holm families. Across the 48 image deletion and 48 image insertion comparisons, 45 and 46, respectively, had Holm-adjusted `p < 0.05`; the per-comparison deltas and intervals are reported without selecting a method.
- Clinical TreeSHAP order versus the 20-order random baseline had paired AUC deltas (TreeSHAP minus random) of `-0.148` (95% patient-cluster CI `[-0.159, -0.137]`) for XGBoost deletion, `+0.155` (`[0.142, 0.168]`) for XGBoost insertion, `-0.071` (`[-0.076, -0.066]`) for CWF deletion, and `+0.074` (`[0.068, 0.080]`) for CWF insertion; each comparison has Holm-adjusted `p = 0.00020`. Treat these as exploratory.

The prior debugging use of one test image remains a protocol deviation: none of these real-data test results is confirmatory, and no further method development may use this split. Phase G, Phase H, the remaining Phase I literature audit, and project-wide Phase J work are separate pending workstreams; they are not represented as complete here. The old guardrail requiring approval for any new exploratory pass is satisfied for this one fixed run only. An untouched external cohort remains necessary for confirmatory claims.
