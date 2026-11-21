# Saved Artifact Audit

Audit date: 2026-11-18. This report inventories the current saved project outputs and distinguishes execution evidence from validation. The complete SHA-256 manifest is `artifact_manifest.json`.

## High-level status

| Work item | Status | Evidence / limitation |
|---|---|---|
| Frozen split and base outputs | Preserved | `split_v1.json` and `base_outputs_v1.npz` remain present; hashes included in manifest |
| Final XGBoost and ViT checkpoints | Preserved | Hashes match `MODEL_REGISTRY.json`; no fit or retraining in this audit |
| Task 2 TreeSHAP notebook | Executed and core checks validated | 10/10 code cells, no saved cell errors; SHAP 0.52.0; raw and probability efficiency checks pass |
| Task 2 image explainer notebook | Executed and arrays validated | 6/6 code cells, no saved cell errors; image-output provenance caveat and test-image debugging deviation are documented |
| Modality Shapley v1/v2 and OOF checkpoint | Saved artifacts validated | All five OOF folds complete; v1/v2 fusion weights match saved fusion weights; no fold-specific model files |
| Shortcut audit and global importance derivatives | Completed from saved TreeSHAP arrays | Three audit-only CSVs; no model inference or fitting |
| Visual audit grid | Generated from saved image maps | 18 test examples; report-only, no method selection |
| Additional TreeSHAP background/raw comparisons | Not validated | Not present in saved outputs; not run |
| Planted-cue benchmark | Not started | No saved outputs found; retraining/pilot not performed |
| Additional analysis of the frozen test set | Executed and validated once under an approved exploratory protocol | Phase F outputs are saved under `outputs/phase_f_v1/`; the latest report is `summary_v3/`; all real-test findings remain nonconfirmatory because a test image was used during debugging |

## Master brief phase status

| Phase | Objective | Status | Evidence / blocker |
|---|---|---|---|
| A | Repository and artifact audit | **Complete (read-only scope)** | Files inventoried, hashes recorded, frozen models/split checked; license remains unverified |
| B | Validate existing explanation artifacts | **Partial** | Saved TreeSHAP, modality Shapley, and image arrays checked; unavailable 50-vs-200 TreeSHAP sensitivity and 20-row raw grouped comparison remain unvalidated |
| C | Image-explainer validation | **Partial** | Shapes/ranges/alignment checked; source-to-NPZ timestamp mismatch and test-image debugging deviation prevent stronger confirmation |
| D | Honest explanation-method inventory | **Complete** | Branch-versus-fusion targets and interpretation limits recorded in `NOTEBOOK_04_05_AUDIT.md` |
| E | Pre-register evaluation plan | **Approved for one exploratory Phase F pass** | The fixed protocol, baselines, perturbations, patient bootstrap, multiplicity rules, and test-exposure limitation are recorded in `ANALYSIS_PLAN.md`; results are not confirmatory |
| F | Faithfulness and stability tests | **Executed and validated (exploratory)** | Image curves, stability/randomization, grouped clinical interventions, the training-only label-permuted control, and statistical reports cover the 461 frozen test rows; evidence is under `outputs/phase_f_v1/` |
| G | Planted-cue benchmark | **Not executed** | No synthetic pilot has run; its generator, training, pilot gate, and validation remain pending |
| H | Second-dataset feasibility only | **Not executed in this phase** | The earlier repository inventory found no candidate files; accessible external locations, licensing, and compatibility have not been newly audited |
| I | Literature and novelty audit | **Partial** | [Focused source-checked matrix](LITERATURE_NOVELTY_AUDIT.md); LEMON, GridVQA-X, cross-modal synergy, and same-dataset fusion papers identified; O-Shap/PRISM/Jin references remain unresolved and full-text review is incomplete |
| J | Reproducibility and handoff | **Partial** | Phase F scripts, fixed configs, version records, logs, and hashes are saved; literature/second-dataset work and external validation remain pending |

## Output inventory

`artifact_manifest.json` contains all files currently in `outputs/`, including:

- Frozen assets: `split_v1.json`, `base_outputs_v1.npz`, `MODEL_REGISTRY.json`, `xgb_final_v1.joblib`, `vit_final_v1.pt`, and `fusion_v1.*`.
- Explanation results: `treeshap_v1.npz`, `image_explainers_v1.npz`, `modality_shapley_v1.npz`, `modality_shapley_v2.npz`, `modality_shapley_v2_oof_checkpoint.npz`, and `modality_shapley_v2b_randombg.npz`.
- Audit-only derivatives: `global_importance_by_output_class_v2.csv`, `shortcut_audit_by_diagnosis_v2.csv`, `shortcut_attribution_by_predicted_class_v2.csv`, and `visual_audit_image_explainers_v1.png`.
- Phase F exploratory results: 116 image-curve chunks, 348 image-stability chunks, 348 ViT-randomization chunks, grouped clinical intervention/control outputs, and report tables/figures under `outputs/phase_f_v1/`. `summary_v1/` retains the failed report attempt, `summary_v2/` the first validated report, and `summary_v3/` the latest validated report with source-code hashing.
- Existing partial/temporary arrays `oof_vit_partial.npy` and `oof_vit_tmp.npy` are retained in the inventory as legacy artifacts. They are not represented as a complete OOF model checkpoint.

The manifest records each file's path, size, last-write time, and SHA-256 hash, plus notebook execution counts and recorded library versions. Hashes of the final model files match the model registry. The 04/05 execution evidence and methodological caveats are detailed in `NOTEBOOK_04_05_AUDIT.md`.

### Saved-result code-cell map

| Result | Notebook cell(s) | Notes |
|---|---|---|
| `outputs/treeshap_v1.npz` | `04_treeshap.ipynb`, code cell 10 | Raw-margin check: cell 5; exact coalition function and small-batch timing: cell 6; probability Shapley/efficiency: cell 7; shortcut audit: cell 8; native rank check and original global table: cell 9 |
| `outputs/image_explainers_v1.npz` | `05_image_explainers.ipynb`, code cell 6 | Map computation/targets are in cells 2–5; saved output includes eight test maps. Latest-source correspondence remains uncertain due the timestamp mismatch |
| Audit-only CSVs and visual grid | No generator script saved in the repository | Derived from the saved NPZs and frozen metadata; hashes are recorded, but the exact derivative-generation code is not available for rerun |

## Validation highlights

- `treeshap_v1.npz`: 461 aligned test rows, six classes, eight grouped variables; contribution arrays finite; raw additivity max residual `4.293333836e-6`; probability efficiency max residual `6.661338148e-16`; 200 unique training background rows. OOF TreeSHAP is explicitly unavailable because fold model weights were not persisted.
- `image_explainers_v1.npz`: eight finite, bounded `(461,14,14)` maps; row keys, labels, class order, and targets align with frozen artifacts. The ViT probability recomputation differs from frozen base output by max `0.0007751882`, with one argmax disagreement.
- `modality_shapley_v1.npz` and `modality_shapley_v2.npz`: efficiency residuals are at floating-point precision; both fusion-weight arrays match `fusion_v1.npz`. All five OOF folds are marked complete in the saved checkpoint.
- `modality_shapley_v2b_randombg.npz`: saved random-background input-level modality results satisfy efficiency at floating-point precision. Their share-rank correlation with v2 is 0.9535 on OOF and 0.9778 on test; tabular-sign agreement is 95.7% and 95.0%, respectively. This is a saved-output sensitivity check, not a new experiment.
- Four audit-only derivatives were created from saved explanation arrays and metadata; they do not modify any frozen model, split, notebook, or original output.
- Phase F input validation confirmed 461 ordered test IDs, 40 encoded clinical columns partitioned into eight groups, a class-stratified 200-row training background, exact frozen XGBoost test-prediction reproduction, and zero train/test patient overlap.
- Frozen and label-permuted-control grouped probability explanations each evaluated all 256 coalitions against the same 200 background rows. Maximum efficiency residuals were below `6.7e-16`; the control was fitted only on training rows and permuted training labels.
- The report contains 194 pooled and 1,164 per-class metrics plus 100 paired comparisons in four Holm families, using 2,000 patient-cluster bootstrap resamples and 10,000 patient-level sign flips. Independent validation confirmed output hashes and unchanged frozen-input hashes.

## No-rerun guarantee

This audit performed file, array, metadata, hash, and saved-notebook-output checks only. It did not train or retrain any model, rerun either notebook, generate new model predictions, or use test data to choose an explainer, background, or threshold. The prior one-test-image attention implementation smoke test is disclosed as a protocol deviation in `DATA_AUDIT.md` and `NOTEBOOK_04_05_AUDIT.md`.

## Subsequent approved Phase F execution (2026-11-19)

After the read-only audit, the user authorized one fixed exploratory Phase F run. The saved final ViT and XGBoost checkpoints, frozen split, v1 explanation arrays, and image cache were not overwritten. New result files were written to versioned Phase F paths. The latest combined report is `outputs/phase_f_v1/summary_v3/report.md`; its CSVs contain pooled and per-class patient-cluster intervals and Holm-adjusted comparisons.

The test set contains 461 rows across 275 patients (ACK 127, BCC 174, MEL 7, NEV 49, SCC 49, SEK 55). The Task 4 control's exploratory accuracy and macro-F1 were 0.236 and 0.105; grouped-rank Spearman versus the frozen model was 0.298 for XGBoost and 0.289 for CWF. Image and clinical branch outputs remain distinct from recomputed CWF scores. The latest summary manifest also hashes `experiments/phase_f/summarize.py`; all ten listed summary outputs and frozen-input hashes were independently rechecked. All test-set results remain exploratory because of the disclosed debugging deviation; external validation is still required.

## Remaining limitations

- The 05 notebook's timestamp is later than `image_explainers_v1.npz`; exact latest-source-to-output provenance is unverified.
- The test image-explainer results are descriptive/exploratory because the implementation smoke check used a test image.
- TreeSHAP's saved native rank-check label is inaccurate; the code compares native contributions with raw grouped TreeSHAP, not probability grouped Shapley.
- TreeSHAP 50-vs-200 sensitivity and the separate 20-row raw grouped comparison are not saved.
- No planted-cue benchmark output or external validation exists.
- Dataset licensing terms remain unverified; see `DATA_AUDIT.md`.
- The paper-plan copy still says 05 had not run; that statement is stale and is superseded by the saved execution counts and outputs documented here.
