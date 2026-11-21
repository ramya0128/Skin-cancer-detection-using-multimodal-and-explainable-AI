# Notebook 04/05 Execution and Method Audit

Audit date: 2026-11-18. This is a read-only audit of saved notebooks and outputs. No model was fit, no notebook was rerun, and no new test-set inference was performed.

## Execution evidence

| Notebook | Code cells executed | Saved cell errors | Saved output | Status |
|---|---:|---:|---|---|
| `04_treeshap.ipynb` | 10/10 | 0 | `outputs/treeshap_v1.npz` | Executed; core numerical checks pass |
| `05_image_explainers.ipynb` | 6/6 | 0 | `outputs/image_explainers_v1.npz` | Executed; arrays validate; provenance/method limitations remain |

Saved output is evidence that the notebook cells ran, not by itself proof that the output NPZ corresponds to the latest notebook source. In particular, the image NPZ timestamp (15:01) predates the notebook's last-write time (20:54) on 2026-11-18.

## 04 — clinical branch TreeSHAP

**Status: executed and numerically validated for the saved final XGBoost model.** The notebook uses SHAP 0.52.0 and XGBoost 3.2.0, loads the frozen final model, and does not retrain it. The model was saved in an environment whose registry records XGBoost 3.4.1; the notebook records exact prediction reproduction for its loaded model.

- Frozen test rows: 461; classes: ACK, BCC, MEL, NEV, SCC, SEK.
- Grouped variables: `grew`, `itch`, `changed`, `elevation`, `bleed`, `hurt`, `region`, `age`.
- Raw-margin TreeSHAP background: 200 class-stratified training rows.
- SHAP 0.52 raw additivity maximum absolute residual: `4.293333836e-6`, below `1e-4`.
- Exact grouped probability Shapley: all 256 coalitions, replacing all one-hot columns of each variable together; 200-row background; softmax probabilities evaluated for each coalition.
- Probability efficiency maximum absolute error: `6.661338148e-16`, below `1e-5` (and `1e-4`).
- The notebook contains an additive 8-variable/6-output toy-game unit test; it asserts exact recovery and efficiency at `1e-12`.
- It times a two-case coalition batch before the full saved test evaluation.
- OOF TreeSHAP remains unavailable because fold-specific XGBoost weights were not persisted. The notebook correctly marks this not computed; no retraining was performed.

Per-class residual in `margin - (sum(phi) + expected_value)`; “constant” uses peak-to-peak `< 1e-4`:

| Class | Mean residual | Std. dev. | Max absolute residual | Constant offset? |
|---|---:|---:|---:|---|
| ACK | -1.4873668e-08 | 6.8594047e-07 | 4.1107408e-06 | Yes |
| BCC | -3.5798834e-08 | 4.3090157e-07 | 2.0387125e-06 | Yes |
| MEL | -4.7846278e-07 | 1.4156409e-06 | 4.2933338e-06 | Yes |
| NEV | -3.6313653e-07 | 1.1900931e-06 | 3.8695735e-06 | Yes |
| SCC | -9.1635643e-08 | 5.9097939e-07 | 2.5212893e-06 | Yes |
| SEK | 7.3477158e-08 | 4.6232715e-07 | 1.9709087e-06 | Yes |

### 04 findings and caveats

1. **Mislabelled rank-check output.** The notebook prints “native pred_contribs vs exact probability grouped rank Spearman: 0.9190”, but the code compares grouped native contributions to `raw_grouped`, not `probability_grouped`. The value is a raw-margin rank cross-check and should be relabelled if the notebook is next intentionally revised. The saved primary results remain the interventional grouped explanations, not native `pred_contribs`.
2. **Importance denominator mismatch in original console output.** The class-level mean absolute attribution is averaged over all 461 test rows, while the console line also prints the count of true diagnoses for that class. The audit-only `global_importance_by_output_class_v2.csv` corrects the aggregation label and explicitly states that every output class is averaged over all test rows.
3. **Shortcut grouping semantics.** The audit identifies encoded `UNK`/`NAN` dummy activations, not raw nulls. The eight source metadata variables have no raw missing values. Counts are highly sparse for some variables; e.g., no test rows activate the `UNK`/`NAN` dummy for elevation, region, or age. Audit-only diagnosis-stratified tables are available in `outputs/shortcut_audit_by_diagnosis_v2.csv` and `outputs/shortcut_attribution_by_predicted_class_v2.csv`.
4. The requested 50-vs-200 TreeSHAP background sensitivity and a separate 20-row raw grouped-coalition-vs-TreeSHAP comparison are not present in the saved artifact. They were not run during this audit.

## 05 — image branch explainers

**Status: executed; saved arrays structurally and numerically validated; method scope and provenance are limited.**

- Eight arrays have shape `(461, 14, 14)`, are finite, and lie in `[0,1]`.
- Test-row keys, true labels, class order, and CWF targets align with frozen artifacts.
- Methods saved: Grad-CAM, class-independent attention rollout, Chefer-style gradient-weighted attention rollout, and ViT occlusion. Occlusion replaces raw 16x16 patches at stride 16 with Gaussian-blurred pixels.
- The notebook records Python 3.11.15, NumPy 2.4.6, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA 12.8, and RTX 5050 Laptop GPU.
- Recomputed ViT probabilities differ from frozen base-output probabilities by a maximum of `0.0007751882`, with one argmax disagreement.
- The saved fused-target and ViT-top-target maps use different target classes on 140/461 rows. This is not the same statistic as the one argmax disagreement between recomputed ViT predictions and frozen base predictions.
- The two attention-rollout arrays are exactly identical, as expected for class-independent rollout.
- The notebook does not persist an attention-wrapper-vs-normal-forward assertion.
- A prior one-image manual attention-wrapper smoke check showed max logit delta `1.788e-6` and connected gradients through all 12 blocks. It used a test image during implementation debugging; this is a protocol deviation, not clean confirmatory validation.
- `Chefer` is accurately described as a Chefer-style positive attention-times-class-logit-gradient residual rollout, not established here as an exact reproduction of the published method.
- Neither the fused target names nor the saved maps imply attribution of the fused function: maps use ViT branch logits/probabilities, with the CWF class used only to choose the target.

The saved visual audit grid, `outputs/visual_audit_image_explainers_v1.png`, is report-only and must not be used to select a method.

## Branch-versus-fusion explanation inventory

| Explanation/result | Function actually explained | Test/final or OOF | Correct interpretation |
|---|---|---|---|
| `treeshap_v1.npz` raw TreeSHAP | Final XGBoost clinical/tabular branch, raw margins | Final-model test | Grouped clinical-variable contributions; not image or fused-model explanations |
| `treeshap_v1.npz` grouped probability Shapley | Final XGBoost clinical/tabular branch probabilities | Final-model test | Exact grouped interventional Shapley over 8 variables and 256 coalitions; not CWF fusion |
| `image_explainers_v1.npz` Grad-CAM / Chefer-style / rollout / occlusion | Final ViT image branch | Final-model test only | “Fused” means CWF-selected target class; image evidence is still measured on ViT |
| `modality_shapley_v1.npz` and `modality_shapley_v2.npz` | CWF late-fusion prediction with tabular and image as two players | OOF and final test | Actual modality-level fusion explanation; different estimand from branch-level pixel/feature explanations |
| `modality_shapley_v2b_randombg.npz` | Two-player input-level modality game with random 50-row background | OOF and final test | Background sensitivity result; not the 200-row TreeSHAP background analysis |

## Scope and next-step status

- Audit-only diagnostic CSVs and visual grid: **completed** from saved artifacts.
- 50-vs-200 TreeSHAP sensitivity and 20-row raw grouped comparison: **not validated; not rerun**.
- Additional analysis of the frozen test set: **paused pending explicit approval** and would be exploratory only; confirmatory claims require an untouched external/holdout set.
- Planted-cue benchmark: **not started**; user instruction prohibits model retraining, and no saved cue-run outputs were found.
- No outputs, frozen model files, split files, or original notebook files were overwritten by this audit.
