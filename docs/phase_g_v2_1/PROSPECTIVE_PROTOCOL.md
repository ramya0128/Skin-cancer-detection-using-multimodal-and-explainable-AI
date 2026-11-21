# Phase G v2.1 Prospective Protocol for the Remaining Five Conditions

**Status:** Frozen as protocol structure; not approved for execution.
**Existing v2 diagnostic:** The 72-patient subset is a validation diagnostic,
not preregistered or untouched. No result from that subset is carried into
method selection for the remaining conditions.

## 1. Condition definitions

| Condition | Image cue reliability | Clinical cue reliability | Construction |
|---|---:|---:|---|
| image_dominant | 0.95 | 0.60 | Independent cue corruption, uniform among the five other classes |
| clinical_dominant | 0.60 | 0.95 | Independent cue corruption, uniform among the five other classes |
| redundant | 0.85 shared both cues | 0.85 shared both cues | Shared latent noisy class |
| conflicting | Marginal 0.675 | Marginal 0.675 | Half image-correct/clinical-decoy and half clinical-correct/image-decoy; remaining noise 0.85 |
| synergistic | Individual share has no label information | Individual share has no label information | Joint components sum to label modulo 6 |

## 2. Data and partitions

- Synthetic 12,000-patient design, 6 classes, fixed master seed 20261119.
- 1,400 train, 150 model-selection, 150 calibration, and 300 test patients per class.
- Splits are sampled before rendering and remain disjoint.
- A fresh prospective untouched evaluation subset must be declared in the
  protocol before condition results are inspected.
- The existing v2 72-patient subset is excluded from untouched-evaluation claims.

## 3. Frozen models

- ViT image encoder: 224x224, 16x16 patches, 4 encoder layers, hidden 192,
  3 heads, MLP 384.
- Clinical embedding dimension 16; joint fusion hidden dimension 128.
- Independent image/clinical training and joint cross-entropy only.
- AdamW learning rate 0.0003, weight decay 0.0001, batch 64, maximum 50 epochs,
  model-selection early stopping patience 8.
- Seeds remain 20261120 image-only, 20261121 clinical-only, and 20261122 joint.
- Calibration temperatures are fit only on validation-calibration logits and
  frozen before test evaluation.

## 4. Explainers and invalid-map handling

Prespecified methods:

- Grad-CAM final block, gradient-global;
- Grad-CAM final block, token-local;
- Grad-CAM penultimate block, gradient-global;
- Grad-CAM penultimate block, token-local;
- Chefer;
- attention rollout;
- occlusion.

Invalid maps are excluded from localization and faithfulness. Recorded reasons
include nonfinite, no positive support, constant map, negative attribution,
zero map, and upstream invalidity.

## 5. Localization

Only valid maps enter lesion, marker, and complementary-background shares.
Shares sum to one. Region-area enrichment and pointing-game remain separate.
Translated-control mass uses availability `(N, R)`:

- requested translations count every repeat;
- available translations count repeats with valid placement;
- unavailable translations count repeats with no valid placement;
- patient means require at least one valid repeat;
- no-valid-patient result is NaN;
- valid zero-mass controls remain valid zeros;
- the aggregate denominator is the count of valid available repeats.

## 6. Faithfulness

- 128 connected SLIC superpixels, compactness 10.0, sigma 0.
- 224x224 uint8 RGB, `convert2lab=False`, `enforce_connectivity=True`,
  `start_label=0`, channel-axis -1.
- SLIC seed 20261144; exact seed and scikit-image version are manifest fields.
- 21 affected-area fractions.
- Baselines: renderer background and mid-gray.
- Controls: 20 patient-specific random superpixel orders and center-first order.
- Patient-level curves and AUC are the statistical unit.

## 7. Geometry

Controls are exact integer translations preserving lesion shape and area.
Offsets must leave no lesion/marker overlap and remain in image bounds.
Unavailable controls remain unavailable, never random-pixel substitutions.
Random-pixel area-matched masks remain a separately named exploratory null.

## 8. Predictive contribution and calibration

Primary predictive analysis remains calibrated held-out multiclass NLL.
Temperature scaling is fit only on validation-calibration data and frozen.
Paired patient-level bootstrap uses 10,000 replicates only after test
calibration freeze. It refits only validation-calibration temperatures in
bootstrap draws.

Existing 0.10-nat threshold and Holm family are **not automatically carried**
into this protocol. They require documented approval.

## 9. Statistical analysis

- Primary statistical unit: patient.
- Paired bootstrap resamples entire patient IDs.
- Multiplicity families must be frozen before results:
  1. predictive NLL contrasts;
  2. localization metrics;
  3. faithfulness methods and directions;
  4. cue intervention sensitivity.
- One planned family correction is Holm or another prespecified method.
- Small-sample and missing validity counts are reported by denominator.
- No test data may be used for method or threshold selection.

## 10. Outcomes

**Primary:** incremental calibrated NLL contribution per condition.

**Secondary:** localization, independent superpixel faithfulness, cue
intervention sensitivity, geometry availability.

**Exploratory:** stability, attribution mass, and v2 72-patient diagnostic
comparisons.

A condition success criterion is unresolved until the researcher approves
the predictive threshold and multiplicity-adjusted interpretation.

## 11. Execution gate

No condition is authorized by this document alone. Execution requires:

1. Explicit condition allowlist;
2. Protocol approval;
3. Configuration hash;
4. Test-data access guard;
5. Frozen checkpoints/calibrators;
6. Separate authorization for any condition run.
