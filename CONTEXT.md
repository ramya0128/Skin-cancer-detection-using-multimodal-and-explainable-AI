# Project Context for Future Agents

## Project

This repository studies skin-lesion classification from images and clinical
features, with an emphasis on explainable multimodal fusion. The frozen split
contains 1,837 training and 461 test rows across six classes. The split is
patient-level; do not change it or use the test set for model selection.

## Notebook pipeline

- `01_base_outputs.ipynb` trains the image and clinical branch models and saves
  their OOF/test probabilities, final checkpoints, frozen split, and image
  cache. ViT fitting is expensive; the notebook's code can choose CPU, but
  training all ViT folds and the final ViT on CPU will be slow.
- `02_fusion.ipynb` reads the frozen base predictions and compares fusion
  approaches. It does not retrain the branch models.
- `03_modality_shapley.ipynb` computes probability-level modality Shapley
  explanations from the saved branch probabilities.
- `03b_modality_shapley.ipynb` computes input-level Shapley explanations. For
  each OOF fold it refits branch models on the other folds and evaluates full
  and blurred inputs. Each completed fold is atomically saved to
  `outputs/modality_shapley_v2_oof_checkpoint.npz`, and a later run restores
  the checkpoint and skips completed folds.

## Current generated artifacts

The current working copy contains the base outputs, fusion outputs, v1
probability-level Shapley results, final XGBoost and ViT checkpoints, image
cache, v2 OOF checkpoint, and final v2 results under `outputs/` and `cache/`.
The v2 OOF checkpoint was inspected on 2026-11-17 and marked folds
`[True, True, True, True, True]`.

The checkpoint contains result arrays and fold state, not fold model weights.
The notebook still needs the dataset metadata and image directory through
`common.py` to validate the frozen data and run test-set inference.

## CPU-only collaborator handoff

The current 03b setup has a hard CUDA check before the OOF checkpoint is
restored. Therefore, a CPU-only collaborator cannot run 03b unchanged, even if
they receive the completed checkpoint. The final ViT test inference also runs
when the notebook is executed.

For a complete artifact bundle, the collaborator does not need to rerun 01 or
02. The intended sequence is to configure local dataset paths, then run 03b
after a CPU-aware agent update that:

1. Selects CPU when CUDA is unavailable.
2. Restores and validates the OOF checkpoint.
3. Allows CPU execution only when all five folds are complete.
4. Stops before the fold loop if any fold would need training on CPU.
5. Uses a memory-conscious CPU inference batch for final test evaluation.

Do not silently remove the CUDA requirement for missing-fold training. If only
the finished results are needed, load `outputs/modality_shapley_v2.npz` rather
than rerunning the notebook. `README.md` has the artifact checklist and
collaborator instructions; its Drive URL is intentionally a placeholder until
the owner provides a restricted link.

## Existing implementation details

- `outputs/xgb_final_v1.joblib` is a dictionary with `model` and `columns`.
  Code loading it must use `artifact['model']` for model methods and
  `artifact['columns']` for feature alignment.
- The final ViT checkpoint is `outputs/vit_final_v1.pt`.
- `03b_modality_shapley.ipynb` checks that the final model probabilities
  approximately match the frozen base outputs. A CUDA run previously showed a
  maximum ViT probability difference of `0.000776`, mean difference
  `0.0000417`, and one changed argmax among 461 rows. The current notebook
  allows an absolute maximum deviation of `1e-3` and reports argmax
  disagreements. This tolerance accommodates observed numerical drift; it does
  not establish that the models are identical.
- No `requirements.txt` is present. Use a compatible Python/Jupyter environment
  with NumPy, pandas, scikit-learn, SciPy, Pillow, joblib, XGBoost, PyTorch, and
  torchvision.

## Research direction discussed

The user is exploring a novel **image-only explainable architecture** for a
research paper, not another multimodal fusion model. Candidate directions
discussed were hierarchical counterfactual prototypes, sufficient-and-necessary
lesion-region rationales, and an unsupervised visual-concept bottleneck. These
are brainstorming hypotheses only; no novelty claim or architecture
implementation has been completed.
