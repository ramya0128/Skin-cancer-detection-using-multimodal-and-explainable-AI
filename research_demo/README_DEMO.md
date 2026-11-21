# Skin Research Lab — presentation dashboard

## Launch

Open PowerShell and run:

```powershell
Set-Location "D:\skin_project"
& "D:\vit_gpu_env\Scripts\python.exe" -m streamlit run "D:\skin_project\research_demo\app.py" --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

The app is local-only and read-only with respect to the research outputs. It
does not train models, run inference on the frozen dataset, generate patients,
or change Phase G authorization. The **Upload case** page runs the existing
saved models only on a case explicitly supplied by the user.

The existing environment supplies PyTorch 2.7.1+cu128 for loading the saved
image cache on CPU (`weights_only=True`, `map_location="cpu"`) and running
uploaded-image inference on the available GPU. PyTorch is intentionally not
included in the dashboard requirements to avoid replacing the project's
GPU-specific build.

The upload page reads the metadata path configured by `common.META`. To use a
different local metadata file for a session, set `SKIN_RESEARCH_METADATA` in
PowerShell before launching Streamlit:

```powershell
$env:SKIN_RESEARCH_METADATA = "D:\path\to\metadata.csv"
```

If dependencies need to be restored in the existing environment:

```powershell
& "D:\vit_gpu_env\Scripts\python.exe" -m pip install -r "D:\skin_project\research_demo\requirements.txt"
```

## Pages

- **Overview** — verified real-data and synthetic-track status, with the three
  explanation targets clearly separated.
- **Real-data results** — computes descriptive accuracy, macro-F1, NLL, class
  support, and confusion matrices from aligned saved OOF/test probabilities.
- **Explainability gallery** — sample-level CWF modality Shapley, clinical
  XGBoost grouped probability Shapley, and ViT maps with their target and
  artifact provenance. ViT maps are paired with the matching frozen test image
  through the saved cache and split row keys.
- **Upload case** — accepts a PNG/JPEG and seven clinical fields plus age,
  predicts with the saved ViT and XGBoost branches and CWF fusion, computes a
  fresh ViT Grad-CAM map and grouped clinical probability-Shapley values, and
  shows the class-specific weighted fusion terms. No per-upload accuracy or
  F1 is reported.
- **Synthetic benchmark** — shows the five configured v2.2 condition designs,
  current blocked authorization state, and historical balanced v1 pilot
  metrics separately.
- **Limitations** — surfaces test-set reuse, MEL support, split caveats,
  explanation scope, and clinical-use limitations.

## Artifact safety and interpretation

- Files are read from the existing `outputs/` and `cache/` directories; no
  research artifact is copied or overwritten.
- NPZ arrays are opened with `allow_pickle=False`. Object-valued members are
  skipped and reported; numeric arrays remain available. Class order comes
  from the safe Unicode/JSON fusion metadata.
- Displayed real-data metrics are computed from the saved labels and
  probabilities only. They are descriptive/exploratory because the frozen
  test set has already been used for reporting and explanation work.
- Upload inference uses the saved `vit_final_v1.pt`, `xgb_final_v1.joblib`,
  `fusion_v1.json`, training split, and saved 200-row training background.
  Uploaded bytes and clinical values are processed in the Streamlit session
  and are not written to disk. The page labels CWF, ViT, and XGBoost outputs
  separately; Grad-CAM targets the CWF class, while grouped probability-SHAP
  explains the XGBoost branch's own predicted class.
- Per-upload accuracy/F1 is not meaningful without verified ground truth.
  Saved cohort metrics remain on **Real-data results** and do not measure an
  uploaded case.
- CWF modality Shapley explains the specified two-player fused prediction
  game. TreeSHAP explains the clinical XGBoost branch. Pixel maps explain the
  ViT branch, even when the target class was selected by CWF.
- The historical balanced Phase G v1 pilot is not a v2.2 result or a Phase G
  completion. v2.2 remains draft/unapproved with execution disabled and test
  access forbidden.
- This is a research presentation, not a diagnostic or clinical
  decision-support tool.

## Generated demo evidence

The demo build stores its artifact inventory, smoke-test log, and local
screenshots under this folder:

```text
artifact_inventory.json
logs\dashboard_smoke_test.txt
screenshots\
```

The inventory records file presence, size, timestamps, and safe NPZ
array-schema details. Screenshots are presentation evidence only; they are not
research results.

The upload page's blank-form screenshot is `screenshots\06_upload_case.png`.

## Upload-flow verification

The complete dashboard suite is run with:

```powershell
& "D:\vit_gpu_env\Scripts\python.exe" -m unittest discover -s research_demo\tests -v
& "D:\vit_gpu_env\Scripts\python.exe" -m unittest discover -s experiments\phase_f\tests -p test_shapley.py -v
& "D:\vit_gpu_env\Scripts\python.exe" -m py_compile research_demo\app.py research_demo\inference.py research_demo\data_utils.py
```

The upload smoke test uses a generated, non-dataset RGB image and synthetic
clinical inputs. It does not evaluate a research cohort or create a research
result. See `logs\dashboard_smoke_test.txt` for the latest run details and
verified launch/health-check commands.
