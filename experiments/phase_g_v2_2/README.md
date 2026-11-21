# Phase G v2.2 condition runner

The v2.2 package contains condition-specific data generation, CUDA-only
training/calibration orchestration, safe checkpoint resume, and a fail-closed
runner for the five remaining conditions. It does not rerun the balanced
pilot, change historical Phase G implementations, or overwrite their outputs.

## Status

The checked-in configuration keeps `execution_enabled: false`, an empty
condition allowlist, and test-data access forbidden. The prospective v2.1
protocol is still a draft, so a valid run authorization cannot currently be
issued. The runner also blocks training until protocol compatibility is
resolved: the protocol requires four Grad-CAM variants and SLIC superpixel
faithfulness with a renderer-background/mid-gray baseline pair, while the
available Phase G analysis implements one Grad-CAM method and a 14x14 patch
grid with different baselines. Its legacy primary gate also contains an
unapproved accuracy floor and a fixed two-contrast family.

No condition can run until the protocol, analysis methods, untouched
evaluation cohort, and predictive threshold/multiplicity policy are approved
and bound in an authorization record.

The test-stage entry point currently enforces the authorization, frozen-model,
calibration, and cohort gates, then refuses before generating test records.
It is not a substitute for the unresolved v2.1 predictive/explanation analysis
path; do not treat the runner as ready for scientific execution.

## Commands

From the repository root, use the project environment:

```powershell
python -m experiments.phase_g_v2_2.runner
python -m experiments.phase_g_v2_2.runner --condition image_dominant
python -m experiments.phase_g_v2_2.runner --preflight-only
python -m experiments.phase_g_v2_2.runner --preflight-only --preflight-output outputs/synthetic_phase_g_v2_2/preflight
python -m experiments.phase_g_v2_2.runner --run --condition image_dominant --authorization-file path\to\authorization.json
python -m experiments.phase_g_v2_2.runner --run --condition image_dominant --authorization-file path\to\authorization.json --resume
```

Replace `image_dominant` with `clinical_dominant`, `redundant`,
`conflicting`, or `synergistic` as appropriate. The default and
`--condition` modes are validation-only. `--preflight-only` is explicitly
non-evidentiary and never writes to a condition result directory.
`--run` requires a separate authorization file and fails before data
generation while the protocol or implementation checks remain unresolved.

The supported condition output paths are
`outputs/synthetic_phase_g_v2_2/<condition>`. Existing or mismatched run
directories are not overwritten. Resume requires matching configuration and
source hashes; completed per-model checkpoints are hash-verified before reuse.
Training requires CUDA device 0 and the configured batch size 64. CPU fallback
and mixed precision are not allowed.

## Preflight

The final non-evidentiary batch-64 preflight is at
`outputs/synthetic_phase_g_v2_2/preflight-2026-11-21-final2/preflight_report.json`.
It trained each of the three Phase G models for one epoch on 1,024 disposable
training and 256 model-selection patients. It did not train a full model,
read test data, or execute a condition. Runtime estimates are extrapolations
to the configured 50-epoch maximum and exclude calibration, test evaluation,
explanations, statistics, and report generation.

The runner calculates a lower bound for host RAM required to retain the
training and model-selection image/mask arrays. The observed free RAM was
below this raw-array floor, so an actual condition run would also stop for
insufficient memory even if protocol authorization were supplied.
