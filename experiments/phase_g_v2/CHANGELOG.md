# Phase G v2 change log

## v2.0.0 — validation-only explanation re-evaluation

- Preserved Phase G v1 source and artifacts; added a before/after SHA-256
  verification gate.
- Added four explicit Grad-CAM variants and per-map validity diagnostics so
  zero, non-finite, or constant maps cannot silently enter localization.
- Made localization patient-validity aware and report lesion, marker,
  background, enrichment, pointing-game, and translated-control attribution
  measures.
- Replaced the prior random-pixel geometry fallback with exact
  shape-preserving lesion translations that are disjoint from lesion and
  marker; unavailable controls are reported without fallback.
- Added connected-SLIC deletion/insertion evaluation, renderer-background and
  mid-gray replacement baselines, random-order and center-first controls, and
  paired patient-bootstrap summaries.
- Added separate renderer-based glyph-swap and marker-removal sensitivity
  interventions.
- Added a runner restricted to the frozen image-only and joint checkpoints and
  the `validation_model_selection` partition. It does not retrain models, load
  test-set rows/labels, or run the remaining synthetic conditions.

The v2 outputs are a separate validation-only diagnostic. They do not overwrite
or supersede the original v1 test results.
