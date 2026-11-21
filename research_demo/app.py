from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import META as PROJECT_METADATA_PATH
from data_utils import classification_metrics, load_npz_safely
from research_demo.inference import (
    CasePrediction,
    CaseRuntime,
    ClinicalInputs,
    load_case_runtime,
    predict_uploaded_case,
    prepare_uploaded_image,
)


st.set_page_config(
    page_title="Skin Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_METADATA_PATH = Path(
    os.environ.get("SKIN_RESEARCH_METADATA", PROJECT_METADATA_PATH)
)
PHASE_G_CONFIG = PROJECT_ROOT / "experiments" / "phase_g_v2_2" / "config.json"
PILOT_REPORT = (
    OUTPUTS
    / "synthetic_phase_g_v1"
    / "pilot_balanced_085_085_v1"
    / "pilot_report.json"
)

REAL_ARTIFACTS = {
    "Frozen split": ("outputs/split_v1.json", "Real-data data"),
    "Base predictions": ("outputs/base_outputs_v1.npz", "Real-data models"),
    "Fusion predictions": ("outputs/fusion_v1.npz", "Real-data fusion"),
    "ViT checkpoint": ("outputs/vit_final_v1.pt", "Real-data models"),
    "XGBoost checkpoint": ("outputs/xgb_final_v1.joblib", "Real-data models"),
    "CWF modality Shapley v1": (
        "outputs/modality_shapley_v1.npz",
        "CWF explanation",
    ),
    "CWF modality Shapley v2": (
        "outputs/modality_shapley_v2.npz",
        "CWF explanation",
    ),
    "XGBoost TreeSHAP": ("outputs/treeshap_v1.npz", "XGBoost explanation"),
    "ViT image maps": ("outputs/image_explainers_v1.npz", "ViT explanation"),
    "Matched image cache": ("cache/images_uint8_224.pt", "Image gallery"),
    "Historical balanced pilot": (
        "outputs/synthetic_phase_g_v1/pilot_balanced_085_085_v1/pilot_report.json",
        "Synthetic Phase G v1",
    ),
    "Phase G v2.2 configuration": (
        "experiments/phase_g_v2_2/config.json",
        "Synthetic Phase G v2.2",
    ),
}


@st.cache_data(show_spinner=False)
def _cached_npz(path_string: str, modified_ns: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    del modified_ns
    return load_npz_safely(Path(path_string))


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}")
    return _cached_npz(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_json(path_string: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _case_runtime_signature() -> tuple[Path, tuple[tuple[str, int], ...]]:
    metadata_path = DEFAULT_METADATA_PATH
    artifact_paths = (
        metadata_path,
        OUTPUTS / "xgb_final_v1.joblib",
        OUTPUTS / "split_v1.json",
        OUTPUTS / "treeshap_v1.npz",
        OUTPUTS / "vit_final_v1.pt",
        OUTPUTS / "fusion_v1.json",
    )
    signature = []
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required upload inference artifact is missing: {path}")
        signature.append((str(path.resolve()), path.stat().st_mtime_ns))
    return metadata_path, tuple(signature)


@st.cache_resource(show_spinner="Loading the saved ViT and XGBoost checkpoints...")
def _cached_case_runtime(
    project_root: str,
    metadata_path: str,
    artifact_signature: tuple[tuple[str, int], ...],
) -> CaseRuntime:
    del artifact_signature
    return load_case_runtime(Path(project_root), Path(metadata_path))


def required_array(
    arrays: dict[str, np.ndarray], key: str, artifact_name: str
) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"{artifact_name} does not contain required array {key!r}")
    return arrays[key]


def load_real_bundle() -> dict[str, Any]:
    base, base_skipped = load_npz(OUTPUTS / "base_outputs_v1.npz")
    fusion, fusion_skipped = load_npz(OUTPUTS / "fusion_v1.npz")
    fusion_meta = load_json(
        str(OUTPUTS / "fusion_v1.json"),
        (OUTPUTS / "fusion_v1.json").stat().st_mtime_ns,
    )
    classes = required_array(fusion, "classes", "fusion_v1.npz")
    class_names = [str(value) for value in classes.tolist()]
    selected = str(fusion_meta.get("selected", "unspecified"))
    return {
        "base": base,
        "fusion": fusion,
        "base_skipped": base_skipped,
        "fusion_skipped": fusion_skipped,
        "classes": class_names,
        "selected_fusion": selected,
    }


def _partition_arrays(bundle: dict[str, Any], partition: str) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, str]]]:
    base = bundle["base"]
    fusion = bundle["fusion"]
    if partition == "OOF":
        labels_key = "y_train"
        keys = {
            "XGBoost clinical": ("oof_xgb", "base_outputs_v1.npz"),
            "ViT image": ("oof_vit", "base_outputs_v1.npz"),
            "CWF fusion": ("cwf_oof", "fusion_v1.npz"),
            "MFM fusion": ("mfm_oof", "fusion_v1.npz"),
        }
    else:
        labels_key = "y_test"
        keys = {
            "XGBoost clinical": ("test_xgb", "base_outputs_v1.npz"),
            "ViT image": ("test_vit", "base_outputs_v1.npz"),
            "CWF fusion": ("cwf_test", "fusion_v1.npz"),
            "MFM fusion": ("mfm_test", "fusion_v1.npz"),
        }

    labels = required_array(base, labels_key, "base_outputs_v1.npz")
    result: dict[str, tuple[np.ndarray, str]] = {}
    for model_name, (key, source) in keys.items():
        source_arrays = base if source == "base_outputs_v1.npz" else fusion
        result[model_name] = (
            required_array(source_arrays, key, source),
            f"{source}:{key}",
        )
    return labels, result


def _per_class_table(
    metrics: dict[str, Any], classes: list[str]
) -> pd.DataFrame:
    matrix = metrics["confusion_matrix"].astype(np.float64)
    true_positive = np.diag(matrix)
    predicted_count = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return pd.DataFrame(
        {
            "Class": classes,
            "Support": support.astype(int),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


def _plot_confusion(
    matrix: np.ndarray, classes: list[str], title: str
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#24324a",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_signed_bars(
    values: np.ndarray, labels: list[str], title: str
) -> plt.Figure:
    order = np.argsort(np.abs(values))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = ["#d46a5f" if value < 0 else "#378d83" for value in values[order]]
    axis.barh(np.asarray(labels)[order], values[order], color=colors)
    axis.axvline(0, color="#526174", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Saved contribution value")
    fig.tight_layout()
    return fig


def _display_provenance(path: str, key: str, detail: str) -> None:
    st.caption(f"Source: `{path}` · array: `{key}` · {detail}")


@st.cache_resource(show_spinner="Loading the saved test-image cache...")
def load_matched_test_images(
    cache_path: str,
    cache_modified_ns: int,
    split_path: str,
    split_modified_ns: int,
    explanation_path: str,
    explanation_modified_ns: int,
) -> np.ndarray:
    del cache_modified_ns, split_modified_ns, explanation_modified_ns
    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or "te" not in cache:
        raise ValueError("The image cache does not contain a test tensor named 'te'.")
    images = cache["te"]
    if tuple(images.shape) != (461, 3, 224, 224) or images.dtype != torch.uint8:
        raise ValueError(f"Unexpected test-image cache shape or dtype: {images.shape}, {images.dtype}")

    split = load_json(split_path, Path(split_path).stat().st_mtime_ns)
    explanation, _ = load_npz(Path(explanation_path))
    row_keys = required_array(explanation, "test_row_keys", "image_explainers_v1.npz")
    split_keys = [str(value) for value in split["test_img_ids"]]
    if row_keys.tolist() != split_keys:
        raise ValueError("Image maps and test-image cache do not share the frozen row order.")
    return images.numpy()


def _section_header(kicker: str, title: str, subtitle: str) -> None:
    st.caption(kicker.upper())
    st.title(title)
    st.write(subtitle)


def render_overview() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESEARCH DEMONSTRATION · READ-ONLY</div>
          <h1>Skin-lesion Multimodal Explainability Lab</h1>
          <p>Two research tracks. Three explanation targets. One evidence-first view.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    try:
        bundle = load_real_bundle()
        train_rows = len(required_array(bundle["base"], "y_train", "base_outputs_v1.npz"))
        test_rows = len(required_array(bundle["base"], "y_test", "base_outputs_v1.npz"))
        class_count = len(bundle["classes"])
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.error(f"Could not load the frozen real-data summary: {exc}")
        train_rows = test_rows = class_count = None

    model_count = sum(
        (OUTPUTS / name).is_file()
        for name in ("vit_final_v1.pt", "xgb_final_v1.joblib")
    )
    explanation_count = sum(
        (OUTPUTS / name).is_file()
        for name in (
            "modality_shapley_v1.npz",
            "modality_shapley_v2.npz",
            "treeshap_v1.npz",
            "image_explainers_v1.npz",
        )
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Saved OOF rows", f"{train_rows:,}" if train_rows is not None else "N/A")
    metric_columns[1].metric("Frozen test rows", f"{test_rows:,}" if test_rows is not None else "N/A")
    metric_columns[2].metric("Classes", f"{class_count}" if class_count is not None else "N/A")
    metric_columns[3].metric("Frozen branch checkpoints", f"{model_count}/2")

    st.markdown("## Two separate research tracks")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
            <div class="track-card real">
              <div class="eyebrow">REAL DATA · PAD-UFES-20</div>
              <h3>Image + clinical classification</h3>
              <p>Smartphone clinical photographs paired with patient-level metadata.</p>
              <div class="flow">ViT image branch <span>+</span> XGBoost clinical branch <span>→</span> CWF fusion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(f"{explanation_count}/4 saved explanation artifacts are available.")
        st.caption("Frozen test metrics are descriptive/exploratory; this is not a clean confirmatory holdout.")
    with right:
        st.markdown(
            """
            <div class="track-card synthetic">
              <div class="eyebrow">CONTROLLED SYNTHETIC · PHASE G</div>
              <h3>Planted image and clinical cues</h3>
              <p>A historical balanced pilot is saved. The remaining v2.2 conditions are still blocked.</p>
              <div class="flow">Image cue <span>+</span> clinical cue <span>→</span> model behavior</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error("Phase G v2.2: NOT AUTHORIZED")
        st.caption("No remaining-condition experiment or fresh evaluation cohort has run.")

    st.markdown("## Keep the explanation targets distinct")
    targets = st.columns(3)
    targets[0].markdown("**CWF modality Shapley**  \nExplains the specified two-player fused prediction game.")
    targets[1].markdown("**XGBoost TreeSHAP**  \nExplains the clinical/tabular branch, not the fused model.")
    targets[2].markdown("**ViT image maps**  \nExplain the image branch; a CWF-selected class is only the map target.")
    st.info("Research software only—not a diagnostic tool or clinical decision-support system.")


def render_real_results() -> None:
    _section_header(
        "01 · Real-data results",
        "Saved predictions, not a new model run",
        "Compare out-of-fold predictions with the already observed frozen test outputs. Every metric below is calculated from aligned saved labels and probabilities.",
    )
    st.warning(
        "The 461-row frozen test set has already been used for reporting and exploratory analyses. "
        "Treat these results as descriptive, not confirmatory."
    )

    try:
        bundle = load_real_bundle()
        classes = bundle["classes"]
        partition = st.radio(
            "Saved prediction partition",
            ["OOF", "Frozen test"],
            horizontal=True,
            help="OOF = cross-fitted predictions for training rows. Frozen test = saved final-model outputs; no inference is run here.",
        )
        labels, predictions = _partition_arrays(bundle, partition)
        metrics_by_model: dict[str, dict[str, Any]] = {}
        rows = []
        for model_name, (probabilities, source) in predictions.items():
            metrics = classification_metrics(labels, probabilities, classes)
            metrics_by_model[model_name] = metrics
            rows.append(
                {
                    "Model branch/function": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Macro-F1": metrics["macro_f1"],
                    "NLL (lower is better)": metrics["nll"],
                    "Rows": len(labels),
                    "Saved source": source,
                }
            )

        st.markdown(f"### {partition} summary")
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Macro-F1": "{:.3f}",
                    "NLL (lower is better)": "{:.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Metrics are computed from the saved probabilities only. No model is loaded, trained, calibrated, or run by this page."
        )

        selected_model = st.selectbox(
            "Inspect per-class results",
            list(metrics_by_model),
            index=list(metrics_by_model).index("CWF fusion"),
        )
        metrics = metrics_by_model[selected_model]
        detail_left, detail_right = st.columns([1.2, 1], gap="large")
        with detail_left:
            st.markdown("#### Class support and per-class scores")
            st.dataframe(
                _per_class_table(metrics, classes).style.format(
                    {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
                ),
                hide_index=True,
                width="stretch",
            )
        with detail_right:
            fig = _plot_confusion(
                metrics["confusion_matrix"],
                classes,
                f"{selected_model} · {partition}",
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.markdown("#### Frozen split context")
        split_path = OUTPUTS / "split_v1.json"
        split = load_json(str(split_path), split_path.stat().st_mtime_ns)
        frozen_test_labels = required_array(
            bundle["base"], "y_test", "base_outputs_v1.npz"
        )
        frozen_test_support = {
            class_name: int(np.count_nonzero(frozen_test_labels == index))
            for index, class_name in enumerate(classes)
        }
        split_columns = st.columns(3)
        split_columns[0].metric("Train patients", f"{len(split['train_patients']):,}")
        split_columns[1].metric("Test patients", f"{len(split['test_patients']):,}")
        split_columns[2].metric(
            "MEL frozen-test support",
            f"{frozen_test_support.get('MEL', 0):,}",
        )
        st.caption(
            "Patient-level split is disjoint. The audit also flags 79 lesion IDs across partitions and one perceptual-hash collision."
        )
        st.caption(
            "Source files: `outputs/base_outputs_v1.npz`, `outputs/fusion_v1.npz`, "
            "and `outputs/fusion_v1.json`. Class labels are six-way; no binary regrouping is applied."
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.error(f"Saved result could not be displayed: {exc}")


def _render_cwf_explanation(sample_index: int, version: str) -> None:
    filename = "modality_shapley_v1.npz" if version == "v1 · probability-level" else "modality_shapley_v2.npz"
    path = OUTPUTS / filename
    try:
        arrays, skipped = load_npz(path)
        if version.startswith("v1"):
            tabular_key, image_key = "test_phiT", "test_phiI"
            explanation_name = "probability-level two-player Shapley"
        else:
            tabular_key, image_key = "test_input_phiT", "test_input_phiI"
            explanation_name = "input-level two-player Shapley"
        tabular = float(required_array(arrays, tabular_key, filename)[sample_index])
        image = float(required_array(arrays, image_key, filename)[sample_index])
        payoff_key = "test_f" if version.startswith("v1") else "test_input_f"
        payoff = float(required_array(arrays, payoff_key, filename)[sample_index])
        fig = _plot_signed_bars(
            np.asarray([tabular, image]),
            ["Clinical modality", "Image modality"],
            f"CWF modality contributions · {version}",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.metric("Saved CWF payoff", f"{payoff:.4f}")
        _display_provenance(
            f"outputs/{filename}",
            f"{tabular_key}, {image_key}, {payoff_key}",
            explanation_name,
        )
        st.caption("These are modality-level contributions to the specified CWF game—not feature-level or pixel-level attributions.")
        if skipped:
            st.caption(f"Safely skipped object arrays: {', '.join(sorted(skipped))}")
    except (FileNotFoundError, KeyError, ValueError, OSError, IndexError) as exc:
        st.error(f"CWF explanation unavailable: {exc}")


def _render_treeshap(sample_index: int, classes: list[str]) -> None:
    filename = "treeshap_v1.npz"
    try:
        arrays, _ = load_npz(OUTPUTS / filename)
        variables = [str(value) for value in required_array(arrays, "variables", filename).tolist()]
        contributions = required_array(
            arrays, "test_grouped_probability_predicted", filename
        )[sample_index]
        predicted = int(
            required_array(arrays, "test_predicted_class", filename)[sample_index]
        )
        true_label = int(required_array(arrays, "test_true_y", filename)[sample_index])
        fig = _plot_signed_bars(
            contributions,
            variables,
            f"XGBoost grouped probability Shapley · predicted {classes[predicted]}",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.write(
            f"True class: **{classes[true_label]}** · predicted class: **{classes[predicted]}**"
        )
        _display_provenance(
            f"outputs/{filename}",
            "test_grouped_probability_predicted",
            "Final clinical XGBoost branch · 8 grouped variables",
        )
        st.caption("This explains the XGBoost branch, not the CWF fused function.")
    except (FileNotFoundError, KeyError, ValueError, OSError, IndexError) as exc:
        st.error(f"TreeSHAP explanation unavailable: {exc}")


def _render_vit_map(sample_index: int, classes: list[str]) -> None:
    filename = "image_explainers_v1.npz"
    try:
        arrays, _ = load_npz(OUTPUTS / filename)
        method = st.selectbox(
            "ViT explanation method",
            ["Grad-CAM", "Attention rollout", "Chefer-style rollout", "Occlusion"],
        )
        target = st.radio(
            "Map target class",
            ["CWF-selected class", "ViT top class"],
            horizontal=True,
        )
        suffix = "fused" if target == "CWF-selected class" else "vit_top"
        map_key = {
            "Grad-CAM": f"gradcam_{suffix}",
            "Attention rollout": f"attention_rollout_{suffix}",
            "Chefer-style rollout": f"chefer_{suffix}",
            "Occlusion": f"occlusion_{suffix}",
        }[method]
        heatmap = np.asarray(required_array(arrays, map_key, filename)[sample_index], dtype=np.float32)
        if heatmap.shape != (14, 14) or not np.isfinite(heatmap).all():
            raise ValueError(f"Expected a finite 14x14 map; got shape {heatmap.shape}")
        target_key = (
            "target_fused_class"
            if target == "CWF-selected class"
            else "target_vit_top_class"
        )
        target_class = int(required_array(arrays, target_key, filename)[sample_index])
        true_class = int(required_array(arrays, "test_true_y", filename)[sample_index])

        map_left, map_right = st.columns(2, gap="large")
        map_left.markdown("#### Matched frozen test photograph")
        map_right.markdown("#### Saved 14 × 14 map")
        cache_path = CACHE_DIR / "images_uint8_224.pt"
        split_path = OUTPUTS / "split_v1.json"
        image = None
        try:
            cache = load_matched_test_images(
                str(cache_path),
                cache_path.stat().st_mtime_ns,
                str(split_path),
                split_path.stat().st_mtime_ns,
                str(OUTPUTS / filename),
                (OUTPUTS / filename).stat().st_mtime_ns,
            )
            image = np.transpose(cache[sample_index], (1, 2, 0))
        except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError, ImportError) as exc:
            map_left.info(f"Image pairing unavailable: {exc}")

        if image is not None:
            fig, axis = plt.subplots(figsize=(5.4, 5.2))
            axis.imshow(image)
            axis.imshow(heatmap, cmap="magma", alpha=0.48, interpolation="bilinear")
            axis.set_title(f"{method} overlay")
            axis.axis("off")
            fig.tight_layout()
            map_left.pyplot(fig, width="stretch")
            plt.close(fig)
        fig, axis = plt.subplots(figsize=(5.4, 5.2))
        artist = axis.imshow(heatmap, cmap="magma", interpolation="nearest")
        axis.set_title(f"{method} · {target}")
        axis.axis("off")
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        map_right.pyplot(fig, width="stretch")
        plt.close(fig)

        st.write(
            f"True class: **{classes[true_class]}** · target class: **{classes[target_class]}** · "
            f"test row: **{sample_index + 1} of 461**"
        )
        _display_provenance(
            f"outputs/{filename}",
            map_key,
            "Final ViT image branch · CWF class only selects the target",
        )
        if method == "Attention rollout":
            st.caption("Attention rollout is class-independent; target naming does not make it class-specific.")
        if method == "Chefer-style rollout":
            st.caption("Saved method is described as Chefer-style gradient-weighted attention rollout.")
    except (FileNotFoundError, KeyError, ValueError, OSError, IndexError) as exc:
        st.error(f"ViT explanation unavailable: {exc}")


def render_explanation_gallery() -> None:
    _section_header(
        "02 · Explainability gallery",
        "Three targets, shown separately",
        "Choose a saved explanation artifact and test-row index. The page never runs an explainer or model inference.",
    )
    selection = st.selectbox(
        "Explanation target",
        [
            "CWF modality Shapley",
            "Clinical XGBoost TreeSHAP",
            "ViT image maps",
        ],
    )
    sample_index = st.number_input(
        "Frozen test row (1-based index)",
        min_value=1,
        max_value=461,
        value=1,
        step=1,
    ) - 1

    try:
        classes = load_real_bundle()["classes"]
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.error(f"Could not load class order: {exc}")
        return

    if selection == "CWF modality Shapley":
        version = st.radio(
            "Saved CWF explanation",
            ["v1 · probability-level", "v2 · input-level"],
            horizontal=True,
        )
        _render_cwf_explanation(sample_index, version)
    elif selection == "Clinical XGBoost TreeSHAP":
        _render_treeshap(sample_index, classes)
    else:
        _render_vit_map(sample_index, classes)

    st.info(
        "The frozen test maps are descriptive/exploratory. Their use does not authorize further test-set method development."
    )


def _condition_rows(config: dict[str, Any]) -> pd.DataFrame:
    definitions = config.get("condition_definitions")
    if not isinstance(definitions, dict):
        raise ValueError("v2.2 config has no condition_definitions object")

    rows = []
    for name in (
        "image_dominant",
        "clinical_dominant",
        "redundant",
        "conflicting",
        "synergistic",
    ):
        if name not in definitions:
            raise KeyError(f"v2.2 config is missing condition {name!r}")
        details = definitions[name]
        phase = details.get("phase_g_condition", {})
        if name == "image_dominant":
            design = (
                f"Image {phase['image_reliability']:.2f}; "
                f"clinical {phase['clinical_reliability']:.2f}"
            )
        elif name == "clinical_dominant":
            design = (
                f"Image {phase['image_reliability']:.2f}; "
                f"clinical {phase['clinical_reliability']:.2f}"
            )
        elif name == "redundant":
            design = f"Shared noisy cue · reliability {phase['shared_latent_noisy_class_reliability']:.2f}"
        elif name == "conflicting":
            design = (
                f"{details['image_correct_fraction']:.0%} image-correct/clinical-decoy; "
                f"{details['clinical_correct_fraction']:.0%} clinical-correct/image-decoy; "
                f"{details['independent_fraction']:.0%} independent 0.85/0.85"
            )
        else:
            design = "Complementary U and (Y − U) mod 6; each share retained with probability 0.85"
        rows.append({"Condition": name.replace("_", " "), "Configured cue design": design})
    return pd.DataFrame(rows)


def render_synthetic_status() -> None:
    _section_header(
        "03 · Synthetic benchmark",
        "Historical pilot vs. blocked v2.2",
        "The synthetic benchmark is a controlled research track; its outcomes must not be mixed with clinical PAD-UFES-20 results.",
    )
    try:
        config = load_json(str(PHASE_G_CONFIG), PHASE_G_CONFIG.stat().st_mtime_ns)
        st.error(
            f"PHASE G v2.2 · {config.get('protocol_status', 'status unavailable').upper()} · "
            f"{'EXECUTION ENABLED' if config.get('execution_enabled') else 'NOT AUTHORIZED'}"
        )
        status_columns = st.columns(3)
        status_columns[0].metric("Execution enabled", str(config.get("execution_enabled", "not specified")))
        status_columns[1].metric("Authorized conditions", str(len(config.get("authorized_conditions", []))))
        status_columns[2].metric("Test access", str(config.get("test_data_access", "not specified")))
        st.dataframe(
            _condition_rows(config),
            hide_index=True,
            width="stretch",
        )
        if config.get("full_training_started") is False and config.get("remaining_condition_runs_started") is False:
            st.caption("Checked-in config records no full training and no remaining-condition runs.")
        st.caption("v2.2 explanation and faithfulness code is engineering infrastructure—not completed condition results.")
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.error(f"Could not load v2.2 status: {exc}")

    st.divider()
    st.markdown("### Historical Phase G v1 · balanced 0.85 / 0.85 pilot")
    try:
        report = load_json(str(PILOT_REPORT), PILOT_REPORT.stat().st_mtime_ns)
        primary = report["primary"]
        accuracy = primary["accuracy"]
        nll = primary["calibrated_nll"]
        effect = primary["nll_effects"]
        lower = primary["holm_adjusted_lower_bounds"]
        p_values = primary["holm_adjusted_p_values_at_0_10"]
        st.caption(
            f"Saved experiment: `{report['experiment_id']}` · "
            f"{primary['n_test_patients']:,} synthetic test patients · "
            f"{primary['bootstrap_replicates']:,} bootstrap replicates"
        )
        model_columns = st.columns(3)
        for column, model, label in zip(
            model_columns,
            ("image_only", "clinical_only", "joint"),
            ("Image-only", "Clinical-only", "Joint"),
            strict=True,
        ):
            column.metric(f"{label} accuracy", f"{accuracy[model]:.3f}")
            column.metric(f"{label} calibrated NLL", f"{nll[model]:.3f}")

        contrast_table = pd.DataFrame(
            [
                {
                    "Contrast (NLL difference)": "Image-only − joint",
                    "Saved difference": effect["image_only_minus_joint"],
                    "Adjusted lower bound": lower["image_only_minus_joint"],
                    "Saved adjusted p": p_values["image_only_minus_joint"],
                },
                {
                    "Contrast (NLL difference)": "Clinical-only − joint",
                    "Saved difference": effect["clinical_only_minus_joint"],
                    "Adjusted lower bound": lower["clinical_only_minus_joint"],
                    "Saved adjusted p": p_values["clinical_only_minus_joint"],
                },
            ]
        )
        st.dataframe(
            contrast_table.style.format(
                {
                    "Saved difference": "{:.3f}",
                    "Adjusted lower bound": "{:.3f}",
                    "Saved adjusted p": "{:.5f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Source: `outputs/synthetic_phase_g_v1/pilot_balanced_085_085_v1/pilot_report.json`")
        deviation = report.get("methodological_deviation", {})
        if deviation:
            st.warning(
                f"Historical pilot limitation — {deviation.get('title', 'methodological deviation')}: "
                f"{deviation.get('implication', 'See the saved pilot report.')}"
            )
        if report.get("not_a_phase_g_completion"):
            st.info("The saved report explicitly says this balanced pilot is not a Phase G completion.")
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.warning(f"Historical pilot result is not available: {exc}")


def render_limitations() -> None:
    _section_header(
        "04 · Interpretation",
        "What these results can—and cannot—say",
        "This page keeps the scientific boundaries visible during the evaluator walkthrough.",
    )
    items = [
        (
            "Exploratory test-set reuse",
            "The frozen 461-row PAD-UFES-20 test set has already been used for final reporting and exploratory explanation work. It is not a clean confirmatory holdout.",
        ),
        (
            "MEL support is small",
            "Only seven MEL cases occur in the frozen test partition. Avoid strong class-specific conclusions and show support alongside metrics.",
        ),
        (
            "Split caveats",
            "Patients are disjoint between train and test, while 79 lesion IDs cross partitions and one perceptual-hash collision is flagged in the audit.",
        ),
        (
            "Explanations have different targets",
            "TreeSHAP explains the clinical XGBoost branch; image maps explain the ViT branch; modality Shapley explains the specified CWF game. They are not interchangeable.",
        ),
        (
            "Synthetic work is separate",
            "The historical balanced Phase G v1 pilot is saved. The five remaining v2.2 conditions and fresh evaluation cohort are blocked pending approval and authorization.",
        ),
        (
            "Not a clinical system",
            "This research prototype is not a diagnostic tool, a clinical decision-support system, or a general image-tampering detector.",
        ),
        (
            "Follow-on plan",
            "The one-month research backlog is intentionally outside this presentation sprint and has not been started.",
        ),
    ]
    for title, body in items:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.write(body)

    st.markdown("### Provenance and safeguards")
    st.write(
        "The dashboard reads frozen outputs under `outputs/` and the existing image cache. "
        "It does not train models, run held-out inference, generate patients, or change authorization settings."
    )
    st.caption("For exact artifact status and hashes, see the repository's existing `artifact_manifest.json` and `ARTIFACT_AUDIT.md`.")


def _render_case_prediction(
    prediction: CasePrediction,
    image_rgb: np.ndarray,
    inputs: ClinicalInputs,
    class_names: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> None:
    cwf_index = prediction.cwf_predicted_class_index
    vit_index = prediction.vit_predicted_class_index
    xgb_index = prediction.clinical.predicted_class_index
    st.divider()
    st.markdown("## Latest submitted case")
    st.caption(
        "Single-case model outputs from this upload. They are not cohort performance metrics "
        "and are not a diagnosis."
    )
    prediction_columns = st.columns(3)
    prediction_columns[0].metric(
        "CWF fused prediction",
        class_names[cwf_index],
        help=f"Model probability: {prediction.cwf_probabilities[cwf_index]:.1%}",
    )
    prediction_columns[0].caption(
        f"Model probability: {prediction.cwf_probabilities[cwf_index]:.1%}"
    )
    prediction_columns[1].metric(
        "XGBoost clinical branch",
        class_names[xgb_index],
        help=f"Model probability: {prediction.clinical.probabilities[xgb_index]:.1%}",
    )
    prediction_columns[1].caption(
        f"Model probability: {prediction.clinical.probabilities[xgb_index]:.1%}"
    )
    prediction_columns[2].metric(
        "ViT image branch",
        class_names[vit_index],
        help=f"Model probability: {prediction.vit_probabilities[vit_index]:.1%}",
    )
    prediction_columns[2].caption(
        f"Model probability: {prediction.vit_probabilities[vit_index]:.1%}"
    )

    st.markdown("### Six-class model probabilities")
    probability_table = pd.DataFrame(
        {
            "Class": class_names,
            "CWF fused": prediction.cwf_probabilities,
            "XGBoost clinical": prediction.clinical.probabilities,
            "ViT image": prediction.vit_probabilities,
        }
    )
    st.dataframe(
        probability_table.style.format(
            {
                "CWF fused": "{:.1%}",
                "XGBoost clinical": "{:.1%}",
                "ViT image": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "These are model probability outputs for this submitted case, not validated clinical confidence."
    )

    st.markdown("### Explanations — keep the model branches distinct")
    image_column, clinical_column = st.columns(2, gap="large")
    with image_column:
        st.markdown("#### ViT image explanation · Grad-CAM")
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(image_rgb)
        axis.imshow(
            prediction.gradcam_map,
            cmap="magma",
            alpha=0.43,
            interpolation="bilinear",
        )
        axis.axis("off")
        axis.set_title(f"Grad-CAM target: CWF class {class_names[cwf_index]}")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Grad-CAM is computed from the ViT branch for the CWF-predicted class "
            f"{class_names[cwf_index]}; the ViT branch argmax is {class_names[vit_index]}."
        )

    with clinical_column:
        st.markdown("#### XGBoost clinical explanation · grouped probability-SHAP")
        if len(feature_names) != len(prediction.clinical.grouped_shapley_values):
            raise ValueError("clinical explanation feature names do not match its values")
        submitted_values = {
            "grew": inputs.grew,
            "itch": inputs.itch,
            "changed": inputs.changed,
            "elevation": inputs.elevation,
            "bleed": inputs.bleed,
            "hurt": inputs.hurt,
            "region": inputs.region,
            "age": inputs.age,
        }
        explanation_table = pd.DataFrame(
            {
                "Clinical variable": [name.replace("_", " ").title() for name in feature_names],
                "Submitted value": [submitted_values[name] for name in feature_names],
                "Grouped probability contribution": prediction.clinical.grouped_shapley_values,
            }
        )
        st.dataframe(
            explanation_table.style.format(
                {"Grouped probability contribution": "{:+.4f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"These eight grouped Shapley values explain the XGBoost branch's predicted class "
            f"{class_names[xgb_index]}, not the ViT or CWF prediction. Positive values raise "
            "that class probability relative to the background; negative values lower it."
        )

    with st.expander("CWF class-weighted fusion breakdown", expanded=True):
        xgb_probability = float(prediction.clinical.probabilities[cwf_index])
        vit_probability = float(prediction.vit_probabilities[cwf_index])
        fusion_terms = pd.DataFrame(
            [
                {
                    "Branch": "XGBoost clinical",
                    "Class-specific weight": prediction.xgb_weight,
                    "Branch probability": xgb_probability,
                    "Weighted pre-normalization term": prediction.xgb_weighted_term,
                },
                {
                    "Branch": "ViT image",
                    "Class-specific weight": 1.0 - prediction.xgb_weight,
                    "Branch probability": vit_probability,
                    "Weighted pre-normalization term": prediction.vit_weighted_term,
                },
            ]
        )
        st.dataframe(
            fusion_terms.style.format(
                {
                    "Class-specific weight": "{:.3f}",
                    "Branch probability": "{:.3%}",
                    "Weighted pre-normalization term": "{:.5f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"For class {class_names[cwf_index]}, the two weighted terms are divided by "
            f"the six-class normalization constant {prediction.cwf_normalization_constant:.5f} "
            f"to produce CWF probability {prediction.cwf_probabilities[cwf_index]:.3%}. "
            "This is a fusion breakdown, not a feature-level CWF explanation."
        )

    st.markdown("#### Cohort-level context — separate from this upload")
    st.info(
        "Per-upload accuracy or F1 is not available without verified ground truth. "
        "Saved real-data cohort metrics are on **Real-data results** and do not measure this case."
    )
    st.caption(
        "The image and clinical inputs are held only in this Streamlit session; the dashboard does not save them."
    )


def render_upload_case() -> None:
    _section_header(
        "05 · Single-case inference",
        "Upload a lesion image and clinical details",
        "Run the existing saved ViT and XGBoost models, then inspect their separate explanations and CWF fusion.",
    )
    st.warning(
        "Research demonstration only—not a diagnosis or clinical decision-support tool. "
        "Probability outputs are not validated clinical confidence."
    )
    st.caption(
        "The uploaded image and clinical values are processed in memory for this session. "
        "No model is trained and no research artifact is changed."
    )
    with st.form("upload_case_form"):
        image_column, clinical_column = st.columns([1, 2], gap="large")
        with image_column:
            uploaded_image = st.file_uploader(
                "Lesion image (PNG or JPEG)",
                type=["png", "jpg", "jpeg"],
                key="upload_case_image",
            )
        with clinical_column:
            first_row = st.columns(3)
            grew = first_row[0].selectbox("Grew", ["UNK", "True", "False"])
            itch = first_row[1].selectbox("Itches", ["UNK", "True", "False"])
            changed = first_row[2].selectbox("Changed", ["UNK", "True", "False"])
            second_row = st.columns(3)
            elevation = second_row[0].selectbox("Elevated", ["UNK", "True", "False"])
            bleed = second_row[1].selectbox("Bleeds", ["UNK", "True", "False"])
            hurt = second_row[2].selectbox("Hurts", ["UNK", "True", "False"])
            last_row = st.columns(2)
            region = last_row[0].selectbox(
                "Body region",
                [
                    "ABDOMEN",
                    "ARM",
                    "BACK",
                    "CHEST",
                    "EAR",
                    "FACE",
                    "FOOT",
                    "FOREARM",
                    "HAND",
                    "LIP",
                    "NECK",
                    "NOSE",
                    "SCALP",
                    "THIGH",
                ],
            )
            age = last_row[1].number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=45,
                step=1,
                help="Values outside the observed 6–94 year range are flagged, not clamped.",
            )
        submitted = st.form_submit_button("Predict and explain", type="primary")

    if submitted:
        for key in (
            "upload_case_result",
            "upload_case_rgb",
            "upload_case_inputs",
            "upload_case_classes",
            "upload_case_features",
        ):
            st.session_state.pop(key, None)
        if uploaded_image is None:
            st.error("Choose a PNG or JPEG image before requesting a prediction.")
        else:
            try:
                case_inputs = ClinicalInputs(
                    grew=grew,
                    itch=itch,
                    changed=changed,
                    elevation=elevation,
                    bleed=bleed,
                    hurt=hurt,
                    region=region,
                    age=int(age),
                )
                image_tensor = prepare_uploaded_image(uploaded_image.getvalue())
                metadata_path, signature = _case_runtime_signature()
                runtime = _cached_case_runtime(
                    str(PROJECT_ROOT),
                    str(metadata_path),
                    signature,
                )
                prediction = predict_uploaded_case(
                    image_tensor,
                    case_inputs,
                    runtime,
                )
                image_rgb = (
                    image_tensor[0]
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                st.session_state["upload_case_result"] = prediction
                st.session_state["upload_case_rgb"] = image_rgb
                st.session_state["upload_case_inputs"] = case_inputs
                st.session_state["upload_case_classes"] = runtime.clinical.class_names
                st.session_state["upload_case_features"] = tuple(runtime.clinical.groups)
                if case_inputs.age_out_of_range:
                    st.warning(
                        "This age is outside the 6–94 year training range; it was not clamped."
                    )
                st.success("Prediction and branch explanations are ready for this submitted case.")
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"Could not predict this uploaded case: {error}")

    if "upload_case_result" in st.session_state:
        if st.button("Clear submitted case", key="clear_upload_case"):
            for key in (
                "upload_case_result",
                "upload_case_rgb",
                "upload_case_inputs",
                "upload_case_classes",
                "upload_case_features",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        _render_case_prediction(
            st.session_state["upload_case_result"],
            st.session_state["upload_case_rgb"],
            st.session_state["upload_case_inputs"],
            st.session_state["upload_case_classes"],
            st.session_state["upload_case_features"],
        )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7fb; color: #17243b; }
        .block-container { max-width: 1440px; padding-top: 1.25rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #111d31; }
        [data-testid="stSidebar"] * { color: #eef4fb; }
        .hero {
          padding: 1.8rem 2rem; border-radius: 20px; color: #f8fbff;
          background: linear-gradient(118deg, #10243b 0%, #174e5a 58%, #398679 100%);
          box-shadow: 0 16px 38px rgba(18, 40, 61, .14);
        }
        .hero h1 { font-size: 2.35rem; margin: .45rem 0; color: #fff; }
        .hero p { color: #d7e8ee; font-size: 1.05rem; margin: .2rem 0; }
        .eyebrow { letter-spacing: .13em; font-weight: 700; font-size: .73rem; opacity: .8; }
        .track-card {
          min-height: 205px; padding: 1.35rem 1.5rem; border-radius: 16px;
          border: 1px solid #dbe3ec; background: #fff;
        }
        .track-card h3 { margin: .35rem 0 .45rem; color: #1b2b42; }
        .track-card p { color: #53647a; }
        .track-card .flow { margin-top: 1.15rem; font-weight: 650; color: #1c4656; }
        .track-card .flow span { color: #2e9887; padding: 0 .35rem; }
        div[data-testid="stMetric"] {
          background: white; border: 1px solid #e0e6ee; padding: .8rem 1rem;
          border-radius: 14px; box-shadow: 0 5px 15px rgba(25, 45, 70, .04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## SKIN RESEARCH LAB")
    st.sidebar.caption("PRESENTATION DEMO · READ-ONLY")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Real-data results",
            "Explainability gallery",
            "Upload case",
            "Synthetic benchmark",
            "Limitations",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Local only · saved models · ephemeral upload inference")

    if page == "Overview":
        render_overview()
    elif page == "Real-data results":
        render_real_results()
    elif page == "Explainability gallery":
        render_explanation_gallery()
    elif page == "Upload case":
        render_upload_case()
    elif page == "Synthetic benchmark":
        render_synthetic_status()
    else:
        render_limitations()


main()
