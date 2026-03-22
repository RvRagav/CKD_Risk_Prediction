from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.canonical import CANONICAL_FEATURES
from src.counterfactual import (
    generate_counterfactual,
    load_model,
    load_preprocessor,
    prepare_patient_canonical,
    predict_proba_cf,
    _resolve_target_from_prediction,
    IMMUTABLE_FEATURES,
    LossWeights,
)
from src.config import PREPROCESSOR_PATH

_DEFAULT_MODEL = "models/rf_augmented_42_v1.joblib"
_DEFAULT_CF_WEIGHTS = LossWeights()


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "age":  "Age (years)",
    "hemo": "Hemoglobin (g/dL)",
    "sc":   "Serum Creatinine (mg/dL)",
    "al":   "Albumin (g/dL)",
    "htn":  "Hypertension",
    "dm":   "Diabetes Mellitus",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age":  "Patient age in years",
    "hemo": "Normal range: 12–17 g/dL. Low levels may indicate anemia, common in CKD.",
    "sc":   "Normal: ~0.6–1.2 mg/dL. Elevated levels indicate reduced kidney filtration.",
    "al":   "Normal: 2.5–5.0 g/dL. Low albumin may reflect kidney protein loss.",
    "htn":  "Hypertension is a leading cause of CKD progression.",
    "dm":   "Diabetes is the most common cause of CKD worldwide.",
}

FEATURE_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "hemo": (12.0, 17.0),
    "sc":   (0.6, 1.2),
    "al":   (2.5, 5.0),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _as_yes_no(value: Any) -> str:
    try:
        return "Yes" if int(round(float(value))) == 1 else "No"
    except Exception:
        return "No"


def _fmt_value(feature: str, value: Any) -> str:
    if feature in {"htn", "dm"}:
        return _as_yes_no(value)
    try:
        if feature == "age":
            return str(int(round(float(value))))
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _get_float(row: pd.Series, feature: str, *, default: float = 0.0) -> float:
    try:
        v = row.get(feature)
        return float(default) if v is None else float(v)
    except Exception:
        return float(default)


def _get_int01(row: pd.Series, feature: str, *, default: int = 0) -> int:
    try:
        v = row.get(feature)
        return int(default) if v is None else (1 if int(round(float(v))) == 1 else 0)
    except Exception:
        return int(default)


def _risk_band(p_ckd: float) -> tuple[str, str, str]:
    """Returns (label, hex_color, emoji)."""
    if p_ckd < 0.33:
        return "Low Risk",      "#1F8F4A", "🟢"
    if p_ckd < 0.66:
        return "Moderate Risk", "#C7A100", "🟡"
    return     "High Risk",     "#B42318", "🔴"


def _risk_advice(p_ckd: float) -> str:
    if p_ckd < 0.33:
        return (
            "The model predicts a **low probability of CKD**. "
            "Regular health check-ups are still recommended."
        )
    if p_ckd < 0.66:
        return (
            "The model predicts a **moderate probability of CKD**. "
            "Consider further investigation and lifestyle modifications."
        )
    return (
        "The model predicts a **high probability of CKD**. "
        "Prompt clinical evaluation and specialist referral is strongly advised."
    )


# Threshold below which a SHAP value is considered zero / no contribution.
_SHAP_ZERO_THRESH = 1e-4


def _feature_change_chart(original_row: pd.Series, cf_row: pd.Series) -> go.Figure:
    numeric_feats = ["hemo", "sc", "al"]
    labels  = [FEATURE_LABELS.get(f, f) for f in numeric_feats]
    originals = [_get_float(original_row, f) for f in numeric_feats]
    cfs       = [_get_float(cf_row, f)       for f in numeric_feats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current",
        x=labels, y=originals,
        marker_color="#1E40AF",
        text=[f"{v:.2f}" for v in originals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Suggested (CF)",
        x=labels, y=cfs,
        marker_color="#1F8F4A",
        text=[f"{v:.2f}" for v in cfs],
        textposition="outside",
    ))

    # Normal range bands (subtle)
    for i, f in enumerate(numeric_feats):
        if f in FEATURE_NORMAL_RANGES:
            lo, hi = FEATURE_NORMAL_RANGES[f]
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=i - 0.5, x1=i + 0.5, y0=lo, y1=hi,
                fillcolor="rgba(100,200,100,0.08)",
                line=dict(color="rgba(100,200,100,0.4)", width=1, dash="dot"),
            )

    fig.update_layout(
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Value",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig


def _probability_bar_chart(explanation: dict[str, Any]) -> go.Figure | None:
    orig = explanation.get("original_proba")
    cf   = explanation.get("counterfactual_proba")
    if not (isinstance(orig, list) and isinstance(cf, list)):
        return None

    labels = ["Not CKD", "CKD"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before (Original)",
        x=labels,
        y=[orig[0] * 100, orig[1] * 100],
        marker_color=["#1F8F4A", "#B42318"],
        opacity=0.55,
        text=[f"{v*100:.1f}%" for v in orig],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="After (Counterfactual)",
        x=labels,
        y=[cf[0] * 100, cf[1] * 100],
        marker_color=["#1F8F4A", "#B42318"],
        opacity=1.0,
        text=[f"{v*100:.1f}%" for v in cf],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Probability (%)", range=[0, 115]),
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig


# ── Comparison Table ──────────────────────────────────────────────────────────

def _build_comparison_df(original_row: pd.Series, cf_row: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for f in CANONICAL_FEATURES:
        orig_val = _fmt_value(f, original_row.get(f))
        cf_val   = _fmt_value(f, cf_row.get(f))
        changed  = orig_val != cf_val
        arrow    = "🔄" if changed else "—"
        rows.append({
            "Feature":         FEATURE_LABELS.get(f, f),
            "Current":         orig_val,
            "Suggested":       cf_val,
            "Changed":         arrow,
            "_changed":        changed,
        })
    return pd.DataFrame(rows)


def _style_comparison(df: pd.DataFrame) -> Any:
    visible_cols = [c for c in df.columns if c != "_changed"]

    def _row(row: pd.Series) -> list[str]:
        changed = bool(df.at[row.name, "_changed"])
        if changed:
            return [
                "background-color:#DBEAFE; font-weight:600" if c == "Feature"    else
                "background-color:#FEF9C3" if c == "Current"   else
                "background-color:#DCFCE7; font-weight:700" if c == "Suggested"  else
                "background-color:#DBEAFE" if c == "Changed"   else
                ""
                for c in visible_cols
            ]
        return [""] * len(visible_cols)

    return df[visible_cols].style.apply(_row, axis=1)


# ── Download Bundle ───────────────────────────────────────────────────────────

def _build_download_bundle(
    *,
    patient_input: dict[str, Any],
    cf_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    explanation: dict[str, Any],
    comparison_df: pd.DataFrame,
) -> bytes:
    mem   = io.BytesIO()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("patient_input.json",  json.dumps(patient_input, indent=2))
        z.writestr("counterfactuals.csv", cf_df.to_csv(index=False))
        z.writestr("metrics.csv",         metrics_df.to_csv(index=False))
        z.writestr("comparison.csv",      comparison_df.to_csv(index=False))
        z.writestr("explanation.json",    json.dumps(explanation, indent=2))
        z.writestr("metadata.json",       json.dumps({"generated_at": stamp}, indent=2))
    return mem.getvalue()


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
  .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1100px; }
  h1  { font-size: 2.0rem; font-weight: 800; color: #0F172A; }
  h2  { font-size: 1.25rem; font-weight: 700; color: #1E293B; margin-top: 0.5rem; }
  h3  { font-size: 1.05rem; font-weight: 600; color: #334155; }

  .hero-subtitle { color: #475569; font-size: 1.05rem; margin-bottom: 0.4rem; }

  .card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
  }
  .card-blue  { border-left: 6px solid #1E40AF; }
  .card-green { border-left: 6px solid #1F8F4A; }
  .card-amber { border-left: 6px solid #C7A100; }
  .card-red   { border-left: 6px solid #B42318; }

  .risk-badge {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 9999px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.02em;
  }

  .step-label {
    background: #1E40AF;
    color: white;
    border-radius: 50%;
    width: 1.75rem; height: 1.75rem;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.92rem;
    margin-right: 0.45rem;
  }

  .metric-pill {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    text-align: center;
  }
  .metric-pill .mp-value { font-size: 1.35rem; font-weight: 700; color: #1E40AF; }
  .metric-pill .mp-label { font-size: 0.78rem; color: #64748B; margin-top: 0.15rem; }

  .disclaimer {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #78350F;
    font-size: 0.88rem;
    margin-top: 1rem;
  }

  div[data-testid="stForm"] { border: none; padding: 0; box-shadow: none; }
</style>
"""


# ── SHAP helper (prediction-only, no CF needed) ──────────────────────────────

def _shap_class1_values(
    explainer: Any,
    shap_values: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float] | None:
    """Normalise SHAP outputs to (n_samples, n_features) for class 1 (CKD).

    SHAP output formats vary across versions:
    - list of arrays [class0, class1] with shape (n_samples, n_features)
    - ndarray with shape (n_classes, n_samples, n_features)
    - ndarray with shape (n_samples, n_features, n_classes)  (common in SHAP>=0.48)
    """

    feats = CANONICAL_FEATURES

    # Base value for class 1.
    base = getattr(explainer, "expected_value", None)
    base_arr = np.asarray(base) if base is not None else np.asarray([0.0])
    base_flat = base_arr.reshape(-1)
    base_val = float(base_flat[1] if base_flat.size > 1 else base_flat[0])

    # SHAP values for class 1.
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        if arr.ndim != 2:
            return None
        return arr, base_val

    arr = np.asarray(shap_values)
    if arr.ndim == 1:
        # (n_features,) → single row
        arr2 = arr.reshape(1, -1)
        return arr2, base_val

    if arr.ndim == 2:
        # (n_samples, n_features)
        return arr, base_val

    if arr.ndim == 3:
        n_samples = int(X.shape[0])
        n_features = int(len(feats))

        # Case A: (n_samples, n_features, n_classes)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            cls_axis = arr.shape[2]
            idx = 1 if cls_axis > 1 else 0
            return arr[:, :, idx], base_val

        # Case B: (n_classes, n_samples, n_features)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            idx = 1 if arr.shape[0] > 1 else 0
            return arr[idx, :, :], base_val

    return None

def _compute_shap_original(
    model: Any,
    patient_canon: pd.DataFrame,
    target_index: int,  # kept for signature compatibility; not used for class selection
) -> dict[str, float] | None:
    """Compute SHAP values for the CKD class (class 1) for the original patient.

    We always explain CKD risk (class 1): positive SHAP = pushes towards CKD,
    negative SHAP = pushes away from CKD.  `target_index` is intentionally
    ignored for SHAP class selection to avoid accidentally using the CF-flip
    class (class 0 / Not-CKD) when the patient is already predicted as CKD.
    """
    try:
        import shap as _shap
    except ImportError:
        return None
    try:
        explainer = _shap.TreeExplainer(model)
        sv = explainer.shap_values(patient_canon)
        norm = _shap_class1_values(explainer, sv, patient_canon)
        if norm is None:
            return None
        mat, _base = norm
        if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] != len(CANONICAL_FEATURES):
            return None
        s0 = mat[0]
        return {f: float(v) for f, v in zip(CANONICAL_FEATURES, s0)}
    except Exception:
        return None


def _compute_shap_force_html(
    model: Any,
    patient_canon: pd.DataFrame,
) -> str | None:
    """Build an embeddable SHAP force plot (HTML) for the CKD class (class 1)."""
    try:
        import shap as _shap
    except ImportError:
        return None

    try:
        explainer = _shap.TreeExplainer(model)
        sv = explainer.shap_values(patient_canon)
        norm = _shap_class1_values(explainer, sv, patient_canon)
        if norm is None:
            return None
        mat, base_val = norm
        if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] != len(CANONICAL_FEATURES):
            return None

        shap_row = mat[0]
        feat_row = patient_canon[CANONICAL_FEATURES].iloc[0]

        vis = _shap.force_plot(
            base_val,
            shap_row,
            feat_row,
            feature_names=CANONICAL_FEATURES,
            matplotlib=False,
            show=False,
        )

        html = (
            "<html><head><meta charset='utf-8'></head><body>"
            + _shap.getjs()
            + vis.html()
            + "</body></html>"
        )
        return html
    except Exception:
        return None


def _compute_shap_original_details(
    model: Any,
    patient_canon: pd.DataFrame,
    target_index: int,
) -> dict[str, Any]:
    """Compute SHAP dict (for bar chart) and force-plot HTML for class 1."""
    shap_map = _compute_shap_original(model, patient_canon, target_index)
    force_html = _compute_shap_force_html(model, patient_canon) if shap_map is not None else None
    return {
        "shap_original": shap_map,
        "force_html": force_html,
        "shap_available": shap_map is not None,
    }


def _compute_lime_original(
    model: Any,
    patient_canon: pd.DataFrame,
) -> dict[str, float] | None:
    """Compute LIME local explanation weights for the CKD class (class 1).

    Uses the preprocessed training set as background data.
    Positive weight = feature increases CKD probability for this patient.
    """
    try:
        from lime import lime_tabular as _lime_tabular  # noqa: WPS433
    except ImportError:
        return None
    try:
        train_path = _repo_root() / "data/processed/preprocessed/X_train_preproc.csv"
        if not train_path.exists():
            return None
        X_train = pd.read_csv(train_path)[CANONICAL_FEATURES].values

        explainer = _lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=CANONICAL_FEATURES,
            class_names=["Not CKD", "CKD"],
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )
        patient_arr = patient_canon[CANONICAL_FEATURES].values[0]

        # LIME calls the predictor with raw numpy arrays.
        # Wrap to preserve feature names (silences sklearn warning).
        def _predict_proba_named(X: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X, columns=CANONICAL_FEATURES)
            return model.predict_proba(X_df)

        exp = explainer.explain_instance(
            patient_arr,
            _predict_proba_named,
            num_features=len(CANONICAL_FEATURES),
        )
        # local_exp[1] → [(feature_index, weight), ...] for class 1 (CKD)
        weights: dict[str, float] = {f: 0.0 for f in CANONICAL_FEATURES}
        for feat_idx, weight in exp.local_exp.get(1, []):
            if 0 <= feat_idx < len(CANONICAL_FEATURES):
                weights[CANONICAL_FEATURES[feat_idx]] = float(weight)
        return weights
    except Exception:
        return None


def _local_expl_bar_chart(
    feat_weights: dict[str, float],
    *,
    top_k: int = 6,
    x_title: str = "Weight (+ → CKD risk, − → Not CKD)",
) -> tuple[go.Figure | None, list[str]]:
    """Generic diverging bar chart for any local explanation (SHAP or LIME).

    Returns (figure, zero_feature_labels).
    """
    all_items = sorted(
        [(f, feat_weights.get(f, 0.0)) for f in CANONICAL_FEATURES],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    active_items = [(f, v) for f, v in all_items if abs(v) > _SHAP_ZERO_THRESH]
    zero_labels  = [
        FEATURE_LABELS.get(f, f) for f, v in all_items if abs(v) <= _SHAP_ZERO_THRESH
    ]

    if not active_items:
        return None, zero_labels

    active_items = active_items[:top_k]
    features = [FEATURE_LABELS.get(f, f) for f, _ in active_items]
    values   = [float(v) for _, v in active_items]
    colors   = ["#B42318" if v > 0 else "#1F8F4A" for v in values]
    labels   = [f"{'+' if v > 0 else ''}{v:.4f}" for v in values]
    max_abs  = max(abs(v) for v in values) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        hovertemplate="%{y}<br>Weight = %{text}<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(
        height=max(260, 65 * len(active_items) + 60),
        margin=dict(l=10, r=90, t=15, b=30),
        xaxis=dict(
            title=x_title,
            zeroline=True,
            zerolinecolor="#94A3B8",
            zerolinewidth=2,
            range=[-(max_abs * 1.6), max_abs * 1.6],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig, zero_labels


# ── Main App ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="CKD Risk Prediction",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("# 🩺 CKD Risk Prediction & Clinical Explanation")
    st.markdown(
        '<p class="hero-subtitle">Predict Chronic Kidney Disease risk, explore what '
        'drives it, and discover what it would take to reverse it.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Disclaimer:</b> This tool is for research and '
        'educational purposes only. It is <b>not</b> a substitute for professional '
        'medical advice, diagnosis, or treatment.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Patient Input  (always visible)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### <span class='step-label'>1</span> Enter Patient Data", unsafe_allow_html=True)

    with st.form("patient_form"):
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])

        with col_a:
            st.markdown("**Demographics**")
            age = st.slider("Age (years)", 1, 100, 45)
            htn = st.selectbox("Hypertension", ["No", "Yes"], index=0,
                               help=FEATURE_DESCRIPTIONS["htn"])
            dm  = st.selectbox("Diabetes", ["No", "Yes"], index=0,
                               help=FEATURE_DESCRIPTIONS["dm"])

        with col_b:
            st.markdown("**Blood Markers**")
            hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, 12.0, 0.1,
                                   format="%.1f", help=FEATURE_DESCRIPTIONS["hemo"])
            al   = st.number_input("Albumin (g/dL)", 0.0, 5.0, 3.5, 0.1,
                                   format="%.1f", help=FEATURE_DESCRIPTIONS["al"])

        with col_c:
            st.markdown("**Kidney Function**")
            sc = st.number_input("Serum Creatinine (mg/dL)", 0.4, 15.0, 1.2, 0.1,
                                 format="%.1f", help=FEATURE_DESCRIPTIONS["sc"])
            # st.markdown(
            #     "<div style='font-size:0.82rem; color:#64748B; margin-top:0.5rem;'>"
            #     "Normal SC: 0.6–1.2 mg/dL<br>Normal Hemo: 12–17 g/dL<br>Normal Albumin: 3.5–5.0 g/dL"
            #     "</div>",
            #     unsafe_allow_html=True,
            # )

        with col_d:
            st.markdown("**Input Summary**")
            st.markdown(
                f"<div class='card' style='font-size:0.9rem; background-color:#DBEAFE; color:black;'>"
                f"<b>Age:</b> {age} yrs<br>"
                f"<b>Hemoglobin:</b> {hemo:.1f} g/dL<br>"
                f"<b>Serum Creatinine:</b> {sc:.1f} mg/dL<br>"
                f"<b>Albumin:</b> {al:.1f} g/dL<br>"
                f"<b>Hypertension:</b> {htn}<br>"
                f"<b>Diabetes:</b> {dm}"
                f"</div>",
                unsafe_allow_html=True,
            )

        form_submitted = st.form_submit_button(
            "🔍 Predict CKD Risk",
            use_container_width=True,
            type="primary",
        )

    if not form_submitted and "prediction" not in st.session_state:
        st.info("👆 Fill in the patient data above and click **Predict CKD Risk** to begin.")
        return

    # ── Run prediction (fast — model is LRU-cached, no CF search) ─────────────
    if form_submitted:
        patient_input: dict[str, Any] = {
            "hemo": float(hemo),
            "sc":   float(sc),
            "al":   float(al),
            "htn":  1 if htn == "Yes" else 0,
            "age":  int(age),
            "dm":   1 if dm  == "Yes" else 0,
        }
        with st.spinner("⏳ Running prediction…"):
            try:
                model        = load_model(_DEFAULT_MODEL)
                preproc      = load_preprocessor(PREPROCESSOR_PATH)
                patient_canon = prepare_patient_canonical(
                    patient_input, preprocessor=preproc, preprocess=True
                )
                proba = predict_proba_cf(model, patient_canon)[0]
                target_label, target_index = _resolve_target_from_prediction(
                    model=model, X_canonical=patient_canon, target_class=None
                )
            except Exception as e:
                st.error("**Prediction error:** could not run the model.")
                with st.expander("Show technical details"):
                    st.exception(e)
                st.stop()

        # Store — clear downstream results so stale data is not shown
        st.session_state["patient_input"]   = patient_input
        st.session_state["patient_canon"]   = patient_canon
        st.session_state["prediction"]      = {
            "proba":        proba.tolist(),
            "target_label": int(target_label),
            "target_index": int(target_index),
        }
        st.session_state.pop("shap_result", None)
        st.session_state.pop("cf_result",   None)
        st.session_state.pop("lime_result",  None)

    # ── Restore prediction state ───────────────────────────────────────────────
    patient_input  = st.session_state["patient_input"]
    patient_canon  = st.session_state["patient_canon"]
    pred           = st.session_state["prediction"]
    proba_list     = pred["proba"]
    target_label   = pred["target_label"]
    target_index   = pred["target_index"]

    p_ckd  = float(proba_list[1]) if len(proba_list) > 1 else float("nan")
    finite = np.isfinite(p_ckd)
    band, band_color, band_emoji = _risk_band(p_ckd) if finite else ("Unknown", "#64748B", "⚪")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — CKD Risk Result  (shown immediately after prediction)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### <span class='step-label'>2</span> CKD Risk Prediction Result", unsafe_allow_html=True)

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        badge_bg = {"Low Risk": "#DCFCE7", "Moderate Risk": "#FEF9C3",
                     "High Risk": "#FEE2E2"}.get(band, "#F1F5F9")
        st.markdown(
            f"<div style='margin-bottom:0.6rem;'>"
            f"<span class='risk-badge' style='background:{badge_bg}; color:{band_color};'>"
            f"{band_emoji} {band}</span></div>",
            unsafe_allow_html=True,
        )

        if finite:
            p_not = 1.0 - p_ckd
            pa, pb = st.columns(2)
            with pa:
                st.markdown(
                    f"<div class='metric-pill'>"
                    f"<div class='mp-value' style='color:#B42318;'>{p_ckd*100:.1f}%</div>"
                    f"<div class='mp-label'>CKD probability</div></div>",
                    unsafe_allow_html=True,
                )
            with pb:
                st.markdown(
                    f"<div class='metric-pill'>"
                    f"<div class='mp-value' style='color:#1F8F4A;'>{p_not*100:.1f}%</div>"
                    f"<div class='mp-label'>Not-CKD probability</div></div>",
                    unsafe_allow_html=True,
                )

        advice_card = {"Low Risk": "card-green", "Moderate Risk": "card-amber",
                        "High Risk": "card-red"}.get(band, "card-blue")
        st.markdown(
            f"<div class='card {advice_card}' style='margin-top:0.85rem;'>"
            f"{_risk_advice(p_ckd) if finite else 'Prediction unavailable.'}"
            f"</div>",
            unsafe_allow_html=True,
        )


    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — SHAP Explanation  (user-triggered)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### <span class='step-label'>3</span> Explanations", unsafe_allow_html=True)
    st.caption(
        "Two complementary methods explain *why* the model made this prediction. "
        "Run either or both — each takes only a few seconds."
    )

    # ── 3a SHAP ───────────────────────────────────────────────────────────────
    with st.expander("🔬 SHAP — SHapley Additive exPlanations", expanded=True):
        st.caption(
            "SHAP uses game-theory to attribute the exact contribution of each feature. "
            "Values are exact and consistent across runs."
        )
        run_shap = st.button(
            "🔬 Explain with SHAP",
            key="btn_shap",
            disabled="shap_result" in st.session_state,
            use_container_width=False,
        )

        if run_shap:
            with st.spinner("⏳ Computing SHAP values…"):
                try:
                    model    = load_model(_DEFAULT_MODEL)
                    shap_details = _compute_shap_original_details(model, patient_canon, target_index)
                except Exception as e:
                    st.error("**SHAP error.**")
                    with st.expander("Details"):
                        st.exception(e)
                    shap_details = {"shap_original": None, "force_html": None, "shap_available": False}
            st.session_state["shap_result"] = shap_details
            st.rerun()

        if "shap_result" in st.session_state:
            shap_data = st.session_state["shap_result"]
            shap_map2: dict[str, float] | None = shap_data.get("shap_original")
            force_html: str | None = shap_data.get("force_html")
            shap_fig, shap_zero = _local_expl_bar_chart(
                shap_map2 or {},
                x_title="SHAP value (+ \u2192 CKD risk, \u2212 \u2192 Not CKD)",
            )
            if shap_fig:
                force_html_to_show = force_html
                shap_left, shap_right = st.columns([1.6, 1], gap="large")
                with shap_left:
                    st.caption(
                        "\U0001f534 **Red / right** \u2192 feature pushes towards **CKD**.  "
                        "\U0001f7e2 **Green / left** \u2192 feature pushes towards **Not CKD**."
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)
                    if shap_zero:
                        st.markdown(
                            f"<div style='font-size:0.83rem; color:#64748B; margin-top:-0.4rem;'>"
                            f"\u26aa <b>No contribution</b> for this patient: {', '.join(shap_zero)}</div>",
                            unsafe_allow_html=True,
                        )
                with shap_right:
                    active_shap = [(f, v) for f, v in (shap_map2 or {}).items() if abs(v) > _SHAP_ZERO_THRESH]
                    top_shap = sorted(active_shap, key=lambda x: abs(x[1]), reverse=True)[:3]
                    st.markdown("**Top driving features:**")
                    if top_shap:
                        for rank, (f, v) in enumerate(top_shap, 1):
                            direction = "increases" if v > 0 else "decreases"
                            color     = "#B42318" if v > 0 else "#1F8F4A"
                            st.markdown(
                                f"<div class='card card-blue' style='margin-bottom:0.4rem; padding:0.65rem 0.9rem;'>"
                                f"<b>{rank}. {FEATURE_LABELS.get(f, f)}</b><br/>"
                                f"<span style='color:{color};'>SHAP = {v:+.4f}</span> \u2014 "
                                f"<span style='font-size:0.88rem; color:#475569;'>"
                                f"This feature <b>{direction}</b> CKD risk.</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No features with measurable SHAP contribution found.")
                if st.button("\U0001f504 Re-run SHAP", key="btn_reshap"):
                    st.session_state.pop("shap_result", None)
                    st.rerun()

                # Render force plot full-width for better horizontal visibility.
                if force_html_to_show:
                    st.markdown("**SHAP force plot (patient-level):**")
                    # App container max-width is 1100px; match that for crisp rendering.
                    components.html(force_html_to_show, height=280, width=1100, scrolling=True)
            else:
                st.info("\u2139\ufe0f SHAP unavailable. Install `shap` (pip install shap).")
        elif not run_shap:
            st.info("\U0001f446 Click **Explain with SHAP** above.")

    # ── 3b LIME ───────────────────────────────────────────────────────────────
    # with st.expander("\U0001f9ea LIME — Local Interpretable Model-agnostic Explanations", expanded=False):
    #     st.caption(
    #         "LIME fits a simple linear model in the neighbourhood of this patient to approximate "
    #         "how each feature influences the prediction locally. "
    #         "Weights show each feature's linear contribution towards CKD (positive) or Not CKD (negative)."
    #     )
    #     run_lime = st.button(
    #         "\U0001f9ea Explain with LIME",
    #         key="btn_lime",
    #         disabled="lime_result" in st.session_state,
    #         use_container_width=False,
    #     )

    #     if run_lime:
    #         with st.spinner("\u23f3 Computing LIME explanation\u2026"):
    #             try:
    #                 model    = load_model(_DEFAULT_MODEL)
    #                 lime_map = _compute_lime_original(model, patient_canon)
    #             except Exception as e:
    #                 st.error("**LIME error.**")
    #                 with st.expander("Details"):
    #                     st.exception(e)
    #                 lime_map = None
    #         if lime_map is None:
    #             st.session_state["lime_result"] = {"lime_weights": None, "lime_available": False}
    #         else:
    #             st.session_state["lime_result"] = {"lime_weights": lime_map, "lime_available": True}
    #         st.rerun()

    #     if "lime_result" in st.session_state:
    #         lime_data = st.session_state["lime_result"]
    #         lime_map2: dict[str, float] | None = lime_data.get("lime_weights")

    #         if not lime_data.get("lime_available") or lime_map2 is None:
    #             st.warning(
    #                 "LIME explanation unavailable. "
    #                 "Install `lime` (pip install lime) and ensure training data exists."
    #             )
    #         else:
    #             lime_fig, lime_zero = _local_expl_bar_chart(
    #                 lime_map2,
    #                 x_title="LIME weight (+ \u2192 CKD risk, \u2212 \u2192 Not CKD)",
    #             )
    #             if lime_fig:
    #                 lime_left, lime_right = st.columns([1.6, 1], gap="large")
    #                 with lime_left:
    #                     st.caption(
    #                         "\U0001f534 **Red / right** \u2192 feature pushes towards **CKD**.  "
    #                         "\U0001f7e2 **Green / left** \u2192 feature pushes towards **Not CKD**."
    #                     )
    #                     st.plotly_chart(lime_fig, use_container_width=True)
    #                     if lime_zero:
    #                         st.markdown(
    #                             f"<div style='font-size:0.83rem; color:#64748B; margin-top:-0.4rem;'>"
    #                             f"\u26aa <b>No contribution</b> for this patient: {', '.join(lime_zero)}</div>",
    #                             unsafe_allow_html=True,
    #                         )
    #                 with lime_right:
    #                     active_lime = [(f, v) for f, v in lime_map2.items() if abs(v) > _SHAP_ZERO_THRESH]
    #                     top_lime = sorted(active_lime, key=lambda x: abs(x[1]), reverse=True)[:3]
    #                     st.markdown("**Top driving features:**")
    #                     if top_lime:
    #                         for rank, (f, v) in enumerate(top_lime, 1):
    #                             direction = "increases" if v > 0 else "decreases"
    #                             color     = "#B42318" if v > 0 else "#1F8F4A"
    #                             st.markdown(
    #                                 f"<div class='card card-blue' style='margin-bottom:0.4rem; padding:0.65rem 0.9rem;'>"
    #                                 f"<b>{rank}. {FEATURE_LABELS.get(f, f)}</b><br/>"
    #                                 f"<span style='color:{color};'>LIME = {v:+.4f}</span> \u2014 "
    #                                 f"<span style='font-size:0.88rem; color:#475569;'>"
    #                                 f"This feature <b>{direction}</b> CKD risk.</span>"
    #                                 f"</div>",
    #                                 unsafe_allow_html=True,
    #                             )
    #                     else:
    #                         st.info("No features with measurable LIME contribution found.")

    #                 # SHAP vs LIME comparison note
    #                 if "shap_result" in st.session_state and lime_map2:
    #                     shap_m = st.session_state["shap_result"].get("shap_original") or {}
    #                     shared_top = set(
    #                         f for f, v in sorted(shap_m.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    #                     ) & set(
    #                         f for f, v in sorted(lime_map2.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    #                     )
    #                     if shared_top:
    #                         names = ", ".join(FEATURE_LABELS.get(f) or f for f in sorted(shared_top))
    #                         st.markdown(
    #                             f"<div class='card card-green' style='margin-top:0.6rem; padding:0.55rem 0.9rem; font-size:0.88rem;'>"
    #                             f"\U0001f4a1 <b>SHAP \u2229 LIME agreement:</b> Both methods rank "
    #                             f"<b>{names}</b> among the top 3 drivers \u2014 high confidence signal."
    #                             f"</div>",
    #                             unsafe_allow_html=True,
    #                         )

    #                 if st.button("\U0001f504 Re-run LIME", key="btn_relime"):
    #                     st.session_state.pop("lime_result", None)
    #                     st.rerun()
    #             else:
    #                 st.info("\u2139\ufe0f No LIME weights computed. Try re-running.")
    #     elif not run_lime:
    #         st.info("\U0001f446 Click **Explain with LIME** above.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Counterfactual  (user-triggered, heaviest step)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### <span class='step-label'>4</span> Counterfactual Recommendation", unsafe_allow_html=True)
    st.markdown(
        "<div class='card card-green' style='margin-bottom:0.8rem;'>"
        "💡 A <b>counterfactual</b> shows the <b>smallest realistic changes</b> to clinical "
        "features that would flip the prediction from <b>CKD → Not CKD</b>. "
        "This step uses a search algorithm and may take 15–30 seconds."
        "</div>",
        unsafe_allow_html=True,
    )

    run_cf = st.button(
        "🔄 Generate Counterfactual",
        key="btn_cf",
        disabled="cf_result" in st.session_state,
        use_container_width=False,
    )

    if run_cf:
        # Live debug log (helps when CF search takes long or returns empty)
        debug_lines: list[str] = []
        with st.expander("Show counterfactual debug log", expanded=True):
            log_placeholder = st.empty()

        def _cf_log(msg: str) -> None:
            debug_lines.append(str(msg))
            # Keep memory bounded
            if len(debug_lines) > 250:
                del debug_lines[:-250]
            # Show tail only to keep UI responsive
            log_placeholder.text("\n".join(debug_lines[-40:]))

        with st.spinner("⏳ Searching for counterfactuals… (this may take 15–30 seconds)"):
            try:
                cf_df, metrics_df, explanation, comparison_df = generate_counterfactual(
                    patient_input,
                    model_path=_DEFAULT_MODEL,
                    preprocessor_path=PREPROCESSOR_PATH,
                    target_class=0,  # Not CKD
                    k=2,
                    max_iter=60,
                    n_neighbors=12,
                    pool_size=4,
                    max_attempts=25,
                    target_prob=0.60,
                    proximity_mode="zscore",
                    selection="pareto",
                    compute_explanation=False,
                    explanation_stability_runs=0,
                    seed=42,
                    weights=_DEFAULT_CF_WEIGHTS,
                    progress_callback=_cf_log,
                )
            except Exception as e:
                st.error("**Counterfactual error:** the search raised an exception.")
                with st.expander("Show technical details"):
                    st.exception(e)
                st.stop()
        st.session_state["cf_result"] = dict(
            cf_df=cf_df,
            metrics_df=metrics_df,
            explanation=explanation,
            comparison_df=comparison_df,
            debug_log=debug_lines,
        )
        st.rerun()

    if "cf_result" not in st.session_state:
        st.info("👆 Click **Generate Counterfactual** above to find what changes would lower CKD risk.")
        st.stop()

    # ── Restore CF results ────────────────────────────────────────────────────
    cf_res       = st.session_state["cf_result"]
    cf_df        = cf_res["cf_df"].copy()
    metrics_df   = cf_res["metrics_df"].copy()
    explanation  = dict(cf_res["explanation"])
    comparison_df = cf_res["comparison_df"].copy()

    debug_log = cf_res.get("debug_log")
    if isinstance(debug_log, list) and debug_log:
        with st.expander("Counterfactual debug log"):
            st.code("\n".join([str(x) for x in debug_log][-250:]), language="text")

    has_cf   = not metrics_df.empty and not cf_df.empty
    has_best = has_cf and (comparison_df["row"] == "best_counterfactual").any()

    orig_row = comparison_df.loc[comparison_df["row"] == "original"].iloc[0]
    best_row = (
        comparison_df.loc[comparison_df["row"] == "best_counterfactual"].iloc[0]
        if has_best else orig_row
    )

    # Button to re-run CF
    if st.button("🔄 Re-run Counterfactual Search", key="btn_recf"):
        st.session_state.pop("cf_result", None)
        st.rerun()

    if not has_best:
        st.warning(explanation.get(
            "note",
            "No valid counterfactual found. Try adjusting inputs closer to normal clinical ranges.",
        ))
    else:
        comp_df = _build_comparison_df(orig_row, best_row)
        n_changed = int(comp_df["_changed"].sum())
        cf_proba  = explanation.get("counterfactual_proba")
        p_cf_ckd  = float(cf_proba[1]) if isinstance(cf_proba, list) and len(cf_proba) > 1 else None

        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown(
                f"<div class='metric-pill'>"
                f"<div class='mp-value'>{n_changed}</div>"
                f"<div class='mp-label'>features to change</div></div>",
                unsafe_allow_html=True,
            )
        with info_cols[1]:
            if p_cf_ckd is not None:
                st.markdown(
                    f"<div class='metric-pill'>"
                    f"<div class='mp-value' style='color:#1F8F4A;'>{(1-p_cf_ckd)*100:.1f}%</div>"
                    f"<div class='mp-label'>Not-CKD probability after suggested changes</div></div>",
                    unsafe_allow_html=True,
                )
        with info_cols[2]:
            if not metrics_df.empty:
                rob = float(metrics_df.iloc[0].get("robustness", 0.0))
                st.markdown(
                    f"<div class='metric-pill'>"
                    f"<div class='mp-value'>{rob:.2f}</div>"
                    f"<div class='mp-label'>prediction robustness</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.dataframe(_style_comparison(comp_df), use_container_width=True, hide_index=True)

        clinical_text = explanation.get("clinical_text", "")
        if isinstance(clinical_text, str) and clinical_text.strip():
            clines = clinical_text.replace("\r\n", "\n").replace("\n", "<br/>")
            st.markdown(
                f"<div class='card card-green' style='margin-top:0.9rem;'>{clines}</div>",
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Feature Change Visualisation
    # ─────────────────────────────────────────────────────────────────────────
    if has_best:
        st.divider()
        st.markdown(
            "### <span class='step-label'>5</span> Feature Change Visualisation",
            unsafe_allow_html=True,
        )
        st.caption(
            "Shaded green bands = approximate normal clinical ranges. "
            "Numeric features only (binary features shown in table above)."
        )
        col_chart, col_prob = st.columns([1.4, 1], gap="large")
        with col_chart:
            st.plotly_chart(_feature_change_chart(orig_row, best_row), use_container_width=True)
        with col_prob:
            prob_fig = _probability_bar_chart(explanation)
            if prob_fig:
                st.caption("Probability shift: original → counterfactual")
                st.plotly_chart(prob_fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — CF Quality Metrics
    # ─────────────────────────────────────────────────────────────────────────
    if not metrics_df.empty:
        st.divider()
        st.markdown(
            "### <span class='step-label'>6</span> Counterfactual Quality Metrics",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown(
                "| Metric | Meaning |\n"
                "|---|---|\n"
                "| **Validity** | Does the CF flip the prediction to Not-CKD? (1 = yes) |\n"
                "| **Proximity** | How close is the CF to the original? Lower = better |\n"
                "| **Sparsity** | Number of features changed — fewer is simpler |\n"
                "| **Target Prob.** | Model's Not-CKD confidence for the CF |\n"
                "| **Robustness** | Prediction stability under slight noise — higher is better |\n"
                "| **Pareto Optimal** | Non-dominated across all objectives? |"
            )

        display_cols = [c for c in
                        ["validity", "proximity", "sparsity", "p_target", "robustness", "pareto_optimal"]
                        if c in metrics_df.columns]
        nice_names = {
            "validity": "Validity", "proximity": "Proximity",
            "sparsity": "Sparsity", "p_target": "Target Prob.",
            "robustness": "Robustness", "pareto_optimal": "Pareto Optimal",
        }
        st.dataframe(
            metrics_df[display_cols].rename(columns=nice_names),
            use_container_width=True, hide_index=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — All CF Options
    # ─────────────────────────────────────────────────────────────────────────
    if has_cf and len(cf_df) > 1:
        st.divider()
        st.markdown(
            "### <span class='step-label'>7</span> All Counterfactual Options",
            unsafe_allow_html=True,
        )
        st.caption("Row 0 (highlighted) is the recommended option. All rows satisfy the Pareto filter.")

        show_cols = CANONICAL_FEATURES + [
            c for c in ["p_target", "proximity", "robustness"] if c in metrics_df.columns
        ]
        merged = pd.concat(
            [cf_df[CANONICAL_FEATURES].reset_index(drop=True),
             metrics_df.reset_index(drop=True)],
            axis=1,
        )
        merged = merged[[c for c in show_cols if c in merged.columns]].copy()
        merged.rename(columns={"p_target": "Not-CKD Prob."}, inplace=True)

        def _hl_best(row: pd.Series) -> list[str]:
            return (
                ["background-color:#EFF6FF; font-weight:700"] * len(merged.columns)
                if row.name == 0 else [""] * len(merged.columns)
            )

        st.dataframe(merged.style.apply(_hl_best, axis=1), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⬇️ Download Full Report", unsafe_allow_html=True)

    bundle = _build_download_bundle(
        patient_input=patient_input,
        cf_df=cf_df,
        metrics_df=metrics_df,
        explanation=explanation,
        comparison_df=comparison_df,
    )
    st.download_button(
        label="📥 Download patient report (ZIP: CSV + JSON)",
        data=bundle,
        file_name=f"ckd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption("Bundle includes: patient input, counterfactuals, quality metrics, SHAP explanation, and comparison table.")


if __name__ == "__main__":
    try:
        import os
        os.chdir(_repo_root())
    except Exception:
        pass
    main()
