"""
pages/4_explainability.py
==========================
SHAP-based explainability page:
  - Global: summary (bar) plot, beeswarm, mean |SHAP| table
  - Local: waterfall for individual prediction
  - Dependence plot for any feature pair
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Explainability | Student AI",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0D0D1A !important; border-right: 1px solid #7C3AED22; }
.shap-info {
    background: #1A1A2E;
    border: 1px solid #7C3AED44;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    color: #94a3b8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

_DARK  = "#0F0F1A"
_PANEL = "#1A1A2E"
_TEXT  = "#E2E8F0"


# ── Data / model cache ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_defaults():
    from src.preprocessing import load_raw_data, prepare_data
    from src.feature_engineering import engineer_features
    from src.models import get_all_models

    df = load_raw_data()
    df = engineer_features(df)
    data = prepare_data(df, target="pass_fail", test_size=0.20, apply_smote=True)

    models   = get_all_models(n_classes=2)
    trained  = {}
    for name, mdl in models.items():
        mdl.fit(data["X_train"].values, data["y_train"])
        trained[name] = (mdl, 0.0)

    return trained, data


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Explainable AI – SHAP Analysis")

st.markdown("""
<div class="shap-info">
<strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) assigns each feature a contribution score
for each prediction. Positive values push toward <em>Pass</em>, negative values toward <em>Fail</em>.
Global plots show average importance across all students; Local waterfall plots explain one individual prediction.
</div>
""", unsafe_allow_html=True)

# Load
if "trained_models" in st.session_state and "data_bundle" in st.session_state:
    all_trained   = st.session_state["trained_models"]
    data_bundle   = st.session_state["data_bundle"]
    st.info("Using models from the **Model Training** page.")
else:
    with st.spinner("Loading default models for SHAP …"):
        all_trained, data_bundle = load_defaults()

feature_names = data_bundle["feature_names"]
X_test        = data_bundle["X_test"]
X_test_raw    = data_bundle["X_test_raw"]
y_test        = data_bundle["y_test"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SHAP Config")
    tree_models = {
        k: (v[0] if isinstance(v, tuple) else v)
        for k, v in all_trained.items()
        if any(x in type(v[0] if isinstance(v, tuple) else v).__name__
               for x in ["Forest","Boosting","XGB","Tree","Voting"])
    }
    if not tree_models:
        tree_models = {
            k: (v[0] if isinstance(v, tuple) else v)
            for k, v in all_trained.items()
        }

    model_name   = st.selectbox("Model for SHAP", list(tree_models.keys()))
    selected_mdl = tree_models[model_name]
    max_samples  = st.slider("Max samples for SHAP", 50, 300, 150, step=50)
    class_idx    = st.selectbox("Class index", [0, 1],
                                 format_func=lambda x: "Fail (0)" if x == 0 else "Pass (1)")
    max_display  = st.slider("Max features shown", 5, 30, 15)

# ── Compute SHAP ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_shap(_model, _X_test, _X_train, _model_name: str, _max_samples: int):
    from src.explainability import compute_shap_values
    return compute_shap_values(
        _model, _X_test, X_background=_X_train,
        model_name=_model_name, max_samples=_max_samples,
    )


X_arr    = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
X_bg_arr = data_bundle["X_train"].values if isinstance(data_bundle["X_train"], pd.DataFrame) \
           else data_bundle["X_train"]

with st.spinner(f"Computing SHAP values for **{model_name}** …"):
    try:
        explainer, shap_vals = compute_shap(
            selected_mdl, X_arr, X_bg_arr, model_name, max_samples
        )
        shap_ok = True
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        shap_ok = False

if not shap_ok:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🌐 Global Summary", "🐝 Beeswarm", "📊 Mean |SHAP| Table",
                "🔬 Local Waterfall", "🔗 Dependence Plot"])

# Subsample X for plots
n_plot = min(max_samples, len(X_arr))
idx_plot = np.random.default_rng(42).choice(len(X_arr), n_plot, replace=False)
X_plot  = X_arr[idx_plot]

sv = shap_vals
if isinstance(sv, list):
    sv = sv[min(class_idx, len(sv) - 1)]

# ─── Global Summary (bar) ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🌐 Global Feature Importance (Bar)")
    st.markdown("The bar chart shows the **mean absolute SHAP value** — how much each feature "
                "contributes on average to the model output.")
    from src.explainability import shap_summary_plot
    with st.spinner("Rendering …"):
        fig_bar = shap_summary_plot(
            shap_vals, X_plot, feature_names,
            plot_type="bar", class_idx=class_idx,
            max_display=max_display,
        )
        st.pyplot(fig_bar, width="stretch")

# ─── Beeswarm ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🐝 SHAP Beeswarm Plot")
    st.markdown("Each point is one student. **Colour = feature value** (red=high, blue=low). "
                "Horizontal position = SHAP impact.")
    from src.explainability import shap_summary_plot
    with st.spinner("Rendering …"):
        fig_bee = shap_summary_plot(
            shap_vals, X_plot, feature_names,
            plot_type="violin", class_idx=class_idx,
            max_display=max_display,
        )
        st.pyplot(fig_bee, width="stretch")

# ─── Mean |SHAP| Table ────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📊 Mean |SHAP| Value Table")
    from src.explainability import shap_to_dataframe
    df_shap = shap_to_dataframe(shap_vals, feature_names, class_idx=class_idx)
    df_shap_disp = df_shap.head(25)

    st.dataframe(
        df_shap_disp.style.background_gradient(subset=["Mean |SHAP|"], cmap="Purples"),
        width="stretch",
    )

    fig_shap_bar = px.bar(
        df_shap_disp.iloc[::-1],
        x="Mean |SHAP|", y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Purples",
        template="plotly_dark",
        title="Top Feature Importances (Mean |SHAP|)",
    )
    fig_shap_bar.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
        font_color=_TEXT, showlegend=False,
        height=max(400, len(df_shap_disp) * 22),
    )
    st.plotly_chart(fig_shap_bar, width="stretch")

# ─── Local Waterfall ─────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🔬 Local Explanation – Waterfall Plot")
    st.markdown("Select a test student index to see which features **pushed** the prediction "
                "toward Pass or Fail.")

    sample_idx = st.slider(
        "Test student index", 0, len(X_arr) - 1, 0
    )
    X_instance = X_arr[sample_idx]
    true_label = y_test[sample_idx]
    pred_label = selected_mdl.predict(X_instance.reshape(1,-1))[0]

    c1, c2 = st.columns(2)
    c1.metric("True Label",      "PASS ✅" if true_label == 1 else "FAIL ❌")
    c2.metric("Predicted Label", "PASS ✅" if pred_label == 1 else "FAIL ❌")

    from src.explainability import shap_waterfall_plot
    with st.spinner("Rendering waterfall …"):
        fig_wf = shap_waterfall_plot(
            explainer, X_instance, feature_names, class_idx=class_idx
        )
        st.pyplot(fig_wf, width="stretch")

    # Feature value table for this student
    with st.expander("📋 Feature values for this student"):
        row_df = pd.DataFrame({
            "Feature": feature_names,
            "Value":   X_instance,
            "SHAP":    sv[idx_plot[0]] if sv.shape[0] > 0 else [0]*len(feature_names),
        })
        row_df["Value"] = row_df["Value"].round(4)
        row_df["SHAP"]  = row_df["SHAP"].round(5)
        st.dataframe(row_df.sort_values("SHAP", key=abs, ascending=False),
                     width="stretch")

# ─── Dependence Plot ─────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🔗 SHAP Dependence Plot")
    st.markdown("Shows how a feature's SHAP value changes with its raw value, "
                "and optionally colour-codes by an interaction feature.")

    col_dep1, col_dep2 = st.columns(2)
    with col_dep1:
        dep_feat = st.selectbox("Primary Feature", feature_names,
                                 index=feature_names.index("G2") if "G2" in feature_names else 0,
                                 key="dep_feat")
    with col_dep2:
        interaction_opts = ["auto"] + [f for f in feature_names if f != dep_feat]
        dep_interact = st.selectbox("Interaction Feature (colour)", interaction_opts,
                                     key="dep_interact")

    from src.explainability import shap_dependence_plot
    with st.spinner("Rendering dependence plot …"):
        try:
            fig_dep = shap_dependence_plot(
                shap_vals, X_plot, feature_names,
                feature=dep_feat,
                interaction_feature=dep_interact,
                class_idx=class_idx,
            )
            st.pyplot(fig_dep, width="stretch")
        except Exception as e:
            st.error(f"Dependence plot failed: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#475569; font-size:0.8rem; text-align:center;'>"
    "SHAP explanations help academic staff understand <em>why</em> a student is predicted "
    "to pass or fail, enabling targeted and fair interventions.</p>",
    unsafe_allow_html=True,
)
