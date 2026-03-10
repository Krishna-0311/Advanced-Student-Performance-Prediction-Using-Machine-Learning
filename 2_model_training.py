"""
pages/2_model_training.py
==========================
Train and compare 7 ML models:
  - Select target (binary / multiclass)
  - Toggle hyperparameter tuning
  - Display metrics table with highlighting
  - ROC curves, confusion matrix, learning curve, feature importance
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Model Training | Student AI",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0D0D1A !important; border-right: 1px solid #7C3AED22; }
.metric-highlight {
    background: linear-gradient(135deg, #7C3AED22, #3B82F622);
    border: 1px solid #7C3AED44;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-highlight .val { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-highlight .lbl { color: #64748b; font-size: 0.8rem; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

_DARK  = "#0F0F1A"
_PANEL = "#1A1A2E"
_TEXT  = "#E2E8F0"

# ── Sidebar: training config ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training Config")
    target_mode = st.radio(
        "Prediction Target",
        ["Pass / Fail (binary)", "Grade Label A-F (multiclass)"],
    )
    target = "pass_fail" if "binary" in target_mode else "grade_label"

    apply_smote   = st.toggle("Apply SMOTE",          value=True)
    apply_feature = st.toggle("Feature Engineering",  value=True)
    scale_method  = st.selectbox("Scaling", ["standard", "minmax"])
    test_size     = st.slider("Test set size (%)", 10, 40, 20) / 100

    st.markdown("---")
    st.markdown("### 🤖 Models to Train")
    available_models = [
        "Logistic Regression", "Decision Tree", "SVM (RBF)",
        "Random Forest", "Gradient Boosting", "XGBoost", "Voting Classifier",
    ]
    selected_models = st.multiselect(
        "Select models",
        available_models,
        default=available_models,
    )

    tune_model_name = st.selectbox(
        "Tune model (RandomizedSearchCV)",
        ["None"] + ["Random Forest", "Gradient Boosting", "XGBoost"],
    )
    tune_iters = st.slider("Tuning iterations", 5, 30, 10)


# ── Load & Preprocess ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(target, smote, feat_eng, scale, test_sz):
    from src.preprocessing import load_raw_data, prepare_data
    from src.feature_engineering import engineer_features

    df = load_raw_data()
    if feat_eng:
        df = engineer_features(df)
    return prepare_data(df, target=target, test_size=test_sz,
                        apply_smote=smote, scale_method=scale)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Model Training & Comparison")

if not selected_models:
    st.warning("Please select at least one model in the sidebar.")
    st.stop()

train_btn = st.button("🚀 Train Selected Models", type="primary", width="stretch")

if train_btn or "trained_models" in st.session_state:

    if train_btn:
        # Clear old results
        for key in ["trained_models", "metrics_df", "data_bundle"]:
            if key in st.session_state:
                del st.session_state[key]

    # ── Preprocessing ──────────────────────────────────────────────────────────
    with st.spinner("Preprocessing data …"):
        data = load_and_preprocess(target, apply_smote, apply_feature, scale_method, test_size)
        st.session_state["data_bundle"] = data

    n_classes    = data["n_classes"]
    X_train      = data["X_train"].values
    X_test       = data["X_test"].values
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    feature_names = data["feature_names"]

    st.info(
        f"**Train:** {len(y_train):,}  |  **Test:** {len(y_test):,}  |  "
        f"**Features:** {len(feature_names)}  |  **Classes:** {n_classes}"
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    if train_btn or "trained_models" not in st.session_state:
        from src.models import get_all_models, train_all_models, tune_model

        all_models = get_all_models(n_classes=n_classes)
        models_to_train = {k: v for k, v in all_models.items() if k in selected_models}

        prog = st.progress(0, text="Training …")
        trained: dict = {}
        for i, (name, mdl) in enumerate(models_to_train.items()):
            prog.progress((i) / len(models_to_train), text=f"Training {name} …")
            t0 = time.perf_counter()
            mdl.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            trained[name] = (mdl, elapsed)

        # Optional tuning
        if tune_model_name != "None" and tune_model_name in selected_models:
            prog.progress(0.95, text=f"Tuning {tune_model_name} …")
            best = tune_model(tune_model_name, X_train, y_train,
                              n_classes=n_classes, n_iter=tune_iters, cv=3)
            if best is not None:
                t0 = time.perf_counter()
                best.fit(X_train, y_train)
                trained[f"{tune_model_name} (Tuned)"] = (best, time.perf_counter() - t0)

        prog.progress(1.0, text="Done!")
        time.sleep(0.3)
        prog.empty()
        st.session_state["trained_models"] = trained

    trained = st.session_state["trained_models"]

    # ── Metrics ────────────────────────────────────────────────────────────────
    from src.evaluation import (
        compute_all_metrics,
        plot_metrics_comparison,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_learning_curve,
        plot_feature_importance,
        get_classification_report,
    )

    metrics_df = compute_all_metrics(trained, X_test, y_test, n_classes)
    st.session_state["metrics_df"] = metrics_df

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    st.markdown("## 🏆 Results Summary")
    best_name = metrics_df["F1-Score"].idxmax()
    best_row  = metrics_df.loc[best_name]

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, metric in zip([k1, k2, k3, k4, k5],
                            ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]):
        val = best_row[metric]
        col.markdown(
            f'<div class="metric-highlight">'
            f'<div class="val">{val:.3f}</div>'
            f'<div class="lbl">{metric}<br><span style="color:#a78bfa;font-size:.7rem">({best_name})</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.markdown("## 📊 Full Metrics Table")

    def _style_df(df):
        return df.style.background_gradient(
            subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
            cmap="Purples",
        ).format(precision=4)

    st.dataframe(_style_df(metrics_df), width="stretch")

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown("## 📈 Visualisations")
    v_tabs = st.tabs(["Comparison","ROC Curves","Confusion Matrix","Learning Curve","Feature Importance"])

    with v_tabs[0]:
        fig_cmp = plot_metrics_comparison(metrics_df)
        st.pyplot(fig_cmp, width="stretch")

    with v_tabs[1]:
        if n_classes == 2:
            fig_roc = plot_roc_curves(trained, X_test, y_test, n_classes)
            st.pyplot(fig_roc, width="stretch")
        else:
            st.info("ROC curve plot available for binary classification only.")

    with v_tabs[2]:
        model_sel = st.selectbox("Model", list(trained.keys()), key="cm_model")
        class_names = (
            ["Fail","Pass"] if n_classes == 2
            else ["F","D","C","B","A"]
        )
        fig_cm = plot_confusion_matrix(
            trained[model_sel][0], X_test, y_test,
            class_names=class_names, model_name=model_sel,
        )
        st.pyplot(fig_cm, width="stretch")

        report_str = get_classification_report(
            trained[model_sel][0], X_test, y_test, class_names
        )
        with st.expander("📄 Classification Report"):
            st.code(report_str)

    with v_tabs[3]:
        lc_model_name = st.selectbox("Model for learning curve", list(trained.keys()), key="lc_model")
        lc_model = trained[lc_model_name][0]
        with st.spinner("Computing learning curve …"):
            try:
                fig_lc = plot_learning_curve(lc_model, X_train, y_train,
                                              model_name=lc_model_name, cv=3)
                st.pyplot(fig_lc, width="stretch")
            except Exception as ex:
                st.error(f"Learning curve error: {ex}")

    with v_tabs[4]:
        fi_model_name = st.selectbox("Model", list(trained.keys()), key="fi_model")
        fi_model = trained[fi_model_name][0]
        fig_fi = plot_feature_importance(fi_model, feature_names, top_n=20, model_name=fi_model_name)
        if fig_fi:
            st.pyplot(fig_fi, width="stretch")
        else:
            st.info("Feature importance not available for this model type.")

    # ── Download metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    csv = metrics_df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Metrics CSV",
        data=csv,
        file_name="model_metrics.csv",
        mime="text/csv",
        width="stretch",
    )

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color:#64748b;">
            <div style="font-size:5rem;">🤖</div>
            <h3 style="color:#a78bfa;">Configure & Train</h3>
            <p>Use the sidebar to configure training options, then click
            <strong>Train Selected Models</strong> above.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
