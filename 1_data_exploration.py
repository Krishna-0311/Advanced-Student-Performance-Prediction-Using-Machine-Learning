"""
page/1_data_exploration.py
============================
Interactive EDA page:
  - Dataset overview & statistics
  - Grade distribution (pass/fail + A-F)
  - Feature distributions
  - Correlation heatmap
  - Feature vs target scatter/box plots
"""

import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Data Exploration | Student AI", page_icon="📊", layout="wide")

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.section-header {
    color: #a78bfa;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 1.5rem 0 0.8rem 0;
}
[data-testid="stSidebar"] { background: #0D0D1A !important; border-right: 1px solid #7C3AED22; }
</style>
""", unsafe_allow_html=True)

_DARK = "#0F0F1A"
_PANEL = "#1A1A2E"
_TEXT = "#E2E8F0"
_GRID = "#2D2D44"
_PURPLE = "#7C3AED"
_PALETTE = ["#7C3AED","#3B82F6","#10B981","#F59E0B","#EF4444"]


def set_dark():
    plt.rcParams.update({
        "figure.facecolor": _DARK, "axes.facecolor": _PANEL,
        "axes.edgecolor": _GRID, "axes.labelcolor": _TEXT,
        "xtick.color": _TEXT, "ytick.color": _TEXT,
        "text.color": _TEXT, "grid.color": _GRID,
        "legend.facecolor": _PANEL, "legend.edgecolor": _GRID,
    })


@st.cache_data(show_spinner=False)
def load_data():
    from src.preprocessing import load_raw_data
    from src.feature_engineering import engineer_features
    df_raw = load_raw_data()
    df     = engineer_features(df_raw.copy())
    return df_raw, df


# ── Load ──────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Data Exploration")

with st.spinner("Loading data …"):
    df_raw, df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Filters")
    gender_filter = st.multiselect("Gender", ["M", "F"], default=["M", "F"])
    address_filter = st.multiselect("Address", ["Urban", "Rural"], default=["Urban", "Rural"])
    show_engineered = st.checkbox("Show engineered features", value=True)

df_f = df_raw[
    df_raw["gender"].isin(gender_filter) &
    df_raw["address"].isin(address_filter)
].copy()

st.markdown(f"**Filtered dataset:** {len(df_f):,} students")

tabs = st.tabs(["📋 Overview", "📈 Distributions", "🔥 Correlations", "🎯 Feature vs Target", "📐 Engineered"])

# ─── Tab 1: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df_f):,}")
    c2.metric("Features (raw)", df_raw.shape[1] - 3)
    c3.metric("Pass Rate", f"{df_f['pass_fail'].mean()*100:.1f}%")
    c4.metric("Avg Final Grade", f"{df_f['G3'].mean():.2f}/20")

    st.markdown("#### Sample rows")
    st.dataframe(df_f.head(10), width="stretch")

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_f.describe().round(3), width="stretch")

    # Grade label counts
    gc = df_f["grade_label"].value_counts().reset_index()
    gc.columns = ["Grade", "Count"]
    fig_gc = px.bar(
        gc, x="Grade", y="Count", color="Grade",
        color_discrete_sequence=_PALETTE,
        title="Grade Distribution (A-F)",
        template="plotly_dark",
    )
    fig_gc.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
        font_color=_TEXT, showlegend=False,
    )
    st.plotly_chart(fig_gc, width="stretch")

# ─── Tab 2: Distributions ─────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Feature Distributions</p>', unsafe_allow_html=True)

    numeric_cols = df_f.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["pass_fail"]]
    sel = st.selectbox("Select feature", numeric_cols, index=numeric_cols.index("G3") if "G3" in numeric_cols else 0)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            df_f, x=sel, nbins=30,
            color="gender",
            color_discrete_map={"M": "#7C3AED", "F": "#10B981"},
            title=f"Distribution of {sel} by Gender",
            template="plotly_dark",
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL, font_color=_TEXT)
        st.plotly_chart(fig_hist, width="stretch")

    with col_b:
        fig_box = px.box(
            df_f, x="grade_label", y=sel,
            color="grade_label",
            category_orders={"grade_label": ["A","B","C","D","F"]},
            color_discrete_sequence=_PALETTE,
            title=f"{sel} by Grade Label",
            template="plotly_dark",
        )
        fig_box.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
                               font_color=_TEXT, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

    # G1 vs G2 vs G3 scatter
    st.markdown("#### Period Grade Progression (G1 → G2 → G3)")
    fig_scatter = px.scatter_3d(
        df_f, x="G1", y="G2", z="G3",
        color="grade_label",
        category_orders={"grade_label": ["A","B","C","D","F"]},
        color_discrete_sequence=_PALETTE,
        opacity=0.7, size_max=5,
        template="plotly_dark",
        title="3D: G1 × G2 × G3",
    )
    fig_scatter.update_layout(paper_bgcolor=_DARK, font_color=_TEXT)
    st.plotly_chart(fig_scatter, width="stretch")

# ─── Tab 3: Correlations ──────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)

    num_df = df_f.select_dtypes(include=np.number)
    corr = num_df.corr()

    set_dark()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig_corr, ax = plt.subplots(figsize=(16, 12), facecolor=_DARK)
    ax.set_facecolor(_PANEL)
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdPu", center=0, vmax=1, vmin=-1,
        linewidths=0.3, linecolor=_GRID,
        annot=(corr.shape[0] <= 20),
        fmt=".2f" if corr.shape[0] <= 20 else "",
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, color=_TEXT, pad=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(_TEXT)
        lbl.set_fontsize(8)
    fig_corr.tight_layout()
    st.pyplot(fig_corr, width="stretch")

    # Top correlated with G3
    st.markdown("#### Top Features Correlated with G3")
    corr_g3 = corr["G3"].drop("G3").sort_values(key=abs, ascending=False).head(15)
    fig_corr_bar = px.bar(
        x=corr_g3.values, y=corr_g3.index,
        orientation="h",
        color=corr_g3.values,
        color_continuous_scale="RdPu",
        title="Correlation with G3 (Final Grade)",
        template="plotly_dark",
        labels={"x": "Pearson r", "y": "Feature"},
    )
    fig_corr_bar.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
                                font_color=_TEXT, showlegend=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_corr_bar, width="stretch")

# ─── Tab 4: Feature vs Target ─────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Feature vs Target Analysis</p>', unsafe_allow_html=True)

    target_col = st.radio("Target", ["pass_fail (binary)", "grade_label (A-F)"], horizontal=True)
    target     = "pass_fail" if "binary" in target_col else "grade_label"

    num_features = [c for c in df_f.select_dtypes(include=np.number).columns
                    if c not in ["pass_fail", "G3"]]
    feat = st.selectbox("Feature", num_features, key="fvt_feat",
                        index=num_features.index("G2") if "G2" in num_features else 0)

    col_p, col_v = st.columns(2)

    with col_p:
        fig_viol = px.violin(
            df_f, x=target, y=feat, color=target,
            box=True, points="outliers",
            color_discrete_sequence=_PALETTE,
            template="plotly_dark",
            title=f"{feat} by {target}",
        )
        fig_viol.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
                                font_color=_TEXT, showlegend=False)
        st.plotly_chart(fig_viol, width="stretch")

    with col_v:
        group_mean = (
            df_f.groupby(target)[feat]
            .agg(["mean", "std"])
            .reset_index()
        )
        fig_bar2 = px.bar(
            group_mean, x=target, y="mean",
            color=target,
            error_y="std",
            color_discrete_sequence=_PALETTE,
            template="plotly_dark",
            title=f"Mean {feat} ± std by {target}",
        )
        fig_bar2.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
                                font_color=_TEXT, showlegend=False)
        st.plotly_chart(fig_bar2, width="stretch")

    # Categorical breakdown
    st.markdown("#### Categorical Feature Breakdown")
    cat_feat = st.selectbox(
        "Categorical Feature",
        ["gender", "address", "mother_job", "father_job",
         "internet_access", "higher_ed_aspiration", "romantic"],
        key="cat_feat",
    )
    fig_grp = px.histogram(
        df_f, x=cat_feat, color=target,
        barmode="group",
        color_discrete_sequence=_PALETTE,
        template="plotly_dark",
        title=f"{cat_feat} vs {target}",
    )
    fig_grp.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL, font_color=_TEXT)
    st.plotly_chart(fig_grp, width="stretch")

# ─── Tab 5: Engineered Features ───────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">Engineered Features Overview</p>', unsafe_allow_html=True)

    eng_cols = [
        "learning_index", "grade_momentum", "academic_risk_score",
        "parental_edu_avg", "digital_engagement", "study_efficiency",
        "social_index", "behavioural_composite", "G1_sq", "G2_sq",
    ]
    eng_present = [c for c in eng_cols if c in df.columns]

    if eng_present:
        df_eng_disp = df[eng_present + ["pass_fail", "grade_label"]].copy()
        st.dataframe(df_eng_disp.describe().round(3), width="stretch")

        sel_eng = st.selectbox("Engineered Feature to Visualise", eng_present)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_eh = px.histogram(
                df_eng_disp, x=sel_eng, color="grade_label",
                color_discrete_sequence=_PALETTE,
                template="plotly_dark",
                title=f"{sel_eng} distribution by grade",
                barmode="overlay", opacity=0.7,
                category_orders={"grade_label": ["A","B","C","D","F"]},
            )
            fig_eh.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL, font_color=_TEXT)
            st.plotly_chart(fig_eh, width="stretch")

        with col_e2:
            fig_eb = px.box(
                df_eng_disp, x="pass_fail", y=sel_eng,
                color="pass_fail",
                color_discrete_map={0: "#EF4444", 1: "#10B981"},
                template="plotly_dark",
                title=f"{sel_eng} – Pass vs Fail",
                labels={"pass_fail": "Pass (1) / Fail (0)"},
            )
            fig_eb.update_layout(paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
                                  font_color=_TEXT, showlegend=False)
            st.plotly_chart(fig_eb, width="stretch")
    else:
        st.info("Run the app from the project root to load engineered features.")
