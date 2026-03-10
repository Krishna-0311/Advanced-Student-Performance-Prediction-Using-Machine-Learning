"""
app.py – Main Streamlit entry point
=====================================
Home page with project overview, dataset statistics, and quick navigation.
"""

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: #0F0F1A; }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #7C3AED44;
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, #7C3AED22 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #7C3AED, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero p { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1A2E, #16213e);
        border: 1px solid #7C3AED33;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    .metric-card:hover { border-color: #7C3AED; }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Nav cards */
    .nav-card {
        background: #1A1A2E;
        border: 1px solid #2D2D44;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s, transform 0.2s;
        cursor: pointer;
    }
    .nav-card:hover { border-color: #7C3AED; transform: translateY(-2px); }
    .nav-card h4 { color: #a78bfa; margin-bottom: 0.4rem; }
    .nav-card p  { color: #64748b; font-size: 0.88rem; margin: 0; }

    /* Tags */
    .tag {
        display: inline-block;
        background: #7C3AED22;
        color: #a78bfa;
        border: 1px solid #7C3AED55;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        margin: 0.2rem;
        font-weight: 500;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D0D1A !important;
        border-right: 1px solid #7C3AED22;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #2D2D44;
        margin: 2rem 0;
    }

    /* Streamlit elements */
    .stMetric { background: #1A1A2E; border-radius: 10px; padding: 1rem; }
    div[data-testid="stMetricValue"] { color: #a78bfa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Dataset cache ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    from src.preprocessing import load_raw_data
    from src.feature_engineering import engineer_features
    df_raw = load_raw_data()
    df_eng = engineer_features(df_raw.copy())
    return df_raw, df_eng


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
            <span style="font-size:3rem;">🎓</span>
            <h2 style="color:#a78bfa; font-size:1.3rem; margin:0.5rem 0 0 0;">
                Student AI
            </h2>
            <p style="color:#64748b; font-size:0.8rem; margin:0;">
                Performance Prediction
            </p>
        </div>
        <hr style="border-color:#2D2D44;"/>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 Navigation")
    st.page_link("app.py",                      label="🏠 Home",               icon=None)
    st.page_link("pages/1_data_exploration.py",  label="📊 Data Exploration",   icon=None)
    st.page_link("pages/2_model_training.py",    label="🤖 Model Training",     icon=None)
    st.page_link("pages/3_prediction.py",        label="🔮 Predict Student",    icon=None)
    st.page_link("pages/4_explainability.py",    label="🔍 Explainability",     icon=None)

    st.markdown("<hr style='border-color:#2D2D44;'/>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#475569; font-size:0.75rem; text-align:center;'>"
        "ML Framework · Scikit-learn · XGBoost · SHAP</p>",
        unsafe_allow_html=True,
    )


# ── Main content ───────────────────────────────────────────────────────────────
# Hero
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Performance AI</h1>
        <p>Advanced Machine Learning framework for predicting academic success<br>
        using ensemble models, feature engineering, and explainable AI (SHAP).</p>
        <div style="margin-top:1.5rem;">
            <span class="tag">🌲 Random Forest</span>
            <span class="tag">⚡ XGBoost</span>
            <span class="tag">📈 Gradient Boosting</span>
            <span class="tag">🗳️ Voting Ensemble</span>
            <span class="tag">🔍 SHAP</span>
            <span class="tag">📊 SMOTE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
with st.spinner("Loading dataset …"):
    df_raw, df_eng = load_data()

# Quick stats
st.markdown("### 📊 Dataset at a Glance")
c1, c2, c3, c4, c5 = st.columns(5)

pass_pct = df_raw["pass_fail"].mean() * 100

with c1:
    st.markdown(
        f'<div class="metric-card"><div class="value">{len(df_raw):,}</div>'
        f'<div class="label">Students</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="value">{df_raw.shape[1]}</div>'
        f'<div class="label">Raw Features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="value">{df_eng.shape[1]}</div>'
        f'<div class="label">Engineered Features</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="value">{pass_pct:.1f}%</div>'
        f'<div class="label">Pass Rate</div></div>', unsafe_allow_html=True)
with c5:
    avg_g3 = df_raw["G3"].mean()
    st.markdown(
        f'<div class="metric-card"><div class="value">{avg_g3:.1f}</div>'
        f'<div class="label">Avg Final Grade</div></div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

# Feature groups and navigation cards
col_nav, col_info = st.columns([1, 1.4])

with col_nav:
    st.markdown("### 🗺️ Explore the App")
    st.markdown(
        """
        <div class="nav-card">
            <h4>📊 Data Exploration</h4>
            <p>Distribution plots, correlation heatmap, grade breakdown, feature-target relationships.</p>
        </div>
        <div class="nav-card">
            <h4>🤖 Model Training</h4>
            <p>Train 7 ML models, compare metrics table, ROC curves, confusion matrices, learning curves.</p>
        </div>
        <div class="nav-card">
            <h4>🔮 Student Prediction</h4>
            <p>Input any student profile and get instant Pass/Fail + grade prediction with confidence.</p>
        </div>
        <div class="nav-card">
            <h4>🔍 Explainability (SHAP)</h4>
            <p>Global feature importance and per-student waterfall explanations using SHAP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_info:
    st.markdown("### 🏗️ ML Pipeline Architecture")
    st.markdown(
        """
        ```
        Raw Dataset (1200 students)
               ↓
        Data Cleaning & Imputation
               ↓
        Feature Engineering (+9 composite features)
               ↓
        Label Encoding + Standard Scaling
               ↓
        SMOTE (class imbalance correction)
               ↓
        Train / Test Split  (80/20, stratified)
               ↓
        ┌─────────────────────────────────────┐
        │  Baseline     │   Ensemble          │
        │  • LR         │   • Random Forest   │
        │  • DT         │   • Grad. Boosting  │
        │  • SVM        │   • XGBoost         │
        │               │   • Voting (RF+GBM  │
        │               │          +XGB)      │
        └─────────────────────────────────────┘
               ↓
        Evaluation: Accuracy / F1 / ROC-AUC
               ↓
        SHAP Explainability
        ```
        """
    )

    st.markdown("### 📐 Feature Groups")
    feat_groups = {
        "👤 Demographic":   "age, gender, address, family size, parental edu & job",
        "📚 Academic":      "G1, G2, study time, failures, absences",
        "💻 LMS Behaviour": "login freq, submission rate, attendance, engagement",
        "🧮 Engineered":    "learning index, grade momentum, risk score, behavioural composite",
        "🌍 Socio-econ":    "internet access, extra activities, alcohol, health",
    }
    for group, desc in feat_groups.items():
        with st.expander(group):
            st.write(desc)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

# Literature references
st.markdown("### 📚 Research Foundation")
refs = [
    ("Ahmed et al. (2021)", "Ensemble models outperform traditional classifiers on academic data – *Education & IT*"),
    ("Alhusban et al. (2022)", "Systematic review: deep learning + ensembles dominate post-2020 EDM – *IEEE Access*"),
    ("Khan et al. (2022)", "Hybrid ML + feature selection reduces overfitting in HE datasets – *ESA*"),
    ("Alqahtani & Rajkumar (2023)", "LMS behavioural logs significantly boost prediction accuracy – *C&E: AI*"),
    ("Wang et al. (2023)", "SHAP-integrated ensembles for interpretable student prediction – *KBS*"),
]
for author, text in refs:
    st.markdown(f"**{author}** — {text}")
