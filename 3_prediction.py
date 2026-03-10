"""
pages/3_prediction.py
======================
Interactive single-student prediction page:
  - Step-by-step input form (tabs by category)
  - Real-time Pass/Fail prediction with all 7 models
  - Probability confidence bars
  - Grade label prediction
  - Model confidence comparison
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Predict Student | Student AI",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0D0D1A !important; border-right: 1px solid #7C3AED22; }

.pred-card {
    border: 2px solid;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.pred-pass {
    border-color: #10B981;
    background: #10B98111;
}
.pred-fail {
    border-color: #EF4444;
    background: #EF444411;
}
.pred-label {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.pred-grade {
    font-size: 1.2rem;
    color: #a78bfa;
}
.conf-bar-label {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

_DARK  = "#0F0F1A"
_PANEL = "#1A1A2E"
_TEXT  = "#E2E8F0"
GRADE_LABELS = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_trained_models():
    """Train default models if not already in session state."""
    from src.preprocessing import load_raw_data, prepare_data
    from src.feature_engineering import engineer_features
    from src.models import get_all_models

    df = load_raw_data()
    df = engineer_features(df)
    data = prepare_data(df, target="pass_fail", test_size=0.20, apply_smote=True)

    models = get_all_models(n_classes=2)
    for name, mdl in models.items():
        mdl.fit(data["X_train"].values, data["y_train"])

    return models, data


def build_input_vector(inputs: dict, data_bundle: dict) -> np.ndarray:
    """Convert raw form inputs to scaled feature vector."""
    from src.feature_engineering import engineer_features
    from src.preprocessing import CATEGORICAL_COLS

    df_row = pd.DataFrame([inputs])
    # Add dummy targets for pipeline compatibility
    df_row["G3"]         = 10
    df_row["pass_fail"]  = 1
    df_row["grade_label"] = "D"

    df_row = engineer_features(df_row)

    # Encode categoricals manually
    cat_map = {
        "gender":        {"M": 1, "F": 0},
        "address":       {"Urban": 1, "Rural": 0},
        "family_size":   {"GT3": 1, "LE3": 0},
        "parent_status": {"Together": 1, "Apart": 0},
        "mother_job":    {"at_home":0,"health":1,"other":2,"services":3,"teacher":4},
        "father_job":    {"at_home":0,"health":1,"other":2,"services":3,"teacher":4},
    }
    for col, mapping in cat_map.items():
        if col in df_row.columns:
            df_row[col] = df_row[col].map(mapping).fillna(0)

    feature_names = data_bundle["feature_names"]
    # Add any missing columns as 0
    for col in feature_names:
        if col not in df_row.columns:
            df_row[col] = 0

    X_raw = df_row[feature_names].values.astype(float)
    scaler = data_bundle["scaler"]
    X_scaled = scaler.transform(X_raw)
    return X_scaled


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Student Performance Predictor")
st.markdown("Fill in the student profile below to get an instant performance prediction.")

# Load or re-use models from training page
if "trained_models" in st.session_state and "data_bundle" in st.session_state:
    trained_models = {k: v[0] for k, v in st.session_state["trained_models"].items()}
    data_bundle    = st.session_state["data_bundle"]
    st.info("Using models trained in the **Model Training** page.")
else:
    with st.spinner("Loading default pre-trained models …"):
        trained_models, data_bundle = get_trained_models()

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📝 Student Profile")

form_tabs = st.tabs(["👤 Demographic", "📚 Academic", "💻 LMS / Behaviour", "🌍 Socio-economic"])

inputs: dict = {}

with form_tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["age"]           = st.slider("Age", 15, 22, 17)
        inputs["gender"]        = st.selectbox("Gender", ["M", "F"])
        inputs["address"]       = st.selectbox("Address", ["Urban", "Rural"])
    with c2:
        inputs["family_size"]   = st.selectbox("Family Size", ["GT3", "LE3"])
        inputs["parent_status"] = st.selectbox("Parent Status", ["Together", "Apart"])
        inputs["romantic"]      = st.selectbox("In Relationship", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c3:
        inputs["mother_education"] = st.select_slider("Mother Education", [0,1,2,3,4],
                                                       value=2, format_func=lambda x: ["None","Primary","5-9 grade","Secondary","Higher"][x])
        inputs["father_education"] = st.select_slider("Father Education", [0,1,2,3,4],
                                                       value=2, format_func=lambda x: ["None","Primary","5-9 grade","Secondary","Higher"][x])
        inputs["mother_job"] = st.selectbox("Mother's Job", ["at_home","health","other","services","teacher"])
        inputs["father_job"] = st.selectbox("Father's Job", ["at_home","health","other","services","teacher"])

with form_tabs[1]:
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["G1"]            = st.slider("G1 – Period 1 Grade (0-20)", 0, 20, 12)
        inputs["G2"]            = st.slider("G2 – Period 2 Grade (0-20)", 0, 20, 13)
        inputs["study_time"]    = st.select_slider("Study Time", [1,2,3,4],
                                                    format_func=lambda x: ["<2h","2-5h","5-10h",">10h"][x-1])
    with c2:
        inputs["past_failures"]      = st.selectbox("Past Failures", [0,1,2,3])
        inputs["extra_support"]      = st.selectbox("Extra Educational Support", [0,1], format_func=lambda x: "Yes" if x else "No")
        inputs["family_support"]     = st.selectbox("Family Support", [0,1], format_func=lambda x: "Yes" if x else "No")
    with c3:
        inputs["paid_classes"]       = st.selectbox("Paid Extra Classes", [0,1], format_func=lambda x: "Yes" if x else "No")
        inputs["activities"]         = st.selectbox("Extra Curricular Activities", [0,1], format_func=lambda x: "Yes" if x else "No")
        inputs["higher_ed_aspiration"] = st.selectbox("Wants Higher Education", [1,0], format_func=lambda x: "Yes" if x else "No")
        inputs["absences"]           = st.slider("Number of Absences", 0, 75, 5)

with form_tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["login_frequency"]           = st.slider("Login Frequency (per month)", 0, 30, 15)
        inputs["assignment_submission_rate"] = st.slider("Assignment Submission Rate", 0.0, 1.0, 0.75)
    with c2:
        inputs["attendance_rate"]  = st.slider("Attendance Rate", 0.0, 1.0, 0.80)
        inputs["engagement_score"] = st.slider("Engagement Score (0-10)", 0.0, 10.0, 6.0)
    with c3:
        inputs["forum_posts"]     = st.slider("Forum Posts", 0, 25, 5)
        inputs["resource_access"] = st.slider("Resources Accessed", 0, 60, 25)

with form_tabs[3]:
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["internet_access"] = st.selectbox("Internet Access", [1,0], format_func=lambda x: "Yes" if x else "No")
        inputs["nursery"]         = st.selectbox("Attended Nursery", [1,0], format_func=lambda x: "Yes" if x else "No")
        inputs["travel_time"]     = st.select_slider("Travel Time", [1,2,3,4],
                                                      format_func=lambda x: ["<15min","15-30min","30min-1h",">1h"][x-1])
    with c2:
        inputs["free_time"] = st.slider("Free Time (1-5)", 1, 5, 3)
        inputs["go_out"]    = st.slider("Going Out (1-5)", 1, 5, 3)
        inputs["health"]    = st.slider("Health Status (1-5)", 1, 5, 3)
    with c3:
        inputs["dalc"] = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
        inputs["walc"] = st.slider("Weekend Alcohol (1-5)", 1, 5, 2)

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Performance", type="primary", width="stretch")

if predict_btn:
    with st.spinner("Computing predictions …"):
        try:
            X_input = build_input_vector(inputs, data_bundle)
        except Exception as e:
            st.error(f"Error building feature vector: {e}")
            st.stop()

    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_input)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_input)[0]
            pass_prob = prob[1] if len(prob) > 1 else float(y_pred)
        else:
            pass_prob = float(y_pred)
        results[name] = {"pred": int(y_pred), "pass_prob": pass_prob}

    # Majority vote
    votes      = [v["pred"] for v in results.values()]
    majority   = int(np.round(np.mean(votes)))
    avg_prob   = np.mean([v["pass_prob"] for v in results.values()])
    # Grade prediction from G1/G2 heuristic
    g_approx = 0.5 * inputs["G2"] + 0.3 * inputs["G1"] + inputs["attendance_rate"] * 3 - inputs["past_failures"] * 2
    g_approx = np.clip(g_approx, 0, 20)
    if   g_approx >= 16: grade = "A"
    elif g_approx >= 14: grade = "B"
    elif g_approx >= 12: grade = "C"
    elif g_approx >= 10: grade = "D"
    else:                grade = "F"

    # ── Display result ─────────────────────────────────────────────────────────
    st.markdown("## 🎯 Prediction Results")
    rc1, rc2 = st.columns([1, 1.5])

    with rc1:
        if majority == 1:
            st.markdown(
                '<div class="pred-card pred-pass">'
                '<div class="pred-label" style="color:#10B981">✅ PASS</div>'
                f'<div class="pred-grade">Predicted Grade: <strong>{grade}</strong></div>'
                f'<div style="color:#94a3b8;margin-top:.5rem;">Avg Pass Confidence: {avg_prob*100:.1f}%</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-card pred-fail">'
                '<div class="pred-label" style="color:#EF4444">❌ FAIL</div>'
                f'<div class="pred-grade">Predicted Grade: <strong>{grade}</strong></div>'
                f'<div style="color:#94a3b8;margin-top:.5rem;">Avg Pass Confidence: {avg_prob*100:.1f}%</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Model Agreement")
        pass_count = sum(v == 1 for v in votes)
        st.progress(pass_count / len(votes), text=f"{pass_count}/{len(votes)} models predict PASS")

    with rc2:
        # Confidence bars per model
        st.markdown("#### Per-Model Confidence (Pass Probability)")
        names_list = list(results.keys())
        probs_list = [results[n]["pass_prob"] * 100 for n in names_list]
        preds_list = [results[n]["pred"] for n in names_list]
        colors     = ["#10B981" if p == 1 else "#EF4444" for p in preds_list]

        fig_conf = go.Figure(go.Bar(
            x=probs_list,
            y=names_list,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs_list],
            textposition="outside",
        ))
        fig_conf.add_vline(x=50, line_dash="dash", line_color="white", line_width=1,
                            annotation_text="50%", annotation_position="top")
        fig_conf.update_layout(
            paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
            font_color=_TEXT,
            xaxis=dict(range=[0,110], title="Pass Probability (%)"),
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=10,r=10,t=10,b=30),
        )
        st.plotly_chart(fig_conf, width="stretch")

    # ── Intervention suggestions ───────────────────────────────────────────────
    st.markdown("### 💡 Intervention Recommendations")
    recs = []
    if inputs["attendance_rate"] < 0.75:
        recs.append("📅 **Improve attendance** – currently below 75%. Attendance strongly correlates with final grade.")
    if inputs["assignment_submission_rate"] < 0.70:
        recs.append("📝 **Submit more assignments** – submission rate is below 70%.")
    if inputs["study_time"] <= 1:
        recs.append("⏱️ **Increase study time** – recommended minimum 2-5 hours/week.")
    if inputs["past_failures"] > 0:
        recs.append(f"⚠️ **Past failures ({inputs['past_failures']})** – consider extra tutoring support.")
    if inputs["engagement_score"] < 5:
        recs.append("💬 **Boost LMS engagement** – low engagement score detected.")
    if inputs["absences"] > 15:
        recs.append(f"🚨 **High absences ({inputs['absences']})** – reducing absences can significantly impact performance.")
    if not recs:
        recs.append("✅ **Student profile looks strong** – maintain current performance and engagement levels.")
    for r in recs:
        st.markdown(f"- {r}")
