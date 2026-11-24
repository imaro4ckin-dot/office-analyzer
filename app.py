# app.py — Streamlit UI for "Who Said It?" NLP Classifier

import streamlit as st
import plotly.express as px
from backend import load_artifacts, predict

st.set_page_config(
    page_title="Who Said It? — The Office Classifier",
    page_icon="🏢",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        font-weight: 700;
    }
    .sub-text {
        text-align: center;
        color: #888888;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None

artifacts = get_artifacts()

st.markdown('<p class="main-header">🏢 Who Said It? — The Office NLP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Type a quote or dialogue from <b>The Office</b> and let AI guess the speaker!</p>', unsafe_allow_html=True)

if artifacts is None:
    st.error("⚠️ Model artifacts not found! Please run `python3 backend.py` in the terminal first.")
    st.stop()

user_input = st.text_area(
    "💬 Enter dialogue text here:",
    placeholder="Example: Bears. Beets. Battlestar Galactica.",
    height=100
)

col_space1, col_btn, col_space2 = st.columns([1, 2, 1])
with col_btn:
    predict_button = st.button("✨ Predict Character", type="primary", use_container_width=True)

if predict_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid text.")
    else:
        with st.spinner("🔍 Analyzing dialogue patterns..."):
            result = predict(user_input, artifacts)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

    st.success("🎉 Analysis successful!")

    with st.expander("🔍 View Preprocessed Text (NLP Tokenization)"):
        st.write(f"**Cleaned text:** `{result['clean_text']}`")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📌 Baseline (Logistic Regression)", value=result["lr_prediction"])
    with col2:
        st.metric(label="🚀 Advanced (XGBoost)", value=result["xgb_prediction"])

    if result["vibe"]:
        st.info(f"🎭 **Detected Dialogue Vibe (Cluster {result['cluster_id']}):** {result['vibe']}")

    st.markdown("---")
    st.subheader("📊 Probability Breakdown")

    model_choice = st.selectbox(
        "Select model to view probability distribution:",
        ["Logistic Regression", "XGBoost"]
    )

    if model_choice == "Logistic Regression":
        probs = result["lr_probs"]
        chart_title = "Probability Distribution (Logistic Regression)"
        color_scale = "Blues"
    else:
        probs = result["xgb_probs"]
        chart_title = "Probability Distribution (XGBoost)"
        color_scale = "Greens"

    fig = px.bar(
        x=list(probs.keys()),
        y=list(probs.values()),
        labels={"x": "Character", "y": "Probability"},
        title=chart_title,
        color=list(probs.values()),
        color_continuous_scale=color_scale
    )
    fig.update_layout(xaxis_title="Character", yaxis_title="Probability", title_x=0.1)
    st.plotly_chart(fig, use_container_width=True)
