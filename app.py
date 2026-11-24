# app.py — Streamlit UI for "Who Said It?" NLP Classifier

import streamlit as st
import plotly.express as px
from backend import load_artifacts, predict

# Page configuration
st.set_page_config(
    page_title="Who Said It? — The Office Classifier",
    page_icon="📋",
    layout="centered"
)

# Load models and artifacts (cached for performance)
@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None

artifacts = get_artifacts()

# Header section
st.title("🏢 Who Said It? — The Office NLP Classifier")
st.markdown("Type a quote or dialogue from **The Office** to predict which character said it!")

if artifacts is None:
    st.error("⚠️ Model artifacts not found! Please run `python3 backend.py` in the terminal first.")
    st.stop()

# User input text area
user_input = st.text_area("Enter dialogue text here:", placeholder="Example: Bears. Beets. Battlestar Galactica.")

if st.button("Predict Character", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a valid text.")
    else:
        with st.spinner("Analyzing dialogue..."):
            # Call predict function from backend
            result = predict(user_input, artifacts)
            
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Analysis successful!")
            
            # Show preprocessed text for transparency
            with st.expander("🔍 View Preprocessed Text"):
                st.write(f"**Cleaned text:** `{result['clean_text']}`")
            
            # Display predictions in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Baseline (Logistic Regression)", value=result["lr_prediction"])
            with col2:
                st.metric(label="Advanced (XGBoost)", value=result["xgb_prediction"])
                
            # Display cluster vibe / tone
            if result["vibe"]:
                st.info(f"🎭 **Dialogue Vibe (Cluster {result['cluster_id']}):** {result['vibe']}")
                
            st.markdown("---")
            st.subheader("📊 Probability Breakdown")
            
            # Selectbox to switch between Logistic Regression and XGBoost probabilities
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
            
            # Prepare data for Plotly chart
            fig = px.bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                labels={"x": "Character", "y": "Probability"},
                title=chart_title,
                color=list(probs.values()),
                color_continuous_scale=color_scale
            )
            st.plotly_chart(fig, use_container_width=True)