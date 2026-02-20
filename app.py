# ============================================================
# 🏥 Health Risk Intelligence Platform
# Tuned Decision Tree | Hyperparameter Optimized
# Developed by Akshit Gajera
# ============================================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Health Risk Intelligence",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------------------------
# BLUE ENTERPRISE THEME (UNCHANGED)
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
}
.result-box {
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("hyper.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

feature_names = [
    "age","gender","bmi","daily_steps","sleep_hours",
    "water_intake","calories","smoker","alcohol",
    "resting_hr","systolic_bp","diastolic_bp",
    "cholesterol","family_history"
]

# ------------------------------------------------------------
# SAFE SESSION STATE INIT
# ------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "input_features" not in st.session_state:
    st.session_state.input_features = None

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:

    st.markdown("## 🏥 Project Overview")

    st.markdown("""
    <div style='
        background:#334155;
        padding:15px;
        border-radius:12px;
        color:#60a5fa;
        font-weight:500;
    '>
    Decision Tree Classifier <br><br>
    Hyperparameter Tuned (RandomizedSearchCV) <br><br>
    Accuracy: ~75% <br><br>
    Features: 14
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧠 Model Type")
    st.markdown("<h2 style='color:white;'>Decision Tree</h2>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Performance Monitor")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "75%")
    col2.metric("Features", "14")

    st.progress(0.75)

    st.markdown("---")

    # SAFE LIVE RISK SCORE
    st.markdown("### 🌡️ Live Risk Score")

    if (
        st.session_state.probabilities is not None
        and len(st.session_state.probabilities) > 1
    ):

        if 1 in model.classes_:
            pos_index = list(model.classes_).index(1)
        else:
            pos_index = 0

        risk_percent = round(
            st.session_state.probabilities[pos_index] * 100, 2
        )

        st.progress(risk_percent / 100)
        st.caption(f"Risk Probability: {risk_percent}%")

    else:
        st.progress(0.0)
        st.caption("Run prediction to calculate risk")

    st.markdown("---")

    with st.expander("🧠 About Decision Tree Model"):
        st.write("""
• Non-linear classification  
• Tuned using RandomizedSearchCV  
• Handles mixed data types  
• Interpretable splits  
• Suitable for medical prediction  
""")

    st.markdown("---")

    st.markdown("""
    <div style='
        background: linear-gradient(90deg,#1e40af,#2563eb);
        padding:10px;
        border-radius:10px;
        text-align:center;
        color:white;
        font-weight:600;
    '>
    🚀 Enterprise ML Health System
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("# 🏥 Health Risk Intelligence Platform")
st.caption("AI-Based Lifestyle Disease Risk Assessment System")

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Analytics", "Model Insights"])

# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        daily_steps = st.number_input("Daily Steps", 0, 30000, 8000)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)

    with col2:
        water_intake = st.number_input("Water Intake (L)", 0.0, 10.0, 2.0)
        calories = st.number_input("Calories Consumed", 1000, 6000, 2200)
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        resting_hr = st.number_input("Resting Heart Rate", 40, 150, 75)

    with col3:
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        family_history = st.selectbox("Family History", ["No", "Yes"])

    gender = 1 if gender == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    alcohol = 1 if alcohol == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0

    center_btn = st.columns([1,2,1])
    with center_btn[1]:
        predict = st.button("🔍 Run AI Prediction", use_container_width=True)

    if predict:

        features = np.array([[age, gender, bmi, daily_steps,
                              sleep_hours, water_intake, calories,
                              smoker, alcohol, resting_hr,
                              systolic_bp, diastolic_bp,
                              cholesterol, family_history]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        st.session_state.prediction = prediction
        st.session_state.probabilities = probabilities
        st.session_state.input_features = features

    if st.session_state.prediction is not None:

        probs = st.session_state.probabilities

        if 1 in model.classes_:
            pos_index = list(model.classes_).index(1)
        else:
            pos_index = 0

        risk_probability = probs[pos_index] * 100
        confidence = round(np.max(probs) * 100, 2)

        if st.session_state.prediction == 1:
            label = "⚠️ Health Risk Detected"
            color = "#7f1d1d"
        else:
            label = "✅ No Significant Health Risk"
            color = "#065f46"

        st.markdown(
            f"""
            <div class="result-box" style="background:{color};">
                <h2>{label}</h2>
                <p>Confidence: {confidence}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🎯 Risk Scoring Meter")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability,
            title={'text': "Risk Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "#065f46"},
                    {'range': [40, 70], 'color': "#92400e"},
                    {'range': [70, 100], 'color': "#7f1d1d"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

#============================================================
# TAB 2 - ANALYTICS
# ============================================================
with tab2:

    if st.session_state.input_features is None:
        st.info("Run prediction first to view analytics.")
    else:

        st.markdown("### 📊 Probability Distribution")

        probs = st.session_state.probabilities

        fig_bar = px.bar(
            x=["No Risk", "Risk"],
            y=probs,
            color=["No Risk", "Risk"],
            color_discrete_map={"No Risk":"#10b981","Risk":"#ef4444"}
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # INTERACTIVE PDP
        st.markdown("### 📈 Interactive Partial Dependence")

        selected_feature = st.selectbox(
            "Select Feature",
            feature_names
        )

        base_input = st.session_state.input_features.copy()
        feature_index = feature_names.index(selected_feature)

        if selected_feature in ["gender","smoker","alcohol","family_history"]:
            values = [0,1]
        else:
            base_val = base_input[0][feature_index]
            values = np.linspace(base_val*0.8, base_val*1.2, 30)

        risk_probs = []

        for val in values:
            temp = base_input.copy()
            temp[0][feature_index] = val
            prob = model.predict_proba(temp)[0][1]
            risk_probs.append(prob * 100)

        fig_line = px.line(
            x=values,
            y=risk_probs,
            markers=True,
            labels={"x":selected_feature,"y":"Risk Probability (%)"},
            template="plotly_dark"
        )

        st.plotly_chart(fig_line, use_container_width=True)

# ============================================================
# TAB 3 - MODEL INSIGHTS
# ============================================================
with tab3:

    st.markdown("### 🌳 Decision Tree Visualization")

    fig_tree, ax = plt.subplots(figsize=(18,10))
    plot_tree(model,
              filled=True,
              feature_names=feature_names,
              class_names=["No Risk","Risk"])
    st.pyplot(fig_tree)

    st.markdown("### 📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### ⚙ Best Hyperparameters")

    params_df = pd.DataFrame(model.get_params().items(),
                             columns=["Parameter","Value"])

    st.dataframe(params_df, use_container_width=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2026 Akshit Gajera | Decision Tree Health Risk Intelligence")