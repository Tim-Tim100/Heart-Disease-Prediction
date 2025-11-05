# streamlit_app.py
import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ---------- LOAD MODELS ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")

log_reg_path = os.path.join(models_dir, "logistic_regression_model.pkl")
rf_path = os.path.join(models_dir, "random_forest_model.pkl")

try:
    log_reg = joblib.load(log_reg_path)
except FileNotFoundError:
    st.error(f"❌ Could not find Logistic Regression model at {log_reg_path}")

try:
    rf = joblib.load(rf_path)
except FileNotFoundError:
    st.error(f"❌ Could not find Random Forest model at {rf_path}")


# ---------- FUNCTION: RESET ----------
def reset_inputs():
    st.session_state["Age"] = 100
    st.session_state["Sex"] = "Male"
    st.session_state["Chest_Pain_Type"] = 0
    st.session_state["Resting_BP"] = 120
    st.session_state["Cholesterol"] = 200
    st.session_state["Fasting_BS"] = 0
    st.session_state["Resting_ECG"] = 0
    st.session_state["Max_Heart_Rate"] = 150
    st.session_state["Exercise_Angina"] = 0
    st.session_state["ST_Depression"] = 1.0
    st.session_state["Slope"] = 0
    st.session_state["Major_Vessels"] = 0
    st.session_state["Thalassemia"] = 0


# ---------- USER INPUT ----------
st.title("💓 Heart Disease Risk Prediction")

st.markdown("Provide the patient's health details below:")

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.get("Age", 50), key="Age")
    Sex = st.selectbox("Sex", ["Male", "Female"], key="Sex")
    Chest_Pain_Type = st.selectbox("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)",
                                   [0, 1, 2, 3], key="Chest_Pain_Type")
    Resting_BP = st.number_input("Resting Blood Pressure (mmHg)", value=st.session_state.get("Resting_BP", 120), key="Resting_BP")
    Cholesterol = st.number_input("Cholesterol (mg/dl)", value=st.session_state.get("Cholesterol", 200), key="Cholesterol")
    Fasting_BS = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1], key="Fasting_BS")
    Resting_ECG = st.selectbox("Resting ECG Results (0: Normal, 1: ST-T Wave Abnormality, 2: LV Hypertrophy)",
                               [0, 1, 2], key="Resting_ECG")

with col2:
    Max_Heart_Rate = st.number_input("Max Heart Rate Achieved", value=st.session_state.get("Max_Heart_Rate", 150), key="Max_Heart_Rate")
    Exercise_Angina = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1], key="Exercise_Angina")
    ST_Depression = st.number_input("ST Depression Induced by Exercise", value=st.session_state.get("ST_Depression", 1.0), key="ST_Depression")
    Slope = st.selectbox("Slope of ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)", [0, 1, 2], key="Slope")
    Major_Vessels = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3,
                                    value=st.session_state.get("Major_Vessels", 0), key="Major_Vessels")
    Thalassemia = st.selectbox("Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect)",
                               [0, 1, 2], key="Thalassemia")

Sex = 1 if Sex == "Male" else 0

input_data = pd.DataFrame([[
    Age, Sex, Chest_Pain_Type, Resting_BP, Cholesterol,
    Fasting_BS, Resting_ECG, Max_Heart_Rate, Exercise_Angina,
    ST_Depression, Slope, Major_Vessels, Thalassemia
]], columns=[
    'Age', 'Sex', 'Chest_Pain_Type', 'Resting_BP', 'Cholesterol',
    'Fasting_BS', 'Resting_ECG', 'Max_Heart_Rate', 'Exercise_Angina',
    'ST_Depression', 'Slope', 'Major_Vessels', 'Thalassemia'
])


# ---------- PREDICTION ----------
colA, colB = st.columns([1, 1])

with colA:
    predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)
with colB:
    reset_button = st.button("🔄 Reset Inputs", use_container_width=True, on_click=reset_inputs)

if predict_button:
    try:
        lr_pred = log_reg.predict_proba(input_data)[0][1]
        rf_pred = rf.predict_proba(input_data)[0][1]
        combined_prob = (lr_pred + rf_pred) / 2  # Combine probability from both models

        # Determine category
        if combined_prob >= 0.7:
            result = "High Risk 💔"
            color = "red"
            tone = "🚨 **High Risk Detected:** Both models show a high probability of heart disease. Seek medical attention soon."
        elif combined_prob >= 0.4:
            result = "⚠️ Moderate Risk"
            color = "orange"
            tone = "⚠️ **Moderate Risk:** Some indicators suggest potential issues. Consider a medical check-up."
        else:
            result = "Low Risk ❤️"
            color = "green"
            tone = "✅ **Low Risk:** Your heart indicators appear normal. Keep up a healthy lifestyle."

        # Display results
        st.markdown(f"## 🧠 Prediction Result: **{result}**")
        st.write(tone)

        # ---------- RISK GAUGE ----------
        st.subheader("📊 Risk Level Visualization")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=combined_prob * 100,
            title={'text': "Predicted Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': combined_prob * 100
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
