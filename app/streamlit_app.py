import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Diabetes Risk Predictor",
                   page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational demo — not medical advice.")

# Locate the model relative to this file or project root
HERE = Path(__file__).resolve().parent
candidate_paths = [
    HERE.parent / 'models' / 'best_pipeline.joblib',
    HERE / 'best_pipeline.joblib'
]
model_path = None
for p in candidate_paths:
    if p.exists():
        model_path = p
        break

if model_path is None:
    st.error(
        "Model file not found. Please run `python src/train.py --csv data/raw/diabetes.csv` first.")
    st.stop()

pipeline = joblib.load(model_path)

st.subheader("Input patient data")

col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1)
    Glucose = st.number_input("Glucose", min_value=0,
                              max_value=300, value=120, step=1)
    BloodPressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=70, step=1)
    SkinThickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    Insulin = st.number_input("Insulin", min_value=0,
                              max_value=900, value=80, step=1)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0,
                          value=28.5, step=0.1, format="%.1f")
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
    Age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

if st.button("Predict"):
    # wrap input in a DataFrame with correct column names
    X = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])

    prob = pipeline.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    st.metric("Estimated probability of diabetes", f"{prob*100:.1f}%")
    if pred == 1:
        st.warning(
            "Model prediction: **Diabetic (1)** — please consult a healthcare professional.")
    else:
        st.success(
            "Model prediction: **Non-diabetic (0)** — keep up a healthy lifestyle!")

st.markdown("---")
st.caption(
    "Built for IT41033 mini project. Model and thresholds are for learning purposes only.")
