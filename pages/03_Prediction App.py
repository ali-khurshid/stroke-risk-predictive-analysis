import streamlit as st
import pandas as pd
from helpers import load_model, NUM_RANGES, CAT_RANGES

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="Stroke Prediction Calculator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Stroke Risk Prediction")
st.write("Adjust the patient information below to estimate the stroke likelihood.")

# -------------------- Load Model -------------------- #
pipeline = load_model()
preprocessor = pipeline.named_steps['preprocess']
clf = pipeline.named_steps['clf']

# -------------------- User Inputs in Columns -------------------- #
st.header("🏥 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=NUM_RANGES["age"][0],
        max_value=NUM_RANGES["age"][1],
        value=(NUM_RANGES["age"][0] + NUM_RANGES["age"][1]) // 2
    )
    bmi = st.slider(
        "BMI",
        min_value=NUM_RANGES["bmi"][0],
        max_value=NUM_RANGES["bmi"][1],
        value=(NUM_RANGES["bmi"][0] + NUM_RANGES["bmi"][1]) // 2
    )
    gender = st.selectbox("Gender", CAT_RANGES["gender"])
    ever_married = st.selectbox("Ever Married", CAT_RANGES["ever_married"])
    work_type = st.selectbox("Work Type", CAT_RANGES["work_type"])

with col2:
    avg_glucose_level = st.slider(
        "Average Glucose Level",
        min_value=NUM_RANGES["avg_glucose_level"][0],
        max_value=NUM_RANGES["avg_glucose_level"][1],
        value=(NUM_RANGES["avg_glucose_level"][0] + NUM_RANGES["avg_glucose_level"][1]) // 2
    )
    hypertension = st.radio(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    heart_disease = st.radio(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    residence_type = st.selectbox("Residence Type", CAT_RANGES["residence_type"])
    smoking_status = st.selectbox("Smoking Status", CAT_RANGES["smoking_status"])

# -------------------- Prepare Input DataFrame -------------------- #
input_df = pd.DataFrame({
    "id": [1],  # dummy ID
    "age": [age],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender": [gender],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "residence_type": [residence_type],
    "smoking_status": [smoking_status],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease]
})

# -------------------- Prediction -------------------- #
if st.button("Predict Stroke Risk"):
    # Transform user input
    X_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = clf.predict(X_transformed)[0]
    probability = clf.predict_proba(X_transformed)[0][1]

    # -------------------- Results -------------------- #
    col1, col2 = st.columns([1, 2])

    with col1:
        if probability >=0.25:
            risk_label = "🔴 HIGH Stroke Risk"
        elif probability < 0.25 and probability >= 0.10:
            risk_label = "🟡 Moderate Stroke Risk"
        else:
            risk_label = "🟢 Low Stroke Risk"
        st.metric(label="Predicted Risk", value=risk_label)

    with col2:
        st.progress(probability)
        st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

    # -------------------- Recommendations -------------------- #
    with st.expander("💡 Recommendations / Advice"):
        if probability >= 0.25:
            st.warning("🚨 High Risk — Medical consultation recommended.")
        elif probability < 0.25 and probability >= 0.10:
            st.info("⚠️ Moderate risk — Consider lifestyle improvements and monitoring.")
        else:
            st.success(" ✅ Low risk — Maintain healthy habits.")

# -------------------- Show Input Data -------------------- #
st.write("### 🤒 Patient Input Data")
st.dataframe(input_df.drop(columns=["id"]))


