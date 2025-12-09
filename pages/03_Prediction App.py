import streamlit as st
import pandas as pd
from helpers import load_model, NUM_RANGES, CAT_RANGES

st.title( " 🧠 Stroke Risk Prediction")
st.write("Adjust the patient information below to estimate the stroke likelihood.")

#load and prepare data
model = load_model()

# -------------------- User Inputs -------------------- #

# Numerical sliders using NUM_RANGES
age = st.slider(
    "Age",
    min_value=NUM_RANGES["age"][0],
    max_value=NUM_RANGES["age"][1],
    value=(NUM_RANGES["age"][0] + NUM_RANGES["age"][1]) / 2
)

avg_glucose_level = st.slider(
    "Average Glucose Level",
    min_value=NUM_RANGES["avg_glucose_level"][0],
    max_value=NUM_RANGES["avg_glucose_level"][1],
    value=(NUM_RANGES["avg_glucose_level"][0] + NUM_RANGES["avg_glucose_level"][1]) / 2
)

bmi = st.slider(
    "BMI",
    min_value=NUM_RANGES["bmi"][0],
    max_value=NUM_RANGES["bmi"][1],
    value=(NUM_RANGES["bmi"][0] + NUM_RANGES["bmi"][1]) / 2
)

# Categorical inputs using CAT_RANGES
gender = st.selectbox("Gender", CAT_RANGES["gender"])
ever_married = st.selectbox("Ever Married", CAT_RANGES["ever_married"])
work_type = st.selectbox("Work Type", CAT_RANGES["work_type"])
residence_type = st.selectbox("Residence Type", CAT_RANGES["residence_type"])
smoking_status = st.selectbox("Smoking Status", CAT_RANGES["smoking_status"])

# Binary inputs
hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# -------------------- Prepare Input DataFrame -------------------- #
input_df = pd.DataFrame({
    "id": 1,
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

# -------------------- PREDICTION -------------------- #
if st.button("Predict Stroke Risk"):
    # Load pipeline without SMOTE
    pipeline = load_model() 
    
    # Separate preprocessor and classifier for prediction
    preprocessor = pipeline.named_steps['preprocess']
    clf = pipeline.named_steps['clf']

    # Transform user input
    X_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = clf.predict(X_transformed)[0]
    probability = clf.predict_proba(X_transformed)[0][1]

    # Display results
    threshold = 0.25
    if probability >= threshold:
        st.error(f"🔴 HIGH Stroke Risk")
    else:
        st.success(f"🟢 Low Stroke Risk")

    st.write(f"Prediction Confidence: {round(probability*100, 2)}%")

    st.write("""
    **Note**: This prediction is based on a machine learning model and should not replace professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment.
    """)

    st.write("""
    ⚠ **Disclaimer**: This model prioritizes recall to minimize missed stroke cases, which may lead to some false positives.
    """)

    st.write("### 🤒 Patient Input Data")
    st.dataframe(input_df.drop(columns=["id"]))

