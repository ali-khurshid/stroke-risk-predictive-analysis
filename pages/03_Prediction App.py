import streamlit as st
import pandas as pd
from helpers import load_model, preprocess_input, get_ranges

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="Stroke Prediction Calculator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Stroke Risk Prediction")
st.write("Adjust the patient information below to estimate the stroke likelihood.")

# -------------------- Load Model and Ranges -------------------- #
pipeline = load_model()
preprocessor = pipeline.named_steps['preprocess']
clf = pipeline.named_steps['clf']

NUM_RANGES, CAT_RANGES = get_ranges()

# -------------------- User Inputs with session_state -------------------- #
st.header("🏥 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
    "Age",
    min_value=int(NUM_RANGES["age"][0]),
    max_value=int(NUM_RANGES["age"][1]),
    value=int(st.session_state.get("age", (NUM_RANGES["age"][0] + NUM_RANGES["age"][1]) // 2)),
    step=1,
    key="age"
)
    bmi = st.slider(
    "BMI",
    min_value=int(NUM_RANGES["bmi"][0]),
    max_value=int(NUM_RANGES["bmi"][1]),
    value=int(st.session_state.get("bmi", (NUM_RANGES["bmi"][0] + NUM_RANGES["bmi"][1]) // 2)),
    step=1,
    key="bmi"
)
    gender = st.selectbox(
        "Gender",
        CAT_RANGES["gender"],
        index=st.session_state.get("gender_index", 0),
        key="gender"
    )
    ever_married = st.selectbox(
        "Ever Married",
        CAT_RANGES["ever_married"],
        index=st.session_state.get("ever_married_index", 0),
        key="ever_married"
    )
    work_type = st.selectbox(
        "Work Type",
        CAT_RANGES["work_type"],
        index=st.session_state.get("work_type_index", 0),
        key="work_type"
    )

with col2:
    avg_glucose_level = st.slider(
    "Average Glucose Level",
    min_value=int(NUM_RANGES["avg_glucose_level"][0]),
    max_value=int(NUM_RANGES["avg_glucose_level"][1]),
    value=int(st.session_state.get(
        "avg_glucose_level",
        (NUM_RANGES["avg_glucose_level"][0] + NUM_RANGES["avg_glucose_level"][1]) // 2
    )),
    step=1,
    key="avg_glucose_level"
)
    hypertension = st.radio(
        "Hypertension",
        [0, 1],
        index=st.session_state.get("hypertension", 0),
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="hypertension"
    )
    heart_disease = st.radio(
        "Heart Disease",
        [0, 1],
        index=st.session_state.get("heart_disease", 0),
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="heart_disease"
    )
    residence_type = st.selectbox(
        "Residence Type",
        CAT_RANGES["residence_type"],
        index=st.session_state.get("residence_type_index", 0),
        key="residence_type"
    )
    smoking_status = st.selectbox(
        "Smoking Status",
        CAT_RANGES["smoking_status"],
        index=st.session_state.get("smoking_status_index", 0),
        key="smoking_status"
    )

# -------------------- Prepare Input DataFrame -------------------- #
input_df = pd.DataFrame({
    "id": [1],
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
    # Preprocess input
    X_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = clf.predict(X_transformed)[0]
    probability = clf.predict_proba(X_transformed)[0][1]

    # -------------------- Results -------------------- #
    col1, col2 = st.columns([1, 2])

    with col1:
        if probability >= 0.25:
            risk_label = "🔴 HIGH Stroke Risk"
        elif probability >= 0.10:
            risk_label = "🟡 Moderate Stroke Risk"
        else:
            risk_label = "🟢 Low Stroke Risk"
        st.metric(label="Predicted Risk", value=risk_label)

    with col2:
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

    # -------------------- Recommendations -------------------- #
    with st.expander("💡 Recommendations / Advice"):
        if probability >= 0.25:
            st.warning("🚨 High Risk — Medical consultation recommended.")
        elif probability >= 0.10:
            st.info("⚠️ Moderate risk — Consider lifestyle improvements and monitoring.")
        else:
            st.success("✅ Low risk — Maintain healthy habits.")

   
# -------------------- Show Input Data -------------------- #
st.write("### 🤒 Patient Input Data")
st.dataframe(input_df)
