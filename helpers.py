import joblib
import streamlit as st
import pandas as pd


@st.cache_resource  # better for models
def load_model():
    model = joblib.load("Model/final_stroke_prediction_model.pkl")
    return model


@st.cache_resource
def load_training_data():
    return pd.read_parquet(r"Data/processed_stroke_data.parquet")

train = load_training_data()

NUM_RANGES = {
     "age": (train["age"].min(), train["age"].max()),
    "avg_glucose_level": (train["avg_glucose_level"].min(), train["avg_glucose_level"].max()),
    "bmi": (train["bmi"].min(), train["bmi"].max())
}

CAT_RANGES = {
    "gender": train["gender"].unique().tolist(),
    "ever_married": train["ever_married"].unique().tolist(),
    "work_type": train["work_type"].unique().tolist(),
    "residence_type": train["residence_type"].unique().tolist(),
    "smoking_status": train["smoking_status"].unique().tolist()
}
