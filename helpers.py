import joblib
import streamlit as st

@st.cache_resource  # better for models!
def load_model():
    model = joblib.load("Model/final_stroke_prediction_model.pkl")
    return model