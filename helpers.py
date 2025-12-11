import joblib
import pandas as pd
import streamlit as st
import os

BASE_DIR = os.path.dirname(__file__)  # the folder where helpers.py lives
DATA_PATH = os.path.join(BASE_DIR, "Data", "feature_engineered_stroke_data.parquet")

# -------------------- Load Training Data -------------------- #
@st.cache_resource
def load_training_data():
    """
    Loads the processed stroke dataset.
    Returns a pandas DataFrame.
    """
    df = pd.read_parquet(DATA_PATH)
    return df

# -------------------- Numeric and Categorical Ranges -------------------- #
@st.cache_resource
def get_ranges():
    """
    Returns numeric and categorical ranges from the training data.
    """
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

    return NUM_RANGES, CAT_RANGES

# -------------------- Load Model -------------------- #
@st.cache_resource  # caches the model for faster Streamlit performance
def load_model():
    """
    Loads the pre-trained stroke prediction pipeline.
    Returns a scikit-learn pipeline object.
    """
    model = joblib.load("Model/final_stroke_prediction_model.pkl")
    return model

# -------------------- Preprocess Input -------------------- #


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for user input before prediction.
    Rounds numeric values and ensures correct data types.
    """
    df['age'] = df['age'].round(0).astype(int)
    df['bmi'] = df['bmi'].round(0).astype(int)
    df['avg_glucose_level'] = df['avg_glucose_level'].round(0).astype(int)
    return df


