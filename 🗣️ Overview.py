import streamlit as st
import joblib
import pandas as pd
from helpers import preprocess_input,load_model


# Main entry page for streamlit app


st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load model ---
@st.cache_resource
def load_model():
    model = joblib.load('Model/final_stroke_prediction_model.pkl')
    return model

model = load_model()

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_parquet('Data/feature_engineered_stroke_data.parquet')
    df = preprocess_input(df)  # apply your rounding/casting here
    return df

df = load_data()


st.title("""
🧠 Stroke Risk Prediction Application.
Navigate using the sidebar to explore different sections of the app.
 - **Developed by:** Ali Khurshid.
    - **Data Source:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
    - **Home** - Project Overview and insights
    - **Data EDA** - Exploratory Data Analysis
    - **Feature Engineering** - Data preprocessing and feature engineering steps
    - **Stroke Prediction Calculator** - Predict your risk of stroke
    - ** Model Performance** - Evaluation metrics and performance analysis
""")

# Home page for streamlit app

st.title("🏠" "Project Overview and Insights")
st.write("""
This application uses machine learning to predict the risk of stroke in patients based on key medica and lifestyle factors. 
""")





### 🔧 Technology Stack
st.header("🔧Technology Stack")
st.write("""
The application is built using the following technologies:
- **Python**: Programming language used for data analysis and model development.
- **Streamlit**: Framework for building interactive web applications.
- **Pandas & NumPy**: Libraries for data manipulation and numerical computations.
- **SMOTE**: Technique for handling imbalanced datasets via oversampling.
- **Scikit-learn**: Machine learning library for model training and evaluation.
         Logistic Regression and Random Forest algorithms were used.
- **Imbalanced-learn**: Library for handling imbalanced datasets.
- **Matplotlib & Seaborn**: Libraries for data visualization.
""")

### 📌 Business Problem
st.header("📌 Business Problem")
st.write("""
Stroke is a leading cause of death and disability worldwide. Early prediction of stroke risk can help in timely intervention and prevention strategies. 
This application aims to provide a user-friendly tool for individuals and healthcare professionals to assess stroke risk based on various health parameters.
""")

st.write("""
         ⚠ **Note**: This model prioritizes recall to reduce missed stroke cases, meaning some false positives are possible.
""")