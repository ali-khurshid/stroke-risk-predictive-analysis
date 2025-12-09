import streamlit as st
import joblib
import pandas as pd

# Main entry page for streamlit app


st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

#load and prepare data
@st.cache_resource # Cache the model loading function to improve performance
def load_data():
    model = joblib.load('Model/final_stroke_prediction_model.pkl') # load the model
    return model

df = load_data()

st.title("""
🧠 Stroke Risk Prediction Application.
Navigate using the sidebar to explore different sections of the app.
 - **Developed by:** Ali Khurshid.
    - **Data Source:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
    - **Home** - Project Overview and insights
    - **EDA** - Exploratory Data Analysis
    - **Stroke Prediction Calculator** - Predict your risk of stroke
    - **Model Details** - Information about the predictive model
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