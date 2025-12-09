import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="Model Performance", page_icon="📈")
st.title("📈 Model Performance Overview")
st.write("""
This page summarizes the modeling journey and performance of the final trained model.
The workflow included experimenting with different algorithms, hyperparameter tuning, and threshold optimization.
""")
# -------------------- Modeling Journey -------------------- #
st.subheader("🔹 Modeling Investigation")
st.write("""
1. **Logistic Regression:**  
   - Started with Logistic Regression as a baseline model.
   - Performance was limited due to non-linear relationships and imbalanced target.

2. **Random Forest:**  
   - Switched to Random Forest to capture non-linear interactions and improve classification.
   - Balanced class weights to account for imbalanced stroke occurrence.

3. **SMOTE Oversampling:**  
   - Applied SMOTE to generate synthetic minority class samples for better balance.

4. **Hyperparameter Tuning:**  
   - Used `GridSearchCV` to find the best combination of:
     - `clf__n_estimators`
     - `clf__max_depth`
     - `clf__min_samples_split`
     - `clf__min_samples_leaf`
     - `clf__class_weight`
   - Optimized for F1 score to balance precision and recall.

5. **Threshold Adjustment:**  
   - Explored different classification thresholds to improve sensitivity for the minority class (stroke=1).

6. **Final Model:**  
   - Random Forest with optimal parameters, SMOTE preprocessing, and tuned threshold.
""")

# -------------------- Load Model -------------------- #
@st.cache_resource
def load_model():
    return joblib.load("Model/final_stroke_prediction_model.pkl")


model = load_model()

# -------------------- Load Data -------------------- #
@st.cache_data
def load_data():
    return pd.read_parquet("Data/feature_engineered_stroke_data.parquet")

df = load_data()

# Separate features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# -------------------- Predictions -------------------- #
# Use the pipeline directly; let it handle preprocessing
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# -------------------- Metrics -------------------- #
st.subheader("✅ Accuracy and ROC-AUC")
accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**ROC-AUC:** {roc_auc:.4f}")

# -------------------- Optional: Display prediction distribution -------------------- #
st.subheader("Prediction Distribution")
st.bar_chart(pd.Series(y_pred).value_counts())
st.write("Green = No Stroke (0), Red = Stroke (1)")



