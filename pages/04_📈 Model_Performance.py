import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score
)

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
   - Applied SMOTE to generate synthetic minority class samples.

4. **Hyperparameter Tuning:**  
   - Used `GridSearchCV` to optimize:
     - `clf__n_estimators`
     - `clf__max_depth`
     - `clf__min_samples_split`
     - `clf__min_samples_leaf`
     - `clf__class_weight`
   - Tuned for F1 score due to class imbalance.

5. **Threshold Adjustment:**  
   - Explored different classification thresholds to improve recall for stroke prediction.

6. **Final Model:**  
   - A Random Forest classifier inside a preprocessing pipeline.
""")


# -------------------- Load Model -------------------- #
@st.cache_resource
def load_model():
    return joblib.load("Model/final_stroke_prediction_model.pkl")

model = load_model()


# -------------------- Load Evaluation Data -------------------- #
@st.cache_data
def load_data():
    return pd.read_parquet("Data/feature_engineered_stroke_data.parquet")

df = load_data()


# -------------------- Prepare Features & Target -------------------- #
X = df.drop(columns=['stroke'])
y = df['stroke']


# -------------------- Predictions -------------------- #
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]


# -------------------- Metrics -------------------- #
st.subheader("📊 Performance Metrics")

accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**ROC-AUC:** {roc_auc:.4f}")


# -------------------- Classification Report (Table) -------------------- #
st.subheader("📄 Classification Report")

report_dict = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

st.dataframe(report_df)


# -------------------- Confusion Matrix (Table) -------------------- #
st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(cm_df)

st.markdown(
    """
<hr>
<p style='font-size:12px; color:gray;'>
*Disclaimer: The metrics shown are for demonstration purposes. page is probably computing metrics on the full dataset, which includes training data → inflates numbers. 
They reflect performance on the provided evaluation dataset and may not generalize to new, unseen data.*
</p>
"""
)




