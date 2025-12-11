# pages/feature_engineering.py

import streamlit as st
import pandas as pd
import joblib

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="Feature Engineering", page_icon="🛠️")

st.title("🛠️ Feature Engineering Overview")
st.write("""
This page summarizes the feature engineering steps performed on the stroke dataset. 
We use the **original dataset** and the **trained model** to illustrate the transformations.
""")

# -------------------- Load Data & Model -------------------- #
@st.cache_data
def load_data():
    df = pd.read_parquet(r"Data/feature_engineered_stroke_data.parquet")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("Model/final_stroke_prediction_model.pkl")
    return model

df = load_data()
pipeline = load_model()

# -------------------- Tabs -------------------- #

tab1, tab2, = st.tabs([
    " 💫Feature Engineering", 
    "🪈 Pipeline & Model Overview"
])


# -------------------- Section 1: Original Features -------------------- #
with tab1:

    st.subheader("Original Features")
    st.write("The dataset initially contains the following columns:")
    st.write(df.head(5))
    st.write("Columns:", df.columns.tolist())
    st.write("The stroke column is our target variable.")
    st.write("Target distribution:")
    st.bar_chart(df['stroke'].value_counts())

# -------------------- Section 2: Numeric Feature Transformations -------------------- #
    st.subheader("Numeric Feature Engineering")
    st.write("""
    We applied the following transformations to numeric features:
    - **Median Imputation** for missing values.
    - Scaling is handled internally in the pipeline (if any scaling applied).
    - SMOTE oversampling to balance the target class for training.
    """)

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    st.write("Numeric features:", numeric_cols)

    # Optional: summary statistics
    st.write("Summary statistics of numeric features:")
    st.dataframe(df[numeric_cols].describe())

# -------------------- Section 3: Categorical Feature Transformations -------------------- #
    st.subheader("Categorical Feature Engineering")
    st.write("""
    We applied the following transformations to categorical features:
    - **Most frequent imputation** for missing values.
    For example:
        - Missing entries in `gender` were replaced with the most frequent category (`Female`).  
        - Missing entries in `smoking_status` were replaced with the most frequent category (`never smoked`).  
    - **One-hot encoding** with `drop='first'` to avoid dummy variable trap.
    """)

    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    st.write("Categorical features:", categorical_cols)

    # Optional: display counts
    st.write("Category counts for each feature:")
    for col in categorical_cols:
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

# -------------------- Section 4: Pipeline Overview -------------------- #
    st.subheader("Pipeline Steps")
    st.write("Our trained model pipeline includes:")
    st.write("""
    1. **Preprocessing** (`ColumnTransformer`)  
    - Numeric: median imputer  
    - Categorical: most frequent imputer + one-hot encoding  

    2. **SMOTE**: synthetic oversampling for the minority class  

    3. **Classifier**: RandomForestClassifier (balanced, max_depth=10, 200 trees)
    """)

    st.write("Pipeline structure from the saved model:")
    st.write(pipeline)

    # -------------------- Section 5: Why These Transformations -------------------- #
    st.subheader("Why These Steps Were Applied")
    st.write("""
    - **Imputation** ensures the model can handle missing values without errors.  
    - **One-hot encoding** converts categorical data into numeric format for the model.
    - **SMOTE** balances the stroke target class because strokes are rare events in the dataset.  
    - **RandomForest** is robust to feature scaling, can handle categorical features via preprocessing, and balances feature importance across all predictors.  
    """)

    # -------------------- Section 6: Optional Visualizations -------------------- #
    st.subheader("Feature Distributions")
    st.write("Visualizing numeric features to show data ranges and distributions:")
    st.dataframe(df[numeric_cols].describe())

    st.write("Categorical feature distributions:")
    for col in categorical_cols:
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

# -------------------- Tab 2 -------------------- #
import joblib
import streamlit as st

with tab2:

    st.header("Pipeline")

    file_path = 'Model/final_stroke_prediction_model.pkl'

    Pipeline = joblib.load(file_path)
     
    # Display the pipeline code
    pipeline_code = repr(Pipeline)  # convert pipeline to code-like string
    with st.expander("Use the scroller at the bottom ", expanded=True):
            st.code(pipeline_code, language='python')

    st.header("Model Overview")

    st.markdown("""
    **Model type**: Random Forest Classifier  
**Pipeline: Preprocessing** → SMOTE → RandomForest  
**Hyperparameter search**: GridSearchCV (3-fold CV)  
**Best parameters identified:** 

- `n_estimators` = 200
- `max_depth` = 20
- `min_samples_split` = 2
- `min_samples_leaf` = 1
- `class_weight` = balanced
""")








