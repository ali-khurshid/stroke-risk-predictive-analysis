import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import joblib

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="Data EDA", page_icon="📊")

st.title("📊 Exploratory Data Analysis")
st.write("Explore patterns, distributions, and relationships within the dataset used for model training.")

# -------------------- Load Data -------------------- #
@st.cache_data
def load_data():
    df = pd.read_parquet(r"Data/feature_engineered_stroke_data.parquet")
    return df

df = load_data()

# Proper column detection
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if "id" in numeric_cols:  # Exclude ID; it is meaningless for EDA
    numeric_cols.remove("id")

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# -------------------- Tabs -------------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Raw Data", 
    "📈 Numerical Features", 
    "📊 Categorical Features",
    "⭐ Feature Importance", 
    "🔥 Correlation Heatmap"
])

# -------------------- Tab 1: Raw Data -------------------- #
with tab1:
    st.subheader("Dataset Preview")
    st.write("Showing the first 35 rows of the dataset.")
    st.dataframe(df.head(35))

    st.write("### Dataset Summary")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Columns:**", df.columns.tolist())

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    st.write("### Target Class Distribution (Stroke)")
    st.bar_chart(df['stroke'].value_counts())

# -------------------- Tab 2: Numerical -------------------- #
with tab2:
    st.subheader("Distribution of Numerical Features")

    selected_num = st.selectbox("Select a numerical column", numeric_cols)

    st.write(f"### Histogram — {selected_num}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)


# -------------------- Tab 3: Categorical -------------------- #
with tab3:
    st.subheader("Distribution of Categorical Features")

    selected_cat = st.selectbox("Select a categorical column", cat_cols)

    st.write(f"### Count Plot — {selected_cat}")
    fig, ax = plt.subplots()
    sns.countplot(x=df[selected_cat], order=df[selected_cat].value_counts().index, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)


# -------------------- Tab 4: Feature Importance -------------------- #
# -------------------- Load data -------------------- #
@st.cache_data
def load_data():
    return pd.read_parquet(r"Data/feature_engineered_stroke_data.parquet")

df = load_data()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
all_features = [c for c in df.columns if c not in ['id', 'stroke']]

# -------------------- Tab 4: Feature Importance -------------------- #
with tab4:
    st.subheader("Feature Importance Visualization")
    st.write("The risk of having a stroke is not linearly related to any single feature...")

   

    # Header
    st.subheader("📊 Two Features vs Stroke")

    # Feature selection
    feature_x = st.selectbox("Select first feature (X-axis):", all_features)
    feature_y = st.selectbox("Select second feature (Y-axis):", all_features, index=1)

    # Determine feature types
    x_numeric = feature_x in numeric_cols
    y_numeric = feature_y in numeric_cols

    # Plot logic
    if x_numeric and y_numeric:
        fig = px.scatter(df, x=feature_x, y=feature_y, color='stroke',
                        color_discrete_map={0: 'blue', 1: 'orange'},
                        title=f"{feature_x} vs {feature_y} grouped by Stroke",
                        opacity=0.7)
    elif x_numeric and not y_numeric:
        fig = px.box(df, x=feature_y, y=feature_x, color='stroke',
                    color_discrete_map={0: 'blue', 1: 'orange'},
                    title=f"{feature_x} vs {feature_y} grouped by Stroke")
    elif not x_numeric and y_numeric:
        fig = px.box(df, x=feature_x, y=feature_y, color='stroke',
                    color_discrete_map={0: 'blue', 1: 'orange'},
                    title=f"{feature_x} vs {feature_y} grouped by Stroke")
    else:
        df['combined'] = df[feature_x].astype(str) + " | " + df[feature_y].astype(str)
        ct = pd.crosstab(df['combined'], df['stroke']).reset_index()
        ct_melted = ct.melt(id_vars='combined', value_vars=[0, 1], var_name='stroke', value_name='count')
        fig = px.bar(ct_melted, x='combined', y='count', color='stroke',
                    color_discrete_map={0: 'blue', 1: 'orange'},
                    title=f"{feature_x} + {feature_y} grouped by Stroke")
        fig.update_layout(xaxis_tickangle=45)

    # Show plot
    st.plotly_chart(fig, use_container_width=True)


# -------------------- Tab 5: Correlation Heatmap -------------------- #
with tab5:
    st.subheader("Correlation Between Numerical Features")

    numeric_cols_no_target = [col for col in numeric_cols if col not in ['stroke','id']]
    corr = df[numeric_cols_no_target + ["stroke"]].corr()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("Correlations closer to **1** or **-1** show stronger relationships.")

    st.subheader("🗣️Discussion")
    st.write("""
    From the correlation heatmap, we can observe that **Age** and **Average Glucose Level**
    show the strongest positive relationship with stroke occurrence, meaning that as these
    values increase, the likelihood of stroke also tends to increase.

    On the other hand, features such as **BMI** show very weak or near-zero correlation,
    suggesting that, based on this dataset alone, BMI has little linear relationship with
    stroke risk.
    """)

    st.subheader("📝Notes")
    st.write("""   
    It is important to note that correlation does not imply direct causation, and weak
    correlation does not mean the feature is unimportant — some machine learning models,
    such as Random Forests, can capture **non-linear** patterns that correlation cannot reveal.

    It is also important to note that grouped features may have a combined correlation to stroke occurence 
    and that individual features may not seem correlated on their own.
    """)