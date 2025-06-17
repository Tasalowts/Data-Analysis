import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Title
st.title("04 - Shit with Linear Regression")

# Load dataset
df2 = pd.read_csv("./data/Data.csv")

# Drop missing values
df2 = df2.dropna()

# Fix: Convert all values in object or mixed columns to string before encoding
for col in df2.select_dtypes(include=['object']).columns:
    df2[col] = df2[col].astype(str)  # Ensures uniform string type
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col])

# Show cleaned data
st.subheader("Preview of Cleaned Dataset")
st.dataframe(df2.head())

# Sidebar - Feature & Target selection
st.sidebar.header("Feature & Target Selection")
list_var = list(df2.columns)
features_selection = st.sidebar.multiselect("Select Features (X)", list_var)
target_selection = st.sidebar.selectbox("Select Target Variable (Y)", list_var)
selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Display",
    ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R2 Score"]
)

# Proceed if user selects inputs
if features_selection and target_selection:
    X = df2[features_selection]
    y = df2[target_selection]

    st.subheader("Selected Features and Target")
    st.write("**Features (X)**")
    st.dataframe(X.head())
    st.write("**Target (y)**")
    st.dataframe(y.head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE:** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE:** {mae:,.2f}")
    else:
        mae = None
    if "R2 Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R² Score:** {r2:.3f}")

    if mae is not None:
        st.success(f"My model performance has MAE = {np.round(mae, 2)}")

    # Plot
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

else:
    st.warning("Please select at least one feature and a target variable from the sidebar.")
