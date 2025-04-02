import streamlit as st
import pickle
import numpy as np

# Title
st.title("Future Sales Prediction App")

# User selection: Choose model type
model_choice = st.radio(
    "Choose a model for prediction:",
    ("Sales Prediction with Newspaper", "Sales Prediction without Newspaper")
)

# Load the selected model
if model_choice == "Sales Prediction with Newspaper":
    with open("random_forest_sales_model_with_newspaper.pkl", "rb") as f:
        model = pickle.load(f)
    input_features = ["TV Budget ($)", "Radio Budget ($)", "Newspaper Budget ($)"]
else:
    with open("random_forest_sales_model_without_newspaper.pkl", "rb") as f:
        model = pickle.load(f)
    input_features = ["TV Budget ($)", "Radio Budget ($)"]

# Input fields based on model type
st.write("Enter advertising budgets to predict future sales.")

inputs = []
for feature in input_features:
    value = st.number_input(feature, min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    inputs.append(value)

# Predict Button
if st.button("Predict Sales"):
    # Convert inputs to NumPy array for prediction
    input_data = np.array([inputs]).reshape(1, -1)

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Show result
    st.success(f"Predicted Sales: **{predicted_sales:.2f}** units")

st.write("Powered by Machine Learning")
