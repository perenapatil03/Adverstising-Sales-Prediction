import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("model.pkl")


# Page title
st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📊",
    layout="centered"
)


# Main heading
st.title("📊 Advertising Sales Prediction")
st.write("Enter advertising expenditure to predict product sales.")


# Input section
st.subheader("Enter Advertising Details")


tv = st.number_input(
    "TV Advertising Expenditure",
    min_value=0.0,
    value=100.0,
    step=1.0
)

radio = st.number_input(
    "Radio Advertising Expenditure",
    min_value=0.0,
    value=20.0,
    step=1.0
)

newspaper = st.number_input(
    "Newspaper Advertising Expenditure",
    min_value=0.0,
    value=20.0,
    step=1.0
)


# Prediction button
if st.button("🔮 Predict Sales"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(f"Predicted Sales: {prediction:.2f}")


    # Input Summary
    st.subheader("📋 Input Summary")

    st.write(f"**TV Advertising:** {tv:.2f}")
    st.write(f"**Radio Advertising:** {radio:.2f}")
    st.write(f"**Newspaper Advertising:** {newspaper:.2f}")


# Model information
st.divider()

st.subheader("📈 Model Information")

st.write("Model: Linear Regression")
st.write("Target: Sales")
st.write("Features: TV, Radio, Newspaper")
st.write("R² Score: 0.8332")