
import streamlit as st
import pickle
import pandas as pd

# Load model
with open("temperature_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌍 Temperature Prediction ML App")

st.write("Predict annual temperature using Machine Learning")

year = st.number_input(
    "Enter Year",
    min_value=1950,
    max_value=2100,
    step=1
)

if st.button("Predict Temperature"):

    input_data = pd.DataFrame([[year]], columns=["Year"])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Temperature: {prediction[0]:.2f} °C"
    )
