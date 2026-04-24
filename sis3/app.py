# app.py
import streamlit as st
import requests

st.title("Iris Prediction App")

# Input fields
features = []
for i in range(4):
    value = st.number_input(f"Feature {i+1}")
    features.append(value)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": features}
    )

    result = response.json()
    st.write("Prediction:", result["prediction"])