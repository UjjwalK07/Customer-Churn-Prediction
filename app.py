import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import pickle

# Load the model
model = load_model("model.keras", compile = False)

# Load encoders and scaler
with open('encoder_gender.pkl', 'rb') as f:
    encoder_gender = pickle.load(f)

with open('encoder_geo.pkl', 'rb') as f:
    encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App title
st.title("Customer Churn Prediction")

# User Inputs
st.header("Enter Customer Details")

geography = st.selectbox("Geography", encoder_geo.categories_[0])
gender = st.selectbox("Gender", encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, format="%.2f")
credit_score = st.number_input("Credit Score", min_value=0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Predict on button click
if st.button("Predict"):

    # Transform gender
    gender_encoded = encoder_gender.transform([gender])[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(['Geography']))

    # Combine
    full_input = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(full_input)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # Output 
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{prediction_proba:.2%}**")

    if prediction_proba > 0.5:
        st.error("The customer is **likely to churn**.")
    else:
        st.success("The customer is **not likely to churn**.")

    # View input inside button block to avoid error
    with st.expander("View Input Data"):
        st.dataframe(input_data)
