# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y-i7CCHxL4Sh7rbjrqvzz3GwBQJtQ7yE
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import numpy as np
from datetime import datetime

# Load the car price model from the pickle file
with open("car_price_model.pkl", "rb") as f:
    car_price_model = pickle.load(f)

# Set the title of the Streamlit app
st.title("Car Price Prediction")

# Create input fields for Year, Make, Model, Odometer, and Condition
year = st.number_input(
    "Year", min_value=1900, max_value=datetime.now().year, value=2020
)
odometer = st.number_input("Odometer (miles)", min_value=0, value=10000)
make = st.text_input("Make (e.g., Toyota, Ford)")
model = st.text_input("Model (e.g., Corolla, Mustang)")
condition = st.selectbox(
    "Condition", ["Excellent", "Good", "Fair", "Poor"]
)

# Encode the condition to a numerical value (example mapping)
condition_dict = {
    "Excellent": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1
}

# Output a prediction based on the inputs
if st.button("Predict"):
    # Calculate vehicle age
    current_year = datetime.now().year
    age = current_year - year

    # Encode the condition
    encoded_condition = condition_dict[condition]

    # Create a feature array for the model
    features = np.array(
        [
            [
                year,
                odometer,
                age,
                encoded_condition
            ]
        ]
    )

    # Make a prediction using the car price model
    predicted_price = car_price_model.predict(features)

    # Display the predicted price
    st.write(
        f"The predicted price for a {year} {make} {model} in {condition} condition with {odometer} miles is ${predicted_price[0]:,.2f}."
    )



