import streamlit as st
import pickle
import numpy as np
import requests
from datetime import datetime

# URL of the pickle file on GitHub (raw file link)
url = "https://github.com/Rajkaran713/car_price_prediction/blob/main/car_price_model.pkl"

# Download the file from the GitHub URL
response = requests.get(url)
if response.status_code == 200:
    car_price_model = pickle.loads(response.content)
else:
    st.error("Error: Could not download the model file from GitHub.")

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
