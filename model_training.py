import pandas as pd

startup = pd.read_csv(r"C:\Users\pavil\OneDrive\Desktop\50_Startups (1).csv")
startup = pd.get_dummies(startup, columns=['State'], drop_first=True)  # as state is a categprical variable and the system dont read tha 
print(startup.isnull().sum())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Features and target
X = startup.drop('Profit', axis=1)
y = startup['Profit']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "profit_model.pkl")
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("profit_model.pkl")

st.title("Startup's Profit Prediction App")

# User input
rd = st.number_input("R&D Spend", 0)
admin = st.number_input("Administration Spend", 0)
marketing = st.number_input("Marketing Spend", 0)
state = st.selectbox("State", ["New York", "California", "Florida"])  # adjust based on your dataset

# Convert state to one-hot manually
state_dict = {"New York": [0,0], "California": [1,0], "Florida": [0,1]}  # example if 3 states
state_encoded = state_dict[state]

# Combine features
features = [rd, admin, marketing] + state_encoded
features = np.array([features])  # 2D array

# Predict
if st.button("Predict Profit"):
    prediction = model.predict(features)
    st.success(f"Predicted Profit: ${prediction[0]:.2f}")

    st.balloons()
    st.info("Keep investing wisely for better profits!")