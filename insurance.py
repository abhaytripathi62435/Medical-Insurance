import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('medical_insurance_with_income.csv')  # Ensure you have the updated CSV with income column

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Preprocess data
X = df.drop('charges', axis=1)
y = df['charges']

# Train a Random Forest model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Function to get user input
def get_user_input():
    # Collect user input for each feature
    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    income = st.number_input("Income (in ₹ Rupees)", min_value=1000, max_value=1000000, value=50000)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    # Prepare the input data for prediction
    sex_male = 1 if sex == 'male' else 0
    sex_female = 1 if sex == 'female' else 0
    smoker_true = 1 if smoker == 'yes' else 0
    smoker_false = 1 if smoker == 'no' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0
    region_northeast = 1 if region == 'northeast' else 0
    region_northwest = 1 if region == 'northwest' else 0

    user_input = {
        'age': [age],
        'children': [children],
        'income': [income],
        'sex_male': [sex_male],
        'sex_female': [sex_female],
        'smoker_true': [smoker_true],
        'smoker_false': [smoker_false],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest]
    }

    # Convert the input dictionary to a DataFrame
    user_df = pd.DataFrame(user_input)

    # Ensure that the columns of the user input match the training data
    missing_cols = set(X.columns) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0  # Add missing columns with default value 0
    
    # Reorder the columns to match the training set (important for consistency)
    user_df = user_df[X.columns]

    return user_df

# Streamlit user interface
st.title("Onelife ")
st.write("Medical Insurance Price Prediction")

# Get user input
user_df = get_user_input()

# Scale the user input data using the same scaler as before
user_scaled = scaler.transform(user_df)

# Predict the charges for the current year
current_year_charge = rf_model.predict(user_scaled)[0]

# Conversion rate from USD to INR (update this rate as necessary)
usd_to_inr_rate = 82

# Convert the predicted charge from USD to INR
current_year_charge_inr = current_year_charge * usd_to_inr_rate

st.write(f"Predicted charge for current year (2025): ₹{current_year_charge_inr:.2f}")

# Predict for future years by incrementing the 'age' feature
years_ahead = 5  # Predict for 5 years ahead
predicted_charges_inr = []

for i in range(1, years_ahead + 1):
    future_sample = user_df.copy()
    future_sample['age'] = future_sample['age'] + i  # Increment age for future years
    
    # Scale the future sample data
    future_scaled = scaler.transform(future_sample)
    
    # Predict the charge for the future year
    future_charge = rf_model.predict(future_scaled)[0]
    
    # Convert future charges from USD to INR
    future_charge_inr = future_charge * usd_to_inr_rate
    predicted_charges_inr.append(future_charge_inr)

# Plot the predicted charges over the years
years = [2025 + i for i in range(years_ahead + 1)]  # Assuming the current year is 2025
charges_inr = [current_year_charge_inr] + predicted_charges_inr

# Display the predicted charges for each year
for year, charge_inr in zip(years, charges_inr):
    st.write(f"Predicted charge for {year}: ₹{charge_inr:.2f}")

# Plot the charges for current year and next years
plt.figure(figsize=(8, 6))
plt.plot(years, charges_inr, marker='o', color='b', linestyle='-', label="Predicted Charges")
plt.xlabel('Year')
plt.ylabel('Predicted Insurance Charges (₹)')
plt.title('Predicted Medical Insurance Charges for Future Years (in INR)')
plt.grid(True)
plt.legend()
st.pyplot(plt)


