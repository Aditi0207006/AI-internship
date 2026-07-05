import streamlit as st
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
st.title("House Rent Prediction")
import joblib 
import pandas as pd
model = joblib.load("house_rent_prediction.pkl")
BHK = st.number_input("Enter the number of BHK", min_value=1, max_value=10, step=1)
Size = st.number_input("Enter the size of the house in square feet", min_value=100, max_value=10000, step=10)
Area_Type = st.selectbox("Select the area type", ["Super built-up  Area", "Built-up  Area", "Plot  Area"])
City = st.selectbox("Select the city", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"])
Furnishing_Status = st.selectbox("Select the furnishing status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant_preferred = st.selectbox("Select the tenant preferred", ["Bachelors", "Family", "Any"])
Bathrooms = st.number_input("Enter the number of bathrooms", min_value=1, max_value=10, step=1)
Point_of_Contact = st.selectbox("Select the point of contact", ["Contact Owner", "Contact Agent"])
if st.button("Predict Rent"):
    input = pd.DataFrame({
        "BHK": [BHK],
        "Size": [Size],
        "Area_Type": [Area_Type],
        "City": [City],
        "Furnishing_Status": [Furnishing_Status],
        "tenant_preferred": [tenant_preferred],
        "Bathrooms": [Bathrooms],
        "Point_of_Contact": [Point_of_Contact]
    })
    prediction = model.predict(input)
    st.write(f"The predicted rent is: {prediction[0]}")