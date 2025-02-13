import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("lasso_model.pkl")

st.title("Car Price Prediction System")
st.write("Write Features of the car you want to buy:")

year = st.number_input("Year", min_value=2000, step=1)
Present_price = st.number_input("Present Price", min_value=0)
Kms_Drive = st.number_input("Km Driven", min_value=0, step=100)
FuelType = st.selectbox("Fuel Type", ["Petrol", "Disel", "CNG"])
SellerType = st.selectbox("Seller Type", ["Dealer", "Individual"])
Transmission = st.selectbox("Transmission", ["Mannual", "Automatic"])
Owner = st.number_input("Owner", min_value=0, step=1)

Present_price = Present_price/100000
FuelType = 1 if FuelType == "Disel" else (0 if FuelType == "Petrol" else 2)
SellerType = 1 if SellerType == "Dealer" else 0
Transmission = 1 if Transmission == "Automatic" else 0

if st.button("Predict"):
    input_data = pd.DataFrame({"Year":[year],"Present Price":[Present_price],"Km Driven":[Kms_Drive], "Fuel Type":[FuelType], "Seller Type":[SellerType], "Transmission":[Transmission], "Owner":[Owner]})
    array = np.asarray(input_data)
    reshape = array.reshape(1,-1)
    prediction = model.predict(reshape)

    price = round(prediction[0],2)*100000

    st.write("The Price of Car is estimated to be: ", price)