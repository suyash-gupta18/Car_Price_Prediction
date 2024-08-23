import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st


def main():
   lin_reg_model=pk.load(open('lin_reg_model.pkl','rb')) 
   st.header('Car Price Prediction Model')
   cars_data=pd.read_csv('car data.csv')

   Year=st.selectbox('Select Year',cars_data['Year'].sort_values().unique())
   Present_Price=st.text_input('Select Price(in lakhs with 2 decimal places)')
   Kms_Driven=st.slider('Select Kms Driven',1000,150000)
   Fuel_Type=st.selectbox('Select Fuel Type',cars_data['Fuel_Type'].unique())
   Seller_Type=st.selectbox('Select Seller Type',cars_data['Seller_Type'].unique())
   Transmission=st.selectbox('Select Transmission',cars_data['Transmission'].unique())
   Owner=st.selectbox('Select Owner',cars_data['Owner'].unique())

   if st.button("Predict"):
     input_data_model = pd.DataFrame(
     [[Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]],
     columns=['Year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
     input_data_model.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
     input_data_model.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
     input_data_model.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

     car_price = lin_reg_model.predict(input_data_model)
     st.markdown('Car Price is going to be '+ str(car_price[0]))
if __name__ == "__main__":
    main()
