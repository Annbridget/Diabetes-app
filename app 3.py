# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:35:30 2025

@author: Ann
"""

import numpy as np
import streamlit as st
import pickle
loaded_model = pickle.load(open("C:/Users/Ann/ML/trained_model.sav",'rb'))
                                
 # create a function for prediction                               
def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return('The person is not diabetic')
    else:
        return('The person is diabetic')


def main():
    # givint title to UI Interface
    st.title('diabetie prediction web app')
    
    #get input data for our web
    Pregnancies = st.text_input('No of Pregnancies')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('BMI')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BloodPressure')
    Age = st.text_input('Age')
    SkinThickness = st.text_input('SkinThickness')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    
    #code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('diabetes_test_result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    
    
    