#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:57:03 2023

@author: alex
"""


import numpy as np
from flask import Flask, request, render_template
import joblib
import pandas as pd

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model and scaler
model = joblib.load(open('Models/model.pkl', 'rb'))
scaler = joblib.load(open('Models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] 
    features = [np.array(int_features)]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    if prediction[0] == 0:
        output = 'Low'
    elif prediction[0] == 1:
        output = 'Medium'
    else:
        output = 'High'


    return render_template('index.html', prediction_text='Lung Cancer Risk is: {}'.format(output))



if __name__ == "__main__":
    app.run()