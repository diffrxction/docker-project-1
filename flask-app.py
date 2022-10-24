#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:49:02 2022

@author: diffract
"""

from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
pickle_in = open('classifier_weights.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome Everyone!"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is: {}".format(prediction)

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("filePath"))
    prediction = classifier.predict(df_test)
    return "The predicted values for the file are {}".format(list(prediction))
                                

if __name__ == "__main__":
    app.run()