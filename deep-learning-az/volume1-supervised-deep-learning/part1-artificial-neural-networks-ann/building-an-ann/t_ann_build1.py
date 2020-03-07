#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:34:37 2020

@author: tapiwamaruni
"""

# ARTIFICIAL NEURAL NETWORKS

# Inital loads always first
from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import keras and other packages
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler

# ===========
# Globals
 
sc = StandardScaler()
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
dummies = ['Geography', 'Gender']
final_cols = []

# ===========

def prepareDataset(dataset2):
    
    print('==================================')
    print('Preparing dataset...')
    print(f'Total Records: {len(dataset2.values)}')
    print(f'Info:')
    print(dataset2.info())
    print('==================================')
    
    # Get dependent varaibles
    y = dataset2['Exited'].values
    
    # Get independent features
    # RowNumber	CustomerId	Surname	CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
    dataset = dataset2[columns]
    
    # Handle categorical variables
    dataset = pd.get_dummies(dataset, columns=dummies, drop_first=True)
    
    # Get list of final columns 
    final_cols = dataset.columns.values.tolist()
    
    # Get final list
    X = dataset.values
    
    return X, y, final_cols

def getPredictionSet(data_dict):
    return pd.DataFrame(data_dict)
    

def featureScale(X_train, X_test):
    
    print('Scalling fetures in the data')
    
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test

def buildArtificialNeuralNet():
    
    print('Building ANN structure')
    
    # Initialize the ANN
    classifier = Sequential()
    
    # Step 1 - Add inpit layer and firsthidden layer. [activation function - Rectifier function]
    '''
    activation - activation function used by the layer
    units - (Avergae of #of inputs and # outputs so (11 + 1)/2 = 6)
    kernel_initializer - How teh layer is initialized
    input_dim - Number of inputs - Tuned to th e# of independent variables we extracted
    '''
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    # Step 2 - Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    
    # Adding the output layer
    '''
    -->> If you need more outputs -> Use Softmax -> used when you have more than 2 categories. Then the units will alos need to be increated by OneHotEncoder
    '''
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compile and ANN
    '''
    optimizer - Stcj=hastic optimer called adam
    loss - Use the Logarithmic loss for classification
        - Binary - binary_crossentropy
        - None binary - categorical_crossentropy
    metrics - criteria you use to evaulate/improve your model
    '''
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

def buildConfusionMatrix(y_test, y_pred):
    
    print('Building Confusion Matrix to check accuracy of model with')
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

# ===========
# ===========
# ===========
    
file_name = 'Churn_Modelling.csv'

# Part 1 - Data preprocessing
print('..... START: Building Model for churn classification .......')
print(f'Processing file: {file_name}')


# Importing the dataset
dataset2 = pd.read_csv(file_name)
X, y, final_cols = prepareDataset(dataset2)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
X_train, X_test = featureScale(X_train, X_test)

# ===========
# Part 2 - Model Fitting with and ANN

classifier = buildArtificialNeuralNet()

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=30)

print(f'..... Classifer Built from file: {file_name}. Starting Predictions .....')

# Predicting the Test set results
# Convert probabilities as true or false
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = buildConfusionMatrix(y_test, y_pred)

print('..... TEST: Predicting sample .......')

d1 = {'CreditScore': [600], 
      'Age': [40], 
      'Tenure': [3], 
      'Balance': [60000], 
      'NumOfProducts': [2], 
      'HasCrCard': [1], 
      'IsActiveMember': [1], 
      'EstimatedSalary': [50000],
      'Geography_Germany': [0], 
      'Geography_Spain': [0], 
      'Gender_Male': [1]
      }

d1x = pd.DataFrame(d1).values
# print(d1x)
X1 = sc.transform(d1x)
# print(X1)

y1 = classifier.predict(X1)
y1x = (y1 > 0.5)

print(f'Prediction: {"{:.3%}".format(y1[0,0])} chance to exit. Will Exit={y1x[0,0]}')

print('..... END: Building Model for churn classification .......')

# ===========
# ===========
# ===========
