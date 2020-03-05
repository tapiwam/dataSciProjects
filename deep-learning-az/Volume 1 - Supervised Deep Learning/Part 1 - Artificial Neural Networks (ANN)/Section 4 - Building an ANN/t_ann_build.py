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

# ===========
# Part 1 - Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()

# Fields to keep
# credit score, Geography (c), gender (c), age, tenure, balance, numProd, hasCreCard, ActiveMember, EstinatedSal

dataset2 = pd.read_csv('Churn_Modelling.csv')
dataset2.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Get dependent varaibles
y = dataset2['Exited'].values

# Handle categorical variables
dataset2.drop(['Exited'], axis=1, inplace=True)
dataset2 = pd.get_dummies(dataset2, columns=['Geography', 'Gender'], drop_first=True)
# dataset2.drop(['Gender_Male'], axis=1, inplace=True)
X = dataset2.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ===========
# Part 2 - Model Fitting with and ANN


# Import keras and other packages
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

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


# ===========

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=40)

# Predicting the Test set results
# Convert probabilities as true or false
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# ===========



# ===========
