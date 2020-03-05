#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:34:37 2020

@author: tapiwamaruni
"""


# ARTIFICIAL NEURAL NETWORKS

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
X = dataset2.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ===========
# Part 2 - Model Fitting with and ANN

# Import keras and other packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

# Add inpit layer and firsthidden layer
'''
activation - activation function used by the layer
units - 
kernel_initializer - How teh layer is initialized
input_dim - Number of inputs - Tuned to th e# of independent variables we extracted
'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Step 1 - Randomly Initialize the weights of the NN with small numbers close to 0 (but not 0)

# Step 2 - Add input layer -> activation function - Rectifier function

# Step 3 - Pass activation function value to y - Sigmoid AF




# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)





# ===========




# ===========




# ===========



# ===========
