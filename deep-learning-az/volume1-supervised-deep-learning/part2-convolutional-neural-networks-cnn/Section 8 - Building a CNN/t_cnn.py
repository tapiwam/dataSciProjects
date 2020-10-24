# -*- coding: utf-8 -*-

# Part 1 - Build the CNN

# Inital loads always first
from __future__ import print_function

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
epochs = 12


# =================
# Initialize the CNN
classifier = Sequential()

# =================
# Add layers

# Step 1 - Convolution
#classifier.add(Conv2D(filters=32, kernel_size=(3, 3),strides=2, border_mode='same', input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dropout(0.5))

# Step 4 - Full Connection - Classic ANN
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# =================

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=2000)


