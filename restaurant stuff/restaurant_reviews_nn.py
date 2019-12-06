#!/usr/bin/env python3

# Imports

from keras.layers import Dense
from keras.models import Sequential

from keras.optimizers import SGD
from keras.callbacks import callbacks

import numpy as np
import tensorflow as tf

import json_loader


def normalize(data_set, data_max=None, data_min=None):
    """Normalize values in array to range [0, 1]"""

    if data_min is None:
        data_min = np.min(data_set)

    if data_max is None:
        data_max = np.max(data_set)

    def norm_el(el):

        nonlocal data_min, data_max

        return (el - data_min) / (data_max - data_min)

    return np.array(list(map(norm_el, data_set)))


def stars_given_location(num_epochs):
    """Generates a Model that predicts the amount of stars based
       on the latitude of the restaurant's location"""

    data = json_loader.get_restaurants_data()

    # Kind of a cheap way of dealing with big data,
    # but this all works for our relatively small
    # sample sets
    lats = [data[i]['latitude'] for i in data]
    longs = [data[i]['longitude'] for i in data]
    stars = [data[i]['stars'] for i in data]

    locs = np.array(zip(lats, longs), dtype=float)

    model = create_simple_model(locs, stars, num_epochs, 10)

    return model, max(stars), min(stars), max(lats), min(lats), max(longs), min(longs)


def length_given_stars(num_epochs):
    """Generates a Model that predicts the length of a review based
       on the number of stars given"""

    data = json_loader.get_reviews_data()

    # Kind of a cheap way of dealing with big data,
    # but this all works for our relatively small
    # sample sets
    lens = [len(data[i]['text'].split(' ')) for i in data]
    stars = [data[i]['stars'] for i in data]

    model = create_simple_model(stars, lens, num_epochs, 10)

    return model, max(lens), min(lens), max(stars), min(stars)


def create_simple_model(x, y, num_epochs, batch_size, shape):
    """Creates a simple three-layer sequential model from given input and output data,
       using a given batch size and number of epochs to fit the model to the data"""

    sample_size = len(x)

    # Normalize the data
    nx = normalize(x)
    ny = normalize(y)

    # Model creation:
    model = Sequential()

    # Input layer
    model.add(Dense(units=100, activation='sigmoid', input_shape=shape))

    # Filtering layers, used as a way to deal with comllexity beyond 
    # simple linear relationships
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=20, activation='relu'))

    # Output layer
    model.add(Dense(units=1, activation='tanh'))

    # Model compilation. The optimizer, loss, and other
    # variables to this function can be tweaked or altered to get
    # a better model to relate the input and output
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
    
    # This is where the magic happens:
    m_callbacks = [callbacks.EarlyStopping(patience=3)]
    model.fit(x=nx, y=ny, epochs=num_epochs, validation_data=[nx, ny], verbose=0, shuffle=False, batch_size=batch_size, use_multiprocessing=True, callbacks=m_callbacks)

    return model
