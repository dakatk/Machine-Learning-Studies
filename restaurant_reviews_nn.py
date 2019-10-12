#!/usr/bin/env python3

# Imports

from keras.layers import Dense
from keras.models import Sequential

from keras.optimizers import Adam

import numpy as np

import json_loader


def normalize(data_set, data_max=None, data_min=None):
    """Normalize values in array to range [0, 1]"""

    if data_min is None:
        data_min = min(data_set)

    if data_max is None:
        data_max = max(data_set)

    def norm_el(el):

        nonlocal data_min, data_max

        return (el - data_min) / (data_max - data_min)

    return np.array(list(map(norm_el, data_set)))


def stars_given_latitude(num_epochs):
    """Generates a Model that predicts the amount of stars based
       on the latitude of the restaurant's location"""

    data = json_loader.get_restaurants_data()

    # Kind of a cheap way of dealing with big data,
    # but this all works for our relatively small
    # sample sets
    lats = [data[i]['latitude'] for i in data]
    stars = [data[i]['stars'] for i in data]

    model = create_simple_model(lats, stars, num_epochs, 1)

    return model, max(stars), min(stars), max(lats), min(lats)


def stars_given_length(num_epochs):
    """Generates a Model that predicts the amount of stars based
       on the length of a given review"""

    data = json_loader.get_reviews_data()

    # Kind of a cheap way of dealing with big data,
    # but this all works for our relatively small
    # sample sets
    lens = [len(data[i]['text']) for i in data]
    stars = [data[i]['stars'] for i in data]

    model = create_simple_model(lens, stars, num_epochs, 1)

    return model, max(stars), min(stars), max(lens), min(lens)


def create_simple_model(x, y, num_epochs, batch_size):
    """Creates a simple three-layer sequential model from given input and output data,
       using a given batch size and number of epochs to fit the model to the data"""

    sample_size = len(x)

    # Normalize the data
    nx = normalize(x)
    ny = normalize(y)

    # Model creation:
    model = Sequential()

    # Input layer
    model.add(Dense(units=sample_size, activation='relu', input_shape=(1,)))

    # Filtering layer, used as a way to deal with comllexity beyond 
    # simple linear relationships
    model.add(Dense(units=sample_size // 2, activation='sigmoid'))

    # Output layer
    model.add(Dense(units=1, activation='relu'))

    # Model compilation. The optimizer, loss, and other
    # variables to this function can be tweaked or altered to get
    # a better model to relate the input and output
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    
    # This is where the magic happens:
    model.fit(x=nx, y=ny, batch_size=1, epochs=num_epochs, validation_data=[nx, ny], verbose=0)

    return model
