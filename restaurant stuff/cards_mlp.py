#!/usr/bin/env python3

"""
This is an example of a very simple Neural Network that achieves a 1-1 function
between a set of numbers and a set of strings. This approach shows off how encoding
and decoding can still maintain an accurate relationship between data. This
specific example also highlights an issue that can happen with current libraries
and setups for Neural Netowrks, which is having no protection against over-training.

The set that the function is based on is the entire data set and does not have a domain
beyond what is used in this program. That means that over time, the Neural Network will
train such that there is no leniency and each input is exactly correlated with it's output.
This may sound ideal, but for data sets that can expand, this will not work as well, because an over-trained
Network won't accomodate correctly for data points that do not lie directly on the sample data set.

TLDR: sample data shouldn't be exactly equal to entire data set, because then the NN thinks the
sample data is the only possible data that could exist with the given correlation.
"""

# Imports

from keras.layers import Dense
from keras.models import Sequential

from keras.optimizers import SGD

import numpy as np

# Model Creation

model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=(13,)))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=13, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Main Code

CARD_SYMBOLS = np.array(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
CARD_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


def one_hot_encode(sym, group):
    
    encoded = np.zeros(group.shape[0])
    encoded[np.where(group == sym)] = 1

    return encoded


def test_model(test_symbols):

    global CARD_SYMBOLS

    X_test = np.array([one_hot_encode(s, CARD_SYMBOLS) for s in test_symbols])

    predictions = model.predict(X_test)

    for (s, p) in zip(test_symbols, predictions):
        print(f'{s}: ', CARD_VALUES[np.argmax(p)])
        

X_train = np.array(
    [one_hot_encode(card, CARD_SYMBOLS) for card in CARD_SYMBOLS]
)

y_train = np.array(
    [one_hot_encode(value, CARD_VALUES) for value in CARD_VALUES]
)

model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=[X_train, y_train], shuffle=True)

test_model(CARD_SYMBOLS)
