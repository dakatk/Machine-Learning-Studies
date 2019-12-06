## TODO add optimization function support

import numpy as np
import json


_class_registry = dict()


class Relu(object):
    """Rectified Linear Unit activation function"""

    def call(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class Sigmoid(object):
    """Logisitic Sigmoid activation function"""

    def __init__(self, k, x0):

        self.args = (k, x0)

        self.k = k
        self.x0 = x0

    def call(self, x):
        return 1 / (1 + np.exp(-self.k * (x - self.x0)))

    def derivative(self, x):

        y = self.call(x)
        
        return y * (1 - y)


class Tanh(object):
    """Hyperbolic tangent activation function"""

    def call(self, x):
        return np.tanh(x)

    def derivative(self, x):

        y = self.call(x)
        
        return 1 - (y * y)


class MSE(object):
    """Mean squared error loss function"""

    def call(self, y, o):
        return np.sum(np.square(y - o)) / 2

    def gradient(self, y, o):
        return y - o


class BinaryCrossentropy(object):
    """Binary Crossentropy loss function"""
    
    def call(self, y, o):

        a = y * np.log(o)
        b = (1 - y) * np.log((1 - o))

        return np.sum(a + b) / -len(y)

    def gradient(self, y, o):
        return y * (o - 1) + ((1 - y) * o)


class NN(object):
    """Multi-layer perceptron neural network"""

    learning_rate = 0.01

    def __init__(self, inputs, outputs, activation_fns, cost, neurons, *, skip_setup=False):

        # Total number of layers (hidden layers plus output layer)
        self.layers = len(activation_fns)

        # Neurons for each layer
        self.neurons = neurons

        # Input values
        self.inputs = inputs

        # Expected output values
        self.outputs = outputs

        # Activation (transfer) functions
        self.activation_fns = activation_fns

        # Cost (loss) function
        self.cost = cost

        if not skip_setup:

            ## Layer order:
            #    input layer
            #    hidden layer(s)
            #    output layer

            # Setup of weight and bias matrices:
            self.weights = []
            self.biases = []

            for i in range(self.layers):

                rows = neurons[i - 1] if i != 0 else inputs.shape[1]
                cols = neurons[i] if i != self.layers - 1 else outputs.shape[1]

                weight = np.random.random((rows, cols))
                bias = np.random.random((inputs.shape[0], cols))

                self.weights.append(weight)
                self.biases.append(bias)

            # Values calculated between layers
            self.a = [None] * (self.layers + 1)
            self.z = [None] * self.layers

            self.a[0] = np.copy(inputs)

    def feed_forward(self):

        for i in range(self.layers):

            self.z[i] = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.a[i + 1] = self.activation_fns[i].call(self.z[i])

        return self.a[self.layers]

    def back_propogate(self, pred_outputs):

        dC_dW = [None] * self.layers
        dC_db = [None] * self.layers

        for i in range(self.layers - 1, -1, -1):

            if i == self.layers - 1:
                error = self.cost.gradient(self.outputs, pred_outputs)

            else:
                error = np.dot(delta, self.weights[i + 1].T)

            y_prime = self.activation_fns[i].derivative(self.a[i + 1])
            delta = error * y_prime

            dC_dW[i] = np.dot(self.a[i].T, delta)
            dC_db[i] = delta

        for i in range(self.layers):
            
            self.weights[i] += self.learning_rate * dC_dW[i]
            self.biases[i] += self.learning_rate * dC_db[i]

    def prediction(self):

        pred_outputs = self.feed_forward()
        
        return {'output': pred_outputs, 'error': self.cost.call(self.outputs, pred_outputs)}

    def train(self, epochs=1):

        for _ in range(epochs):
            self.back_propogate(self.feed_forward())

    def save(self):

        global _class_registry

        data = {
            'neurons': self.neurons,
            'inputs': self.inputs.tolist(),
            'outputs': self.outputs.tolist(),
            'weights': {},
            'biases': {},
            'a': {},
            'z': {},
            'activation_fns': {},
            'cost': None
        }

        def jsonify_array(array, name):

            nonlocal data

            for (i, el) in enumerate(array):
                data[name][str(i)] = el.tolist()

        jsonify_array(self.weights, 'weights')
        jsonify_array(self.biases, 'biases')
        jsonify_array(self.a, 'a')
        jsonify_array(self.z, 'z')

        for (i, f) in enumerate(self.activation_fns):

            f_class = f.__class__
            class_name = f_class.__name__

            _class_registry[class_name] = f_class
            data['activation_fns'][str(i)] = {'name': class_name}

            class_args = tuple()

            if hasattr(f_class, 'args'):
                class_args = f.args

            data['activation_fns'][str(i)]['args'] = class_args

        cost_class = self.cost.__class__
        cost_name = cost_class.__name__
        
        _class_registry[cost_name] = cost_class
        data['cost'] = cost_name

        return json.dumps(data)


if __name__ == '__main__':

    from copy import deepcopy

    '''
    for input [a, b, c]:
        output = [a xor b, b nor c]
    '''

    inputs = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]) # 8 rows, 3 cols

    outputs = np.array([
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0]
    ]) # 8 rows, 2 cols

    activation_fns = [Sigmoid(-1, 0), Sigmoid(-1, 0), Relu()]
    neurons = [16, 16]
    
    nn = NN(inputs, outputs, activation_fns, BinaryCrossentropy(), neurons)

    def best_prediction(nn, sequences):

        min_loss = nn.prediction()['error']
        best_fit = (0, deepcopy(nn))

        for i in range(sequences):
        
            nn.train(1)
            loss = nn.prediction()['error']

            if loss < min_loss:

                min_loss = loss
                best_fit = (i + 1, deepcopy(nn))

        return best_fit

    index, best_fit = best_prediction(nn, 5000)
    pred = best_fit.prediction()
    
    print(f'Best prediction (at epoch {index}):\n', pred, '\n')

    with open('NN.json', 'w') as f:
        f.write(best_fit.save())
