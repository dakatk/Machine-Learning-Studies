import numpy as np


class Relu(object):
    """Rectified Linear Unit activation function"""

    def call(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class Sigmoid(object):
    """Logisitic Sigmoid activation function"""

    def __init__(self, k, x0):

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
    reg_factor = 0.002

    def __init__(self, input_size, output_rows, activation_fns, cost, neurons):

        # Total number of layers (hidden layers plus output layer)
        self.layers = len(activation_fns)

        # Neurons for each layer
        self.neurons = neurons

        # Activation (transfer) functions
        self.activation_fns = activation_fns

        # Cost (loss) function
        self.cost = cost

        ## Layer order:
        #    input layer
        #    hidden layer(s)
        #    output layer

        # Setup of weight and bias matrices:
        self.weights = []
        self.biases = []

        for i in range(self.layers):

            rows = neurons[i - 1] if i != 0 else input_size
            cols = neurons[i]

            weight = np.random.random((rows, cols))
            bias = np.random.random((output_rows, cols))

            self.weights.append(weight)
            self.biases.append(bias)

        # Values calculated between layers
        self.a = [None] * (self.layers + 1)

    def feed_forward(self, inputs):

        self.a[0] = inputs[:]

        for i in range(self.layers):

            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.a[i + 1] = self.activation_fns[i].call(z)

        return self.a[self.layers]

    def back_propogate(self, actual_outputs, pred_outputs):

        dC_dW = [None] * self.layers
        dC_db = [None] * self.layers

        for i in range(self.layers - 1, -1, -1):

            if i == self.layers - 1:
                
                error = self.cost.gradient(actual_outputs, pred_outputs)
                error += self.reg_factor * np.sum(np.absolute(self.weights[i]))

            else:
                error = np.dot(delta, self.weights[i + 1].T)

            y_prime = self.activation_fns[i].derivative(self.a[i + 1])
            delta = error * y_prime

            transposed = self.a[i].T
            shape = transposed.shape[0]

            dC_dW[i] = np.dot(transposed.view().reshape((shape, 1)), delta)
            dC_db[i] = delta

        for i in range(self.layers):
            
            self.weights[i] += self.learning_rate * dC_dW[i]
            self.biases[i] += self.learning_rate * dC_db[i]

    def predict(self, test_input):

        return self.feed_forward(test_input)

    def loss(self, test_input, test_output):

        return self.cost.call(test_output, self.predict(test_input))

    def train(self, train_inputs, train_outputs, epochs=1):

        for _ in range(epochs):

            for (inputs, outputs) in zip(train_inputs, train_outputs):
                self.back_propogate(outputs, self.feed_forward(inputs)) 


if __name__ == '__main__':

    from copy import deepcopy

    train_inputs = [
        np.array([0, 0, 0]),
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        np.array([1, 0, 0]),
        np.array([1, 0, 1])
    ]

    test_inputs = [
        np.array([1, 1, 0]),
        np.array([1, 1, 1])
    ]

    train_outputs = [
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 1]),
        np.array([1, 0])
    ]

    test_outputs = [
        np.array([0, 0]),
        np.array([0, 0])
    ]

    activation_fns = [Tanh(), Sigmoid(1, 0), Sigmoid(1, 0), Sigmoid(1, 0)]
    neurons = [4, 3, 3, 2]
    
    nn = NN(3, 1, activation_fns, BinaryCrossentropy(), neurons)
    nn.train(train_inputs, train_outputs, 2000)

    for (i, el) in enumerate(train_inputs):
        
        print(nn.predict(el))
        print(nn.loss(el, train_outputs[i]))

    for (i, el) in enumerate(test_inputs):
        
        print(nn.predict(el))
        print(nn.loss(el, test_outputs[i]))

