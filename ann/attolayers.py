# Implementation of the following neural network layers
# 1. Fully connected dense layer (Linear)
# 2. Activation layer (tanh, sigmoid)
# 3. Convolution Layers
# 4. Max Pooling Layer

import numpy as np
from optimizers import SGD

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.gradWeights = None
        self.gradBias = None

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    def backprop(self, error, lr, optimizer):
        if (optimizer == "SGD"):
            self.grad = SGD()
            self.weights = self.weights - lr * self.gradWeights
            self.bias = self.bias - lr * self.grqadBias