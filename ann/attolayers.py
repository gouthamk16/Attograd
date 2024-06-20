# Implementation of the following neural network layers
# 1. Fully connected dense layer (Linear)
# 2. Activation layer (tanh, sigmoid)
# 3. Convolution Layers
# 4. Max Pooling Layer

import numpy as np
from optimizers import SGD
from tensor import Tensor

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.gradWeights = None
        self.gradBias = None

class Linear(Layer):

    def __init__(self, input_size, output_size, bias=True):
        self.weight = Tensor.random((input_size, output_size)) / input_size**0.5
        self.bias = Tensor.zeros(output_size) if bias else None

    def forward(self, x):
        self.input = x
        self.output = x.matmul(self.weight)
        if self.bias is not None:
            self.output += self.bias
        return self.output
    
    def params(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
    ### BACKPROP - HOW? ########
    # Gradient descent implemented in the tensor class - how tf do we connect them?
    # Check pytorch docs to see how they did it

    # def backprop(self, error, lr, optimizer):
    #     if (optimizer == "SGD"):
    #         self.grad = SGD()
    #         self.weights = self.weights - lr * self.gradWeights
    #         self.bias = self.bias - lr * self.gradBias

    ## Don't push yet, keep this class implementation as it is

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.params()]