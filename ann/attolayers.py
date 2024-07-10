# Implementation of the following neural network layers
# 1. Fully connected dense layer (Linear)
# 2. Activation layer (tanh, sigmoid)
# 3. Convolution Layers
# 4. Max Pooling Layer
# 5. Implement Batch Normalization

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from tensor import Tensor

class Neuron:

    def __init__(self, n_inputs):
        self.w = [Tensor.random(1, value_range=(-1, 1)) for _ in range(n_inputs)] 
        self.b = Tensor.random(1, value_range=(-1, 1))

    # Parameters
    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x, activation="tanh"):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) # pairing the w's and the x's
        if activation=="tanh":
            out = act.tanh()
        elif activation=="sigmoid":
            out = act.sigmoid()
        return out

class Linear:

    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.activation = activation
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __call__(self, x):
        out = [n(x, activation=self.activation) for n in self.neurons]
        return out

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # Function to update weights
    def update(self, lr=0.01, optimizer = "SGD"):
        for layer in self.layers:
            if isinstance(layer, Linear):
                for neuron in layer.neurons:
                    for i in range(len(neuron.w)):
                        neuron.w[i] -= lr * neuron.w[i].grad
                        # reset the gradient after weight updation
                        neuron.w[i].grad = 0
                    neuron.b -= lr * neuron.b.grad
                    # Reset the bias after updation
                    neuron.b.grad = 0