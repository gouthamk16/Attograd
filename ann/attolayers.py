# Implementation of the following neural network layers
# 1. Fully connected dense layer (Linear)
# 2. Activation layer (tanh, sigmoid)
# 3. Convolution Layers
# 4. Max Pooling Layer

import numpy as np
from tensor import Tensor

class Neuron:

    def __init__(self, n_inputs):
        self.w = [Tensor.random(1, value_range=(-1, 1)) for _ in range(n_inputs)] 
        self.b = Tensor.random(1, value_range=(-1, 1))

    def __call__(self, x, activation="tanh"):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) # pairing the w's and the x's
        if activation=="tanh":
            out = act.tanh()
        elif activation=="sigmoid":
            out = act.sigmoid()
        return out

class Linear:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x