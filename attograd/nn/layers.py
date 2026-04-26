"""
Neural network layer implementations.
"""

import numpy as np
from ..tensor import Tensor
from ..cuda.cuda_interface import matrixMultiply, is_cuda_available


_VALID_ACTIVATIONS = {None, 'tanh', 'sigmoid', 'relu'}


class Neuron:
    def __init__(self, n_inputs):
        self.w = Tensor.random(n_inputs, value_range=(-1, 1))
        self.b = Tensor(float(np.random.uniform(-1, 1)))

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x, activation=None):
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Expected one of {_VALID_ACTIVATIONS}.")
        if is_cuda_available.cuda and hasattr(x, 'data'):
            x_arr = x.data.flatten().astype('float32').reshape(1, -1)
            w_arr = self.w.data.reshape(-1, 1)
            act = Tensor(matrixMultiply(x_arr, w_arr).item()) + self.b
        else:
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if activation == 'tanh':
            return act.tanh()
        if activation == 'sigmoid':
            return act.sigmoid()
        if activation == 'relu':
            return act.relu()
        return act


class Linear:
    def __init__(self, nin, nout, activation=None):
        if not (isinstance(nin, int) and nin > 0):
            raise ValueError(f"nin must be a positive integer, got {nin!r}")
        if not (isinstance(nout, int) and nout > 0):
            raise ValueError(f"nout must be a positive integer, got {nout!r}")
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Expected one of {_VALID_ACTIVATIONS}.")
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.activation = activation

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __call__(self, x):
        return [n(x, activation=self.activation) for n in self.neurons]


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def update(self, lr=0.01):
        for p in self.parameters():
            p.data -= lr * p.grad


class TanhLayer:
    def parameters(self):
        return []

    def __call__(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        return x.tanh()


class SigmoidLayer:
    def parameters(self):
        return []

    def __call__(self, x):
        if isinstance(x, list):
            return [xi.sigmoid() for xi in x]
        return x.sigmoid()


class ReLULayer:
    def parameters(self):
        return []

    def __call__(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()


class Flatten:
    def parameters(self):
        return []

    def __call__(self, x):
        if isinstance(x, Tensor):
            return x.flatten()
        raise TypeError(f"Flatten expects a Tensor, got {type(x).__name__}")
