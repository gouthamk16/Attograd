"""
Core Tensor implementation for Attograd framework.
"""

import math
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __getitem__(self, idx):
        if self.data.ndim == 0:
            return self.data
        return Tensor(self.data[idx])

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = np.array(value, dtype=self.data.dtype)

    @staticmethod
    def toTensor(arr):
        return Tensor(arr)

    def toNumpy(self):
        return self.data

    def shape(self):
        return self.data.shape

    def ndim(self):
        return self.data.ndim

    def __len__(self):
        return len(self.data)

    @staticmethod
    def zeros(shape, dtype="float32"):
        return Tensor(np.zeros(shape, dtype))

    @staticmethod
    def ones(shape, dtype="float32"):
        return Tensor(np.ones(shape, dtype))

    @staticmethod
    def random(shape, dtype="float32", value_range=None):
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "float32":
            if value_range:
                low, high = value_range
                return Tensor(np.random.uniform(low, high, size=shape).astype(np.float32))
            return Tensor(np.random.rand(*shape).astype(np.float32))
        if dtype == "int32":
            if value_range:
                low, high = value_range
                return Tensor(np.random.randint(low, high + 1, size=shape, dtype=np.int32))
            return Tensor(np.random.randint(0, 2, size=shape, dtype=np.int32))
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected 'float32' or 'int32'.")

    def reshape(self, new_shape):
        return Tensor(np.reshape(self.data, new_shape))

    def flatten(self):
        original_shape = self.data.shape
        out = Tensor(self.data.flatten(), (self,), 'flatten')
        def _backward():
            self.grad += out.grad.reshape(original_shape)
        out._backward = _backward
        return out

    def reshape(self, new_shape):
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(original_shape)
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other).__name__}")
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')
        def _backward():
            # dL/dA = dL/dout @ B.T,  dL/dB = A.T @ dL/dout
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        return out

    @staticmethod
    def multinomial(probabilities, num_samples, replacement=True):
        if not isinstance(probabilities, Tensor):
            raise TypeError(f"Expected Tensor, got {type(probabilities).__name__}")
        n = len(probabilities)
        idx = np.random.choice(n, size=num_samples, replace=replacement, p=probabilities.data)
        return Tensor(idx)

    def _check_scalar(self, op_name):
        if self.data.ndim != 0:
            raise ValueError(
                f"{op_name}() only supports scalar tensors, got shape {self.data.shape}. "
                "Vectorized ops are not yet implemented."
            )

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Exponent must be int or float, got {type(other).__name__}")
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return Tensor(other) * self ** -1

    def exp(self):
        self._check_scalar('exp')
        x = self.data.item()
        out = Tensor(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        self._check_scalar('log')
        x = self.data.item()
        if x <= 0:
            raise ValueError(f"log() requires positive input, got {x}")
        out = Tensor(math.log(x), (self,), 'log')
        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        self._check_scalar('tanh')
        x = self.data.item()
        t = math.tanh(x)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        self._check_scalar('sigmoid')
        x = self.data.item()
        s = 1 / (1 + math.exp(-x))
        out = Tensor(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        self._check_scalar('relu')
        x = self.data.item()
        r = max(0, x)
        out = Tensor(r, (self,), 'relu')
        def _backward():
            self.grad += (r > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        if self.data.size != 1:
            raise RuntimeError(
                f"backward() can only be called on a single-element tensor, got shape {self.data.shape}. "
                "Call .sum() or use a scalar loss first."
            )
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
