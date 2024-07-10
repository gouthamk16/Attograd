import math
import numpy as np
import cupy as cp

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', device='cpu'):
        self.device = device
        self.data = self.to_device(np.array(data) if isinstance(data, list) else data, device)
        self.grad = self.to_device(np.zeros_like(self.data), device)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Tensor(data={self.data}, device='{self.device}')"

    def to_device(self, data, device):
        if device == 'cpu':
            return np.array(data)
        elif device == 'gpu':
            return cp.array(data)
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'")
    
    def to(self, device):
        if device == self.device:
            return self
        
        if device == 'cpu':
            self.data = cp.asnumpy(self.data)
            self.grad = cp.asnumpy(self.grad)
        elif device == 'gpu':
            self.data = cp.array(self.data)
            self.grad = cp.array(self.grad)
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'")
        
        self.device = device
        return self

    @staticmethod
    def to_tensor(arr, device='cpu'):
        return Tensor(arr, device=device)
    
    @staticmethod
    def zeros(shape, dtype="float32", device='cpu'):
        data = np.zeros(shape, dtype) if device == 'cpu' else cp.zeros(shape, dtype)
        return Tensor(data, device=device)

    @staticmethod
    def ones(shape, dtype="float32", device='cpu'):
        data = np.ones(shape, dtype) if device == 'cpu' else cp.ones(shape, dtype)
        return Tensor(data, device=device)
    
    @staticmethod
    def random(shape, dtype="float32", value_range=None, device='cpu'):
        if dtype == "float32":
            if value_range:
                low, high = value_range
                data = np.random.uniform(low, high, size=shape).astype(np.float32) if device == 'cpu' else cp.random.uniform(low, high, size=shape).astype(cp.float32)
            else:
                data = np.random.rand(*shape).astype(np.float32) if device == 'cpu' else cp.random.rand(*shape).astype(cp.float32)
        elif dtype == "int32":
            if value_range:
                low, high = value_range
                data = np.random.randint(low, high+1, size=shape, dtype=np.int32) if device == 'cpu' else cp.random.randint(low, high+1, size=shape, dtype=cp.int32)
            else:
                data = np.random.randint(0, 100, size=shape, dtype=np.int32) if device == 'cpu' else cp.random.randint(0, 100, size=shape, dtype=cp.int32)
        else:
            raise ValueError("dtype must be int32 or float32")
        return Tensor(data, device=device)
    
    def reshape(self, new_shape):
        return Tensor(self.data.reshape(new_shape), device=self.device)

    @staticmethod
    def multinomial(probabilities, num_samples, replacement=True):
        if not isinstance(probabilities, Tensor):
            raise ValueError("Probabilities should be of type Tensor")
        data = np.random.choice(len(probabilities.data), size=num_samples, replace=replacement, p=probabilities.data) if probabilities.device == 'cpu' else cp.random.choice(len(probabilities.data), size=num_samples, replace=replacement, p=probabilities.data)
        return Tensor(data, device=probabilities.device)
    
    def matmul(self, other):
        if isinstance(other, Tensor):
            data = np.matmul(self.data, other.data) if self.device == 'cpu' else cp.matmul(self.data, other.data)
            return Tensor(data, device=self.device)
        else:
            raise ValueError("Matrix should be of type tensor")

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, (self, other), '+', device=self.device)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data * other.data, (self, other), '*', device=self.device)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "dtype should be int or float"
        out = Tensor(self.data ** other, (self,), f"**{other}", device=self.device)
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # for ops reversed eg; other * self
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def exp(self):
        x = self.data
        out = Tensor(cp.exp(x) if self.device == 'gpu' else np.exp(x), (self,), 'exp', device=self.device)
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Tensor(cp.log(x) if self.device == 'gpu' else np.log(x), (self,), 'log', device=self.device)
        def _backward():
            self.grad += (1/x) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        activated_val = (cp.exp(2*x)-1)/(cp.exp(2*x)+1) if self.device == 'gpu' else (np.exp(2*x)-1)/(np.exp(2*x)+1)
        out = Tensor(activated_val, (self,), 'tanh', device=self.device)
        def _backward():
            self.grad += (1 - activated_val**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        activated_val = 1 / (1 + cp.exp(-x)) if self.device == 'gpu' else 1 / (1 + np.exp(-x))
        out = Tensor(activated_val, (self,), 'sigmoid', device=self.device)
        def _backward():
            self.grad += activated_val * (1 - activated_val) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = self.to_device(np.ones_like(self.data), self.device)
        for node in reversed(topo):
            node._backward()
