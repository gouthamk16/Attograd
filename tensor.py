import math
import numpy as np
import matplotlib.pyplot as plt

######
## THE GRADIENT EQUATION OF THE 'LOG' FUNCTION NEEDS TO BE CHECKED
######

class Tensor:

    def _init_(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def _repr_(self):
        return f"Tensor(data={self.data})"
    
    def toTensor(arr):
        return Tensor(arr)
    
    def zeros(shape, dtype="float32"):
        return Tensor(np.zeros(shape, dtype))

    def ones(shape, dtype="float32"):
        return Tensor(np.ones(shape, dtype))
    
    def random(shape, dtype="float32", value_range=None):
        if dtype=="float32":
            if value_range:
                low, high = value_range
                return Tensor(np.random.rand(low, high, size=shape).astype(np.float32))
            else:
                return Tensor(np.random.rand(size=shape).astype(np.float32))
            
        if dtype=="int32":
            if value_range:
                low, high = value_range
                return Tensor(np.random.randint(low, high+1, size=shape, dtype=np.int32))
            else:
                return Tensor(np.random.randint(size=shape, dtype=np.int32))
        
        else:
            raise ValueError("dtype must be int32 or float32 <Will be compatible with more datatypes later>")
    
    def reshape(self, new_shape):
        return Tensor(np.reshape(self.data, new_shape))

    def multinomial(probabilities, num_samples, replacement=True):
        if not isinstance(probabilities, Tensor):
            raise ValueError("Probabilities should be of type Tensor")
        return Tensor(np.random.choice(len(probabilities.data), size=num_samples, replace=replacement, p=probabilities.data))
    

    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.matmul(self.data, other.data))
        else:
            raise ValueError("Matrix should be of type tensor")

    def _add_(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def _mul_(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def _pow_(self, other):
        assert isinstance(other, (int, float)), "dtype should be int or float"
        out = Tensor(self.data*other, (self,), f"*{other}")
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def _rmul_(self, other): # for ops reversed eg; other * self
        return self * other
    
    def _truediv_(self, other):
        return self * other**-1
    
    def exp(self):
        x = self.data
        out = Tensor(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Tensor(math.log(x), (self,), 'log')
        def _backward():
            self.grad += (1/x) * out.grad ### THIS NEEDS TO BE CHECKED
        out._backward = _backward
        return out

    
    def tanh(self):
        x = self.data
        activated_val = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Tensor(activated_val, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - activated_val**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        activated_val = (math.exp(-x))/(1+math.exp(-x))
        out = Tensor(activated_val, (self, ), 'sig')
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
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
