import math
import numpy as np
import matplotlib.pyplot as plt
# seed
np.random.seed(0)

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
        return Tensor(self.data[idx])
    
    # Function for item assignment
    def __setitem__(self, idx, value):
        # Convert value to a numpy array if it is a Tensor
        if isinstance(value, Tensor):
            value = value.data
        # Ensure value is of compatible shape and type
        value = np.array(value, dtype=self.data.dtype)
        self.data[idx] = value
    
    def toTensor(arr):
        return Tensor(arr)
    
    def toNumpy(self):
        return self.data
    
    def shape(self):
        return self.data.shape
    
    def ndim(self):
        return self.data.ndim

    ## Function to get the shape
    def shape(self):
        return self.data.shape
    
    def zeros(shape, dtype="float32"):
        return Tensor(np.zeros(shape, dtype))

    def ones(shape, dtype="float32"):
        return Tensor(np.ones(shape, dtype))
    
    def random(shape, dtype="float32", value_range=None):
        if isinstance(shape, int):
            shape = (shape,)

        if dtype=="float32":
            if value_range:
                low, high = value_range
                return Tensor(np.random.uniform(low, high, size=shape).astype(np.float32))
            else:
                return Tensor(np.random.rand(*shape).astype(np.float32))
            
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
    
    def __len__(self):
        return len(self.data)
    
    def flatten(self):
        return Tensor(self.data.flatten())

    def multinomial(probabilities, num_samples, replacement=True):
        if not isinstance(probabilities, Tensor):
            raise ValueError("Probabilities should be of type Tensor")
        return Tensor(np.random.choices(len(probabilities), size=num_samples, replace=replacement, p=probabilities.data))
    

    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.matmul(self.data, other.data))
        else:
            raise ValueError("Matrix should be of type tensor")
        
    ## Resolve reverse addition i.e, other comes in place of self (just like we did for __rmul__)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    # subtraction
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
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
        assert isinstance(other, (int, float)), "dtype should be int or float"
        out = Tensor(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): # for ops reversed eg; other * self
        return self * other
    
    def __radd__(self, other): # for ops reversed eg; other * self
        return self + other
    
    def __rsub__(self, other): # for ops reversed eg; other * self
        return self - other

    def __truediv__(self, other):
        return self * other**-1
    
    # def __rtruediv__(self, other): # for ops reversed eg; other * self
    #     return other * self**-1
    
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