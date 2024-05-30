# Implementing SGD and Adam
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

class SGD(Optimizer):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params)
        self.lr = lr
    def descent(self):
        for t in self.params:
            self.grad = t.data
            t.data -= self.lr * t.grad

class Adam(Optimizer):
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        super(Adam, self).__init__(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(t.data) for t in self.params]
        self.v = [np.zeros_like(t.data) for t in self.params]
    
    def descent(self):
        for i, t in enumerate(self.params):
            self.t += 1
            self.m[i]
            # Not imeplemented yet ...................
