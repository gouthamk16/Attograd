# Implementing SGD and Adam
## To do : Numerical check of Adam

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
        self.t += 1
        for i, t in enumerate(self.params):
            grad = t.grad
            
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad #Biased First Moment Estimate
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grad ** 2) #Biased Second Moment Estimate
            
            m_hat = self.m[i] / (1 - self.b1 ** self.t) #Bias-Corrected First Moment Estimate
            v_hat = self.v[i] / (1 - self.b2 ** self.t) #Bias-Corrected Second Moment Estimate
            
            t.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) 
            
            