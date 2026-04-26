import numpy as np


class Optimizer:
    def __init__(self, params):
        if not params:
            raise ValueError("Optimizer received an empty parameter list.")
        self.params = params


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Adam(Optimizer):
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= b1 < 1):
            raise ValueError(f"b1 must be in [0, 1), got {b1}")
        if not (0 <= b2 < 1):
            raise ValueError(f"b2 must be in [0, 1), got {b2}")
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, params, lr, alpha=0.99, eps=1e-8):
        super().__init__(params)
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= alpha < 1):
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + self.eps)
