import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0 * p.grad


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # First and second moment estimates, one per parameter
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            g = p.grad

            # Update biased first moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update biased second moment estimate (adaptive scaling)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0 * p.grad

# add to network/optimizer.py

class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0 * p.grad