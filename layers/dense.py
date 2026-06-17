import numpy as np
from engine.tensor import Tensor
from layers.base import Layer


class Dense(Layer):
    def __init__(self, in_features, out_features):
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        W_data = np.random.uniform(-limit, limit, (in_features, out_features))
        b_data = np.zeros(out_features)

        self.W = Tensor(W_data, label='W')
        self.b = Tensor(b_data, label='b')

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]