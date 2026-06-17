from layers.base import Layer
from engine.tensor import Tensor


class ReLU(Layer):
    def forward(self, x):
        return x.relu()


class Sigmoid(Layer):
    def forward(self, x):
        # sigmoid(x) = 1 / (1 + e^-x)
        return (Tensor(1.0) + (-x).exp()) ** -1


class Tanh(Layer):
    def forward(self, x):
        # tanh(x) = 2*sigmoid(2x) - 1
        sig = (Tensor(1.0) + (x * -2.0).exp()) ** -1
        return sig * 2.0 - 1.0