# tests/test_activations.py
import numpy as np
from engine.tensor import Tensor
from layers.activations import Sigmoid, Tanh

def test_sigmoid_gradient():
    x = Tensor(0.0)
    sig = Sigmoid()
    out = sig.forward(x)
    out.backward()

    # sigmoid(0) = 0.5, sigmoid'(0) = sigmoid(0)*(1-sigmoid(0)) = 0.25
    assert np.isclose(out.data, 0.5)
    assert np.isclose(x.grad, 0.25)
    print("Sigmoid gradient test passed")

def test_tanh_gradient():
    x = Tensor(0.0)
    tanh = Tanh()
    out = tanh.forward(x)
    out.backward()

    # tanh(0) = 0, tanh'(0) = 1 - tanh(0)^2 = 1
    assert np.isclose(out.data, 0.0)
    assert np.isclose(x.grad, 1.0)
    print("Tanh gradient test passed")

if __name__ == "__main__":
    test_sigmoid_gradient()
    test_tanh_gradient()