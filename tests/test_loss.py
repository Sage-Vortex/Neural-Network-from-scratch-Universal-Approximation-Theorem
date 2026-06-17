# tests/test_loss.py
import numpy as np
from engine.tensor import Tensor
from layers.loss import mse_loss

def test_mse():
    pred = Tensor(np.array([[1.0], [2.0], [3.0]]))
    target = np.array([[1.5], [2.5], [2.5]])

    loss = mse_loss(pred, target)

    # manual: errors = [-0.5, -0.5, 0.5] -> squared = [0.25,0.25,0.25] -> mean = 0.25
    assert np.isclose(loss.data, 0.25)

    loss.backward()
    # d(MSE)/d(pred) = (2/n)(pred - target) = (2/3)*[-0.5,-0.5,0.5]
    expected_grad = (2/3) * np.array([[-0.5], [-0.5], [0.5]])
    assert np.allclose(pred.grad, expected_grad)

    print("MSE loss test passed")

if __name__ == "__main__":
    test_mse()