# tests/test_training_loop.py
import numpy as np
from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import SGD

def test_overfit_single_batch():
    np.random.seed(0)

    # Tiny dataset: y = 2x + 1
    X = Tensor(np.array([[0.0], [1.0], [2.0], [3.0]]))
    y = np.array([[1.0], [3.0], [5.0], [7.0]])

    model = Sequential([
        Dense(1, 8),
        Tanh(),
        Dense(8, 1)
    ])

    optimizer = SGD(model.parameters(), lr=0.05)

    losses = []
    for step in range(200):
        pred = model.forward(X)
        loss = mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data)

    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])
    assert losses[-1] < losses[0]
    print("Training loop test passed")

if __name__ == "__main__":
    test_overfit_single_batch()