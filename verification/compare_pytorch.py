# verification/compare_pytorch.py
import numpy as np
import torch
import torch.nn as nn

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential


def run_your_engine(X_data, y_data, W1, b1, W2, b2):
    model = Sequential([
        Dense(1, 4),
        Tanh(),
        Dense(4, 1)
    ])

    # Force identical weights
    model.layers[0].W.data = W1.copy()
    model.layers[0].b.data = b1.copy()
    model.layers[2].W.data = W2.copy()
    model.layers[2].b.data = b2.copy()

    X = Tensor(X_data)
    pred = model.forward(X)
    loss = mse_loss(pred, y_data)
    loss.backward()

    return {
        "loss": loss.data,
        "W1_grad": model.layers[0].W.grad.copy(),
        "b1_grad": model.layers[0].b.grad.copy(),
        "W2_grad": model.layers[2].W.grad.copy(),
        "b2_grad": model.layers[2].b.grad.copy(),
    }


def run_pytorch(X_data, y_data, W1, b1, W2, b2):
    X_t = torch.tensor(X_data, dtype=torch.float64)
    y_t = torch.tensor(y_data, dtype=torch.float64)

    model = nn.Sequential(
        nn.Linear(1, 4),
        nn.Tanh(),
        nn.Linear(4, 1)
    )

    # Force identical weights
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float64))
        model[0].bias.copy_(torch.tensor(b1,    dtype=torch.float64))
        model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float64))
        model[2].bias.copy_(torch.tensor(b2,    dtype=torch.float64))

    model = model.double()

    pred = model(X_t)
    loss = nn.MSELoss()(pred, y_t)
    loss.backward()

    return {
        "loss": loss.item(),
        "W1_grad": model[0].weight.grad.numpy().T.copy(),
        "b1_grad": model[0].bias.grad.numpy().copy(),
        "W2_grad": model[2].weight.grad.numpy().T.copy(),
        "b2_grad": model[2].bias.grad.numpy().copy(),
    }


def compare(yours, pytorch, tolerance=1e-5):
    all_passed = True
    keys = ["loss", "W1_grad", "b1_grad", "W2_grad", "b2_grad"]

    for key in keys:
        a = np.atleast_1d(np.array(yours[key]))
        b = np.atleast_1d(np.array(pytorch[key]))

        max_diff = np.max(np.abs(a - b))
        status = "PASS" if max_diff < tolerance else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"{key:12s} | max absolute diff: {max_diff:.2e} | {status}")

    return all_passed


def main():
    np.random.seed(7)

    X_data = np.linspace(-np.pi, np.pi, 20).reshape(-1, 1)
    y_data = np.sin(X_data)

    # Shared random weights — same for both engines
    W1 = np.random.randn(1, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4, 1)
    b2 = np.random.randn(1)

    yours   = run_your_engine(X_data, y_data, W1, b1, W2, b2)
    pytorch = run_pytorch(X_data, y_data, W1, b1, W2, b2)

    print("Comparing your engine vs PyTorch...")
    print("-" * 55)
    passed = compare(yours, pytorch)
    print("-" * 55)

    if passed:
        print("All values match PyTorch. Engine is externally verified.")
    else:
        print("Mismatch detected. Review the failing parameter.")


if __name__ == "__main__":
    main()