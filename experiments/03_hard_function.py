import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


# The target functions — increasingly difficult
def f1(x):
    return np.sin(x)

def f2(x):
    return x * np.sin(x)

def f3(x):
    return x * np.sin(x) + np.cos(x ** 2)

def f4(x):
    return np.sin(3 * x) * np.exp(-0.1 * x ** 2) + 0.1 * x


FUNCTIONS = {
    "sin(x)":                        f1,
    "x·sin(x)":                      f2,
    "x·sin(x) + cos(x²)":            f3,
    "sin(3x)·exp(-0.1x²) + 0.1x":   f4,
}


def build_model():
    return Sequential([
        Dense(1, 64),
        Tanh(),
        Dense(64, 64),
        Tanh(),
        Dense(64, 1)
    ])


def train(model, X, y_data, epochs=3000, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        pred = model.forward(X)
        loss = mse_loss(pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    return losses


def main():
    np.random.seed(42)

    X_data = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
    X = Tensor(X_data)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (name, fn) in enumerate(FUNCTIONS.items()):
        y_data = fn(X_data)

        model = build_model()
        losses = train(model, X, y_data)
        pred = model.forward(X).data

        print(f"{name:35s} | Final MSE: {losses[-1]:.8f}")

        # top row: approximation vs truth
        axes[0][col].plot(X_data, y_data, 'k-', linewidth=2, label='True')
        axes[0][col].plot(X_data, pred, 'r--', linewidth=2, label='Network')
        axes[0][col].set_title(name, fontsize=9)
        axes[0][col].legend(fontsize=7)
        axes[0][col].grid(True)

        # bottom row: loss curve
        axes[1][col].plot(losses)
        axes[1][col].set_yscale("log")
        axes[1][col].set_title(f"Loss curve — {name}", fontsize=9)
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("MSE")
        axes[1][col].grid(True)

    plt.suptitle(
        "Universal Approximation: One Architecture, Four Functions",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "results_03_hard_functions.png"), dpi=150)
    plt.show()

    print("\nHard function study complete.")


if __name__ == "__main__":
    main()