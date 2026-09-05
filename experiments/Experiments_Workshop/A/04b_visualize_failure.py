# experiments/04b_visualize_failure.py
import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


def build_model(hidden_layers):
    layer_list = []
    in_size = 1
    for h in hidden_layers:
        layer_list.append(Dense(in_size, h))
        layer_list.append(Tanh())
        in_size = h
    layer_list.append(Dense(in_size, 1))
    return Sequential(layer_list)


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

    X_data = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    X = Tensor(X_data)

    hard_fn = lambda x: x * np.sin(x) + np.cos(x**2)
    y_data = hard_fn(X_data)

    architectures = {
        "Wide-Shallow [1->166->1]":         [166],
        "Deep-Narrow  [1->15->14->15->1]":  [15, 14, 15],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (name, hidden) in enumerate(architectures.items()):
        np.random.seed(42)
        model = build_model(hidden)
        losses = train(model, X, y_data, epochs=3000)
        pred = model.forward(X).data

        print(f"{name} | Final MSE: {losses[-1]:.6e}")

        # Top row: prediction vs truth
        axes[0][col].plot(X_data, y_data, 'k-', linewidth=2, label='True function')
        axes[0][col].plot(X_data, pred, 'r--', linewidth=2, label='Network prediction')
        axes[0][col].set_title(f"{name}\nFinal MSE: {losses[-1]:.2e}", fontsize=10)
        axes[0][col].set_xlabel("x")
        axes[0][col].set_ylabel("y")
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True)

        # Bottom row: loss curve
        axes[1][col].plot(losses)
        axes[1][col].set_yscale("log")
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("MSE Loss (log scale)")
        axes[1][col].set_title(f"Training curve — {name}", fontsize=10)
        axes[1][col].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Why Does Wide-Shallow Fail on x*sin(x) + cos(x^2)?\nSame Parameter Budget (~500), Different Shape",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_04b_failure_visualization.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()