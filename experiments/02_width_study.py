import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


def build_model(n_neurons):
    return Sequential([
        Dense(1, n_neurons),
        Tanh(),
        Dense(n_neurons, 1)
    ])


def train(model, X, y_data, epochs=2000, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        pred = model.forward(X)
        loss = mse_loss(pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.data


def main():
    np.random.seed(42)

    X_data = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    y_data = np.sin(X_data)
    X = Tensor(X_data)

    neuron_counts = [2, 4, 8, 16, 32, 64, 128]
    final_losses = []

    for n in neuron_counts:
        model = build_model(n)
        loss = train(model, X, y_data)
        final_losses.append(loss)
        print(f"Neurons: {n:4d} | Final MSE: {loss:.8f}")

    # --- Plot 1: approximation error vs neuron count ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(neuron_counts, final_losses, marker='o', linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of neurons (log scale)")
    axes[0].set_ylabel("Final MSE (log scale)")
    axes[0].set_title("UAT: Approximation Error vs Neuron Count")
    axes[0].grid(True, which='both', linestyle='--', alpha=0.7)

    # --- Plot 2: visual approximation for each neuron count ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(neuron_counts)))
    axes[1].plot(X_data, y_data, 'k-', linewidth=2, label='True sin(x)')

    for i, n in enumerate(neuron_counts):
        model = build_model(n)
        train(model, X, y_data)
        pred = model.forward(X).data
        axes[1].plot(X_data, pred, color=colors[i],
                     linestyle='--', alpha=0.8, label=f'{n} neurons')

    axes[1].set_title("Approximation Quality vs Neuron Count")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "results_02_width_study.png"), dpi=150)
    plt.show()

    print("\nWidth study complete.")
    print(f"Best result: {min(final_losses):.8f} MSE with "
          f"{neuron_counts[final_losses.index(min(final_losses))]} neurons")


if __name__ == "__main__":
    main()