#The actual UAT demo


import numpy as np
import matplotlib.pyplot as plt
import os
from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam

def main():
    np.random.seed(42)

    # ---- Generate data: y = sin(x) ----
    X_data = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    y_data = np.sin(X_data)

    X = Tensor(X_data)

    # ---- Build model ----
    model = Sequential([
        Dense(1, 32),
        Tanh(),
        Dense(32, 32),
        Tanh(),
        Dense(32, 1)
    ])

    optimizer = Adam(model.parameters(), lr=0.01)


    # ---- Train ----
    losses = []
    epochs = 2000

    for epoch in range(epochs):
        pred = model.forward(X)
        loss = mse_loss(pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data)

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")

    print(f"Final loss: {losses[-1]:.6f}")

    # ---- Plot results ----
    final_pred = model.forward(X).data

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(X_data, y_data, label="True sin(x)", linewidth=2)
    axes[0].plot(X_data, final_pred, label="Network approximation", linestyle='--', linewidth=2)
    axes[0].set_title("Universal Approximation: Learning sin(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(losses)
    axes[1].set_yscale("log")
    axes[1].set_title("Training Loss (log scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].grid(True)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_01_sin_WithAdam.png")

    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()