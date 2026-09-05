# experiments/07_training_budget.py
import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


ARCHITECTURES = {
    "Wide-Shallow [1->166->1]":               [166],
    "Deep-Narrow  [1->15->14->15->1]":        [15, 14, 15],
}

FUNCTIONS = {
    "sin(x)":               lambda x: np.sin(x),
    "x*sin(x)+cos(x^2)":   lambda x: x * np.sin(x) + np.cos(x**2),
}

EPOCHS = 15000
LOG_EVERY = 500
THRESHOLD = 1e-4


def build_model(hidden_layers):
    layer_list = []
    in_size = 1
    for h in hidden_layers:
        layer_list.append(Dense(in_size, h))
        layer_list.append(Tanh())
        in_size = h
    layer_list.append(Dense(in_size, 1))
    return Sequential(layer_list)


def train(model, X, y_data, epochs, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
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

    all_losses = {}

    for fn_name, fn in FUNCTIONS.items():
        y_data = fn(X_data)
        all_losses[fn_name] = {}
        print(f"\n--- {fn_name} ---")

        for arch_name, hidden in ARCHITECTURES.items():
            model = build_model(hidden)
            losses = train(model, X, y_data, epochs=EPOCHS)
            all_losses[fn_name][arch_name] = losses

            # Print at checkpoints
            for epoch in range(LOG_EVERY - 1, EPOCHS, LOG_EVERY):
                status = "PASS" if losses[epoch] < THRESHOLD else ""
                print(f"  {arch_name} | epoch {epoch+1:6d} | "
                      f"MSE: {losses[epoch]:.4e} {status}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, fn_name in enumerate(FUNCTIONS):
        for col, arch_name in enumerate(ARCHITECTURES):
            losses = all_losses[fn_name][arch_name]
            ax = axes[row][col]

            ax.plot(losses, linewidth=1.5, color='steelblue')
            ax.axhline(THRESHOLD, color='red', linestyle='--',
                       alpha=0.7, label=f'Threshold ({THRESHOLD})')

            # mark first time threshold is crossed
            crossed = next((i for i, l in enumerate(losses)
                           if l < THRESHOLD), None)
            if crossed is not None:
                ax.axvline(crossed, color='green', linestyle=':',
                           alpha=0.8, label=f'First PASS @ epoch {crossed}')

            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE (log scale)")
            ax.set_title(f"{arch_name}\nTarget: {fn_name}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Experiment D: Does More Training Time Fix the Plateau?\n"
        "15,000 Epochs, Adam, ~500 params each",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_07_training_budget.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    # --- Summary ---
    print("\n" + "=" * 65)
    print("SUMMARY: Final MSE at 15,000 epochs")
    print("=" * 65)
    for fn_name in FUNCTIONS:
        for arch_name in ARCHITECTURES:
            losses = all_losses[fn_name][arch_name]
            final = losses[-1]
            crossed = next((i for i, l in enumerate(losses)
                           if l < THRESHOLD), None)
            crossed_str = f"epoch {crossed}" if crossed else "Never"
            print(f"  {arch_name:35s} | {fn_name:25s} | "
                  f"Final MSE: {final:.4e} | First PASS: {crossed_str}")


if __name__ == "__main__":
    main()