# experiments/04_width_vs_depth.py
import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


# --- Architectures with approximately equal parameter counts (~320 params) ---
ARCHITECTURES = {
    "Wide-Shallow [1->166->1]":               [166],
    "Balanced     [1->20->20->1]":            [20, 20],
    "Deep-Narrow  [1->15->14->15->1]":        [15, 14, 15],
    "Very-Deep    [1->12->12->12->12->1]":    [12, 12, 12, 12],
}


def count_params(hidden_layers):
    layers = [1] + hidden_layers + [1]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i+1] + layers[i+1]
    return total


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


# --- Target functions ---
FUNCTIONS = {
    "sin(x)":               lambda x: np.sin(x),
    "x*sin(x)":             lambda x: x * np.sin(x),
    "x*sin(x)+cos(x^2)":   lambda x: x * np.sin(x) + np.cos(x**2),
    "sin(3x)*exp+0.1x":    lambda x: np.sin(3*x) * np.exp(-0.1*x**2) + 0.1*x,
}


def main():
    np.random.seed(42)

    X_data = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    X = Tensor(X_data)
    epochs = 3000

    # Print parameter counts first
    print("Architecture parameter counts:")
    print("-" * 45)
    for name, hidden in ARCHITECTURES.items():
        print(f"{name} | params: {count_params(hidden)}")
    print("-" * 45)
    print()

    # Results table: architecture x function -> final MSE
    results = {arch: {} for arch in ARCHITECTURES}
    all_losses = {arch: {} for arch in ARCHITECTURES}

    for arch_name, hidden_layers in ARCHITECTURES.items():
        for fn_name, fn in FUNCTIONS.items():
            y_data = fn(X_data)
            np.random.seed(42)
            model = build_model(hidden_layers)
            losses = train(model, X, y_data, epochs=epochs)
            final_mse = losses[-1]
            results[arch_name][fn_name] = final_mse
            all_losses[arch_name][fn_name] = losses
            print(f"{arch_name} | {fn_name:25s} | MSE: {final_mse:.2e}")

    # --- Plot 1: Final MSE heatmap (architecture x function) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    arch_names = list(ARCHITECTURES.keys())
    fn_names = list(FUNCTIONS.keys())

    mse_matrix = np.array([
        [results[a][f] for f in fn_names]
        for a in arch_names
    ])

    im = axes[0].imshow(np.log10(mse_matrix + 1e-10), cmap='viridis_r', aspect='auto')
    axes[0].set_xticks(range(len(fn_names)))
    axes[0].set_xticklabels(fn_names, rotation=20, ha='right', fontsize=9)
    axes[0].set_yticks(range(len(arch_names)))
    axes[0].set_yticklabels([a.split('[')[0].strip() for a in arch_names], fontsize=9)
    axes[0].set_title("Final MSE (log10 scale)\nDarker = Better")
    fig.colorbar(im, ax=axes[0], label="log10(MSE)")

    # Annotate cells with actual MSE values
    for i in range(len(arch_names)):
        for j in range(len(fn_names)):
            axes[0].text(j, i, f"{mse_matrix[i,j]:.1e}",
                        ha='center', va='center', fontsize=7, color='white')

    # --- Plot 2: Loss curves for sin(x) across architectures ---
    for arch_name in arch_names:
        losses = all_losses[arch_name]["sin(x)"]
        label = arch_name.split('[')[0].strip()
        axes[1].plot(losses, label=label, linewidth=1.5)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (log scale)")
    axes[1].set_title("Training Curves for sin(x)\nAll Architectures, Equal Parameter Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Width vs Depth: Approximation Quality Under Fixed Parameter Budget",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_04_width_vs_depth_2.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("\nWidth vs Depth study complete.")
    print("\nSummary — best architecture per function:")
    for fn_name in fn_names:
        best = min(arch_names, key=lambda a: results[a][fn_name])
        print(f"  {fn_name:25s} -> {best.split('[')[0].strip()} "
              f"(MSE: {results[best][fn_name]:.2e})")


if __name__ == "__main__":
    main()