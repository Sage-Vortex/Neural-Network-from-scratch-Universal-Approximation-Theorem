# experiments/06_min_width_per_function.py
import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


FUNCTIONS = {
    "sin(x)":               lambda x: np.sin(x),
    "x*sin(x)":             lambda x: x * np.sin(x),
    "x*sin(x)+cos(x^2)":   lambda x: x * np.sin(x) + np.cos(x**2),
    "sin(3x)*exp+0.1x":    lambda x: np.sin(3*x) * np.exp(-0.1*x**2) + 0.1*x,
}

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
THRESHOLD = 1e-4
EPOCHS = 3000


def build_model(width):
    return Sequential([
        Dense(1, width),
        Tanh(),
        Dense(width, 1)
    ])


def train(model, X, y_data, epochs=EPOCHS, lr=0.01):
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
    X = Tensor(X_data)

    results = {}        # results[fn_name][width] = final MSE
    min_width = {}       # min_width[fn_name] = smallest width hitting threshold

    for fn_name, fn in FUNCTIONS.items():
        y_data = fn(X_data)
        results[fn_name] = {}
        min_width[fn_name] = None

        print(f"\n--- {fn_name} ---")
        for width in WIDTHS:
            model = build_model(width)
            final_mse = train(model, X, y_data)
            results[fn_name][width] = final_mse

            status = "PASS" if final_mse < THRESHOLD else ""
            print(f"  width={width:4d} | MSE: {final_mse:.4e} {status}")

            if final_mse < THRESHOLD and min_width[fn_name] is None:
                min_width[fn_name] = width

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for fn_name in FUNCTIONS:
        ws = list(results[fn_name].keys())
        mses = list(results[fn_name].values())
        axes[0].plot(ws, mses, marker='o', label=fn_name, linewidth=2)

    axes[0].axhline(THRESHOLD, color='red', linestyle='--', alpha=0.6, label=f'Threshold ({THRESHOLD})')
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Width (neurons, log scale)")
    axes[0].set_ylabel("Final MSE (log scale)")
    axes[0].set_title("Approximation Error vs Width, by Function")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

    # Bar chart: minimum width needed per function
    fn_names = list(FUNCTIONS.keys())
    min_widths = [min_width[f] if min_width[f] is not None else max(WIDTHS) * 2 for f in fn_names]
    colors = ['green' if min_width[f] is not None else 'red' for f in fn_names]

    bars = axes[1].bar(range(len(fn_names)), min_widths, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(fn_names)))
    axes[1].set_xticklabels(fn_names, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel(f"Minimum width to reach MSE < {THRESHOLD}")
    axes[1].set_title("Minimum Width Required per Function")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar, fn_name in zip(bars, fn_names):
        label = str(min_width[fn_name]) if min_width[fn_name] is not None else "Never"
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     label, ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        "Experiment C: Does Function Complexity Require More Neurons?",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_06_min_width_per_function.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY: Minimum width to reach MSE < 1e-4")
    print("=" * 60)
    for fn_name in FUNCTIONS:
        w = min_width[fn_name]
        print(f"  {fn_name:25s} -> {w if w is not None else 'Never reached threshold'}")


if __name__ == "__main__":
    main()