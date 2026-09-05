import os
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import SGD, SGDMomentum, Adam


ARCHITECTURES = {
    "Wide-Shallow [1->166->1]":               [166],
    "Balanced     [1->20->20->1]":            [20, 20],
    "Deep-Narrow  [1->15->14->15->1]":        [15, 14, 15],
    "Very-Deep    [1->12->12->12->12->1]":    [12, 12, 12, 12],
}

OPTIMIZERS = {
    "SGD":           lambda params: SGD(params, lr=0.05),
    "SGD+Momentum":  lambda params: SGDMomentum(params, lr=0.05, momentum=0.9),
    "Adam":          lambda params: Adam(params, lr=0.01),
}


def build_model(hidden_layers):
    layer_list = []
    in_size = 1
    for h in hidden_layers:
        layer_list.append(Dense(in_size, h))
        layer_list.append(Tanh())
        in_size = h
    layer_list.append(Dense(in_size, 1))
    return Sequential(layer_list)


def train(model, X, y_data, optimizer_fn, epochs=3000):
    optimizer = optimizer_fn(model.parameters())
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

    results = {}   # results[arch][opt] = final_loss
    all_losses = {}

    for arch_name, hidden in ARCHITECTURES.items():
        results[arch_name] = {}
        all_losses[arch_name] = {}
        for opt_name, opt_fn in OPTIMIZERS.items():
            np.random.seed(42) 
            model = build_model(hidden)
            losses = train(model, X, y_data, opt_fn, epochs=3000)
            results[arch_name][opt_name] = losses[-1]
            all_losses[arch_name][opt_name] = losses
            print(f"{arch_name:35s} | {opt_name:13s} | Final MSE: {losses[-1]:.4e}")

    # --- Plot: grid of architectures x optimizers, loss curves ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (arch_name, hidden) in enumerate(ARCHITECTURES.items()):
        for opt_name in OPTIMIZERS:
            axes[idx].plot(all_losses[arch_name][opt_name], label=opt_name, linewidth=1.5)
        axes[idx].set_yscale("log")
        axes[idx].set_title(arch_name, fontsize=10)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("MSE (log scale)")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Optimizer Comparison Across Architectures\nTarget: x*sin(x) + cos(x^2), ~500 params each",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_05_optimizer_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: Final MSE by Architecture x Optimizer")
    print("=" * 70)
    header = f"{'Architecture':35s}" + "".join(f"{o:>13s}" for o in OPTIMIZERS)
    print(header)
    for arch_name in ARCHITECTURES:
        row = f"{arch_name:35s}"
        for opt_name in OPTIMIZERS:
            row += f"{results[arch_name][opt_name]:>13.2e}"
        print(row)


if __name__ == "__main__":
    main()