# experiments/08_seed_robustness.py
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
    "Balanced     [1->20->20->1]":            [20, 20],
    "Deep-Narrow  [1->15->14->15->1]":        [15, 14, 15],
    "Very-Deep    [1->12->12->12->12->1]":    [12, 12, 12, 12],
}

FUNCTIONS = {
    "sin(x)":               lambda x: np.sin(x),
    "x*sin(x)+cos(x^2)":   lambda x: x * np.sin(x) + np.cos(x**2),
}

SEEDS = [0, 7, 21, 42, 99]
EPOCHS = 3000
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
    X_data = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    X = Tensor(X_data)

    # results[fn][arch][seed] = final_mse
    results = {fn: {arch: [] for arch in ARCHITECTURES} for fn in FUNCTIONS}

    for fn_name, fn in FUNCTIONS.items():
        y_data = fn(X_data)
        print(f"\n--- {fn_name} ---")

        for arch_name, hidden in ARCHITECTURES.items():
            for seed in SEEDS:
                np.random.seed(seed)
                model = build_model(hidden)
                final_mse = train(model, X, y_data)
                results[fn_name][arch_name].append(final_mse)
                status = "PASS" if final_mse < THRESHOLD else ""
                print(f"  {arch_name:35s} | seed={seed:3d} | "
                      f"MSE: {final_mse:.4e} {status}")

    # --- Compute stats ---
    print("\n" + "=" * 75)
    print("SUMMARY: Mean ± Std over 5 seeds")
    print("=" * 75)

    stats = {}
    for fn_name in FUNCTIONS:
        stats[fn_name] = {}
        print(f"\n{fn_name}")
        for arch_name in ARCHITECTURES:
            vals = np.array(results[fn_name][arch_name])
            mean = vals.mean()
            std = vals.std()
            pass_rate = (vals < THRESHOLD).sum()
            stats[fn_name][arch_name] = (mean, std, pass_rate)
            print(f"  {arch_name:35s} | "
                  f"mean: {mean:.4e} | std: {std:.4e} | "
                  f"pass rate: {pass_rate}/{len(SEEDS)}")

    # --- Plot: mean +/- std bar chart for each function ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    arch_names = list(ARCHITECTURES.keys())
    short_names = [a.split('[')[0].strip() for a in arch_names]
    x = np.arange(len(arch_names))
    width = 0.35

    for col, fn_name in enumerate(FUNCTIONS):
        ax = axes[col]

        means = [stats[fn_name][a][0] for a in arch_names]
        stds  = [stats[fn_name][a][1] for a in arch_names]
        passes = [stats[fn_name][a][2] for a in arch_names]

        bars = ax.bar(x, means, width=0.6, yerr=stds,
                      capsize=5, alpha=0.75,
                      color=['#d62728' if m > THRESHOLD else '#2ca02c'
                             for m in means])

        ax.axhline(THRESHOLD, color='red', linestyle='--',
                   alpha=0.7, label=f'Threshold ({THRESHOLD})')
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel("Final MSE (log scale)")
        ax.set_title(f"Target: {fn_name}\nMean ± Std over {len(SEEDS)} seeds",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # annotate pass rate
        for i, (bar, p) in enumerate(zip(bars, passes)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.5,
                    f"{p}/{len(SEEDS)}", ha='center',
                    va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(
        "Experiment E: Seed Robustness\n"
        "Are Results Consistent Across 5 Random Seeds?",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_08_seed_robustness.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()