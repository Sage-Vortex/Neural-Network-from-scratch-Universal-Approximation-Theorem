# visualization/loss_landscape_3d.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential
from network.optimizer import Adam


def build_model():
    return Sequential([
        Dense(1, 8),
        Tanh(),
        Dense(8, 1)
    ])


def train_model(model, X, y_data, epochs=2000, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        pred = model.forward(X)
        loss = mse_loss(pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def compute_loss(model, X, y_data):
    pred = model.forward(X)
    return mse_loss(pred, y_data).data


def get_random_directions(model):
    """
    Two random unit vectors in parameter space.
    Filter normalization: each direction vector is normalized
    layer-by-layer to match the scale of the corresponding parameter.
    This is the Li et al. (2018) method — without it, directions with
    large-norm layers dominate and the landscape looks artificially flat.
    """
    params = model.parameters()
    d1, d2 = [], []

    for p in params:
        v1 = np.random.randn(*p.data.shape)
        v2 = np.random.randn(*p.data.shape)

        # normalize each direction to match parameter norm (filter normalization)
        v1 = v1 / (np.linalg.norm(v1) + 1e-8) * np.linalg.norm(p.data)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8) * np.linalg.norm(p.data)

        d1.append(v1)
        d2.append(v2)

    return d1, d2


def set_params(model, center_params, d1, d2, alpha, beta):
    """Perturb model weights: theta = theta* + alpha*d1 + beta*d2"""
    for i, p in enumerate(model.parameters()):
        p.data = center_params[i] + alpha * d1[i] + beta * d2[i]


def main():
    np.random.seed(42)

    # --- Data ---
    X_data = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y_data = np.sin(X_data)
    X = Tensor(X_data)

    # --- Train to find a good minimum ---
    print("Training model to find loss minimum...")
    model = build_model()
    train_model(model, X, y_data)

    center_loss = compute_loss(model, X, y_data)
    print(f"Trained model loss (center): {center_loss:.6f}")

    # --- Save trained weights as the center point ---
    center_params = [p.data.copy() for p in model.parameters()]

    # --- Random directions in parameter space ---
    d1, d2 = get_random_directions(model)

    # --- Grid of perturbations ---
    grid_size = 30
    alphas = np.linspace(-1.0, 1.0, grid_size)
    betas  = np.linspace(-1.0, 1.0, grid_size)

    loss_grid = np.zeros((grid_size, grid_size))

    print(f"Computing loss landscape ({grid_size}x{grid_size} grid)...")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_params(model, center_params, d1, d2, alpha, beta)
            loss_grid[i, j] = compute_loss(model, X, y_data)

        if i % 5 == 0:
            print(f"  Row {i+1}/{grid_size} done")

    # Restore trained weights
    for i, p in enumerate(model.parameters()):
        p.data = center_params[i]

    # --- Plot ---
    A, B = np.meshgrid(alphas, betas)
    Z = loss_grid.T  # transpose to align axes

    fig = plt.figure(figsize=(18, 6))

    # Plot 1: 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(A, B, Z, cmap=cm.viridis, alpha=0.85,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel("Direction 1 (alpha)")
    ax1.set_ylabel("Direction 2 (beta)")
    ax1.set_zlabel("MSE Loss")
    ax1.set_title("Loss Landscape (3D Surface)")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Plot 2: contour map
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(A, B, Z, levels=40, cmap=cm.viridis)
    ax2.contour(A, B, Z, levels=40, colors='white', alpha=0.2, linewidths=0.5)
    ax2.scatter([0], [0], color='red', s=80, zorder=5, label='Trained minimum')
    ax2.set_xlabel("Direction 1 (alpha)")
    ax2.set_ylabel("Direction 2 (beta)")
    ax2.set_title("Loss Landscape (Contour)")
    ax2.legend()
    fig.colorbar(contour, ax=ax2)

    # Plot 3: 1D slice through center
    ax3 = fig.add_subplot(133)
    center_idx = grid_size // 2
    ax3.plot(alphas, loss_grid[:, center_idx], linewidth=2, label='Slice along d1 (beta=0)')
    ax3.plot(betas,  loss_grid[center_idx, :], linewidth=2, linestyle='--', label='Slice along d2 (alpha=0)')
    ax3.axvline(0, color='red', linestyle=':', alpha=0.7, label='Trained minimum')
    ax3.set_xlabel("Perturbation magnitude")
    ax3.set_ylabel("MSE Loss")
    ax3.set_title("1D Loss Slices Through Minimum")
    ax3.legend(fontsize=8)
    ax3.grid(True)

    plt.suptitle(
        "Loss Landscape Visualization — Trained Neural Network (sin(x) approximation)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results_loss_landscape.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\nSaved to {save_path}")
    print(f"Min loss on grid: {loss_grid.min():.6f} at center (trained minimum)")
    print(f"Max loss on grid: {loss_grid.max():.6f} at edges (perturbed weights)")


if __name__ == "__main__":
    main()