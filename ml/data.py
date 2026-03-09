"""Synthetic spiral dataset for classification experiments."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def make_spirals(n_samples=1500, n_classes=3, noise=0.5, seed=42):
    """Generate interleaved spiral dataset.

    Parameters
    ----------
    n_samples : int
        Total samples (split 2:1 train:test).
    n_classes : int
        Number of spiral arms.
    noise : float
        Standard deviation of Gaussian noise added to coordinates.
    seed : int
        Random seed.

    Returns
    -------
    X_train, y_train, X_test, y_test : torch.Tensor
        Float32 tensors on CPU.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // n_classes

    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)

    for j in range(n_classes):
        ix = range(n_per_class * j, n_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(j * 2 * np.pi / n_classes,
                        j * 2 * np.pi / n_classes + 2.5 * np.pi,
                        n_per_class)
        t += rng.normal(0, noise * 0.1, n_per_class)
        X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        X[ix] += rng.normal(0, noise * 0.05, (n_per_class, 2))
        y[ix] = j

    # Shuffle
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    # Split
    n_train = int(len(y) * 2 / 3)
    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    X_test = torch.tensor(X[n_train:], dtype=torch.float32)
    y_test = torch.tensor(y[n_train:], dtype=torch.long)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_spirals()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {torch.unique(y_train).tolist()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label in torch.unique(y_train):
        mask = y_train == label
        ax1.scatter(X_train[mask, 0], X_train[mask, 1], s=5, alpha=0.6,
                    label=f"Class {label.item()}")
    ax1.set_title("Training data")
    ax1.legend()
    ax1.set_aspect("equal")

    for label in torch.unique(y_test):
        mask = y_test == label
        ax2.scatter(X_test[mask, 0], X_test[mask, 1], s=5, alpha=0.6,
                    label=f"Class {label.item()}")
    ax2.set_title("Test data")
    ax2.legend()
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("results/spirals.png", dpi=150, bbox_inches="tight")
    print("Saved results/spirals.png")
