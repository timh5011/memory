import numpy as np
import matplotlib.pyplot as plt
from bernoulli_shift import empirical_block_distribution


def shannon_entropy(dist):
    """Compute Shannon entropy H = -sum q log2(q) from a distribution dict."""
    h = 0.0
    for q in dist.values():
        if q > 0:
            h -= q * np.log2(q)
    return h


def block_entropy_estimates(system, n_steps=100_000, k_max=12, seed=42):
    """Compute block entropy estimates for a symbolic ergodic system.

    Returns (ks, H_k, h_rate, h_diff) where:
        ks:     [1, ..., k_max]
        H_k:    block entropy H(k) for each k
        h_rate: H(k)/k for each k
        h_diff: H(k) - H(k-1) for each k (H(0) defined as 0)
    """
    trajectory = system.generate_trajectory(n_steps=n_steps, seed=seed)
    symbolic = system.symbolize(trajectory)

    ks = list(range(1, k_max + 1))
    H_k = []
    for k in ks:
        dist = empirical_block_distribution(symbolic, k)
        H_k.append(shannon_entropy(dist))

    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]

    return ks, H_k, h_rate, h_diff


def plot_entropy_convergence(ks, h_rate, analytical_h=None, label=None, ax=None, save_path=None):
    """Plot H(k)/k vs k, optionally with analytical entropy line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, h_rate, 'o-', label=label or 'H(k)/k')

    if analytical_h is not None:
        ax.axhline(analytical_h, linestyle='--', alpha=0.7,
                   label=f'Analytical H = {analytical_h:.4f}')

    ax.set_xlabel('Block length k')
    ax.set_ylabel('H(k)/k  (bits)')
    ax.set_title('Entropy Rate Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax
