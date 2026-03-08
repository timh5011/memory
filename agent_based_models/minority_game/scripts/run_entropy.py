"""Detailed entropy analysis of the Minority Game at different memory lengths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Allow importing from ergodic_systems
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ergodic_systems"))

from sim import MinorityGameModel, MinorityGameConfig
from entropy.block_counting import empirical_block_distribution, shannon_entropy


def run_and_get_outcomes(memory_length: int, n_steps: int, seed: int) -> np.ndarray:
    """Run a Minority Game and return the outcome sequence after burn-in."""
    config = MinorityGameConfig(
        memory_length=memory_length, n_steps=n_steps, seed=seed
    )
    model = MinorityGameModel(config)
    for _ in range(n_steps):
        model.step()
    return np.array(model.outcomes)


def compute_entropy_rates(outcomes: np.ndarray, burn_in: int, k_max: int):
    """Compute block entropy estimates from the post-burn-in outcome sequence."""
    seq = outcomes[burn_in:]
    ks = list(range(1, k_max + 1))
    H_k = []
    for k in ks:
        dist = empirical_block_distribution(seq, k)
        H_k.append(shannon_entropy(dist))
    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]
    return ks, H_k, h_rate, h_diff


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    memory_lengths = [3, 6, 9]  # crowded, critical, uncrowded
    n_steps = 5000
    burn_in = 500
    k_max = 12
    N = 301

    colors = {"3": "crimson", "6": "steelblue", "9": "mediumseagreen"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Minority Game – KS Entropy Analysis (N=301)", fontsize=14)

    for M in memory_lengths:
        alpha = 2 ** M / N
        label = f"M={M} (α={alpha:.3f})"
        color = colors[str(M)]

        print(f"Running M={M} (α={alpha:.3f})...")
        outcomes = run_and_get_outcomes(M, n_steps, seed=42)
        ks, H_k, h_rate, h_diff = compute_entropy_rates(outcomes, burn_in, k_max)

        print(f"  H(k)/k converged to {h_rate[-1]:.4f} bits")
        print(f"  h_diff converged to {h_diff[-1]:.4f} bits")

        axes[0].plot(ks, h_rate, "o-", color=color, label=label, markersize=4)
        axes[1].plot(ks, h_diff, "s-", color=color, label=label, markersize=4)

    # Reference line at 1.0 bit (random coin flip)
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Random (1 bit)")
    axes[0].set_xlabel("Block length k")
    axes[0].set_ylabel("H(k)/k (bits)")
    axes[0].set_title("Entropy Rate H(k)/k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Random (1 bit)")
    axes[1].set_xlabel("Block length k")
    axes[1].set_ylabel("H(k) − H(k−1) (bits)")
    axes[1].set_title("Conditional Entropy h(k)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "entropy_analysis.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
