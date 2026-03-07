"""
Bernoulli Shift — Block Distribution Convergence

Demonstrates how KS entropy controls the rate at which empirical block
distributions converge to the true product measure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.shift import (
    generate_sequence,
    empirical_block_distribution,
    true_block_distribution,
    kl_divergence,
)
from sim.entropy import shannon_entropy

# --- Configuration ---

DISTRIBUTIONS = {
    "Binary peaked [0.95, 0.05]": np.array([0.95, 0.05]),
    "Binary skewed [0.8, 0.2]": np.array([0.8, 0.2]),
    "Binary mild [0.65, 0.35]": np.array([0.65, 0.35]),
    "Binary uniform [0.5, 0.5]": np.array([0.5, 0.5]),
    "Quaternary uniform [1/4]^4": np.array([0.25, 0.25, 0.25, 0.25]),
}

BLOCK_LENGTHS = [1, 2, 3, 4]
SEQ_LENGTHS = np.unique(np.logspace(2, 5, num=20).astype(int))
N_SEEDS = 10
SEED_BASE = 42

# --- Run ---

def run():
    results = {}  # (dist_name, k) -> list of (N, mean_dkl)

    for dist_name, dist in DISTRIBUTIONS.items():
        h = shannon_entropy(dist)
        alphabet_size = len(dist)
        print(f"{dist_name}  h={h:.4f}")

        for k in BLOCK_LENGTHS:
            true_dist = true_block_distribution(dist, k)
            curve = []

            for N in SEQ_LENGTHS:
                dkls = []
                for s in range(N_SEEDS):
                    seq = generate_sequence(alphabet_size, dist, int(N), SEED_BASE + s)
                    emp = empirical_block_distribution(seq, k)
                    dkl = kl_divergence(emp, true_dist)
                    dkls.append(dkl)
                curve.append((int(N), np.mean(dkls)))

            results[(dist_name, k)] = curve
            Ns = [c[0] for c in curve]
            dkls = [c[1] for c in curve]
            print(f"  k={k}  D_KL range: [{min(dkls):.6f}, {max(dkls):.6f}]")

    return results


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("KL Divergence of Empirical Block Distribution vs Sequence Length", fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(DISTRIBUTIONS)))
    dist_names = list(DISTRIBUTIONS.keys())

    for idx, k in enumerate(BLOCK_LENGTHS):
        ax = axes[idx // 2][idx % 2]
        ax.set_title(f"Block length k = {k}")

        for i, dist_name in enumerate(dist_names):
            h = shannon_entropy(DISTRIBUTIONS[dist_name])
            curve = results[(dist_name, k)]
            Ns = [c[0] for c in curve]
            dkls = [c[1] for c in curve]
            label = f"h={h:.3f} {dist_name}"
            ax.loglog(Ns, dkls, "o-", color=colors[i], label=label, markersize=3, linewidth=1.2)

        ax.set_xlabel("Sequence length N")
        ax.set_ylabel("D_KL(empirical || true)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "block_convergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    results = run()
    plot(results)
