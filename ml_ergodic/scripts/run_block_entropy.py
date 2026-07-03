"""Block-counting entropy rate of training dynamics vs learning rate.

For each learning rate: generate a weight trajectory, symbolize the loss
observable into quantile bins, compute block entropies, and plot how the
entropy rate depends on lr. This is the symbolic-dynamics counterpart to the
Lyapunov/Pesin estimates in basic/ml — two independent routes to the same
quantity, exactly as validated on the logistic map in basic/ergodic_systems.

Expected picture: small lr -> loss decreases smoothly -> nearly deterministic
symbol sequence -> h ~ 0. Large lr -> oscillatory/chaotic loss -> positive
entropy rate. The interesting question is what happens in between, at the
learning rates that train best.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system import GradientDescentSystem  # noqa: E402
# system.py has already put basic/ergodic_systems on sys.path
from entropy.block_counting import empirical_block_distribution, shannon_entropy  # noqa: E402


def block_entropies(symbols, k_max):
    ks = list(range(1, k_max + 1))
    H_k = [shannon_entropy(empirical_block_distribution(symbols, k)) for k in ks]
    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]
    return ks, H_k, h_rate, h_diff


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    lrs = [0.005, 0.02, 0.05, 0.1, 0.3, 0.7]
    n_steps = 2000
    burn_in = 200      # drop the initial monotone descent transient
    k_max = 8
    seed = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training dynamics — block-counting entropy of the loss observable",
                 fontsize=14)
    cmap = plt.get_cmap("viridis")
    final_h = []

    for i, lr in enumerate(lrs):
        color = cmap(i / max(len(lrs) - 1, 1))
        print(f"lr={lr}: generating {n_steps}-step trajectory...")
        system = GradientDescentSystem(lr=lr)
        traj = system.generate_trajectory(n_steps=n_steps, seed=seed)
        if len(traj) <= burn_in + 10 * k_max:
            print("  diverged too early for entropy estimation; skipping")
            final_h.append(np.nan)
            continue
        symbols = system.symbolize(traj[burn_in:])
        ks, H_k, h_rate, h_diff = block_entropies(symbols, k_max)
        final_h.append(h_diff[-1])
        print(f"  h(k={k_max}) = {h_diff[-1]:.3f} bits/step")
        axes[0].plot(ks, h_rate, "o-", color=color, ms=4, label=f"lr={lr}")

    axes[0].set_xlabel("Block length k")
    axes[0].set_ylabel("H(k)/k (bits)")
    axes[0].set_title("Entropy rate convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lrs, final_h, "o-", color="crimson")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel(f"h(k={k_max}) (bits/step)")
    axes[1].set_title("Entropy rate vs learning rate")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = results_dir / "block_entropy_vs_lr.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
