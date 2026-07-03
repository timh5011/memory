"""Empirical ergodicity and mixing checks for training dynamics.

Two diagnostics, directly motivated by the theory in doc/PHILOSOPHY.md:

1. Birkhoff check — for several random initializations at the same learning
   rate, plot the running time-average of the loss observable. If the system
   were ergodic, all initializations would converge to the same value (time
   average = ensemble average). Seed-dependent limits are direct evidence of
   broken ergodicity (multiple basins of attraction), which is the expected
   result at small lr — and the interesting question is whether large lr
   restores mixing.

2. Mixing proxy — autocorrelation of post-transient loss fluctuations. Fast
   decay of correlations is the numerical signature of mixing; long-lived
   correlations mean the past keeps influencing the present.
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


def running_mean(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag + 1)
    return np.array([
        np.mean(x[: len(x) - lag] * x[lag:]) / var for lag in range(max_lag + 1)
    ])


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    lrs = [0.05, 0.5]          # stable regime vs near-chaotic regime
    seeds = [0, 1, 2, 3, 4]
    n_steps = 2000
    burn_in = 200
    max_lag = 200

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Is training ergodic? Birkhoff averages and correlation decay",
                 fontsize=14)

    for col, lr in enumerate(lrs):
        print(f"lr={lr}:")
        system = GradientDescentSystem(lr=lr)
        finals = []
        for seed in seeds:
            print(f"  seed={seed} ...")
            traj = system.generate_trajectory(n_steps=n_steps, seed=seed)
            losses = system.loss_series(traj)

            # Birkhoff running time-average (post burn-in)
            post = losses[burn_in:] if len(losses) > burn_in else losses
            rm = running_mean(post)
            finals.append(rm[-1])
            axes[0, col].plot(rm, lw=1.0, label=f"seed {seed}")

            # Autocorrelation of fluctuations
            if len(post) > max_lag + 10:
                ac = autocorrelation(post, max_lag)
                axes[1, col].plot(ac, lw=1.0, label=f"seed {seed}")

        spread = np.max(finals) - np.min(finals)
        print(f"  time-average spread across seeds: {spread:.4f}")

        axes[0, col].set_xlabel(f"Steps after burn-in ({burn_in})")
        axes[0, col].set_ylabel("Running time-average of loss")
        axes[0, col].set_title(f"Birkhoff averages, lr={lr} (spread={spread:.3f})")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].axhline(0, color="gray", ls="--", alpha=0.5)
        axes[1, col].set_xlabel("Lag (steps)")
        axes[1, col].set_ylabel("Loss autocorrelation")
        axes[1, col].set_title(f"Correlation decay (mixing proxy), lr={lr}")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    out = results_dir / "ergodicity_check.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
