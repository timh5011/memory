"""Largest Lyapunov exponent of training dynamics via Benettin perturbation.

Uses the GENERIC estimator from basic/ergodic_systems (entropy/lyapunov.py)
on GradientDescentSystem — no Hessians, no autodiff tricks, just "perturb the
weights, iterate both copies, measure divergence, renormalize."

This is an independent cross-check of basic/ml's Hessian-vector-product
Lyapunov spectrum: the perturbation estimate here should track the largest
exponent from the HVP method (lambda_max = largest eigenvalue direction of
I - lr*H accumulated along the trajectory). Where basic/ml gives the top-k
spectrum for Pesin sums, this gives only lambda_max — but through a totally
different numerical route, which is exactly what makes agreement meaningful.
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
from entropy.lyapunov import lyapunov_perturbation  # noqa: E402


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    lrs = [0.005, 0.02, 0.05, 0.1, 0.3, 0.7]
    n_steps = 1500
    delta = 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Largest Lyapunov exponent of training (Benettin perturbation)",
                 fontsize=14)
    cmap = plt.get_cmap("viridis")
    lambdas = []

    for i, lr in enumerate(lrs):
        color = cmap(i / max(len(lrs) - 1, 1))
        print(f"lr={lr}: running perturbation estimate ({n_steps} steps)...")
        system = GradientDescentSystem(lr=lr)
        lam_nats, lam_bits, running = lyapunov_perturbation(
            system, n_steps=n_steps, delta=delta, renorm_interval=1, seed=42
        )
        lambdas.append(lam_bits)
        print(f"  lambda_max = {lam_nats:.4f} nats/step = {lam_bits:.4f} bits/step")
        axes[0].plot(running / np.log(2), lw=0.9, color=color, label=f"lr={lr}")

    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_xlabel("Renormalization step")
    axes[0].set_ylabel("Running λ_max (bits/step)")
    axes[0].set_title("Convergence of the running estimate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lrs, lambdas, "o-", color="crimson")
    axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("λ_max (bits/step)")
    axes[1].set_title("λ_max vs learning rate")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = results_dir / "lyapunov_perturbation.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
