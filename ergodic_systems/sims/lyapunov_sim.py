"""Validate Lyapunov exponent estimation on the logistic map (r=4).

Compares two methods:
  1. Perturbation (Benettin's algorithm) — generic, works for any system with
     metric() and perturb()
  2. Jacobian — exact for 1D maps, averages ln|f'(x)| along a trajectory

The Bernoulli shift is excluded: as an i.i.d. symbolic process it has no
continuous state to perturb, so the Lyapunov exponent approach does not apply.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from systems import LogisticMap
from entropy import lyapunov_perturbation, lyapunov_jacobian, plot_lyapunov_convergence

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lm = LogisticMap(r=4.0)
    h_analytical_bits = lm.analytical_ks_entropy()  # 1.0 bits
    h_analytical_nats = np.log(2)  # ln(2) nats

    n_steps = 500_000

    # --- Method 1: Perturbation ---
    print("Perturbation method (Benettin's algorithm)...")
    lyap_p_nats, lyap_p_bits, running_p = lyapunov_perturbation(
        lm, n_steps=n_steps, delta=1e-8, renorm_interval=1, seed=42
    )
    print(f"  λ = {lyap_p_nats:.6f} nats/step = {lyap_p_bits:.6f} bits/step")
    print(f"  Analytical: {h_analytical_nats:.6f} nats = {h_analytical_bits:.6f} bits")

    # --- Method 2: Jacobian ---
    print("\nJacobian method...")
    lyap_j_nats, lyap_j_bits, running_j = lyapunov_jacobian(
        lm, n_steps=n_steps, seed=42
    )
    print(f"  λ = {lyap_j_nats:.6f} nats/step = {lyap_j_bits:.6f} bits/step")
    print(f"  Analytical: {h_analytical_nats:.6f} nats = {h_analytical_bits:.6f} bits")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Logistic Map (r=4): Lyapunov Exponent Estimation", fontsize=14)

    plot_lyapunov_convergence(
        running_p, analytical_nats=h_analytical_nats,
        method_label="Perturbation", ax=ax1
    )
    ax1.set_title("Perturbation Method (Benettin)")

    plot_lyapunov_convergence(
        running_j, analytical_nats=h_analytical_nats,
        method_label="Jacobian", ax=ax2
    )
    ax2.set_title("Jacobian Method")

    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'logistic_lyapunov.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
