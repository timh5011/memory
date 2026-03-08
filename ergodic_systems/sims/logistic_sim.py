"""Validate KS entropy estimation on the logistic map (r=4)."""

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from systems import LogisticMap
from entropy import block_entropy_estimates

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    lm = LogisticMap(r=4.0)

    print("Computing block entropy estimates for logistic map (r=4)...")
    ks, H_k, h_rate, h_diff = block_entropy_estimates(lm, n_steps=500_000, k_max=16, seed=42)

    h_analytical = lm.analytical_ks_entropy()

    print(f"\nLogistic map r=4:")
    print(f"  Analytical H = {h_analytical:.4f} bits")
    print(f"  Estimated H(k)/k at k=1: {h_rate[0]:.4f}, k=8: {h_rate[7]:.4f}, k=16: {h_rate[15]:.4f}")
    print(f"  H(k)-H(k-1) at k=1: {h_diff[0]:.4f}, k=8: {h_diff[7]:.4f}, k=16: {h_diff[15]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # H(k)/k convergence
    ax1.plot(ks, h_rate, 'o-', color='steelblue', label='H(k)/k')
    ax1.axhline(h_analytical, linestyle='--', color='coral', alpha=0.7,
                label=f'Analytical H = {h_analytical:.4f}')
    ax1.set_xlabel('Block length k')
    ax1.set_ylabel('H(k)/k  (bits)')
    ax1.set_title('Logistic Map (r=4): Entropy Rate Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # H(k) - H(k-1) convergence
    ax2.plot(ks, h_diff, 's-', color='steelblue', label='H(k) - H(k-1)')
    ax2.axhline(h_analytical, linestyle='--', color='coral', alpha=0.7,
                label=f'Analytical H = {h_analytical:.4f}')
    ax2.set_xlabel('Block length k')
    ax2.set_ylabel('H(k) - H(k-1)  (bits)')
    ax2.set_title('Logistic Map (r=4): Conditional Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'logistic_entropy_rate.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
