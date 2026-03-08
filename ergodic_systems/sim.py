"""Demo: validate KS entropy estimation on Bernoulli shifts."""

import matplotlib.pyplot as plt
from bernoulli_shift import BernoulliShift
from ks_entropy import block_entropy_estimates, plot_entropy_convergence


def main():
    fair = BernoulliShift([0.5, 0.5])
    biased = BernoulliShift([0.9, 0.1])

    print("Computing block entropy estimates...")
    ks_f, H_f, hr_f, hd_f = block_entropy_estimates(fair, n_steps=100_000, k_max=12, seed=42)
    ks_b, H_b, hr_b, hd_b = block_entropy_estimates(biased, n_steps=100_000, k_max=12, seed=42)

    h_fair = fair.analytical_ks_entropy()
    h_biased = biased.analytical_ks_entropy()

    # Print validation
    print(f"\nFair coin  [0.5, 0.5]:")
    print(f"  Analytical H(p) = {h_fair:.4f} bits")
    print(f"  Estimated H(k)/k at k=1: {hr_f[0]:.4f}, k=6: {hr_f[5]:.4f}, k=12: {hr_f[11]:.4f}")

    print(f"\nBiased coin [0.9, 0.1]:")
    print(f"  Analytical H(p) = {h_biased:.4f} bits")
    print(f"  Estimated H(k)/k at k=1: {hr_b[0]:.4f}, k=6: {hr_b[5]:.4f}, k=12: {hr_b[11]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks_f, hr_f, 'o-', color='steelblue', label='Fair coin H(k)/k')
    ax.axhline(h_fair, linestyle='--', color='steelblue', alpha=0.7,
               label=f'Fair analytical H = {h_fair:.4f}')
    ax.plot(ks_b, hr_b, 's-', color='coral', label='Biased coin H(k)/k')
    ax.axhline(h_biased, linestyle='--', color='coral', alpha=0.7,
               label=f'Biased analytical H = {h_biased:.4f}')

    ax.set_xlabel('Block length k')
    ax.set_ylabel('H(k)/k  (bits)')
    ax.set_title('Bernoulli Shift: Entropy Rate vs Block Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = 'entropy_rate.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
