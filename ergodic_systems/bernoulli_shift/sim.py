import numpy as np
import matplotlib.pyplot as plt
from bernoulli_shift import make_distribution, generate_sequence, empirical_block_distribution

# --- Parameters ---
N = 100_000       # sequence length (long enough to estimate high-k blocks)
K_MAX = 12        # max block length to try
SEED = 42

# Two distributions: high entropy (fair coin) vs low entropy (biased)
distributions = {
    "fair (H=1.0)":   make_distribution([0.5, 0.5]),
    "biased (H≈0.47)": make_distribution([0.9, 0.1]),
}


def shannon_entropy(dist):
    """H(dist) = -sum q * log2(q) over observed blocks."""
    return -sum(q * np.log2(q) for q in dist.values())


def entropy_rate_curve(p, N, k_max, seed):
    """For each k in 1..k_max, compute H(empirical k-block dist) / k."""
    seq = generate_sequence(p, length=N, seed=seed)
    ks, rates = [], []
    for k in range(1, k_max + 1):
        emp = empirical_block_distribution(seq, k)
        rate = shannon_entropy(emp) / k
        ks.append(k)
        rates.append(rate)
    return ks, rates


# --- Run and plot ---
fig, ax = plt.subplots(figsize=(8, 5))

for label, p in distributions.items():
    ks, rates = entropy_rate_curve(p, N, K_MAX, SEED)
    true_h = -sum(pi * np.log2(pi) for pi in p if pi > 0)
    ax.plot(ks, rates, marker="o", label=label)
    ax.axhline(true_h, linestyle="--", alpha=0.4)

ax.set_xlabel("Block length k")
ax.set_ylabel("H(empirical k-block) / k  (bits)")
ax.set_title("Empirical entropy rate converging to KS entropy H(p)")
ax.legend()
plt.tight_layout()
plt.savefig("entropy_rate.png", dpi=150)
plt.show()
print("Saved entropy_rate.png")
