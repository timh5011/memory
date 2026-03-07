import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def make_transition_matrix(rows):
    """Validate and return a row-stochastic transition matrix."""
    T = np.array(rows, dtype=float)
    assert T.ndim == 2 and T.shape[0] == T.shape[1], "Must be square"
    assert np.allclose(T.sum(axis=1), 1.0), "Each row must sum to 1"
    assert np.all(T >= 0), "Entries must be non-negative"
    return T


def stationary_distribution(T):
    """Compute stationary distribution by solving pi @ T = pi."""
    n = T.shape[0]
    A = (T.T - np.eye(n))
    A[-1] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def true_entropy_rate(T):
    """h = -sum_i pi_i sum_j T_ij log2(T_ij), the conditional entropy H(X_{n+1}|X_n)."""
    pi = stationary_distribution(T)
    h = 0.0
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] > 0:
                h -= pi[i] * T[i, j] * np.log2(T[i, j])
    return h


def generate_markov_sequence(T, length, seed=None):
    """Generate a Markov chain sequence of given length."""
    rng = np.random.default_rng(seed)
    pi = stationary_distribution(T)
    n = T.shape[0]
    seq = np.empty(length, dtype=int)
    seq[0] = rng.choice(n, p=pi)
    for t in range(1, length):
        seq[t] = rng.choice(n, p=T[seq[t - 1]])
    return seq


def empirical_block_distribution(seq, k):
    """Return empirical distribution over length-k blocks."""
    counts = defaultdict(int)
    for i in range(len(seq) - k + 1):
        counts[tuple(seq[i:i + k])] += 1
    total = sum(counts.values())
    return {block: count / total for block, count in counts.items()}


def shannon_entropy(dist):
    """H(dist) = -sum q * log2(q)."""
    return -sum(q * np.log2(q) for q in dist.values())


# --- Parameters ---
N = 100_000
K_MAX = 12
SEED = 42

# Two Markov chains: low memory (near-independent) vs high memory (sticky)
chains = {
    "low memory":  make_transition_matrix([[0.5, 0.5], [0.5, 0.5]]),   # like fair Bernoulli
    "high memory": make_transition_matrix([[0.95, 0.05], [0.05, 0.95]]),  # sticky: stays in same state
}

# --- Run and plot ---
fig, ax = plt.subplots(figsize=(8, 5))

for label, T in chains.items():
    seq = generate_markov_sequence(T, N, seed=SEED)
    h_true = true_entropy_rate(T)
    ks, rates = [], []
    for k in range(1, K_MAX + 1):
        emp = empirical_block_distribution(seq, k)
        rates.append(shannon_entropy(emp) / k)
        ks.append(k)
    ax.plot(ks, rates, marker="o", label=f"{label} (h={h_true:.2f} bits)")
    ax.axhline(h_true, linestyle="--", alpha=0.4)

ax.set_xlabel("Block length k")
ax.set_ylabel("H(empirical k-block) / k  (bits)")
ax.set_title("Entropy rate convergence — Markov chains")
ax.legend()
plt.tight_layout()
plt.savefig("markov_entropy_rate.png", dpi=150)
plt.show()
print("Saved markov_entropy_rate.png")
