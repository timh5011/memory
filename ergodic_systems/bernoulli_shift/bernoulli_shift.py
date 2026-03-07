import numpy as np


def make_distribution(probs):
    """Validate and return a probability vector as a numpy array."""
    p = np.array(probs, dtype=float)
    assert np.all(p >= 0), "Probabilities must be non-negative"
    assert np.isclose(p.sum(), 1.0), f"Probabilities must sum to 1, got {p.sum()}"
    return p


def generate_sequence(p, length, seed=None):
    """Sample `length` i.i.d. symbols from distribution p over alphabet {0,...,n-1}."""
    rng = np.random.default_rng(seed)
    alphabet = np.arange(len(p))
    return rng.choice(alphabet, size=length, p=p)


def shift(seq):
    """Apply the left-shift map: drop the first symbol, return the rest."""
    return seq[1:]


def true_block_distribution(p, k):
    """Return the true product measure over all length-k blocks."""
    from itertools import product
    alphabet = range(len(p))
    return {block: np.prod([p[s] for s in block]) for block in product(alphabet, repeat=k)}


def empirical_block_distribution(seq, k):
    """Return the empirical distribution over length-k blocks in seq."""
    counts = {}
    for i in range(len(seq) - k + 1):
        block = tuple(seq[i:i + k])
        counts[block] = counts.get(block, 0) + 1
    total = sum(counts.values())
    return {block: count / total for block, count in counts.items()}


def kl_divergence(empirical, true):
    """KL divergence KL(empirical || true), summing only over observed blocks."""
    return sum(q * np.log(q / true[block]) for block, q in empirical.items())


if __name__ == "__main__":
    p = make_distribution([0.5, 0.5])
    seq = generate_sequence(p, length=20, seed=42)
    print("Original sequence: ", seq)
    print("After one shift:   ", shift(seq))
    print("After two shifts:  ", shift(shift(seq)))
