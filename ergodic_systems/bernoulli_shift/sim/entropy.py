import numpy as np


def shannon_entropy(distribution: np.ndarray) -> float:
    """Compute H(p) = -sum(p_i log p_i). Equals the KS entropy h for a Bernoulli shift."""
    p = np.asarray(distribution, dtype=float)
    # Filter out zero entries to avoid log(0)
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask]))
