import numpy as np
from itertools import product as iterproduct


def generate_sequence(alphabet_size: int, distribution: np.ndarray, length: int, seed: int) -> np.ndarray:
    """Generate a length-N i.i.d. sequence from distribution p over {0, ..., alphabet_size-1}."""
    rng = np.random.default_rng(seed)
    return rng.choice(alphabet_size, size=length, p=distribution)


def empirical_block_distribution(sequence: np.ndarray, k: int) -> dict[tuple, float]:
    """Slide a window of length k over sequence, return normalized block frequencies."""
    n = len(sequence)
    if n < k:
        return {}
    counts: dict[tuple, int] = {}
    for i in range(n - k + 1):
        block = tuple(sequence[i:i + k].tolist())
        counts[block] = counts.get(block, 0) + 1
    total = n - k + 1
    return {block: c / total for block, c in counts.items()}


def true_block_distribution(distribution: np.ndarray, k: int) -> dict[tuple, float]:
    """Compute the exact product measure for all k-blocks: P(b) = prod(p(b_j))."""
    alphabet_size = len(distribution)
    result = {}
    for block in iterproduct(range(alphabet_size), repeat=k):
        prob = 1.0
        for symbol in block:
            prob *= distribution[symbol]
        result[block] = prob
    return result


def kl_divergence(empirical: dict[tuple, float], true: dict[tuple, float]) -> float:
    """Compute D_KL(Q_hat || P), summing only over observed blocks."""
    dkl = 0.0
    for block, q in empirical.items():
        if q > 0 and block in true and true[block] > 0:
            dkl += q * np.log(q / true[block])
    return dkl
