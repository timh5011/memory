import numpy as np
import itertools
from ergodic_system import ErgodicSystem


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


def empirical_block_distribution(seq, k):
    """Sliding window counts over sequence, normalized to frequencies.
    Returns dict keyed by tuples of length k."""
    counts = {}
    for i in range(len(seq) - k + 1):
        block = tuple(seq[i:i + k])
        counts[block] = counts.get(block, 0) + 1
    total = sum(counts.values())
    return {b: c / total for b, c in counts.items()}


def true_block_distribution(p, k):
    """Product measure over all length-k blocks.
    Returns dict keyed by tuples of length k."""
    alphabet = range(len(p))
    dist = {}
    for block in itertools.product(alphabet, repeat=k):
        prob = 1.0
        for symbol in block:
            prob *= p[symbol]
        dist[block] = prob
    return dist


class BernoulliShift(ErgodicSystem):
    """Bernoulli shift: i.i.d. process with distribution p on finite alphabet."""

    def __init__(self, probs, seed=None):
        self.p = make_distribution(probs)
        self._seed = seed
        super().__init__(dimension=1, is_symbolic=True, alphabet_size=len(self.p))

    def iterate(self, state):
        """Sample next symbol (i.i.d., state is ignored)."""
        rng = np.random.default_rng(self._seed)
        return rng.choice(self.alphabet_size, p=self.p)

    def generate_trajectory(self, initial_state=None, n_steps=1000, seed=None):
        s = seed if seed is not None else self._seed
        return generate_sequence(self.p, n_steps, seed=s)

    def sample_initial_state(self, seed=None):
        s = seed if seed is not None else self._seed
        rng = np.random.default_rng(s)
        return rng.choice(self.alphabet_size, p=self.p)

    def analytical_ks_entropy(self):
        """H(p) = -sum p_i log2(p_i) in bits."""
        return -np.sum(self.p[self.p > 0] * np.log2(self.p[self.p > 0]))
