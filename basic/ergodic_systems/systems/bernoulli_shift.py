import numpy as np
from .ergodic_system import ErgodicSystem


class BernoulliShift(ErgodicSystem):
    """Bernoulli shift: i.i.d. process with distribution p on finite alphabet."""

    def __init__(self, probs, seed=None):
        p = np.array(probs, dtype=float)
        assert np.all(p >= 0), "Probabilities must be non-negative"
        assert np.isclose(p.sum(), 1.0), f"Probabilities must sum to 1, got {p.sum()}"
        self.p = p
        self._seed = seed
        super().__init__(dimension=1, is_symbolic=True, alphabet_size=len(self.p))

    def iterate(self, state):
        """Sample next symbol (i.i.d., state is ignored)."""
        rng = np.random.default_rng(self._seed)
        return rng.choice(self.alphabet_size, p=self.p)

    def generate_trajectory(self, initial_state=None, n_steps=1000, seed=None):
        s = seed if seed is not None else self._seed
        rng = np.random.default_rng(s)
        alphabet = np.arange(len(self.p))
        return rng.choice(alphabet, size=n_steps, p=self.p)

    def sample_initial_state(self, seed=None):
        s = seed if seed is not None else self._seed
        rng = np.random.default_rng(s)
        return rng.choice(self.alphabet_size, p=self.p)

    def analytical_ks_entropy(self):
        """H(p) = -sum p_i log2(p_i) in bits."""
        return -np.sum(self.p[self.p > 0] * np.log2(self.p[self.p > 0]))
