import numpy as np
from .ergodic_system import ErgodicSystem


class LogisticMap(ErgodicSystem):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n) on [0, 1].

    For r = 4 the map is fully chaotic and ergodic with respect to the
    arcsine (Beta(1/2, 1/2)) invariant measure, and has a known Lyapunov
    exponent of ln(2).
    """

    def __init__(self, r=4.0):
        assert 0 < r <= 4, f"r must be in (0, 4], got {r}"
        self.r = r
        super().__init__(dimension=1, is_symbolic=False)

    def iterate(self, state):
        return self.r * state * (1 - state)

    def generate_trajectory(self, initial_state=None, n_steps=1000, seed=None):
        if initial_state is None:
            initial_state = self.sample_initial_state(seed)
        traj = np.empty(n_steps)
        traj[0] = initial_state
        for i in range(1, n_steps):
            traj[i] = self.r * traj[i - 1] * (1 - traj[i - 1])
        return traj

    def sample_initial_state(self, seed=None):
        """Sample from Beta(1/2, 1/2) — the invariant measure for r=4."""
        rng = np.random.default_rng(seed)
        return rng.beta(0.5, 0.5)

    def jacobian(self, state):
        """Derivative of the map: dT/dx = r(1 - 2x)."""
        return self.r * (1 - 2 * state)

    def symbolize(self, trajectory, partition=None):
        """Binary partition at x = 0.5 (the critical point)."""
        if partition is None:
            partition = 0.5
        return (trajectory >= partition).astype(int)

    def analytical_ks_entropy(self):
        """For r=4: h_KS = ln(2) (nats) = log2(2) = 1.0 (bits) via Pesin's identity."""
        if self.r == 4.0:
            return np.log2(2)  # 1.0 bits
        return None
