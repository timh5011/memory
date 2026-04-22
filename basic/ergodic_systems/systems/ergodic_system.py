from abc import ABC, abstractmethod


class ErgodicSystem(ABC):
    """Base class for measure-preserving ergodic dynamical systems (X, B, mu, T)."""

    def __init__(self, dimension, is_symbolic=False, alphabet_size=None):
        self.dimension = dimension
        self.is_symbolic = is_symbolic
        self.alphabet_size = alphabet_size

    @abstractmethod
    def iterate(self, state):
        """Apply the map T once: state -> T(state)."""
        ...

    @abstractmethod
    def generate_trajectory(self, initial_state=None, n_steps=1000, seed=None):
        """Generate a trajectory of length n_steps. If initial_state is None, sample from mu."""
        ...

    @abstractmethod
    def sample_initial_state(self, seed=None):
        """Sample a single state from the invariant measure mu."""
        ...

    def jacobian(self, state):
        raise NotImplementedError(f"{type(self).__name__} does not implement jacobian()")

    def metric(self, state_a, state_b):
        """Distance between two states. Required for Lyapunov perturbation method."""
        raise NotImplementedError(f"{type(self).__name__} does not implement metric()")

    def perturb(self, state, delta, rng):
        """Create a nearby state at distance ~delta. Required for Lyapunov perturbation method."""
        raise NotImplementedError(f"{type(self).__name__} does not implement perturb()")

    def symbolize(self, trajectory, partition=None):
        if self.is_symbolic:
            return trajectory
        raise NotImplementedError(f"{type(self).__name__} does not implement symbolize()")

    def analytical_ks_entropy(self):
        return None
