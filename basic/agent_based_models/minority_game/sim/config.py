"""Configuration dataclass for the Minority Game model."""

from dataclasses import dataclass


@dataclass
class MinorityGameConfig:
    """All parameters needed to fully specify a Minority Game simulation."""

    n_agents: int = 301
    memory_length: int = 6  # M — the key parameter
    n_strategies: int = 2   # S — strategies per agent
    n_steps: int = 500
    seed: int = 42

    def __post_init__(self):
        assert self.n_agents % 2 == 1, "n_agents must be odd"

    @property
    def alpha(self) -> float:
        """Complexity ratio α = 2^M / N."""
        return 2 ** self.memory_length / self.n_agents
