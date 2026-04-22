"""Configuration dataclass for the Sugarscape model."""

from dataclasses import dataclass


@dataclass
class SugarscapeConfig:
    """All parameters needed to fully specify a Sugarscape simulation."""

    # Grid
    grid_width: int = 50
    grid_height: int = 50
    alpha: float = 1.0  # sugar regrowth per step per cell

    # Agents
    n_agents: int = 250
    max_vision: int = 6  # upper bound on vision distribution

    # Simulation
    n_steps: int = 500
    seed: int = 42
