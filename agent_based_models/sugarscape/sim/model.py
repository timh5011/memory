"""SugarscapeModel: the main model class."""

from __future__ import annotations

import numpy as np
import mesa

from .config import SugarscapeConfig
from .grid import make_sugar_landscape
from .agents import SugarAgent
from .metrics import gini


class SugarscapeModel(mesa.Model):
    """Sugarscape ABM following Epstein & Axtell 1996."""

    def __init__(self, config: SugarscapeConfig | None = None) -> None:
        """Initialize the model from a SugarscapeConfig."""
        if config is None:
            config = SugarscapeConfig()
        super().__init__(seed=config.seed)
        self.config = config

        # Sugar landscape
        self.sugar_max: np.ndarray = make_sugar_landscape(
            config.grid_width, config.grid_height
        )
        self.sugar_grid: np.ndarray = self.sugar_max.copy().astype(float)

        # Mesa grid (toroidal, MultiGrid allows multiple agents per cell)
        self.grid = mesa.space.MultiGrid(
            config.grid_width, config.grid_height, torus=True
        )

        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "population": lambda m: len(list(m.agents)),
                "mean_sugar": lambda m: float(
                    np.mean([a.sugar for a in m.agents]) if m.agents else 0.0
                ),
                "gini": lambda m: gini(
                    np.array([a.sugar for a in m.agents], dtype=float)
                ),
            }
        )

        # Storage for completed agent wealth trajectories (Approach 2 entropy)
        self.completed_trajectories: list[list[int]] = []

        # Place initial agents at random empty cells
        all_cells = [
            (x, y)
            for x in range(config.grid_width)
            for y in range(config.grid_height)
        ]
        chosen = self.rng.choice(
            len(all_cells), size=config.n_agents, replace=False
        )
        for idx in chosen:
            pos = all_cells[idx]
            agent = SugarAgent(self)
            self.grid.place_agent(agent, pos)

        # Collect initial state
        self.datacollector.collect(self)

    def step(self) -> None:
        """Advance the model by one step: activate agents then grow sugar."""
        # Activate all agents in random order
        agent_list = list(self.agents)
        self.rng.shuffle(agent_list)
        for agent in agent_list:
            if agent.pos is not None:  # agent may have died mid-step
                agent.step()

        # Grow sugar on all cells
        alpha = self.config.alpha
        self.sugar_grid = np.minimum(
            self.sugar_grid + alpha, self.sugar_max.astype(float)
        )

        # Collect data
        self.datacollector.collect(self)
