"""MinorityGameModel: the main model class."""

from __future__ import annotations

from collections import deque

import numpy as np
import mesa

from .config import MinorityGameConfig
from .agents import MinorityGameAgent


class MinorityGameModel(mesa.Model):
    """Minority Game (El Farol Bar Problem) ABM."""

    def __init__(self, config: MinorityGameConfig | None = None) -> None:
        if config is None:
            config = MinorityGameConfig()
        super().__init__(seed=config.seed)
        self.config = config

        M = config.memory_length

        # History deque initialized with random bits
        self.history: deque[int] = deque(maxlen=M)
        for _ in range(M):
            self.history.append(int(self.rng.integers(0, 2)))

        # Create agents (no grid — no spatial structure)
        for _ in range(config.n_agents):
            MinorityGameAgent(self)

        # Outcome sequence for entropy analysis
        self.outcomes: list[int] = []

        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "attendance": lambda m: m._last_attendance,
            }
        )
        self._last_attendance: int = 0

    def step(self) -> None:
        """Advance the model by one step: simultaneous moves."""
        history_tuple = tuple(self.history)

        # All agents choose simultaneously
        agents = list(self.agents)
        choices = [agent.choose(history_tuple) for agent in agents]

        # Count attendance (number choosing 1)
        n1 = sum(choices)
        n0 = self.config.n_agents - n1

        # Winning side = minority
        if n1 < n0:
            winning_action = 1
        elif n0 < n1:
            winning_action = 0
        else:
            # Exact tie shouldn't happen with odd N, but handle it
            winning_action = int(self.rng.integers(0, 2))

        # Update all strategy scores
        for agent in agents:
            agent.update_scores(history_tuple, winning_action)

        # Append winning action to history
        self.history.append(winning_action)
        self.outcomes.append(winning_action)

        # Record attendance for data collector
        self._last_attendance = n1
        self.datacollector.collect(self)
