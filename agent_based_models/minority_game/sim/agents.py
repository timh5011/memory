"""MinorityGameAgent for the Minority Game model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from itertools import product

import mesa
import numpy as np

if TYPE_CHECKING:
    from .model import MinorityGameModel


class MinorityGameAgent(mesa.Agent):
    """An agent with S strategy tables that plays the Minority Game."""

    def __init__(self, model: MinorityGameModel) -> None:
        super().__init__(model)
        rng = model.rng
        M = model.config.memory_length
        S = model.config.n_strategies

        # All possible history patterns (2^M tuples of 0/1)
        all_histories = list(product((0, 1), repeat=M))

        # Generate S random strategy tables
        # Each strategy maps every M-length binary tuple to an action (0 or 1)
        self.strategies: list[dict[tuple, int]] = []
        for _ in range(S):
            actions = rng.integers(0, 2, size=len(all_histories))
            table = {h: int(a) for h, a in zip(all_histories, actions)}
            self.strategies.append(table)

        # Cumulative scores for each strategy
        self.scores: np.ndarray = np.zeros(S, dtype=float)

    def choose(self, history_tuple: tuple) -> int:
        """Return action from highest-scoring strategy (random tie-breaking)."""
        rng = self.model.rng
        best_score = self.scores.max()
        best_indices = np.where(self.scores == best_score)[0]
        chosen_idx = int(rng.choice(best_indices))
        return self.strategies[chosen_idx][history_tuple]

    def update_scores(self, history_tuple: tuple, winning_action: int) -> None:
        """+1/-1 for ALL strategies based on match with winning action."""
        for i, strategy in enumerate(self.strategies):
            if strategy[history_tuple] == winning_action:
                self.scores[i] += 1
            else:
                self.scores[i] -= 1
