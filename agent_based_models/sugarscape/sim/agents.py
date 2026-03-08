"""SugarAgent class for the Sugarscape model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa
import numpy as np

if TYPE_CHECKING:
    from .model import SugarscapeModel


class SugarAgent(mesa.Agent):
    """An agent that harvests sugar, pays metabolism, ages, and may die."""

    def __init__(self, model: SugarscapeModel) -> None:
        """Initialize agent with randomly drawn attributes."""
        super().__init__(model)
        rng = model.rng
        self.sugar: int = int(rng.integers(5, 26))
        self.metabolism: int = int(rng.integers(1, 5))
        self.vision: int = int(rng.integers(1, model.config.max_vision + 1))
        self.max_age: int = int(rng.integers(60, 101))
        self.age: int = 0
        self.wealth_history: list[int] = []

    def _get_candidate_cells(self) -> list[tuple[int, int]]:
        """Return all cells visible in the 4 cardinal directions plus current cell."""
        grid = self.model.grid
        x, y = self.pos
        candidates: list[tuple[int, int]] = [self.pos]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for dist in range(1, self.vision + 1):
                nx = (x + dx * dist) % grid.width
                ny = (y + dy * dist) % grid.height
                candidates.append((nx, ny))
        return candidates

    def step(self) -> None:
        """Execute one agent step: move, harvest, metabolize, age, possibly die."""
        model: SugarscapeModel = self.model
        grid = model.grid
        sugar_grid = model.sugar_grid

        candidates = self._get_candidate_cells()

        # Find best cell: highest sugar, break ties by nearest, then random
        rng = model.rng
        x0, y0 = self.pos

        def cell_key(pos: tuple[int, int]) -> tuple[float, float, float]:
            cx, cy = pos
            # Toroidal distance
            dx = min(abs(cx - x0), grid.width - abs(cx - x0))
            dy = min(abs(cy - y0), grid.height - abs(cy - y0))
            dist = dx + dy
            return (-sugar_grid[cx, cy], dist, rng.random())

        best = min(candidates, key=cell_key)

        # Move to best cell and harvest sugar
        if best != self.pos:
            grid.move_agent(self, best)
        bx, by = best
        self.sugar += sugar_grid[bx, by]
        sugar_grid[bx, by] = 0

        # Metabolize and age
        self.sugar -= self.metabolism
        self.age += 1

        # Record wealth after metabolism, before death check
        self.wealth_history.append(self.sugar)

        # Death check
        if self.sugar < 0 or self.age > self.max_age:
            self._die_and_replace()

    def _die_and_replace(self) -> None:
        """Remove this agent and place a fresh one at a random empty cell."""
        model: SugarscapeModel = self.model
        grid = model.grid

        # Save completed wealth trajectory before removal
        if self.wealth_history:
            model.completed_trajectories.append(self.wealth_history)

        grid.remove_agent(self)
        self.remove()

        # Replace only if there is an empty cell
        empties = list(grid.empties)
        if not empties:
            return

        new_agent = SugarAgent(model)
        pos = empties[int(model.rng.integers(0, len(empties)))]
        grid.place_agent(new_agent, pos)
