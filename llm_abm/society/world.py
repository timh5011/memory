"""The Polis world engine: deterministic consequences of agent decisions.

The LLM (or mock) decides; this engine applies ALL effects — wages,
transfers, tie strengthening, reputation, decay, cost of living. No physics
is ever delegated to the language model (same principle as the Minority Game
pipeline). All randomness flows through the model's seeded Generator.

The societal-memory knobs live here: `reputation_decay` and `tie_decay`
control how fast the institution forgets standing and relationships — the
"forgiveness rate" of the society.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SocietyConfig
from .identity import Identity, ROLES, FACTIONS, DIMENSIONS

ACTIONS = ("WORK", "SOCIALIZE", "HELP", "HOST_GATHERING", "VENTURE",
           "ADVOCATE", "REST")

# Normalization scales mapping raw stocks to [0,1] fulfillment dimensions
WEALTH_SCALE = 40.0     # prosperity = w / (w + WEALTH_SCALE)
TIE_SCALE = 3.0         # belonging  = ties_sum / (ties_sum + TIE_SCALE)
STANDING_SCALE = 4.0    # standing   = s / (s + STANDING_SCALE)
HOUSEHOLD_TIE_FLOOR = 0.3   # family never fully forgets you


@dataclass
class Action:
    kind: str
    target: int | None = None   # agent_id for SOCIALIZE / HELP
    amount: float | None = None  # coins for HELP
    fallback: bool = False       # True if this came from a parse failure


class WorldEngine:
    """Owns the dynamic state and resolves one season at a time."""

    def __init__(self, config: SocietyConfig, identities: list[Identity],
                 rng: np.random.Generator):
        self.config = config
        self.identities = identities
        self.rng = rng
        n = config.n_agents

        self.wealth = np.array(
            [self._start_wealth(i) for i in identities], dtype=float)
        self.standing = np.array(
            [ROLES[i.role][2] for i in identities], dtype=float)
        self.reserve = np.full(n, 0.5)   # security stock, built by REST

        # Symmetric tie-strength matrix in [0,1]
        self.ties = np.zeros((n, n))
        self._household = np.zeros((n, n), dtype=bool)
        for ident in identities:
            for j in ident.household:
                if j != ident.agent_id:
                    self.ties[ident.agent_id, j] = 0.8
                    self._household[ident.agent_id, j] = True
            for j in ident.friends:
                self.ties[ident.agent_id, j] = max(self.ties[ident.agent_id, j], 0.3)
        self.ties = np.maximum(self.ties, self.ties.T)
        self._household |= self._household.T

        # Faction politics: advocacy this season → wage bonus next season
        self._advocacy_counts = {f: 0 for f in FACTIONS}
        self._wage_bonus_faction: str | None = None

        self.bulletin: list[str] = ["The town of Polis begins a new era."]

    def _start_wealth(self, ident: Identity) -> float:
        from .identity import CLASS_WEALTH
        base = CLASS_WEALTH[ident.social_class]
        return base * float(self.rng.uniform(0.8, 1.2))

    # ------------------------------------------------------------------ #
    # Season resolution
    # ------------------------------------------------------------------ #

    def resolve(self, actions: list[Action]) -> list[list[str]]:
        """Apply one season of decisions. Returns per-agent event lists and
        replaces self.bulletin with this season's public news."""
        cfg = self.config
        n = cfg.n_agents
        events: list[list[str]] = [[] for _ in range(n)]
        bulletin: list[str] = []
        names = [i.name for i in self.identities]

        # Wage bonus earned by last season's dominant advocacy
        bonus_faction = self._wage_bonus_faction
        self._advocacy_counts = {f: 0 for f in FACTIONS}

        order = self.rng.permutation(n)
        for i in order:
            act = actions[i]
            ident = self.identities[i]

            if act.kind == "WORK":
                base, var, _ = ROLES[ident.role]
                wage = base * float(self.rng.uniform(1 - var, 1 + var))
                if bonus_faction == ident.faction:
                    wage *= 1.05
                self.wealth[i] += wage
                events[i].append(f"You worked as a {ident.role} and earned {wage:.1f} coins.")

            elif act.kind == "SOCIALIZE":
                t = act.target
                if t is None or t == i:
                    hh = [j for j in ident.household if j != i]
                    t = int(max(hh, key=lambda j: self.ties[i, j])) if hh else None
                if t is None:
                    events[i].append("You had no one to visit and rested instead.")
                    self.reserve[i] = min(1.0, self.reserve[i] + 0.10)
                else:
                    self._bump_tie(i, t, 0.15)
                    events[i].append(f"You spent the season with {names[t]}; you grew closer.")
                    events[t].append(f"{names[i]} spent time with you; you grew closer.")

            elif act.kind == "HELP":
                t = act.target
                if t is None or t == i:
                    neighbors = np.where(self.ties[i] > 0.05)[0]
                    t = int(neighbors[np.argmin(self.wealth[neighbors])]) if len(neighbors) else None
                if t is None or self.wealth[i] < 1.0:
                    events[i].append("You wanted to help someone but could not; you rested.")
                    self.reserve[i] = min(1.0, self.reserve[i] + 0.10)
                else:
                    amount = act.amount if act.amount and act.amount > 0 else 0.2 * self.wealth[i]
                    amount = float(np.clip(amount, 1.0, 0.3 * self.wealth[i]))
                    self.wealth[i] -= amount
                    self.wealth[t] += amount
                    self.standing[i] += 0.6
                    self._bump_tie(i, t, 0.2)
                    events[i].append(f"You gave {amount:.1f} coins to {names[t]}. The town noticed your generosity.")
                    events[t].append(f"{names[i]} gave you {amount:.1f} coins in your need.")

            elif act.kind == "HOST_GATHERING":
                if self.wealth[i] < cfg.gathering_cost:
                    events[i].append("You could not afford to host a gathering; you rested.")
                    self.reserve[i] = min(1.0, self.reserve[i] + 0.10)
                else:
                    self.wealth[i] -= cfg.gathering_cost
                    guests = np.argsort(self.ties[i])[::-1]
                    guests = [int(g) for g in guests if self.ties[i, g] > 0.1][:8]
                    for g in guests:
                        self._bump_tie(i, g, 0.1)
                        events[g].append(f"You attended {names[i]}'s gathering.")
                    self.standing[i] += 1.2
                    events[i].append(f"You hosted a gathering for {len(guests)} people; your standing rose.")
                    bulletin.append(f"{names[i]} hosted a well-attended gathering.")

            elif act.kind == "VENTURE":
                stake = max(2.0, 0.3 * self.wealth[i])
                if self.wealth[i] < 2.0:
                    events[i].append("You were too poor to risk a venture and worked instead.")
                    base, var, _ = ROLES[ident.role]
                    self.wealth[i] += base * float(self.rng.uniform(1 - var, 1 + var))
                elif self.rng.random() < 0.5:
                    self.wealth[i] += stake
                    self.standing[i] += 0.4
                    events[i].append(f"Your venture succeeded! You gained {stake:.1f} coins.")
                    if stake >= 5:
                        bulletin.append(f"{names[i]}'s bold venture paid off handsomely.")
                else:
                    loss = 0.9 * stake
                    self.wealth[i] -= loss
                    events[i].append(f"Your venture failed. You lost {loss:.1f} coins.")
                    if loss >= 5:
                        bulletin.append(f"{names[i]}'s venture ended in visible failure.")

            elif act.kind == "ADVOCATE":
                self.standing[i] += 0.6
                self._advocacy_counts[ident.faction] += 1
                for j in range(n):
                    if j == i or self.ties[i, j] <= 0.05:
                        continue
                    if self.identities[j].faction == ident.faction:
                        self._bump_tie(i, j, 0.05)
                    else:
                        self.ties[i, j] = self.ties[j, i] = max(
                            0.0, self.ties[i, j] - 0.08)
                events[i].append(
                    f"You advocated for the {ident.faction}. Allies drew closer; rivals cooled toward you.")

            else:  # REST (also the parse-failure fallback)
                self.reserve[i] = min(1.0, self.reserve[i] + 0.15)
                for j in ident.household:
                    if j != i:
                        self._bump_tie(i, j, 0.03)
                events[i].append("You rested and spent quiet time with your household.")
                if act.fallback:
                    events[i].append("(You had been indecisive this season.)")

        # --- Cost of living and hardship ---
        hardships = 0
        for i in range(n):
            self.wealth[i] -= cfg.cost_of_living
            if self.wealth[i] < 0:
                self.wealth[i] = 0.0
                self.reserve[i] = max(0.0, self.reserve[i] - 0.2)
                events[i].append("Hardship: you could not cover your cost of living.")
                hardships += 1
        if hardships:
            bulletin.append(f"{hardships} townsfolk fell on hard times this season.")

        # --- Societal memory: decay of standing and ties ---
        self.standing *= (1.0 - cfg.reputation_decay)
        nonhh = ~self._household
        self.ties[nonhh] *= (1.0 - cfg.tie_decay)
        hh = self._household
        self.ties[hh] = HOUSEHOLD_TIE_FLOOR + (
            self.ties[hh] - HOUSEHOLD_TIE_FLOOR) * (1.0 - cfg.tie_decay / 2)
        np.fill_diagonal(self.ties, 0.0)
        self.ties = np.clip(self.ties, 0.0, 1.0)
        self.reserve *= 0.97

        # --- Politics: who sets the tone next season ---
        counts = self._advocacy_counts
        if any(counts.values()):
            top = max(counts, key=counts.get)
            if counts[top] > min(counts.values()):
                self._wage_bonus_faction = top
                bulletin.append(f"The {top} holds sway in town affairs.")
            else:
                self._wage_bonus_faction = None
        else:
            self._wage_bonus_faction = None

        self.bulletin = bulletin if bulletin else ["A quiet season in Polis."]
        return events

    def _bump_tie(self, i: int, j: int, delta: float) -> None:
        v = min(1.0, self.ties[i, j] + delta)
        self.ties[i, j] = self.ties[j, i] = v

    # ------------------------------------------------------------------ #
    # Fulfillment
    # ------------------------------------------------------------------ #

    def dimension_norms(self) -> np.ndarray:
        """(N, 4) matrix of normalized fulfillment dimensions in [0,1],
        columns ordered as identity.DIMENSIONS."""
        cfg = self.config
        tie_sums = self.ties.sum(axis=1)
        prosperity = self.wealth / (self.wealth + WEALTH_SCALE)
        belonging = tie_sums / (tie_sums + TIE_SCALE)
        standing = self.standing / (self.standing + STANDING_SCALE)
        buffer = np.minimum(1.0, self.wealth / (4 * cfg.cost_of_living))
        security = 0.5 * self.reserve + 0.5 * buffer
        return np.column_stack([prosperity, belonging, standing, security])

    def fulfillment(self) -> np.ndarray:
        """F_i = Σ_d w_id · dim_id — each agent scored by their OWN values."""
        norms = self.dimension_norms()
        weights = np.array([[ident.value_weights[d] for d in DIMENSIONS]
                            for ident in self.identities])
        return (norms * weights).sum(axis=1)
