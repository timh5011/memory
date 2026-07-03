"""SocietyAgent: one inhabitant of Polis."""

from __future__ import annotations

from collections import deque

import numpy as np

from .identity import Identity, DIMENSIONS
from .prompts import build_system_prompt, render_observation, parse_action
from .world import Action


class SocietyAgent:
    """Holds an identity and a bounded memory of past seasons' events.

    Each season the agent renders its memory into a fresh single-turn prompt,
    asks its backend for a decision, and returns a validated Action. All world
    consequences are applied by the engine, never here.
    """

    def __init__(self, ident: Identity, all_names: list[str],
                 memory_window: int, backend, max_retries: int = 2):
        self.ident = ident
        self.all_names = all_names
        self.name_to_id = {n: i for i, n in enumerate(all_names)}
        self.memory_window = memory_window
        self.backend = backend
        self.max_retries = max_retries
        self.system_prompt = build_system_prompt(ident, all_names)
        # memory[k] = list of event strings from one past season (oldest first)
        self.memory: deque[list[str]] = deque(maxlen=max(memory_window, 0))
        self.n_fallbacks = 0

    def choose(self, season: int, engine, rng: np.random.Generator,
               recorder=None) -> Action:
        i = self.ident.agent_id
        norms_row = engine.dimension_norms()[i]
        norms = {d: float(norms_row[k]) for k, d in enumerate(DIMENSIONS)}
        tie_row = engine.ties[i]
        top = np.argsort(tie_row)[::-1][:5]
        top_ties = [(self.all_names[j], float(tie_row[j]))
                    for j in top if tie_row[j] > 0.05]

        observation = {
            "season": season,
            "wealth": float(engine.wealth[i]),
            "norms": norms,
            "value_weights": self.ident.value_weights,
            "temperament": self.ident.temperament,
            "top_ties": top_ties,
            "cost_of_living": engine.config.cost_of_living,
            "can_afford_gathering": bool(
                engine.wealth[i] >= engine.config.gathering_cost),
            "memory_signals": self._memory_signals(),
        }
        user_prompt = render_observation(
            season=season,
            wealth=float(engine.wealth[i]),
            norms=norms,
            memory=list(self.memory),
            bulletin=engine.bulletin,
            top_ties=top_ties,
            memory_window=self.memory_window,
        )

        parsed = None
        raw = None
        for _ in range(self.max_retries + 1):
            raw = self.backend.respond(i, self.system_prompt, user_prompt,
                                       observation, rng)
            parsed = parse_action(raw, self.name_to_id)
            if parsed is not None:
                break

        fallback = parsed is None
        if fallback:
            self.n_fallbacks += 1
            action = Action(kind="REST", fallback=True)
        else:
            action = Action(kind=parsed["action"], target=parsed["target"],
                            amount=parsed["amount"])

        if recorder is not None:
            recorder.log({
                "season": season,
                "agent": i,
                "name": self.ident.name,
                "user_prompt": user_prompt,
                "raw_response": raw,
                "action": action.kind,
                "target": action.target,
                "fallback": fallback,
            })
        return action

    def _memory_signals(self) -> dict:
        """Structured signals extracted from remembered event strings.

        The strings are our own engine's fixed formats (world.py), so simple
        substring checks are reliable. This is what makes the memory window w
        matter for the MOCK backend too (reciprocity, venture hot-hand,
        gathering imitation) — for LLM backends the raw text in the prompt is
        the memory, and these signals are ignored.
        """
        helped_by: list[str] = []
        venture_wins = 0
        venture_losses = 0
        gatherings_seen = 0
        for season_events in self.memory:
            for ev in season_events:
                if "gave you" in ev:
                    helped_by.append(ev.split()[0])
                elif "venture succeeded" in ev.lower():
                    venture_wins += 1
                elif "venture failed" in ev.lower():
                    venture_losses += 1
                elif "gathering" in ev and "attended" in ev:
                    gatherings_seen += 1
        return {
            "n_seasons_remembered": len(self.memory),
            "helped_by": helped_by,
            "venture_wins": venture_wins,
            "venture_losses": venture_losses,
            "gatherings_seen": gatherings_seen,
        }

    def observe(self, season_events: list[str]) -> None:
        if self.memory.maxlen and self.memory.maxlen > 0:
            self.memory.append(season_events)
