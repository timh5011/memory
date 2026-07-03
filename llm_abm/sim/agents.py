"""LLMAgent: one player in the LLM Minority Game."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .prompts import build_system_prompt, render_observation, parse_action


@dataclass
class RoundMemory:
    """What an agent remembers about one past round."""
    outcome: int   # the winning (minority) option
    action: int    # what this agent chose
    won: bool      # whether this agent was on the minority side


class LLMAgent:
    """A player whose decisions come from a backend (mock or LLM).

    The agent's memory is a bounded deque of the last `memory_window` rounds.
    Each round it renders that memory into a fresh single-turn prompt — the
    prompt IS the agent's entire memory of the game.
    """

    def __init__(self, agent_id: int, persona: str, n_agents: int,
                 memory_window: int, backend, max_retries: int = 2):
        self.agent_id = agent_id
        self.persona = persona
        self.memory_window = memory_window
        self.backend = backend
        self.max_retries = max_retries
        self.system_prompt = build_system_prompt(n_agents, persona)
        self.memory: deque[RoundMemory] = deque(maxlen=max(memory_window, 0))
        self.n_wins = 0
        self.n_fallbacks = 0  # rounds where the response was unparseable
        self._last_action: int | None = None

    def choose(self, round_idx: int, rng: np.random.Generator,
               recorder=None) -> int:
        history = [m.outcome for m in self.memory]
        own_actions = [m.action for m in self.memory]
        wins = sum(1 for m in self.memory if m.won)
        observation = {
            "round": round_idx,
            "history": history,
            "own_actions": own_actions,
            "wins": wins,
        }
        user_prompt = render_observation(
            round_idx, history, own_actions, wins, self.memory_window
        )

        action = None
        raw = None
        for _ in range(self.max_retries + 1):
            raw = self.backend.respond(
                self.agent_id, self.system_prompt, user_prompt, observation, rng
            )
            action = parse_action(raw)
            if action is not None:
                break

        fallback = action is None
        if fallback:
            action = int(rng.integers(0, 2))
            self.n_fallbacks += 1

        if recorder is not None:
            recorder.log({
                "round": round_idx,
                "agent": self.agent_id,
                "user_prompt": user_prompt,
                "raw_response": raw,
                "action": action,
                "fallback": fallback,
            })

        self._last_action = action
        return action

    def observe(self, outcome: int) -> None:
        """Record the round's result into bounded memory."""
        won = self._last_action == outcome
        if won:
            self.n_wins += 1
        if self.memory.maxlen and self.memory.maxlen > 0:
            self.memory.append(
                RoundMemory(outcome=outcome, action=self._last_action, won=won)
            )
