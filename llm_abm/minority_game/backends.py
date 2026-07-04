"""Minority Game backends.

The general backend abstraction (AgentBackend, CamelBackend) and the
paid-backend guardrail live in the shared `llm_abm/backends.py`. This module holds only the
Minority-Game-specific free backend — MockBackend, whose policies operate on
the game's binary 0/1 action space — plus build_backend() which wires it up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put llm_abm/ on path for shared modules

from backends import AgentBackend, CamelBackend, refuse_paid_backend  # noqa: E402,F401


class MockBackend(AgentBackend):
    """Rule-based stand-in for an LLM. Free and fully local.

    A mixture of simple behavioral policies (contrarian, trend-follower,
    frequency-based, random) that stands in for LLM reasoning so the entire
    pipeline — simulation, transcripts, metrics, KS entropy analysis, sweeps,
    plots — can be built and validated at zero cost before a single API call
    is ever made.
    """

    name = "mock"
    is_free = True

    POLICIES = ("oppose_last", "repeat_last", "window_minority",
                "window_majority", "random")

    def __init__(self, noise: float = 0.1, chatty_prob: float = 0.05):
        # noise: probability of flipping the policy's action (keeps the mock
        # population from freezing into a cycle). chatty_prob: probability of
        # returning a verbose sentence instead of a bare digit, so the
        # response parser and retry path get exercised.
        self.noise = noise
        self.chatty_prob = chatty_prob
        self._policies: dict[int, str] = {}

    def respond(self, agent_id, system_prompt, user_prompt, observation, rng):
        policy = self._policies.setdefault(
            agent_id, self.POLICIES[len(self._policies) % len(self.POLICIES)]
        )
        history = observation["history"]

        if not history or policy == "random":
            action = int(rng.integers(0, 2))
        elif policy == "oppose_last":
            action = 1 - history[-1]
        elif policy == "repeat_last":
            action = history[-1]
        else:
            ones = sum(history)
            zeros = len(history) - ones
            if ones == zeros:
                action = int(rng.integers(0, 2))
            elif policy == "window_minority":
                action = 1 if ones < zeros else 0
            else:  # window_majority
                action = 1 if ones > zeros else 0

        if rng.random() < self.noise:
            action = 1 - action

        if rng.random() < self.chatty_prob:
            return f"Thinking about the recent rounds, I'll go with {action}."
        return str(action)


def build_backend(config) -> AgentBackend:
    """Build the free default backend from a config.

    Deliberately refuses to build paid backends: those must be constructed
    explicitly by the caller so that spending money is always a visible,
    intentional line of code.
    """
    if config.backend == "mock":
        return MockBackend()
    return refuse_paid_backend(config.backend)
