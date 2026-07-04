"""SocietyMockBackend: a free, local stand-in for LLM decision-making.

Scores each action by its rough expected fulfillment gain UNDER THE AGENT'S
OWN VALUE WEIGHTS, plus temperament biases and seeded noise — i.e. it plays
the same character the system prompt describes, just without a language
model. This validates the entire pipeline (world engine, prompts, parsing,
records, entropy analysis, sweeps) at zero cost, and doubles as the
"hand-coded rational baseline" to compare live LLM behavior against.
"""

from __future__ import annotations

import json

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import AgentBackend  # noqa: E402

# Rough per-dimension gains of each action (mirrors world.py magnitudes,
# expressed in normalized-dimension units)
ACTION_GAINS = {
    "WORK":           {"prosperity": 0.06, "security": 0.02},
    "SOCIALIZE":      {"belonging": 0.04},
    "HELP":           {"standing": 0.10, "belonging": 0.03, "prosperity": -0.04},
    "HOST_GATHERING": {"standing": 0.15, "belonging": 0.08, "prosperity": -0.07},
    "VENTURE":        {"prosperity": 0.03, "standing": 0.02, "security": -0.04},
    "ADVOCATE":       {"standing": 0.10, "belonging": -0.01},
    "REST":           {"security": 0.10, "belonging": 0.02},
}


class SocietyMockBackend(AgentBackend):
    name = "mock"
    is_free = True

    def __init__(self, noise: float = 0.35, malformed_prob: float = 0.02):
        # noise: softens the argmax so behavior isn't frozen; malformed_prob:
        # occasionally emit non-JSON to exercise the parser/fallback path.
        self.noise = noise
        self.malformed_prob = malformed_prob

    def respond(self, agent_id, system_prompt, user_prompt, observation, rng):
        if rng.random() < self.malformed_prob:
            return "Hmm, I think I will just work this season."

        weights = observation["value_weights"]
        temperament = observation["temperament"]
        wealth = observation["wealth"]
        col = observation["cost_of_living"]

        scores = {}
        for action, gains in ACTION_GAINS.items():
            s = sum(weights.get(d, 0.0) * g for d, g in gains.items())
            scores[action] = s + self.noise * rng.normal() * 0.05

        # Temperament biases — the mock stays in character
        scores["VENTURE"] += 0.03 * max(temperament["ambition"], 0)
        scores["ADVOCATE"] += 0.02 * abs(temperament["progressivism"])
        scores["SOCIALIZE"] += 0.03 * max(temperament["gregariousness"], 0)
        scores["HOST_GATHERING"] += 0.02 * max(temperament["gregariousness"], 0)
        scores["REST"] += 0.03 * max(-temperament["ambition"], 0)

        # Memory-driven adjustments — this is what makes the memory window w
        # a live parameter for the mock too. With w=0 all signals are empty
        # and the mock is memoryless; with larger w it reciprocates help,
        # chases venture streaks, and imitates gathering norms.
        mem = observation["memory_signals"]
        if mem["helped_by"]:
            scores["HELP"] += 0.05
            scores["SOCIALIZE"] += 0.03
        streak = mem["venture_wins"] - mem["venture_losses"]
        scores["VENTURE"] += 0.04 * np.sign(streak) * min(abs(streak), 2)
        if mem["gatherings_seen"] >= 2:
            scores["HOST_GATHERING"] += 0.03

        # Survival instinct: when poor, work; don't gamble or give
        if wealth < 2.5 * col:
            scores["WORK"] += 0.10
            scores["VENTURE"] -= 0.10
            scores["HELP"] -= 0.10
            scores["HOST_GATHERING"] -= 0.15
        if not observation["can_afford_gathering"]:
            scores["HOST_GATHERING"] -= 1.0

        action = max(scores, key=scores.get)

        target = None
        amount = None
        ties = observation["top_ties"]
        tie_names = [name for name, _ in ties]
        # Reciprocity: prefer whoever helped us most recently
        remembered_helper = next(
            (h for h in reversed(mem["helped_by"]) if h in tie_names), None)
        if action == "SOCIALIZE" and ties:
            target = remembered_helper or min(ties, key=lambda kv: kv[1])[0]
        elif action == "HELP":
            if remembered_helper:
                target = remembered_helper
            else:
                target = ties[int(rng.integers(0, len(ties)))][0] if ties else None
            amount = round(0.2 * wealth, 1)

        return json.dumps({"action": action, "target": target,
                           "amount": amount, "reason": "mock policy"})
