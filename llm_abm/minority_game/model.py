"""LLMMinorityGameModel: orchestrates the LLM Minority Game.

No Mesa dependency — there is no spatial structure, and the round loop is a
plain synchronous loop so it works identically for mock and LLM backends.
(An async gather over agents is the natural optimization for live LLM runs;
keep the sequential loop as the reference implementation.)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put llm_abm/ on path for shared modules

from recorder import TranscriptRecorder  # noqa: E402

from .config import LLMMinorityGameConfig
from .agents import LLMAgent
from .backends import AgentBackend, build_backend
from .prompts import GENERIC_PERSONA, PERSONAS


class LLMMinorityGameModel:
    """N agents, one backend, simultaneous binary choices, minority wins."""

    def __init__(self, config: LLMMinorityGameConfig | None = None,
                 backend: AgentBackend | None = None):
        self.config = config or LLMMinorityGameConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.backend = backend if backend is not None else build_backend(self.config)

        self.recorder = None
        if self.config.transcript_path:
            self.recorder = TranscriptRecorder(self.config.transcript_path)

        self.agents: list[LLMAgent] = []
        for i in range(self.config.n_agents):
            persona = (PERSONAS[i % len(PERSONAS)]
                       if self.config.persona_diversity else GENERIC_PERSONA)
            self.agents.append(LLMAgent(
                agent_id=i,
                persona=persona,
                n_agents=self.config.n_agents,
                memory_window=self.config.memory_window,
                backend=self.backend,
                max_retries=self.config.max_retries,
            ))

        self.outcomes: list[int] = []     # winning option per round (binary symbol sequence)
        self.attendance: list[int] = []   # number of agents choosing 1 per round
        self.round_idx = 0

    def step(self) -> None:
        choices = [agent.choose(self.round_idx, self.rng, self.recorder)
                   for agent in self.agents]

        n1 = sum(choices)
        n0 = self.config.n_agents - n1
        if n1 < n0:
            winning = 1
        elif n0 < n1:
            winning = 0
        else:  # impossible with odd N, but be safe
            winning = int(self.rng.integers(0, 2))

        for agent in self.agents:
            agent.observe(winning)

        self.outcomes.append(winning)
        self.attendance.append(n1)
        self.round_idx += 1

    def run(self) -> "LLMMinorityGameModel":
        for _ in range(self.config.n_steps):
            self.step()
        if self.recorder is not None:
            self.recorder.close()
        return self

    def to_record(self) -> dict:
        """Serializable summary of the run, for saving and post-hoc analysis."""
        return {
            "config": asdict(self.config),
            "backend": self.backend.name,
            "outcomes": self.outcomes,
            "attendance": self.attendance,
            "agent_wins": [a.n_wins for a in self.agents],
            "n_fallbacks": sum(a.n_fallbacks for a in self.agents),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_record(), fh)
        return path
