"""SocietyModel: the Polis simulation loop.

Round structure (a "season"): every agent chooses one action from its bounded
memory + current state; the world engine resolves all consequences
deterministically; events flow back into agent memories; the full fulfillment
state is recorded for ergodic analysis.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import AgentBackend, refuse_paid_backend  # noqa: E402
from recorder import TranscriptRecorder  # noqa: E402

from .config import SocietyConfig
from .identity import sample_population, DIMENSIONS
from .world import WorldEngine
from .agents import SocietyAgent
from .mock_policy import SocietyMockBackend


class BudgetExceededError(RuntimeError):
    pass


class BudgetGuard(AgentBackend):
    """Wraps a paid backend and hard-stops the run at max_api_calls.

    Counts every respond() call (retries included), so the cap is on actual
    API usage, not on seasons. Free backends are never wrapped.
    """

    def __init__(self, inner: AgentBackend, max_calls: int):
        self.inner = inner
        self.max_calls = max_calls
        self.calls = 0
        self.name = inner.name
        self.is_free = inner.is_free

    def respond(self, *args, **kwargs):
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"API call budget of {self.max_calls} exhausted.")
        self.calls += 1
        return self.inner.respond(*args, **kwargs)


class SocietyModel:
    def __init__(self, config: SocietyConfig | None = None,
                 backend: AgentBackend | None = None):
        self.config = config or SocietyConfig()
        self.rng = np.random.default_rng(self.config.seed)

        if backend is None:
            if self.config.backend == "mock":
                backend = SocietyMockBackend()
            else:
                backend = refuse_paid_backend(self.config.backend)  # raises: paid backends must be explicit
        if not backend.is_free and self.config.max_api_calls is not None:
            backend = BudgetGuard(backend, self.config.max_api_calls)
        self.backend = backend

        self.recorder = None
        if self.config.transcript_path:
            self.recorder = TranscriptRecorder(self.config.transcript_path)

        self.identities = sample_population(self.config.n_agents, self.rng)
        self.engine = WorldEngine(self.config, self.identities, self.rng)
        all_names = [i.name for i in self.identities]
        self.agents = [
            SocietyAgent(ident, all_names, self.config.memory_window,
                         self.backend, self.config.max_retries)
            for ident in self.identities
        ]

        # Records for ergodic analysis (lists of per-season arrays)
        self.fulfillment_history: list[np.ndarray] = []   # (N,) per season
        self.wealth_history: list[np.ndarray] = []        # (N,) per season
        self.dims_history: list[np.ndarray] = []          # (N,4) per season
        self.action_history: list[list[str]] = []         # (N,) action names
        self.season = 0
        self.aborted = False

    def step(self) -> None:
        actions = [agent.choose(self.season, self.engine, self.rng,
                                self.recorder)
                   for agent in self.agents]
        events = self.engine.resolve(actions)
        for agent, ev in zip(self.agents, events):
            agent.observe(ev)

        self.fulfillment_history.append(self.engine.fulfillment().copy())
        self.wealth_history.append(self.engine.wealth.copy())
        self.dims_history.append(self.engine.dimension_norms())
        self.action_history.append([a.kind for a in actions])
        self.season += 1

    def run(self) -> "SocietyModel":
        try:
            for _ in range(self.config.n_steps):
                self.step()
        except BudgetExceededError as e:
            # Stop gracefully: keep everything recorded so far.
            print(f"[budget] {e} Stopping at season {self.season}.")
            self.aborted = True
        finally:
            if self.recorder is not None:
                self.recorder.close()
        return self

    def to_record(self) -> dict:
        return {
            "config": asdict(self.config),
            "backend": self.backend.name,
            "dimensions": list(DIMENSIONS),
            "identities": [i.to_record() for i in self.identities],
            "fulfillment": np.array(self.fulfillment_history).tolist(),
            "wealth": np.array(self.wealth_history).tolist(),
            "dims": np.array(self.dims_history).tolist(),
            "actions": self.action_history,
            "n_fallbacks": sum(a.n_fallbacks for a in self.agents),
            "aborted": self.aborted,
            "api_calls": getattr(self.backend, "calls", None),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_record(), fh)
        return path
