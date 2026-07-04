"""Configuration dataclass for the LLM Minority Game."""

from dataclasses import dataclass


@dataclass
class LLMMinorityGameConfig:
    """All parameters needed to fully specify an LLM Minority Game simulation.

    The central research parameter is `memory_window` (w): the number of past
    rounds included in each agent's prompt. It plays the role that memory
    length M plays in the classical Minority Game — the knob that controls how
    much of the system's history conditions the present.
    """

    n_agents: int = 51          # must be odd so a minority always exists
    memory_window: int = 4      # w — rounds of history shown to each agent
    n_steps: int = 300
    seed: int = 42

    # Backend selection. "mock" is free and local. For LLM backends, construct
    # a backend object explicitly and pass it to the model (see backends.py).
    backend: str = "mock"
    model_platform: str = "anthropic"
    model_name: str = "claude-sonnet-5"
    temperature: float = 0.7

    persona_diversity: bool = True  # cycle distinct personas across agents
    max_retries: int = 2            # re-prompts allowed on unparseable output
    transcript_path: str | None = None  # JSONL log of every prompt/response

    def __post_init__(self):
        assert self.n_agents % 2 == 1, "n_agents must be odd"
        assert self.memory_window >= 0

    @property
    def alpha(self) -> float:
        """Information ratio 2^w / N, by analogy with the classical MG.

        For LLM agents there are no strategy tables, so this is not the
        Challet-Zhang phase-transition parameter in a strict sense — but it is
        the natural dimensionless ratio of distinguishable histories to agents.
        """
        return 2 ** self.memory_window / self.n_agents
