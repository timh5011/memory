"""Configuration for the Polis society simulation."""

from dataclasses import dataclass


@dataclass
class SocietyConfig:
    """All parameters needed to fully specify a Polis run.

    The two memory knobs of the experiment:
      - memory_window (w): individual memory — how many past seasons of events
        appear in each agent's prompt. The prompt IS the agent's memory.
      - reputation_decay / tie_decay: societal memory — how fast the
        institution forgets standing and relationships. High decay = a
        forgiving society; low decay = grudges and glory persist.
    """

    n_agents: int = 24
    n_steps: int = 120          # seasons
    seed: int = 42

    # --- memory knobs ---
    memory_window: int = 4      # w — seasons of events in each agent's prompt
    reputation_decay: float = 0.10   # per-season decay of standing
    tie_decay: float = 0.05          # per-season decay of non-household ties

    # --- economy ---
    cost_of_living: float = 3.0
    gathering_cost: float = 8.0

    # --- backend ---
    backend: str = "mock"       # "mock" is free; LLM backends passed explicitly
    model_platform: str = "anthropic"
    model_name: str = "claude-sonnet-5"
    temperature: float = 0.7
    max_retries: int = 2
    transcript_path: str | None = None

    # --- budget guardrail (live backends only) ---
    # Hard cap on total API calls for the run, retries included. The model
    # stops gracefully (saving partial records) rather than exceed it.
    # Ignored for free backends.
    max_api_calls: int | None = 5000

    def __post_init__(self):
        assert self.n_agents >= 6, "need at least a few households"
        assert self.memory_window >= 0
        assert 0.0 <= self.reputation_decay <= 1.0
        assert 0.0 <= self.tie_decay <= 1.0
