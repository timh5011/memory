"""Agent backends: the pluggable "brain" behind each agent.

A backend answers one single-turn question per round: given a system prompt
and an observation, return a text response containing 0 or 1. Because memory
is externalized into the prompt (see prompts.py), backends are stateless
across rounds, which makes mock and LLM backends exactly interchangeable.

Two implementations:

- MockBackend: free, local, no network. A mixture of simple behavioral
  policies (contrarian, trend-follower, frequency-based, random) that stands
  in for LLM reasoning so the entire pipeline — simulation, transcripts,
  metrics, KS entropy analysis, sweeps, plots — can be built and validated at
  zero cost before a single API call is ever made.

- CamelBackend: wraps a CAMEL ChatAgent. Requires `pip install camel-ai` and
  an API key, and must be constructed with allow_api_calls=True as an
  explicit guardrail against accidental spend. UNTESTED until camel-ai is
  installed — verify the ModelFactory/ChatAgent calls against the installed
  version before the first live run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AgentBackend(ABC):
    """Interface for anything that can play one round of the game."""

    name: str = "base"
    is_free: bool = False

    @abstractmethod
    def respond(self, agent_id: int, system_prompt: str, user_prompt: str,
                observation: dict, rng: np.random.Generator) -> str:
        """Return the raw text response for one round.

        `observation` carries the structured form of the user prompt
        (history, own_actions, round) so mock backends don't have to parse
        text. LLM backends should use only the rendered prompts.
        """
        ...


class MockBackend(AgentBackend):
    """Rule-based stand-in for an LLM. Free and fully local."""

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


class CamelBackend(AgentBackend):
    """CAMEL ChatAgent backend. Costs money — guarded by allow_api_calls.

    Each respond() call builds a fresh single-turn ChatAgent so that no
    hidden conversation state accumulates inside CAMEL: the only memory the
    agent has is what render_observation() put in the prompt.
    """

    name = "camel"
    is_free = False

    def __init__(self, model_platform: str = "anthropic",
                 model_name: str = "claude-sonnet-5",
                 temperature: float = 0.7,
                 allow_api_calls: bool = False):
        if not allow_api_calls:
            raise RuntimeError(
                "CamelBackend makes paid API calls. Construct it with "
                "allow_api_calls=True to confirm you intend to spend money. "
                "Run scripts/estimate_tokens.py first to see the expected usage."
            )
        try:
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType
        except ImportError as e:
            raise ImportError(
                "camel-ai is not installed. Install with `pip install camel-ai` "
                "(see llm_abm/requirements.txt). The rest of the pipeline runs "
                "without it via MockBackend."
            ) from e

        self._ChatAgent = ChatAgent
        # NOTE: verify against the installed camel-ai version — the
        # ModelFactory signature has changed across releases.
        self._model = ModelFactory.create(
            model_platform=ModelPlatformType[model_platform.upper()],
            model_type=model_name,
            model_config_dict={"temperature": temperature},
        )

    def respond(self, agent_id, system_prompt, user_prompt, observation, rng):
        agent = self._ChatAgent(system_message=system_prompt, model=self._model)
        response = agent.step(user_prompt)
        return response.msgs[0].content


def build_backend(config) -> AgentBackend:
    """Build the free default backend from a config.

    Deliberately refuses to build paid backends: those must be constructed
    explicitly by the caller so that spending money is always a visible,
    intentional line of code.
    """
    if config.backend == "mock":
        return MockBackend()
    raise ValueError(
        f"backend={config.backend!r} must be constructed explicitly "
        "(e.g. CamelBackend(..., allow_api_calls=True)) and passed to "
        "LLMMinorityGameModel. Only 'mock' is built automatically."
    )
