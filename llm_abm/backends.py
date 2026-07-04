"""Shared agent-backend abstraction — the pluggable "brain" behind an agent.

General across simulations: a backend answers one single-turn question per
round — given a system prompt and an observation, return a text response.
Because memory is externalized into the prompt (each simulation's prompts
module builds it), backends are stateless across rounds, which makes mock and
LLM backends exactly interchangeable.

This module holds only the *general* pieces:

- AgentBackend: the interface every backend implements.
- CamelBackend: wraps a CAMEL ChatAgent for live (paid) runs. Requires
  `pip install camel-ai` and an API key, and must be constructed with
  allow_api_calls=True as an explicit guardrail against accidental spend.
  UNTESTED until camel-ai is installed — verify the ModelFactory/ChatAgent
  calls against the installed version before the first live run.
- refuse_paid_backend(): the shared guardrail that keeps paid backends from
  ever being auto-constructed.

Each simulation supplies its own free MockBackend (its action space differs),
e.g. minority_game.backends.MockBackend and society.mock_policy.SocietyMockBackend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AgentBackend(ABC):
    """Interface for anything that can play one round of a simulation."""

    name: str = "base"
    is_free: bool = False

    @abstractmethod
    def respond(self, agent_id: int, system_prompt: str, user_prompt: str,
                observation: dict, rng: np.random.Generator) -> str:
        """Return the raw text response for one round.

        `observation` carries the structured form of the user prompt so mock
        backends don't have to parse text. LLM backends should use only the
        rendered prompts.
        """
        ...


class CamelBackend(AgentBackend):
    """CAMEL ChatAgent backend. Costs money — guarded by allow_api_calls.

    Each respond() call builds a fresh single-turn ChatAgent so that no
    hidden conversation state accumulates inside CAMEL: the only memory the
    agent has is what the prompt put in it.
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
                "Run the simulation's estimate_tokens script first to see the "
                "expected usage."
            )
        try:
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType
        except ImportError as e:
            raise ImportError(
                "camel-ai is not installed. Install with `pip install camel-ai` "
                "(see llm_abm/requirements.txt). The rest of the pipeline runs "
                "without it via each simulation's MockBackend."
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


def refuse_paid_backend(backend_name: str) -> "AgentBackend":
    """Never auto-construct a paid backend.

    Simulations call this from their build_backend() for any non-mock backend:
    spending money must always be a visible, intentional line of code in the
    caller (e.g. CamelBackend(..., allow_api_calls=True)).
    """
    raise ValueError(
        f"backend={backend_name!r} must be constructed explicitly "
        "(e.g. CamelBackend(..., allow_api_calls=True)) and passed to the "
        "model. Only 'mock' is built automatically."
    )
