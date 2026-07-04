"""Prompt templates and response parsing for the LLM Minority Game.

Design decision: agent memory is EXTERNAL and researcher-controlled. Every
round, each agent receives a fresh single-turn prompt containing exactly the
last `memory_window` rounds of history. We do not rely on the LLM framework's
internal conversation memory (e.g. CAMEL's ChatHistoryMemory), because then
the effective memory depth would be an opaque property of the framework
rather than the controlled variable of the experiment.
"""

from __future__ import annotations

import re

SYSTEM_TEMPLATE = """\
You are playing a repeated game called the Minority Game with {n_others} other players.

Rules:
- Each round, every player simultaneously chooses option 0 or option 1.
- After all choices are made, the side chosen by FEWER players wins that round.
- Your goal is to be on the minority side as often as possible.

{persona}

Each round you will be shown the recent history of winning options and your own
recent record. Reason about what the crowd is likely to do and try to be on the
less-crowded side.

IMPORTANT: Reply with exactly one character: 0 or 1. No explanation, no punctuation."""

GENERIC_PERSONA = "You are a rational player trying to maximize your number of wins."

# Personas are cycled across the population when persona_diversity is enabled.
# Heterogeneity in reasoning style is the LLM analog of the random strategy
# tables in the classical game — without it, identical agents given identical
# prompts tend to collapse onto the same choice (maximal herding).
PERSONAS = [
    "You are a contrarian: you believe crowds are usually wrong, so you bet against apparent trends.",
    "You are a trend-follower: you suspect that whatever has been winning will keep winning.",
    "You are a careful statistician: you count how often each option has won recently and reason from frequencies.",
    "You are impulsive and unpredictable: you like to surprise the other players.",
    "You are a cautious player: you prefer the option that has won less often lately, reasoning it is less crowded.",
]


def build_system_prompt(n_agents: int, persona: str) -> str:
    return SYSTEM_TEMPLATE.format(n_others=n_agents - 1, persona=persona)


def render_observation(round_idx: int, history: list[int],
                       own_actions: list[int], wins: int,
                       memory_window: int) -> str:
    """Render the per-round user prompt from the agent's bounded memory.

    `history` and `own_actions` are the agent's remembered rounds, oldest
    first, both at most `memory_window` long.
    """
    lines = [f"Round {round_idx + 1}."]
    if memory_window == 0 or not history:
        lines.append("You have no information about past rounds.")
    else:
        w = len(history)
        lines.append(
            f"Winning options of the last {w} rounds (oldest first): "
            + ", ".join(str(o) for o in history)
        )
        lines.append(
            f"Your choices in those rounds (oldest first): "
            + ", ".join(str(a) for a in own_actions)
        )
        lines.append(f"You were on the winning (minority) side in {wins} of those {w} rounds.")
    lines.append("Choose 0 or 1.")
    return "\n".join(lines)


def parse_action(text: str | None) -> int | None:
    """Extract a 0/1 action from a raw model response. None if unparseable."""
    if text is None:
        return None
    s = text.strip()
    if s in ("0", "1"):
        return int(s)
    m = re.search(r"\b([01])\b", s)
    if m:
        return int(m.group(1))
    return None
