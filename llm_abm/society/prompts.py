"""Prompts and response parsing for Polis agents.

Same externalized-memory principle as the Minority Game: every backend call
is a fresh single-turn prompt. What the agent "remembers" is exactly the last
`memory_window` seasons of events rendered here — nothing more. That makes
individual memory an exact experimental variable.
"""

from __future__ import annotations

import json
import re

from .identity import Identity, DIMENSIONS
from .world import ACTIONS

SYSTEM_TEMPLATE = """\
You are {name}, a {role} in the small town of Polis.

Who you are:
- Class of origin: {social_class}
- Household: you live with {household_names}
- Political leaning: you support the {faction}
- Temperament: {temperament_desc}

What you care about (your personal values, in order of importance):
{values_desc}
These values are who you are. Let them guide every decision — someone who
cares most about {dominant} should live very differently from someone who
cares most about money or reputation.

Each season you choose ONE action:
- WORK — earn coins at your trade
- SOCIALIZE — deepen one relationship (requires "target": a person's name)
- HELP — give coins to someone in need (requires "target" and "amount"; raises your standing)
- HOST_GATHERING — spend coins to host friends and family (raises ties and standing)
- VENTURE — risk a third of your coins on a gamble that may pay double
- ADVOCATE — campaign for your faction (standing up with allies, down with rivals)
- REST — recover your strength and enjoy quiet time with your household

Respond with ONLY a JSON object, no other text:
{{"action": "<ACTION>", "target": "<name or null>", "amount": <number or null>, "reason": "<one short sentence>"}}"""

TEMPERAMENT_WORDS = {
    "ambition": ("driven and ambitious", "content with what you have"),
    "gregariousness": ("outgoing and sociable", "private and solitary"),
    "progressivism": ("progressive, drawn to change", "traditional, loyal to old ways"),
}

DIMENSION_PHRASES = {
    "prosperity": "financial prosperity",
    "belonging": "family and friendships",
    "standing": "respect and reputation in town",
    "security": "safety and stability",
}


def _temperament_desc(temperament: dict[str, float]) -> str:
    parts = []
    for axis, (pos, neg) in TEMPERAMENT_WORDS.items():
        v = temperament[axis]
        if abs(v) < 0.25:
            continue
        parts.append(pos if v > 0 else neg)
    return ", ".join(parts) if parts else "even-keeled and moderate"


def _values_desc(weights: dict[str, float]) -> str:
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    return "\n".join(f"- {DIMENSION_PHRASES[d]}: {w * 100:.0f}%"
                     for d, w in ranked)


def build_system_prompt(ident: Identity, all_names: list[str]) -> str:
    household_names = ", ".join(all_names[j] for j in ident.household
                                if j != ident.agent_id) or "no one"
    return SYSTEM_TEMPLATE.format(
        name=ident.name,
        role=ident.role,
        social_class=ident.social_class,
        household_names=household_names,
        faction=ident.faction,
        temperament_desc=_temperament_desc(ident.temperament),
        values_desc=_values_desc(ident.value_weights),
        dominant=DIMENSION_PHRASES[ident.dominant_value],
    )


def _level(x: float) -> str:
    if x < 0.25:
        return "very low"
    if x < 0.45:
        return "low"
    if x < 0.65:
        return "moderate"
    if x < 0.85:
        return "high"
    return "very high"


def render_observation(season: int, wealth: float, norms: dict[str, float],
                       memory: list[list[str]], bulletin: list[str],
                       top_ties: list[tuple[str, float]],
                       memory_window: int) -> str:
    lines = [f"Season {season + 1}.", ""]
    lines.append(f"Your coins: {wealth:.1f}")
    lines.append("Your life right now: " + "; ".join(
        f"{DIMENSION_PHRASES[d]} is {_level(norms[d])}" for d in DIMENSIONS))
    if top_ties:
        lines.append("Closest people: " + ", ".join(
            f"{name} ({_level(s)})" for name, s in top_ties))

    if memory_window == 0 or not memory:
        lines.append("")
        lines.append("You recall nothing of past seasons.")
    else:
        lines.append("")
        lines.append(f"What you remember of the last {len(memory)} season(s):")
        for age, season_events in enumerate(memory):
            label = "last season" if age == len(memory) - 1 else f"{len(memory) - age} seasons ago"
            for ev in season_events:
                lines.append(f"- ({label}) {ev}")

    lines.append("")
    lines.append("Town bulletin: " + " ".join(bulletin))
    lines.append("")
    lines.append("Choose your action for this season. JSON only.")
    return "\n".join(lines)


def parse_action(text: str | None, name_to_id: dict[str, int]) -> dict | None:
    """Extract {action, target(id), amount} from a raw response. None if unusable."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    action = str(obj.get("action", "")).strip().upper().replace(" ", "_")
    if action not in ACTIONS:
        return None
    target = obj.get("target")
    target_id = None
    if isinstance(target, str):
        target_id = name_to_id.get(target.strip().capitalize())
    amount = obj.get("amount")
    try:
        amount = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount = None
    return {"action": action, "target": target_id, "amount": amount,
            "reason": str(obj.get("reason", ""))[:200]}
