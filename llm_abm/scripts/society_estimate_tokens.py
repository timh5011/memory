"""Estimate API usage for a live Polis run BEFORE spending anything.

Samples the real population for the given seed, renders the actual system
prompt and a full-memory observation, and reports call and token totals
(chars/4 heuristic). Multiply by current per-token prices yourself — prices
are deliberately not hardcoded so they can't go stale.

Usage:
    python scripts/society_estimate_tokens.py --agents 24 --steps 120 --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from society import SocietyConfig
from society.identity import sample_population
from society.world import WorldEngine
from society.prompts import build_system_prompt, render_observation
from society.identity import DIMENSIONS

TOKENS_PER_CHAR = 0.25
OUTPUT_TOKENS_PER_CALL = 60  # JSON object + one-sentence reason


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", type=int, default=24)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--retry-factor", type=float, default=1.1)
    args = p.parse_args()

    config = SocietyConfig(n_agents=args.agents, n_steps=args.steps,
                           memory_window=args.window)
    rng = np.random.default_rng(config.seed)
    identities = sample_population(config.n_agents, rng)
    engine = WorldEngine(config, identities, rng)
    names = [i.name for i in identities]

    ident = identities[0]
    system = build_system_prompt(ident, names)

    # Worst-case observation: a full memory window of busy seasons
    norms_row = engine.dimension_norms()[0]
    norms = {d: float(norms_row[k]) for k, d in enumerate(DIMENSIONS)}
    busy_season = [
        "You worked as a farmer and earned 4.2 coins.",
        f"{names[1]} gave you 3.0 coins in your need.",
        f"You attended {names[2]}'s gathering.",
    ]
    user = render_observation(
        season=args.steps - 1, wealth=25.0, norms=norms,
        memory=[list(busy_season) for _ in range(args.window)],
        bulletin=["A quiet season in Polis.", f"{names[3]} hosted a gathering."],
        top_ties=[(names[j], 0.5) for j in range(1, 6)],
        memory_window=args.window,
    )

    calls_per_run = args.agents * args.steps
    calls_total = int(calls_per_run * args.seeds * args.retry_factor)
    in_per_call = (len(system) + len(user)) * TOKENS_PER_CHAR
    in_total = int(calls_total * in_per_call)
    out_total = int(calls_total * OUTPUT_TOKENS_PER_CALL)

    print("=== Polis society usage estimate ===")
    print(f"agents={args.agents}  steps={args.steps}  window={args.window}  "
          f"seeds={args.seeds}")
    print(f"API calls (incl. ~{(args.retry_factor - 1) * 100:.0f}% retries): {calls_total:,}")
    print(f"~input tokens/call:   {in_per_call:,.0f}")
    print(f"~input tokens total:  {in_total:,}")
    print(f"~output tokens total: {out_total:,}")
    print(f"suggested --max-api-calls: {int(calls_total * 1.2):,}")
    print()
    print("Multiply by your provider's current $/token to get cost.")
    print("Example system prompt this estimate is based on:")
    print("-" * 50)
    print(system)
    print("-" * 50)
    print(user)


if __name__ == "__main__":
    main()
