"""Estimate API usage for a live LLM run BEFORE spending anything.

Renders the actual prompts a run would send and reports call counts and
approximate token totals (chars/4 heuristic). Multiply by your provider's
current per-token prices to get a dollar figure — prices are deliberately not
hardcoded here so they can't go stale.

Usage:
    python scripts/estimate_tokens.py --agents 9 --steps 30 --window 4 --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.prompts import build_system_prompt, render_observation, PERSONAS

TOKENS_PER_CHAR = 0.25   # rough heuristic
OUTPUT_TOKENS_PER_CALL = 5  # "0" or "1" plus wrapper tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", type=int, default=51)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--retry-factor", type=float, default=1.1,
                   help="Multiplier for re-prompts on unparseable output.")
    args = p.parse_args()

    system = build_system_prompt(args.agents, PERSONAS[0])
    # Worst-case observation: full window of history
    w = args.window
    user = render_observation(
        round_idx=args.steps - 1,
        history=[1] * w, own_actions=[0] * w, wins=w // 2,
        memory_window=w,
    )

    calls_per_run = args.agents * args.steps
    calls_total = int(calls_per_run * args.seeds * args.retry_factor)
    in_tokens_per_call = (len(system) + len(user)) * TOKENS_PER_CHAR
    in_total = int(calls_total * in_tokens_per_call)
    out_total = int(calls_total * OUTPUT_TOKENS_PER_CALL)

    print("=== LLM Minority Game usage estimate ===")
    print(f"agents={args.agents}  steps={args.steps}  window={w}  seeds={args.seeds}")
    print(f"API calls (incl. ~{(args.retry_factor - 1) * 100:.0f}% retries): {calls_total:,}")
    print(f"~input tokens/call:  {in_tokens_per_call:,.0f}")
    print(f"~input tokens total: {in_total:,}")
    print(f"~output tokens total: {out_total:,}")
    print()
    print("Multiply by your provider's current $/token to get cost.")
    print("Example prompts this estimate is based on:")
    print("-" * 40)
    print(system)
    print("-" * 40)
    print(user)


if __name__ == "__main__":
    main()
