"""Sweep the agent memory window w — the central experiment.

For each memory window and seed, run the game, then plot:
  1. efficiency vs w
  2. KS entropy rate vs w
  3. entropy vs efficiency (the "money plot", LLM edition)

Defaults to the free mock backend. A live LLM run via CAMEL requires the
--live flag AND camel-ai installed AND an API key; without --live the script
cannot spend money. Always run estimate_tokens.py before a live run.

Usage:
    python scripts/run_sweep_memory.py                       # free mock sweep
    python scripts/run_sweep_memory.py --windows 0,1,2,4,8   # custom windows
    python scripts/run_sweep_memory.py --backend camel --live \
        --agents 9 --steps 30 --seeds 1                      # small live run ($)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minority_game import LLMMinorityGameConfig, LLMMinorityGameModel, efficiency, volatility
from backends import CamelBackend
from minority_game.analysis.ks_entropy import entropy_rate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "camel"], default="mock")
    p.add_argument("--live", action="store_true",
                   help="Required for paid backends. Guardrail against accidental spend.")
    p.add_argument("--windows", default="0,1,2,3,4,6,8",
                   help="Comma-separated memory windows to sweep.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--agents", type=int, default=51)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def make_backend(args):
    if args.backend == "mock":
        return None  # model builds MockBackend itself
    if not args.live:
        raise SystemExit(
            "Refusing to run a paid backend without --live. "
            "Run scripts/estimate_tokens.py first to see expected usage."
        )
    return CamelBackend(model_name=args.model, temperature=args.temperature,
                        allow_api_calls=True)


def main() -> None:
    args = parse_args()
    windows = [int(w) for w in args.windows.split(",")]
    backend = make_backend(args)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    burn_in = min(50, args.steps // 5)

    records = []
    for w in windows:
        for seed in range(args.seeds):
            config = LLMMinorityGameConfig(
                n_agents=args.agents,
                memory_window=w,
                n_steps=args.steps,
                seed=seed,
                backend=args.backend,
                model_name=args.model,
                temperature=args.temperature,
                transcript_path=str(
                    runs_dir / f"sweep_{args.backend}_w{w}_s{seed}.jsonl"
                ) if args.backend != "mock" else None,
            )
            print(f"w={w} seed={seed} ...", end=" ", flush=True)
            model = LLMMinorityGameModel(config, backend=backend).run()

            attendance = np.array(model.attendance)[burn_in:]
            rec = model.to_record()
            rec["metrics"] = {
                "efficiency": efficiency(attendance, args.agents),
                "volatility": volatility(attendance, args.agents),
                "entropy_rate": entropy_rate(model.outcomes, k_max=8, burn_in=burn_in),
            }
            records.append(rec)
            print(f"eff={rec['metrics']['efficiency']:.3f} "
                  f"h={rec['metrics']['entropy_rate']:.3f} bits")

    out_json = runs_dir / f"sweep_{args.backend}.json"
    with open(out_json, "w") as fh:
        json.dump(records, fh)
    print(f"Saved run records to {out_json}")

    # --- Plots ---
    ws = np.array([r["config"]["memory_window"] for r in records])
    effs = np.array([r["metrics"]["efficiency"] for r in records])
    hs = np.array([r["metrics"]["entropy_rate"] for r in records])
    uniq = np.array(sorted(set(ws)))
    eff_mean = np.array([effs[ws == w].mean() for w in uniq])
    h_mean = np.array([hs[ws == w].mean() for w in uniq])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"LLM Minority Game memory sweep — backend={args.backend}, "
        f"N={args.agents}, {args.seeds} seeds", fontsize=14,
    )

    axes[0].scatter(ws, effs, alpha=0.5, s=25)
    axes[0].plot(uniq, eff_mean, "o-", color="crimson", label="mean")
    axes[0].set_xlabel("Memory window w")
    axes[0].set_ylabel("Efficiency")
    axes[0].set_title("Efficiency vs memory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ws, hs, alpha=0.5, s=25)
    axes[1].plot(uniq, h_mean, "o-", color="crimson", label="mean")
    axes[1].axhline(1.0, color="gray", ls="--", alpha=0.5, label="Random (1 bit)")
    axes[1].set_xlabel("Memory window w")
    axes[1].set_ylabel("Entropy rate (bits/round)")
    axes[1].set_title("KS entropy vs memory")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    sc = axes[2].scatter(hs, effs, c=ws, cmap="viridis", s=30)
    axes[2].plot(h_mean, eff_mean, "-", color="gray", alpha=0.6)
    fig.colorbar(sc, ax=axes[2], label="memory window w")
    axes[2].set_xlabel("Entropy rate (bits/round)")
    axes[2].set_ylabel("Efficiency")
    axes[2].set_title("Efficiency vs KS entropy")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = results_dir / f"sweep_memory_{args.backend}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Plot saved to {out_png}")


if __name__ == "__main__":
    main()
