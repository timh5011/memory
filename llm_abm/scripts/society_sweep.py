"""Sweep a memory knob of the Polis society — the central experiment.

Two knobs, one per invocation:
  --knob memory       sweep memory_window w (individual memory)
  --knob forgiveness  sweep reputation_decay (societal memory / forgiveness)

For each knob value and seed: run the society, then plot entropy (trajectory,
mobility) and success (welfare, Gini, value-alignment) against the knob, plus
the entropy-vs-welfare money plot.

Defaults to the free mock backend. Live runs require --live AND camel-ai AND
an API key, and are hard-capped by max_api_calls. Run
society_estimate_tokens.py first.

Usage:
    python scripts/society_sweep.py                          # mock, memory knob
    python scripts/society_sweep.py --knob forgiveness       # mock, decay knob
    python scripts/society_sweep.py --knob memory --backend camel --live \
        --agents 12 --steps 40 --seeds 1 --values 0,2,6      # small live run ($)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from society import SocietyConfig, SocietyModel
from sim.backends import CamelBackend
from analysis.society_entropy import trajectory_entropy, mobility, welfare_metrics

KNOBS = {
    "memory": ("memory_window", int, "0,1,2,4,8", "Individual memory window w"),
    "forgiveness": ("reputation_decay", float, "0.0,0.05,0.1,0.25,0.5",
                    "Societal forgetting (reputation decay)"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--knob", choices=list(KNOBS), default="memory")
    p.add_argument("--values", default=None,
                   help="Comma-separated knob values (defaults per knob).")
    p.add_argument("--backend", choices=["mock", "camel"], default="mock")
    p.add_argument("--live", action="store_true",
                   help="Required for paid backends. Guardrail against accidental spend.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--agents", type=int, default=24)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-api-calls", type=int, default=5000)
    return p.parse_args()


def make_backend(args):
    if args.backend == "mock":
        return None  # SocietyModel builds SocietyMockBackend itself
    if not args.live:
        raise SystemExit(
            "Refusing to run a paid backend without --live. "
            "Run scripts/society_estimate_tokens.py first.")
    return CamelBackend(model_name=args.model, allow_api_calls=True)


def main() -> None:
    args = parse_args()
    field, cast, default_values, knob_label = KNOBS[args.knob]
    values = [cast(v) for v in (args.values or default_values).split(",")]

    backend = make_backend(args)  # refuses paid backends without --live, up front

    results_dir = Path(__file__).resolve().parents[1] / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    burn_in = min(20, args.steps // 5)

    rows = []
    for value in values:
        for seed in range(args.seeds):
            kwargs = {field: value}
            config = SocietyConfig(
                n_agents=args.agents, n_steps=args.steps, seed=seed,
                backend=args.backend, model_name=args.model,
                max_api_calls=args.max_api_calls,
                transcript_path=str(
                    runs_dir / f"society_{args.knob}_{value}_s{seed}.jsonl"
                ) if args.backend != "mock" else None,
                **kwargs,
            )
            print(f"{field}={value} seed={seed} ...", end=" ", flush=True)
            model = SocietyModel(config, backend=backend).run()
            record = model.to_record()

            F = np.asarray(record["fulfillment"])
            _, _, _, h_diff = trajectory_entropy(F, burn_in=burn_in)
            _, h_mob = mobility(F, burn_in=burn_in)
            wm = welfare_metrics(record)
            rows.append({
                "knob": args.knob, "value": value, "seed": seed,
                "trajectory_entropy": h_diff[-1],
                "mobility_entropy": h_mob,
                **wm,
                "aborted": record["aborted"],
            })
            print(f"h_traj={h_diff[-1]:.3f}  h_mob={h_mob:.3f}  "
                  f"welfare={wm['welfare']:.3f}")

    out_json = runs_dir / f"society_sweep_{args.knob}_{args.backend}.json"
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Saved sweep records to {out_json}")

    # --- Plots ---
    vals = np.array([r["value"] for r in rows], dtype=float)
    uniq = np.array(sorted(set(vals)))

    def mean_of(key):
        arr = np.array([r[key] for r in rows])
        return arr, np.array([arr[vals == v].mean() for v in uniq])

    h_traj, h_traj_m = mean_of("trajectory_entropy")
    h_mob, h_mob_m = mean_of("mobility_entropy")
    welfare, welfare_m = mean_of("welfare")
    gini_v, gini_m = mean_of("fulfillment_gini")
    align, align_m = mean_of("value_alignment")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Polis sweep: {knob_label} — backend={args.backend}, "
                 f"N={args.agents}, {args.seeds} seeds", fontsize=14)

    ax = axes[0]
    ax.scatter(vals, h_traj, alpha=0.5, s=25, color="steelblue")
    ax.plot(uniq, h_traj_m, "o-", color="steelblue", label="trajectory entropy")
    ax.scatter(vals, h_mob, alpha=0.5, s=25, color="crimson")
    ax.plot(uniq, h_mob_m, "s-", color="crimson", label="mobility entropy")
    ax.set_xlabel(knob_label)
    ax.set_ylabel("Entropy rate (bits/season)")
    ax.set_title("KS entropy vs memory knob")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(vals, welfare, alpha=0.5, s=25, color="steelblue")
    ax.plot(uniq, welfare_m, "o-", color="steelblue", label="welfare")
    ax.scatter(vals, align, alpha=0.5, s=25, color="mediumseagreen")
    ax.plot(uniq, align_m, "^-", color="mediumseagreen", label="value alignment")
    ax.scatter(vals, gini_v, alpha=0.5, s=25, color="crimson")
    ax.plot(uniq, gini_m, "s-", color="crimson", label="fulfillment Gini")
    ax.set_xlabel(knob_label)
    ax.set_title("Success metrics vs memory knob")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(h_traj, welfare, c=vals, cmap="viridis", s=35)
    fig.colorbar(sc, ax=ax, label=knob_label)
    ax.set_xlabel("Trajectory entropy rate (bits/season)")
    ax.set_ylabel("Welfare (mean fulfillment)")
    ax.set_title("Welfare vs KS entropy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = results_dir / f"society_sweep_{args.knob}_{args.backend}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Plot saved to {out_png}")


if __name__ == "__main__":
    main()
