"""Single mock run of the Polis society: free, local pipeline validation.

Produces a 4-panel diagnostic: fulfillment trajectories by class of origin,
welfare and inequality over time, the rank-mobility transition matrix, and
entropy-rate curves per sub-population. No API calls, no cost.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from society import SocietyConfig, SocietyModel
from society.analysis.society_entropy import (
    trajectory_entropy, macro_entropy, mobility, partitions,
    welfare_metrics, gini,
)

CLASS_COLORS = {"lower": "crimson", "middle": "steelblue", "upper": "mediumseagreen"}


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = SocietyConfig(
        n_agents=24, n_steps=120, seed=42, backend="mock",
        transcript_path=str(results_dir / "runs" / "mock_society_transcript.jsonl"),
    )
    print(f"Running Polis (mock): N={config.n_agents}, {config.n_steps} seasons, "
          f"w={config.memory_window}, reputation_decay={config.reputation_decay}...")
    model = SocietyModel(config).run()
    record = model.to_record()
    model.save(results_dir / "runs" / "mock_society_run.json")

    F = np.asarray(record["fulfillment"])          # (T, N)
    burn_in = 20
    parts = partitions(record["identities"])

    # --- Console summary ---
    actions = Counter(a for season in record["actions"] for a in season)
    total = sum(actions.values())
    print("  action mix: " + ", ".join(
        f"{k} {100 * v / total:.0f}%" for k, v in actions.most_common()))
    wm = welfare_metrics(record)
    print(f"  welfare={wm['welfare']:.3f}  gini={wm['fulfillment_gini']:.3f}  "
          f"value_alignment={wm['value_alignment']:.3f}")
    P, h_mob = mobility(F, burn_in=burn_in)
    print(f"  mobility entropy rate = {h_mob:.3f} bits/season "
          f"(0=frozen hierarchy, {np.log2(5):.2f}=full reshuffle)")
    print(f"  unparseable responses: {record['n_fallbacks']}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Polis (mock backend) — N={config.n_agents}, w={config.memory_window}, "
        f"reputation_decay={config.reputation_decay}", fontsize=14)

    ax = axes[0, 0]
    for cls, ids in parts["social_class"].items():
        color = CLASS_COLORS[cls]
        for i in ids:
            ax.plot(F[:, i], color=color, alpha=0.15, lw=0.6)
        ax.plot(F[:, ids].mean(axis=1), color=color, lw=2.2,
                label=f"{cls} (n={len(ids)})")
    ax.set_xlabel("Season")
    ax.set_ylabel("Fulfillment F")
    ax.set_title("Fulfillment trajectories by class of origin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(F.mean(axis=1), color="steelblue", label="mean fulfillment (welfare)")
    ginis = [gini(F[t]) for t in range(len(F))]
    ax.plot(ginis, color="crimson", label="fulfillment Gini")
    ax.set_xlabel("Season")
    ax.set_title("Welfare and inequality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, ax=ax, label="P(next class | class)")
    ax.set_xlabel("Rank quintile next season")
    ax.set_ylabel("Rank quintile this season")
    ax.set_title(f"Mobility matrix (h = {h_mob:.3f} bits/season)")

    ax = axes[1, 1]
    for cls, ids in parts["social_class"].items():
        ks, _, h_rate, _ = trajectory_entropy(F, ids, burn_in=burn_in)
        ax.plot(ks, h_rate, "o-", color=CLASS_COLORS[cls], ms=4,
                label=f"trajectories: {cls}")
    ks, _, h_rate_macro, _ = macro_entropy(F, burn_in=burn_in)
    ax.plot(ks, h_rate_macro, "s--", color="gray", ms=4, label="macro state")
    ax.set_xlabel("Block length k")
    ax.set_ylabel("H(k)/k (bits)")
    ax.set_title("Entropy rates by sub-population")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = results_dir / "mock_society_run.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
