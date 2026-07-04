"""Single mock-backend run: free, local pipeline validation.

Produces a 4-panel diagnostic (attendance, attendance histogram, entropy rate,
conditional entropy) analogous to the classical Minority Game single-run plot.
No API calls, no cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minority_game import LLMMinorityGameConfig, LLMMinorityGameModel, volatility, efficiency, predictability
from minority_game.analysis.ks_entropy import block_entropy_analysis


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = LLMMinorityGameConfig(
        n_agents=51,
        memory_window=4,
        n_steps=300,
        seed=42,
        backend="mock",
        transcript_path=str(results_dir / "runs" / "mock_single_transcript.jsonl"),
    )
    print(f"Running mock LLM Minority Game: N={config.n_agents}, "
          f"w={config.memory_window}, {config.n_steps} rounds...")
    model = LLMMinorityGameModel(config).run()

    attendance = np.array(model.attendance)
    N = config.n_agents
    burn_in = min(50, config.n_steps // 5)

    print(f"  volatility σ²/N   = {volatility(attendance[burn_in:], N):.3f}")
    print(f"  efficiency        = {efficiency(attendance[burn_in:], N):.3f}")
    print(f"  predictability    = {predictability(attendance[burn_in:], N):.3f}")
    print(f"  unparseable resp. = {sum(a.n_fallbacks for a in model.agents)}")

    ks, H_k, h_rate, h_diff = block_entropy_analysis(
        model.outcomes, k_max=8, burn_in=burn_in
    )
    print(f"  entropy rate h(k={ks[-1]}) = {h_diff[-1]:.3f} bits/round")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"LLM Minority Game (mock backend) — N={N}, w={config.memory_window}",
        fontsize=14,
    )

    axes[0, 0].plot(attendance, lw=0.8, color="steelblue")
    axes[0, 0].axhline(N / 2, color="gray", ls="--", alpha=0.6, label="N/2")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Attendance (# choosing 1)")
    axes[0, 0].set_title("Attendance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(attendance[burn_in:], bins=20, color="steelblue", alpha=0.8)
    axes[0, 1].axvline(N / 2, color="gray", ls="--", alpha=0.6)
    axes[0, 1].set_xlabel("Attendance")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Attendance distribution (post burn-in={burn_in})")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ks, h_rate, "o-", color="crimson", ms=4)
    axes[1, 0].axhline(1.0, color="gray", ls="--", alpha=0.5, label="Random (1 bit)")
    axes[1, 0].set_xlabel("Block length k")
    axes[1, 0].set_ylabel("H(k)/k (bits)")
    axes[1, 0].set_title("Entropy rate H(k)/k")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ks, h_diff, "s-", color="mediumseagreen", ms=4)
    axes[1, 1].axhline(1.0, color="gray", ls="--", alpha=0.5, label="Random (1 bit)")
    axes[1, 1].set_xlabel("Block length k")
    axes[1, 1].set_ylabel("H(k) − H(k−1) (bits)")
    axes[1, 1].set_title("Conditional entropy h(k)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = results_dir / "mock_single_run.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
