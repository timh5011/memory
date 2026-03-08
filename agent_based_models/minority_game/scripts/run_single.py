"""Run one Minority Game simulation and produce a 4-panel diagnostic plot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import MinorityGameModel, MinorityGameConfig
from sim.metrics import volatility, efficiency, predictability


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = MinorityGameConfig(n_steps=500, seed=42)
    model = MinorityGameModel(config)

    for _ in range(config.n_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    attendance = df["attendance"].values.astype(float)
    N = config.n_agents

    vol = volatility(attendance, N)
    eff = efficiency(attendance, N)
    pred = predictability(attendance, N)

    print(f"M={config.memory_length}, N={N}, α={config.alpha:.4f}")
    print(f"Volatility σ²/N = {vol:.4f}")
    print(f"Efficiency      = {eff:.4f}")
    print(f"Predictability  = {pred:.4f}")

    steps = np.arange(len(attendance))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Minority Game – Single Run (M={config.memory_length}, N={N}, α={config.alpha:.3f})",
        fontsize=14,
    )

    # Panel 1: Attendance over time
    axes[0, 0].plot(steps, attendance, color="steelblue", linewidth=0.5)
    axes[0, 0].axhline(N / 2, color="red", linestyle="--", alpha=0.7, label="N/2")
    axes[0, 0].set_title("Attendance (choosing 1) over Time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("n₁")
    axes[0, 0].legend()

    # Panel 2: Running volatility (trailing window)
    window = 50
    running_vol = np.array([
        np.var(attendance[max(0, i - window):i + 1]) / N
        for i in range(len(attendance))
    ])
    axes[0, 1].plot(steps, running_vol, color="goldenrod", linewidth=0.8)
    axes[0, 1].axhline(1 / 4, color="red", linestyle="--", alpha=0.7, label="Random (N/4)/N")
    axes[0, 1].set_title(f"Running Volatility (window={window})")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("σ²/N")
    axes[0, 1].legend()

    # Panel 3: Strategy score distribution
    all_best_scores = []
    for agent in model.agents:
        all_best_scores.append(float(agent.scores.max()))
    axes[1, 0].hist(all_best_scores, bins=30, color="mediumseagreen", edgecolor="white")
    axes[1, 0].set_title("Best Strategy Score Distribution")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Count")

    # Panel 4: Attendance autocorrelation
    A_centered = attendance - np.mean(attendance)
    autocorr = np.correlate(A_centered, A_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    max_lag = 50
    axes[1, 1].bar(range(max_lag), autocorr[:max_lag], color="coral", width=1.0)
    axes[1, 1].set_title("Attendance Autocorrelation")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out_path = results_dir / "single_run.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
