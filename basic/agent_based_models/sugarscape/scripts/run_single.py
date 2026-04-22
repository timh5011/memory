"""Run one Sugarscape simulation and produce a 4-panel diagnostic plot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import SugarscapeModel, SugarscapeConfig
from sim.metrics import gini


def main() -> None:
    """Run a single simulation and save diagnostic plots."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = SugarscapeConfig(n_steps=500, seed=42)
    model = SugarscapeModel(config)

    for _ in range(config.n_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    steps = np.arange(len(df))
    population = df["population"].values
    mean_sugar = df["mean_sugar"].values
    gini_vals = df["gini"].values

    final_gini = float(np.mean(gini_vals[-50:]))
    final_mean = float(np.mean(mean_sugar[-50:]))

    print(f"Final Gini (last 50 steps avg):       {final_gini:.4f}")
    print(f"Final mean sugar (last 50 steps avg): {final_mean:.4f}")

    # Collect final wealth for histogram
    final_wealth = np.array([a.sugar for a in model.agents], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Sugarscape – Single Run (seed=42, 500 steps)", fontsize=14)

    axes[0, 0].plot(steps, population, color="steelblue")
    axes[0, 0].set_title("Population over Time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Population")

    axes[0, 1].plot(steps, mean_sugar, color="goldenrod")
    axes[0, 1].set_title("Mean Sugar over Time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Mean Sugar")

    axes[1, 0].plot(steps, gini_vals, color="crimson")
    axes[1, 0].set_title("Gini Coefficient over Time")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Gini")

    axes[1, 1].hist(final_wealth, bins=30, color="mediumseagreen", edgecolor="white")
    axes[1, 1].set_title(f"Wealth Distribution (Final Step)\nGini = {gini(final_wealth):.3f}")
    axes[1, 1].set_xlabel("Sugar (wealth)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    out_path = results_dir / "single_run.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
