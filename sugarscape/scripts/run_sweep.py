"""Run the Sugarscape parameter sweep and produce heatmap visualizations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.experiment import run_sweep


PARAM_GRID = {
    "max_vision": [1, 2, 3, 6],
    "alpha": [0.25, 0.5, 1.0, 2.0, 4.0],
}
N_SEEDS = 5
N_STEPS = 300


def _pivot_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot sweep results to a (max_vision x alpha) mean-value table."""
    return (
        df.groupby(["max_vision", "alpha"])[value_col]
        .mean()
        .unstack(level="alpha")
    )


def _save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    fmt: str = ".3f",
) -> None:
    """Render and save a seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Alpha (sugar regrowth rate)")
    ax.set_ylabel("Max Vision")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    """Run sweep and generate heatmaps."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    df = run_sweep(PARAM_GRID, n_steps=N_STEPS, n_seeds=N_SEEDS, output_dir=results_dir)

    # Gini heatmap
    pivot_gini = _pivot_mean(df, "final_gini")
    _save_heatmap(
        pivot_gini,
        "Mean Gini Coefficient (last 50 steps avg)",
        results_dir / "sweep_gini_heatmap.png",
        cmap="Reds",
    )

    # KS entropy heatmap
    pivot_ks = _pivot_mean(df, "ks_entropy")
    _save_heatmap(
        pivot_ks,
        "Approximate KS Entropy of Mean-Sugar Series",
        results_dir / "sweep_ks_heatmap.png",
        cmap="YlOrBr",
        fmt=".4f",
    )

    # Social mobility heatmap
    pivot_smi = _pivot_mean(df, "social_mobility_index")
    _save_heatmap(
        pivot_smi,
        "Social Mobility Index (Spearman rank corr, lower = more mobile)",
        results_dir / "sweep_mobility_heatmap.png",
        cmap="Blues_r",
    )

    print("\nExpected row count:", len(PARAM_GRID["max_vision"]) * len(PARAM_GRID["alpha"]) * N_SEEDS)
    print("Actual row count:  ", len(df))


if __name__ == "__main__":
    main()
