"""Plot social mobility index vs KS entropy from sweep results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    """Scatter plot of social mobility vs KS entropy, with per-param-combo means overlaid."""
    df = pd.read_csv(RESULTS_DIR / "sweep_results.csv")
    df = df.dropna(subset=["ks_entropy", "social_mobility_index"])

    # Colour by max_vision, marker style by alpha
    vision_values = sorted(df["max_vision"].unique())
    alpha_values = sorted(df["alpha"].unique())

    vision_colors = {v: c for v, c in zip(vision_values, ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"])}
    alpha_markers = {a: m for a, m in zip(alpha_values, ["o", "s", "D", "^", "P"])}

    fig, ax = plt.subplots(figsize=(9, 6))

    # Individual seed runs (faint)
    for _, row in df.iterrows():
        ax.scatter(
            row["ks_entropy"],
            row["social_mobility_index"],
            color=vision_colors[row["max_vision"]],
            marker=alpha_markers[row["alpha"]],
            alpha=0.35,
            s=55,
            linewidths=0,
        )

    # Per (max_vision, alpha) means — fully opaque, larger
    means = df.groupby(["max_vision", "alpha"])[["ks_entropy", "social_mobility_index"]].mean().reset_index()
    for _, row in means.iterrows():
        ax.scatter(
            row["ks_entropy"],
            row["social_mobility_index"],
            color=vision_colors[row["max_vision"]],
            marker=alpha_markers[row["alpha"]],
            s=160,
            linewidths=0.8,
            edgecolors="white",
            zorder=5,
        )

    # Linear trend line across all points
    x = df["ks_entropy"].values
    y = df["social_mobility_index"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.4, linestyle="--",
            label=f"Linear fit  (slope = {m:.2f})", zorder=4)

    # Correlation annotation
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.97, 0.97, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    # Legends
    vision_handles = [
        mlines.Line2D([], [], color=vision_colors[v], marker="o", linestyle="None",
                      markersize=8, label=f"max_vision = {v}")
        for v in vision_values
    ]
    alpha_handles = [
        mlines.Line2D([], [], color="gray", marker=alpha_markers[a], linestyle="None",
                      markersize=8, label=f"alpha = {a}")
        for a in alpha_values
    ]
    trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.4,
                                 label=f"Linear fit  (slope = {m:.2f})")

    leg1 = ax.legend(handles=vision_handles, title="Color = max_vision",
                     loc="upper left", fontsize=9, title_fontsize=9)
    leg2 = ax.legend(handles=alpha_handles + [trend_handle], title="Marker = alpha",
                     loc="lower right", fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)

    ax.set_xlabel("Approximate KS Entropy (K2, nats/step)", fontsize=12)
    ax.set_ylabel("Social Mobility Index\n(Spearman rank corr, lower = more mobile)", fontsize=12)
    ax.set_title("Social Mobility vs KS Entropy\n"
                 "Faint points = individual seeds · Bold points = per-condition means",
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    out_path = RESULTS_DIR / "mobility_vs_ks_entropy.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"Pearson r(KS entropy, social mobility) = {r:.4f}")
    print(f"Linear fit: mobility = {m:.3f} * KS_entropy + {b:.3f}")


if __name__ == "__main__":
    main()
