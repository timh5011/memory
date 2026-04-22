"""Approach 1: KS entropy of the Sugarscape wealth distribution state.

The 'state' at each step is the wealth distribution binned into a histogram tuple.
Block counting on this symbolic sequence estimates the entropy rate of the
macro-level wealth dynamics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from project root and ergodic_systems
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ergodic_systems"))

from sim import SugarscapeModel, SugarscapeConfig
from entropy.block_counting import empirical_block_distribution, shannon_entropy

# --- Parameters ---
N_STEPS = 5000
SEED = 42
WEALTH_BIN_EDGES = [0, 10, 25, 50, 100, float("inf")]  # 5 bins
K_MAX = 8
BURNIN = 200  # discard transient


def wealth_histogram_symbol(agents, bin_edges):
    """Bin agent sugar values and return a hashable histogram tuple."""
    sugars = np.array([a.sugar for a in agents], dtype=float)
    counts = np.histogram(sugars, bins=bin_edges)[0]
    return tuple(counts.tolist())


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = SugarscapeConfig(n_steps=N_STEPS, seed=SEED)
    model = SugarscapeModel(config)

    # Run simulation, recording histogram symbol at each step
    symbols = []
    snapshot_steps = [0, 500, 1000, 2500, 5000]
    snapshots = {}  # step -> array of sugar values

    for step in range(1, N_STEPS + 1):
        model.step()
        sym = wealth_histogram_symbol(model.agents, WEALTH_BIN_EDGES)
        symbols.append(sym)
        if step in snapshot_steps:
            snapshots[step] = np.array([a.sugar for a in model.agents], dtype=float)

    # Also grab step 0 from before any stepping (already past, use step 1 area)
    # We already have step 0 not recorded; snapshots start at step 500+

    print(f"Total symbols: {len(symbols)}")
    print(f"Unique symbols observed: {len(set(symbols))}")

    # Discard burn-in
    symbols_ss = symbols[BURNIN:]
    print(f"Symbols after burn-in ({BURNIN}): {len(symbols_ss)}")
    print(f"Unique steady-state symbols: {len(set(symbols_ss))}")

    # Compute block entropies
    ks = list(range(1, K_MAX + 1))
    H_k = []
    for k in ks:
        dist = empirical_block_distribution(symbols_ss, k)
        H_k.append(shannon_entropy(dist))

    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]

    print(f"\nBlock entropy estimates:")
    print(f"{'k':>3}  {'H(k)':>8}  {'H(k)/k':>8}  {'h_diff':>8}")
    for i, k in enumerate(ks):
        print(f"{k:>3}  {H_k[i]:>8.4f}  {h_rate[i]:>8.4f}  {h_diff[i]:>8.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle("Sugarscape KS Entropy — Wealth Distribution State", fontsize=14)

    # Top: example wealth distributions at snapshot steps
    ax = axes[0]
    available_snapshots = sorted(snapshots.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(available_snapshots)))
    bin_edges_plot = [0, 10, 25, 50, 100, 200]  # finite edges for plotting
    for i, step in enumerate(available_snapshots):
        w = snapshots[step]
        w_clipped = np.clip(w, 0, 200)
        ax.hist(w_clipped, bins=bin_edges_plot, alpha=0.4, color=colors[i],
                edgecolor="white", label=f"Step {step}")
    ax.set_xlabel("Sugar (wealth)")
    ax.set_ylabel("Agent count")
    ax.set_title("Wealth Distributions at Selected Time Points")
    ax.legend()

    # Bottom: H(k)/k convergence
    ax = axes[1]
    ax.plot(ks, h_rate, "o-", color="steelblue", label="H(k)/k")
    ax.plot(ks, h_diff, "s--", color="crimson", alpha=0.7, label="H(k) - H(k-1)")
    ax.set_xlabel("Block length k")
    ax.set_ylabel("Bits")
    ax.set_title("Entropy Rate Convergence (Distribution State)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "distribution_entropy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
