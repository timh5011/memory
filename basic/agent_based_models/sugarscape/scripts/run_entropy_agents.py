"""Approach 2: KS entropy of individual agent wealth trajectories.

Track each agent's sugar over its lifetime, discretize into symbols,
and pool block statistics across all agent lifetimes to estimate the
entropy rate of individual wealth dynamics.
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
from entropy.block_counting import (
    empirical_block_distribution, shannon_entropy, symbolize_timeseries
)

# --- Parameters ---
N_STEPS = 5000
SEED = 42
N_BINS = 8
K_MAX = 10
MIN_TRAJ_LEN = 5  # ignore very short trajectories


def pool_block_counts(symbolic_trajectories, k):
    """Accumulate block counts across multiple trajectories.

    Returns a frequency distribution dict (block_tuple -> probability).
    """
    counts = {}
    total = 0
    for traj in symbolic_trajectories:
        if len(traj) < k:
            continue
        for i in range(len(traj) - k + 1):
            block = tuple(traj[i:i + k])
            counts[block] = counts.get(block, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {b: c / total for b, c in counts.items()}


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    config = SugarscapeConfig(n_steps=N_STEPS, seed=SEED)
    model = SugarscapeModel(config)

    # Run simulation
    print(f"Running Sugarscape for {N_STEPS} steps...")
    for step in range(N_STEPS):
        model.step()
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}/{N_STEPS}")

    # Collect all trajectories: completed (dead agents) + still-living
    all_trajectories = list(model.completed_trajectories)
    for agent in model.agents:
        if agent.wealth_history:
            all_trajectories.append(agent.wealth_history)

    print(f"\nTotal trajectories: {len(all_trajectories)}")
    lengths = [len(t) for t in all_trajectories]
    print(f"Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")

    # Filter short trajectories
    trajectories = [t for t in all_trajectories if len(t) >= MIN_TRAJ_LEN]
    print(f"Trajectories with len >= {MIN_TRAJ_LEN}: {len(trajectories)}")

    # Compute global bin edges from all wealth values pooled
    all_values = np.concatenate([np.array(t, dtype=float) for t in trajectories])
    print(f"Total wealth observations: {len(all_values)}")

    percentiles = np.linspace(0, 100, N_BINS + 1)
    global_edges = np.percentile(all_values, percentiles)
    global_edges[0] = all_values.min() - 1e-10
    global_edges[-1] = all_values.max() + 1e-10

    # Symbolize each trajectory using global bin edges
    symbolic_trajectories = []
    for traj in trajectories:
        symbols, _ = symbolize_timeseries(traj, n_bins=N_BINS, bin_edges=global_edges)
        symbolic_trajectories.append(symbols.tolist())

    # Compute pooled block entropies
    ks = list(range(1, K_MAX + 1))
    H_k = []
    for k in ks:
        dist = pool_block_counts(symbolic_trajectories, k)
        H_k.append(shannon_entropy(dist))

    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]

    print(f"\nPooled block entropy estimates:")
    print(f"{'k':>3}  {'H(k)':>8}  {'H(k)/k':>8}  {'h_diff':>8}")
    for i, k in enumerate(ks):
        print(f"{k:>3}  {H_k[i]:>8.4f}  {h_rate[i]:>8.4f}  {h_diff[i]:>8.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle("Sugarscape KS Entropy — Individual Agent Wealth Trajectories",
                 fontsize=14)

    # Top: sample of individual wealth trajectories
    ax = axes[0]
    rng = np.random.default_rng(0)
    # Pick trajectories of varying lengths for visual diversity
    sample_idx = rng.choice(len(trajectories), size=min(20, len(trajectories)),
                            replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(sample_idx)))
    for i, idx in enumerate(sample_idx):
        traj = trajectories[idx]
        ax.plot(range(len(traj)), traj, alpha=0.6, linewidth=0.8, color=colors[i])
    ax.set_xlabel("Agent lifetime step")
    ax.set_ylabel("Sugar (wealth)")
    ax.set_title(f"Sample Agent Wealth Trajectories (n={len(sample_idx)})")

    # Bottom: H(k)/k convergence
    ax = axes[1]
    ax.plot(ks, h_rate, "o-", color="steelblue", label="H(k)/k")
    ax.plot(ks, h_diff, "s--", color="crimson", alpha=0.7, label="H(k) - H(k-1)")
    ax.set_xlabel("Block length k")
    ax.set_ylabel("Bits")
    ax.set_title("Entropy Rate Convergence (Pooled Agent Trajectories)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "agent_entropy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
