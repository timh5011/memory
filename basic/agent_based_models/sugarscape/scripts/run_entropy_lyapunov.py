"""Lyapunov exponent estimate for Sugarscape wealth dynamics.

Method: at regular intervals along a baseline simulation, deep-copy the model,
apply a small wealth perturbation to one agent, run both copies forward, and
measure how the Wasserstein-1 distance between their wealth distributions
grows over time. The exponential growth rate of this divergence estimates the
largest Lyapunov exponent, which by Pesin's identity bounds the KS entropy
from below.

Metric choice: Wasserstein-1 (earth mover's distance) on the raw wealth
distributions. Alternatives considered:
  - L2 norm on binned histograms: doesn't respect ordinal wealth structure
  - KL divergence: not a true metric (asymmetric, unbounded)
Wasserstein-1 is the natural transport metric for ordered 1D distributions.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import SugarscapeModel, SugarscapeConfig
from sim.metrics import wasserstein_1d

# --- Parameters ---
SEED = 42
BURN_IN = 500       # steps before first perturbation trial
N_TRIALS = 50       # number of perturbation experiments
T_HORIZON = 100     # steps to track divergence per trial
SPACING = 50        # steps between trials on the baseline
DELTA = 1           # sugar perturbation magnitude


def get_wealth(model):
    """Extract sorted wealth array from a model."""
    return np.array([a.sugar for a in model.agents], dtype=float)


def run_trial(model, delta, t_horizon):
    """Clone model, perturb one agent, run both forward, return divergence curve.

    Returns array of Wasserstein-1 distances of length t_horizon+1 (including t=0).
    """
    clone = copy.deepcopy(model)

    # Perturb: add delta sugar to a random agent in the clone
    clone_agents = list(clone.agents)
    idx = clone.rng.integers(0, len(clone_agents))
    clone_agents[idx].sugar += delta

    # Record initial distance
    distances = np.zeros(t_horizon + 1)
    distances[0] = wasserstein_1d(get_wealth(model), get_wealth(clone))

    # Run both forward
    for t in range(1, t_horizon + 1):
        model.step()
        clone.step()
        distances[t] = wasserstein_1d(get_wealth(model), get_wealth(clone))

    return distances


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    total_steps = BURN_IN + N_TRIALS * (T_HORIZON + SPACING)
    config = SugarscapeConfig(n_steps=total_steps, seed=SEED)
    model = SugarscapeModel(config)

    # Burn-in
    print(f"Burn-in: {BURN_IN} steps...")
    for _ in range(BURN_IN):
        model.step()

    # Perturbation trials
    all_distances = []
    for trial in range(N_TRIALS):
        print(f"Trial {trial + 1}/{N_TRIALS}...")

        # Save model state before trial (we need to restore after since
        # run_trial advances the model by T_HORIZON steps)
        model_backup = copy.deepcopy(model)

        distances = run_trial(model_backup, DELTA, T_HORIZON)
        all_distances.append(distances)

        # Advance the baseline model by T_HORIZON + SPACING steps
        for _ in range(T_HORIZON + SPACING):
            model.step()

    all_distances = np.array(all_distances)  # (N_TRIALS, T_HORIZON+1)

    # Compute average log divergence (skip t=0 for log)
    # Replace zeros with nan to avoid log(0)
    log_distances = np.full_like(all_distances, np.nan)
    mask = all_distances > 0
    log_distances[mask] = np.log(all_distances[mask])

    mean_log_div = np.nanmean(log_distances, axis=0)
    std_log_div = np.nanstd(log_distances, axis=0)

    # Fit Lyapunov exponent: slope of mean ln(d(t)) for t >= 1
    # Use only the initial growth phase (first ~30 steps) before saturation
    fit_end = min(30, T_HORIZON)
    t_fit = np.arange(1, fit_end + 1)
    log_fit = mean_log_div[1:fit_end + 1]
    valid = ~np.isnan(log_fit)
    if valid.sum() >= 2:
        coeffs = np.polyfit(t_fit[valid], log_fit[valid], 1)
        lyap_nats = coeffs[0]  # nats per step
        lyap_bits = lyap_nats / np.log(2)
        intercept = coeffs[1]
    else:
        lyap_nats = np.nan
        lyap_bits = np.nan
        intercept = np.nan

    print(f"\nLyapunov exponent estimate:")
    print(f"  λ = {lyap_nats:.4f} nats/step = {lyap_bits:.4f} bits/step")
    print(f"  (fitted over t=1..{fit_end})")
    print(f"  d(0) mean = {np.mean(all_distances[:, 0]):.6f}")
    print(f"  d({T_HORIZON}) mean = {np.mean(all_distances[:, -1]):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle("Sugarscape Lyapunov Exponent — Wasserstein-1 Divergence",
                 fontsize=14)

    t = np.arange(T_HORIZON + 1)

    # Top: sample divergence curves (log scale)
    ax = axes[0]
    n_show = min(10, N_TRIALS)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    for i in range(n_show):
        d = all_distances[i]
        d_plot = np.where(d > 0, d, np.nan)
        ax.plot(t, d_plot, alpha=0.4, linewidth=0.8, color=colors[i])
    ax.set_yscale("log")
    ax.set_xlabel("Steps after perturbation")
    ax.set_ylabel("Wasserstein-1 distance")
    ax.set_title(f"Sample Divergence Curves (n={n_show}, δ={DELTA})")
    ax.grid(True, alpha=0.3)

    # Bottom: average log divergence with linear fit
    ax = axes[1]
    ax.plot(t, mean_log_div, "o-", color="steelblue", markersize=3,
            label="Mean ln(d(t))")
    ax.fill_between(t, mean_log_div - std_log_div, mean_log_div + std_log_div,
                    alpha=0.2, color="steelblue")
    if not np.isnan(lyap_nats):
        fit_line = intercept + lyap_nats * t
        ax.plot(t[:fit_end + 1], fit_line[:fit_end + 1], "--", color="crimson",
                linewidth=2,
                label=f"Linear fit: λ = {lyap_nats:.4f} nats = {lyap_bits:.4f} bits")
    ax.set_xlabel("Steps after perturbation")
    ax.set_ylabel("ln(Wasserstein-1 distance)")
    ax.set_title("Average Log Divergence & Lyapunov Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "lyapunov_entropy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
