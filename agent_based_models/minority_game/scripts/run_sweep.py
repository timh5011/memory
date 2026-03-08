"""Sweep memory length M: canonical phase transition plots + entropy analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Allow importing from ergodic_systems
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ergodic_systems"))

from sim import MinorityGameModel, MinorityGameConfig
from sim.metrics import volatility, efficiency, predictability
from entropy.block_counting import empirical_block_distribution, shannon_entropy


def run_trial(M: int, N: int, n_steps: int, burn_in: int, seed: int, k_max: int = 8):
    """Run one trial and return (alpha, vol, eff, pred, ks_entropy)."""
    config = MinorityGameConfig(
        n_agents=N, memory_length=M, n_steps=n_steps, seed=seed
    )
    model = MinorityGameModel(config)
    for _ in range(n_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    attendance = df["attendance"].values[burn_in:].astype(float)

    vol = volatility(attendance, N)
    eff = efficiency(attendance, N)
    pred = predictability(attendance, N)

    # KS entropy from binary outcome sequence
    outcomes = np.array(model.outcomes[burn_in:])
    H_prev = 0.0
    for k in range(1, k_max + 1):
        dist = empirical_block_distribution(outcomes, k)
        H_k = shannon_entropy(dist)
        h_diff = H_k - H_prev
        H_prev = H_k
    ks_entropy = h_diff

    return config.alpha, vol, eff, pred, ks_entropy


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    N = 301
    M_values = list(range(2, 13))  # M = 2 to 12
    n_seeds = 20
    n_steps = 1000
    burn_in = 200

    # Collect results
    all_alpha = []
    all_M = []
    all_vol = []
    all_eff = []
    all_pred = []
    all_entropy = []

    for M in M_values:
        alpha = 2 ** M / N
        print(f"M={M:2d} (α={alpha:.4f}): ", end="", flush=True)
        for s in range(n_seeds):
            seed = 1000 * M + s
            alpha_val, vol, eff, pred, ks_ent = run_trial(M, N, n_steps, burn_in, seed)
            all_alpha.append(alpha_val)
            all_M.append(M)
            all_vol.append(vol)
            all_eff.append(eff)
            all_pred.append(pred)
            all_entropy.append(ks_ent)
        print(f"done (vol={np.mean(all_vol[-n_seeds:]):.4f}, "
              f"eff={np.mean(all_eff[-n_seeds:]):.4f}, "
              f"h={np.mean(all_entropy[-n_seeds:]):.4f})")

    all_alpha = np.array(all_alpha)
    all_M = np.array(all_M)
    all_vol = np.array(all_vol)
    all_eff = np.array(all_eff)
    all_pred = np.array(all_pred)
    all_entropy = np.array(all_entropy)

    # Compute means per M
    unique_M = sorted(set(all_M))
    mean_alpha = [all_alpha[all_M == M].mean() for M in unique_M]
    mean_vol = [all_vol[all_M == M].mean() for M in unique_M]
    mean_eff = [all_eff[all_M == M].mean() for M in unique_M]
    mean_pred = [all_pred[all_M == M].mean() for M in unique_M]
    mean_entropy = [all_entropy[all_M == M].mean() for M in unique_M]
    std_vol = [all_vol[all_M == M].std() for M in unique_M]
    std_eff = [all_eff[all_M == M].std() for M in unique_M]
    std_pred = [all_pred[all_M == M].std() for M in unique_M]
    std_entropy = [all_entropy[all_M == M].std() for M in unique_M]

    # ---- Plot 1: Canonical phase transition (standard literature plots) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))
    fig1.suptitle(
        f"Minority Game – Phase Transition (N={N}, {n_seeds} seeds/M)",
        fontsize=14,
    )

    # Volatility vs α (Challet & Zhang 1997)
    ax = axes1[0, 0]
    ax.errorbar(mean_alpha, mean_vol, yerr=std_vol, fmt="o-", color="steelblue",
                capsize=3, markersize=5)
    ax.axvline(0.34, color="red", linestyle="--", alpha=0.5, label="α_c ≈ 0.34")
    ax.axhline(1 / 4, color="gray", linestyle=":", alpha=0.5, label="Random (1/4)")
    ax.set_xscale("log")
    ax.set_xlabel("α = 2^M / N")
    ax.set_ylabel("σ²/N")
    ax.set_title("Volatility vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predictability vs α
    ax = axes1[0, 1]
    ax.errorbar(mean_alpha, mean_pred, yerr=std_pred, fmt="o-", color="goldenrod",
                capsize=3, markersize=5)
    ax.axvline(0.34, color="red", linestyle="--", alpha=0.5, label="α_c ≈ 0.34")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("α = 2^M / N")
    ax.set_ylabel("H = ⟨A²⟩/N − N/4")
    ax.set_title("Predictability vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Efficiency vs α
    ax = axes1[1, 0]
    ax.errorbar(mean_alpha, mean_eff, yerr=std_eff, fmt="o-", color="mediumseagreen",
                capsize=3, markersize=5)
    ax.axvline(0.34, color="red", linestyle="--", alpha=0.5, label="α_c ≈ 0.34")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("α = 2^M / N")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Volatility vs α (linear scale, zoomed near transition)
    ax = axes1[1, 1]
    ax.errorbar(mean_alpha, mean_vol, yerr=std_vol, fmt="o-", color="steelblue",
                capsize=3, markersize=5)
    ax.axvline(0.34, color="red", linestyle="--", alpha=0.5, label="α_c ≈ 0.34")
    ax.axhline(1 / 4, color="gray", linestyle=":", alpha=0.5, label="Random (1/4)")
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("α = 2^M / N")
    ax.set_ylabel("σ²/N")
    ax.set_title("Volatility vs α (linear, zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    out1 = results_dir / "sweep_phase_transition.png"
    fig1.savefig(out1, dpi=150)
    print(f"\nPlot saved to {out1}")

    # ---- Plot 2: Entropy analysis (this project's contribution) ----
    cmap = plt.cm.viridis
    color_norm = plt.Normalize(vmin=min(all_M), vmax=max(all_M))

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
    fig2.suptitle(
        f"Minority Game – KS Entropy Analysis (N={N}, {n_seeds} seeds/M)",
        fontsize=14,
    )

    # KS entropy vs α
    ax = axes2[0, 0]
    ax.errorbar(mean_alpha, mean_entropy, yerr=std_entropy, fmt="o-", color="crimson",
                capsize=3, markersize=5)
    ax.axvline(0.34, color="red", linestyle="--", alpha=0.5, label="α_c ≈ 0.34")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Random (1 bit)")
    ax.set_xscale("log")
    ax.set_xlabel("α = 2^M / N")
    ax.set_ylabel("KS entropy (bits)")
    ax.set_title("KS Entropy vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy vs Efficiency
    ax = axes2[0, 1]
    sc = ax.scatter(all_entropy, all_eff, c=all_M, cmap=cmap, norm=color_norm,
                    alpha=0.5, s=20, edgecolors="none")
    ax.plot(mean_entropy, mean_eff, "k-o", markersize=6, linewidth=1.5, zorder=5)
    for i, M in enumerate(unique_M):
        ax.annotate(f"M={M}", (mean_entropy[i], mean_eff[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("M")
    ax.set_xlabel("KS Entropy (bits)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Entropy vs Efficiency")
    ax.grid(True, alpha=0.3)

    # Entropy vs Volatility
    ax = axes2[1, 0]
    sc2 = ax.scatter(all_entropy, all_vol, c=all_M, cmap=cmap, norm=color_norm,
                     alpha=0.5, s=20, edgecolors="none")
    ax.plot(mean_entropy, mean_vol, "k-o", markersize=6, linewidth=1.5, zorder=5)
    for i, M in enumerate(unique_M):
        ax.annotate(f"M={M}", (mean_entropy[i], mean_vol[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    cbar2 = fig2.colorbar(sc2, ax=ax)
    cbar2.set_label("M")
    ax.set_xlabel("KS Entropy (bits)")
    ax.set_ylabel("σ²/N")
    ax.set_title("Entropy vs Volatility")
    ax.grid(True, alpha=0.3)

    # Entropy vs Predictability
    ax = axes2[1, 1]
    sc3 = ax.scatter(all_entropy, all_pred, c=all_M, cmap=cmap, norm=color_norm,
                     alpha=0.5, s=20, edgecolors="none")
    ax.plot(mean_entropy, mean_pred, "k-o", markersize=6, linewidth=1.5, zorder=5)
    for i, M in enumerate(unique_M):
        ax.annotate(f"M={M}", (mean_entropy[i], mean_pred[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    cbar3 = fig2.colorbar(sc3, ax=ax)
    cbar3.set_label("M")
    ax.set_xlabel("KS Entropy (bits)")
    ax.set_ylabel("H = ⟨A²⟩/N − N/4")
    ax.set_title("Entropy vs Predictability")
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    out2 = results_dir / "sweep_entropy.png"
    fig2.savefig(out2, dpi=150)
    print(f"Plot saved to {out2}")


if __name__ == "__main__":
    main()
