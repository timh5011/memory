"""Analysis and plotting for KS entropy experiment results."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_results(path="results/experiment_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def aggregate_by_lr(results):
    """Group results by learning rate."""
    by_lr = {}
    for r in results:
        lr = r.config.lr
        if lr not in by_lr:
            by_lr[lr] = []
        by_lr[lr].append(r)
    return dict(sorted(by_lr.items()))


def plot_ks_vs_convergence(results, save_path="results/ks_vs_convergence.png"):
    """Plot 1: KS entropy vs convergence rate, colored by LR."""
    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())
    norm = LogNorm(vmin=min(lrs), vmax=max(lrs))
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 7))

    # Individual points
    for lr, runs in by_lr.items():
        ks_vals = [r.ks_entropy_final for r in runs]
        conv_vals = [r.convergence_step if r.convergence_step is not None
                     else r.config.n_steps for r in runs]
        color = cmap(norm(lr))
        ax.scatter(ks_vals, conv_vals, c=[color] * len(ks_vals),
                   s=30, alpha=0.7, edgecolors="none")

    # Mean trend line
    mean_ks = []
    mean_conv = []
    for lr in lrs:
        runs = by_lr[lr]
        ks = np.mean([r.ks_entropy_final for r in runs])
        conv = np.mean([r.convergence_step if r.convergence_step is not None
                        else r.config.n_steps for r in runs])
        mean_ks.append(ks)
        mean_conv.append(conv)
    ax.plot(mean_ks, mean_conv, "r-o", markersize=4, linewidth=1.5,
            label="Mean trend", zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Learning rate")

    ax.set_xlabel("KS entropy (nats/step)")
    ax.set_ylabel("Convergence step")
    ax.set_title("KS Entropy vs Convergence Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_ks_vs_loss(results, save_path="results/ks_vs_loss.png"):
    """Plot 2: KS entropy vs final train and test loss."""
    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())
    norm = LogNorm(vmin=min(lrs), vmax=max(lrs))
    cmap = plt.cm.viridis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for lr, runs in by_lr.items():
        ks = [r.ks_entropy_final for r in runs]
        train_loss = [r.final_train_loss for r in runs]
        test_loss = [r.final_test_loss for r in runs]
        color = cmap(norm(lr))
        ax1.scatter(ks, train_loss, c=[color] * len(ks), s=20, alpha=0.7)
        ax2.scatter(ks, test_loss, c=[color] * len(ks), s=20, alpha=0.7)

    for ax, title in [(ax1, "Train Loss"), (ax2, "Test Loss")]:
        ax.set_xlabel("KS entropy (nats/step)")
        ax.set_ylabel("Final loss")
        ax.set_title(f"KS Entropy vs {title}")
        ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax2, label="Learning rate")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_ks_timeseries(results, save_path="results/ks_timeseries.png"):
    """Plot 3: KS entropy evolution during training for representative LRs."""
    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())

    # Pick ~5 representative LRs (evenly spaced in log)
    indices = np.linspace(0, len(lrs) - 1, min(5, len(lrs)), dtype=int)
    selected_lrs = [lrs[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(selected_lrs)))

    for lr, color in zip(selected_lrs, colors):
        runs = by_lr[lr]
        # Use first seed with windowed data
        for r in runs:
            if len(r.lyapunov_steps) > 0:
                ax.plot(r.lyapunov_steps, r.ks_entropy_timeseries,
                        color=color, alpha=0.4, linewidth=0.8)
        # Mean across seeds
        valid = [r for r in runs if len(r.lyapunov_steps) > 0]
        if valid:
            min_len = min(len(r.ks_entropy_timeseries) for r in valid)
            mean_ks = np.mean([r.ks_entropy_timeseries[:min_len]
                               for r in valid], axis=0)
            steps = valid[0].lyapunov_steps[:min_len]
            ax.plot(steps, mean_ks, color=color, linewidth=2,
                    label=f"lr={lr:.1e}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("KS entropy (nats/step)")
    ax.set_title("KS Entropy Evolution During Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_lyapunov_spectrum(results, save_path="results/lyapunov_spectrum.png"):
    """Plot 4: Top-k Lyapunov exponents vs learning rate."""
    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())

    k = results[0].config.k_exponents
    mean_exps = np.zeros((len(lrs), k))
    std_exps = np.zeros((len(lrs), k))

    for i, lr in enumerate(lrs):
        exps = np.array([r.lyapunov_exponents for r in by_lr[lr]])
        mean_exps[i] = exps.mean(axis=0)
        std_exps[i] = exps.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for j in range(k):
        ax.errorbar(lrs, mean_exps[:, j], yerr=std_exps[:, j],
                     color=colors[j], marker="o", markersize=3,
                     linewidth=1, capsize=2, label=f"$\\lambda_{{{j+1}}}$")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Lyapunov exponent (nats/step)")
    ax.set_title("Lyapunov Spectrum vs Learning Rate")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_loss_curves(results, save_path="results/loss_curves.png"):
    """Plot 5: Training loss curves for different LRs."""
    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())

    indices = np.linspace(0, len(lrs) - 1, min(7, len(lrs)), dtype=int)
    selected_lrs = [lrs[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(selected_lrs)))

    for lr, color in zip(selected_lrs, colors):
        runs = by_lr[lr]
        # Mean loss curve
        min_len = min(len(r.loss_curve) for r in runs)
        mean_loss = np.mean([r.loss_curve[:min_len] for r in runs], axis=0)
        ax.plot(mean_loss, color=color, linewidth=1.5, label=f"lr={lr:.1e}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training Loss Curves")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def print_summary(results):
    """Print summary statistics table."""
    by_lr = aggregate_by_lr(results)

    print(f"\n{'LR':>10s} | {'Train Loss':>10s} | {'Test Loss':>10s} | "
          f"{'Test Acc':>8s} | {'Conv Step':>9s} | {'KS Entropy':>10s}")
    print("-" * 75)

    for lr in sorted(by_lr.keys()):
        runs = by_lr[lr]
        tl = np.mean([r.final_train_loss for r in runs])
        tel = np.mean([r.final_test_loss for r in runs])
        ta = np.mean([r.final_test_accuracy for r in runs])
        cs = np.mean([r.convergence_step if r.convergence_step is not None
                      else r.config.n_steps for r in runs])
        ks = np.mean([r.ks_entropy_final for r in runs])

        print(f"{lr:10.6f} | {tl:10.4f} | {tel:10.4f} | "
              f"{ta:8.4f} | {cs:9.1f} | {ks:10.6f}")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} results")

    print_summary(results)

    plot_ks_vs_convergence(results)
    plot_ks_vs_loss(results)
    plot_ks_timeseries(results)
    plot_lyapunov_spectrum(results)
    plot_loss_curves(results)

    print("\nAll plots saved to results/")
