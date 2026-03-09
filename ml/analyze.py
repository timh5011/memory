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


def compute_convergence_step(loss_curve, target_loss):
    """Return the first step where loss drops at or below target_loss, else None."""
    for step, loss in enumerate(loss_curve):
        if loss <= target_loss:
            return step
    return None


def plot_ks_vs_convergence(results, target_percentile=25,
                           save_path="results/ks_vs_convergence.png"):
    """Plot 1: KS entropy vs convergence rate, colored by LR.

    Convergence is defined relative to the best loss achieved across all runs:
    the first step where a run's loss drops to or below the target_percentile-th
    percentile of all final training losses. Runs that never reach the target
    are plotted at n_steps (did not converge).
    """
    # Compute relative convergence target
    all_final_losses = [r.final_train_loss for r in results]
    target_loss = np.percentile(all_final_losses, target_percentile)
    n_converged = sum(1 for r in results
                      if compute_convergence_step(r.loss_curve, target_loss) is not None)

    by_lr = aggregate_by_lr(results)
    lrs = sorted(by_lr.keys())
    norm = LogNorm(vmin=min(lrs), vmax=max(lrs))
    cmap = plt.cm.viridis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for lr, runs in by_lr.items():
        color = cmap(norm(lr))
        for r in runs:
            conv = compute_convergence_step(r.loss_curve, target_loss) or r.config.n_steps
            ks = r.ks_entropy_final
            marker = "x" if r.diverged else "o"
            size = 50 if r.diverged else 30
            lw = 1.5 if r.diverged else 0

            # Left panel: KS entropy vs convergence
            ax1.scatter([ks], [conv], c=[color], s=size, marker=marker,
                        linewidths=lw, alpha=0.7, edgecolors="none" if not r.diverged else color)

            # Right panel: learning rate vs convergence
            ax2.scatter([lr], [conv], c=[color], s=size, marker=marker,
                        linewidths=lw, alpha=0.7, edgecolors="none" if not r.diverged else color)

    # Mean line on LR panel to show the U-shape clearly
    mean_lrs, mean_convs = [], []
    for lr in lrs:
        runs = by_lr[lr]
        mean_lrs.append(lr)
        mean_convs.append(np.mean([
            compute_convergence_step(r.loss_curve, target_loss) or r.config.n_steps
            for r in runs
        ]))
    ax2.plot(mean_lrs, mean_convs, "k-", linewidth=1.5, alpha=0.6, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax2, label="Learning rate")

    ax1.set_xlabel("KS entropy (nats/step)")
    ax1.set_ylabel("Steps to convergence")
    ax1.set_title("KS Entropy vs Convergence")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Learning rate")
    ax2.set_ylabel("Steps to convergence")
    ax2.set_xscale("log")
    ax2.set_title("Learning Rate vs Convergence")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Convergence target: {target_percentile}th percentile of final losses "
        f"({target_loss:.4f}) — {n_converged}/{len(results)} runs converged",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
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
        color = cmap(norm(lr))
        converged = [r for r in runs if not r.diverged]
        diverged = [r for r in runs if r.diverged]
        if converged:
            ks = [r.ks_entropy_final for r in converged]
            ax1.scatter(ks, [r.final_train_loss for r in converged],
                        c=[color] * len(ks), s=20, alpha=0.7)
            ax2.scatter(ks, [r.final_test_loss for r in converged],
                        c=[color] * len(ks), s=20, alpha=0.7)
        if diverged:
            ks = [r.ks_entropy_final for r in diverged]
            ax1.scatter(ks, [r.config.n_steps] * len(ks), c=[color] * len(ks),
                        s=40, marker="x", linewidths=1.5, alpha=0.9)
            ax2.scatter(ks, [r.config.n_steps] * len(ks), c=[color] * len(ks),
                        s=40, marker="x", linewidths=1.5, alpha=0.9)

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


def print_summary(results, target_percentile=25):
    """Print summary statistics table."""
    by_lr = aggregate_by_lr(results)
    target_loss = np.percentile([r.final_train_loss for r in results], target_percentile)

    print(f"\nConvergence target: {target_percentile}th percentile of final losses = {target_loss:.4f}")
    print(f"\n{'LR':>10s} | {'Train Loss':>10s} | {'Test Loss':>10s} | "
          f"{'Test Acc':>8s} | {'Conv Step':>9s} | {'KS Entropy':>10s} | {'Diverged':>8s}")
    print("-" * 87)

    for lr in sorted(by_lr.keys()):
        runs = by_lr[lr]
        n_div = sum(r.diverged for r in runs)
        converged = [r for r in runs if not r.diverged]
        tl = np.mean([r.final_train_loss for r in converged]) if converged else float("inf")
        tel = np.mean([r.final_test_loss for r in converged]) if converged else float("inf")
        ta = np.mean([r.final_test_accuracy for r in converged]) if converged else float("nan")
        cs = np.mean([
            compute_convergence_step(r.loss_curve, target_loss) or r.config.n_steps
            for r in converged
        ]) if converged else float("nan")
        ks = np.mean([r.ks_entropy_final for r in runs])

        tl_str = f"{tl:10.4f}" if np.isfinite(tl) else "      inf "
        tel_str = f"{tel:10.4f}" if np.isfinite(tel) else "      inf "
        ta_str = f"{ta:8.4f}" if np.isfinite(ta) else "     nan"
        cs_str = f"{cs:9.1f}" if np.isfinite(cs) else "      nan"
        print(f"{lr:10.6f} | {tl_str} | {tel_str} | "
              f"{ta_str} | {cs_str} | {ks:10.6f} | {n_div}/{len(runs)}")


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
