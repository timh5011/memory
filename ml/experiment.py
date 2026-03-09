"""Experiment runner: sweep learning rates with multiple seeds."""

import pickle
import time
import numpy as np

from train import TrainingConfig, TrainingResult, run_training


def run_experiment(lr_values=None, n_seeds=5, n_steps=500,
                   k_exponents=10, reorth_interval=1, window_size=50,
                   save_path="results/experiment_results.pkl"):
    """Sweep over learning rates with multiple random seeds.

    Parameters
    ----------
    lr_values : list of float
        Learning rates to sweep. Default: 20 log-spaced from 1e-4 to 1.0.
    n_seeds : int
        Number of random initializations per LR.
    n_steps : int
        Training steps per run.
    k_exponents : int
        Number of Lyapunov exponents to track.
    reorth_interval : int
        Reorthonormalization interval.
    window_size : int
        Window size for windowed Lyapunov estimates.
    save_path : str
        Path to save results.

    Returns
    -------
    results : list of TrainingResult
        All results from all runs.
    """
    if lr_values is None:
        lr_values = np.logspace(-4, 0, 20).tolist()

    results = []
    total = len(lr_values) * n_seeds
    count = 0
    t0 = time.time()

    for i, lr in enumerate(lr_values):
        for seed in range(n_seeds):
            count += 1
            print(f"[{count}/{total}] LR={lr:.6f}, seed={seed}", end=" ... ")

            config = TrainingConfig(
                lr=lr,
                n_steps=n_steps,
                seed=seed,
                k_exponents=k_exponents,
                reorth_interval=reorth_interval,
                window_size=window_size,
            )

            try:
                result = run_training(config)
                results.append(result)
                print(f"loss={result.final_train_loss:.4f}, "
                      f"acc={result.final_test_accuracy:.3f}, "
                      f"h_KS={result.ks_entropy_final:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")

        elapsed = time.time() - t0
        remaining = elapsed / count * (total - count)
        print(f"  [{i+1}/{len(lr_values)} LRs done, "
              f"elapsed={elapsed:.0f}s, est. remaining={remaining:.0f}s]")

    # Save
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {len(results)} results to {save_path}")

    return results


if __name__ == "__main__":
    # Full experiment
    results = run_experiment(
        lr_values=np.logspace(-4, 0, 20).tolist(),
        n_seeds=5,
        n_steps=500,
        k_exponents=10,
        reorth_interval=1,
        window_size=50,
    )
