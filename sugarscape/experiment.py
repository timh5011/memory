"""Parameter sweep runner for Sugarscape experiments."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SugarscapeConfig
from .model import SugarscapeModel
from .metrics import gini, approximate_ks_entropy, social_mobility_index


def run_sweep(
    param_grid: dict[str, list],
    n_steps: int,
    n_seeds: int,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run a full-factorial parameter sweep and save results to CSV.

    Args:
        param_grid: dict mapping config field names to lists of values.
        n_steps:    number of steps per simulation.
        n_seeds:    number of random seeds per parameter combination.
        output_dir: directory where sweep_results.csv will be saved.

    Returns:
        DataFrame with one row per (parameter combo, seed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))
    total = len(combos) * n_seeds
    print(f"Running {len(combos)} parameter combinations × {n_seeds} seeds = {total} runs")

    records = []
    run_idx = 0

    for combo in combos:
        params = dict(zip(param_names, combo))

        # Track wealth ranks at step 50 for mobility
        wealth_at_50: np.ndarray | None = None

        for seed_offset in range(n_seeds):
            run_idx += 1
            seed = 1000 + seed_offset

            cfg_kwargs = {**params, "n_steps": n_steps, "seed": seed}
            config = SugarscapeConfig(**cfg_kwargs)
            model = SugarscapeModel(config)

            mean_sugar_series: list[float] = []

            for step_num in range(1, n_steps + 1):
                model.step()
                df_step = model.datacollector.get_model_vars_dataframe()
                mean_sugar_series.append(float(df_step["mean_sugar"].iloc[-1]))

                if step_num == 50:
                    wealth_at_50 = np.array(
                        df_step["wealth_list"].iloc[-1], dtype=float
                    )

            df_all = model.datacollector.get_model_vars_dataframe()

            # Metrics averaged over last 50 steps
            last50 = df_all.iloc[-50:]
            final_gini = float(last50["gini"].mean())
            final_mean_sugar = float(last50["mean_sugar"].mean())

            # KS entropy from full mean-sugar time series
            ks = approximate_ks_entropy(np.array(mean_sugar_series))

            # Social mobility: ranks at step 50 vs final step
            final_wealth = np.array(df_all["wealth_list"].iloc[-1], dtype=float)
            if wealth_at_50 is not None and len(wealth_at_50) >= 3 and len(final_wealth) >= 3:
                # Align lengths by padding/trimming (population may vary)
                min_len = min(len(wealth_at_50), len(final_wealth))
                ranks_50 = np.argsort(np.argsort(wealth_at_50[:min_len]))
                ranks_final = np.argsort(np.argsort(final_wealth[:min_len]))
                smi = social_mobility_index(ranks_50, ranks_final)
            else:
                smi = float("nan")

            record = {
                **params,
                "seed": seed,
                "final_gini": final_gini,
                "final_mean_sugar": final_mean_sugar,
                "ks_entropy": ks,
                "social_mobility_index": smi,
            }
            records.append(record)

            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            ks_str = f"{ks:.4f}" if np.isfinite(ks) else "nan"
            smi_str = f"{smi:.3f}" if np.isfinite(smi) else "nan"
            print(
                f"  [{run_idx}/{total}] {param_str}, seed={seed} | "
                f"gini={final_gini:.3f}, mean_sugar={final_mean_sugar:.2f}, "
                f"ks={ks_str}, smi={smi_str}"
            )

    results_df = pd.DataFrame(records)
    out_path = output_dir / "sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return results_df
