# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run a single simulation (500 steps, default params) → results/single_run.png
python scripts/run_single.py

# Run the full parameter sweep (100 runs) → results/sweep_results.csv + 3 heatmap PNGs
python scripts/run_sweep.py
```

There is no test suite and no linter configured. The project uses standard `anaconda3` Python 3.11. Dependencies: `mesa==3.3.1`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.

## Architecture

This is a research codebase with a single implemented model: **Sugarscape** (`sugarscape/`), an agent-based model of resource competition used to study the relationship between system "memory" (quantified by KS entropy) and social outcomes (inequality, social mobility).

### Data flow

```
SugarscapeConfig  →  SugarscapeModel  →  mesa.DataCollector
      ↑                    ↑                      ↓
   config.py           model.py          pandas DataFrame
                            ↑
                    SugarAgent (agents.py)
                    sugar_grid: np.ndarray (managed directly on model, not in Mesa)
```

**Key design choices:**
- `sugar_grid` is a raw `np.ndarray` on the model, updated in `model.step()`. It is NOT a Mesa `PropertyLayer` — agents read/write it directly via `model.sugar_grid[x, y]`.
- Mesa 3.x API: `Agent.__init__` takes only `model` (no `unique_id`); `model.rng` is a `numpy.random.Generator`; all randomness flows through `model.rng` for reproducibility.
- Population is held constant: dead agents are replaced immediately in `SugarAgent._die_and_replace()`. Replacement is skipped if `grid.empties` is empty.
- The grid is `MultiGrid` (multiple agents per cell allowed). Agents do not restrict movement to unoccupied cells — they compete for sugar in random activation order.

### Metrics (`metrics.py`)

Three standalone functions used by both scripts and `experiment.py`:
- `gini()` — standard Gini coefficient
- `social_mobility_index()` — Spearman rank correlation of wealth ranks at step 50 vs final step (lower = more mobile)
- `approximate_ks_entropy()` — Grassberger-Procaccia K2 correlation entropy from a scalar time series; returns `np.nan` if the series is too short or constant

### Experiment runner (`experiment.py`)

`run_sweep(param_grid, n_steps, n_seeds, output_dir)` runs a full-factorial loop — no Mesa BatchRunner. Each run constructs a fresh `SugarscapeModel` from a `SugarscapeConfig` with fields overridden by the param combo. Seeds are `1000 + seed_offset`.

## Research context

The long-term goal is to find the KS entropy that maximizes a meritocracy/social-mobility metric. The current Sugarscape implementation is the **Phase 1 baseline** — standard model, no agent memory. Phase 2 will add explicit agent memory (history-weighted movement decisions) as a tunable parameter directly linked to KS entropy. See `PHILOSOPHY.md` for the ergodic theory framework and `README.md` for full project description and current results.
