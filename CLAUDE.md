# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

```
memory/
├── agent_based_models/
│   └── sugarscape/          # Agent-based model of resource competition
│       ├── scripts/
│       │   ├── run_single.py
│       │   ├── run_entropy_distribution.py  # Approach 1: distribution-state entropy
│       │   ├── run_entropy_agents.py        # Approach 2: agent-trajectory entropy
│       │   └── run_entropy_lyapunov.py      # Approach 3: Lyapunov exponent via perturbation
│       ├── sim/
│       │   ├── agents.py
│       │   ├── config.py
│       │   ├── grid.py
│       │   ├── metrics.py
│       │   └── model.py
│       └── results/
├── ergodic_systems/
│   ├── systems/                     # dynamical system definitions
│   │   ├── ergodic_system.py        # ABC: ErgodicSystem base class
│   │   ├── bernoulli_shift.py       # BernoulliShift
│   │   └── logistic_map.py          # LogisticMap
│   ├── entropy/                     # KS entropy computation methods
│   │   └── block_counting.py        # block entropy estimation and plotting
│   ├── sims/                        # simulation scripts
│   │   ├── bernoulli_sim.py
│   │   └── logistic_sim.py
│   └── results/                     # plots and output
├── PHILOSOPHY.md
└── README.md
```

## Commands

```bash
# Sugarscape: single simulation (500 steps) → agent_based_models/sugarscape/results/single_run.png
cd agent_based_models/sugarscape && python scripts/run_single.py

# Sugarscape: KS entropy of wealth distribution (5000 steps) → results/distribution_entropy.png
cd agent_based_models/sugarscape && python scripts/run_entropy_distribution.py

# Sugarscape: KS entropy of agent trajectories (5000 steps) → results/agent_entropy.png
cd agent_based_models/sugarscape && python scripts/run_entropy_agents.py

# Sugarscape: Lyapunov exponent via perturbation (50 trials) → results/lyapunov_entropy.png
cd agent_based_models/sugarscape && python scripts/run_entropy_lyapunov.py

# Bernoulli shift → ergodic_systems/results/bernoulli_trajectories.png, bernoulli_entropy_rate.png
cd ergodic_systems && python sims/bernoulli_sim.py

# Logistic map → ergodic_systems/results/logistic_trajectories.png, logistic_entropy_rate.png
cd ergodic_systems && python sims/logistic_sim.py
```

There is no test suite and no linter configured. The project uses standard `anaconda3` Python 3.11. Dependencies: `mesa==3.3.1`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.

## Architecture

This is a research codebase exploring the relationship between dynamical system memory (quantified by KS entropy) and social outcomes. It contains two categories of experiments:

### Agent-Based Models — Sugarscape (`agent_based_models/sugarscape/`)

An agent-based model of resource competition studying inequality and social mobility.

**Data flow:**

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
- Each agent tracks `wealth_history` (list of sugar values after each step). On death, `_die_and_replace()` appends the history to `model.completed_trajectories` before removal.
- The grid is `MultiGrid` (multiple agents per cell allowed). Agents do not restrict movement to unoccupied cells — they compete for sugar in random activation order.

**Metrics (`metrics.py`):** `gini()`, `wasserstein_1d()` (earth mover's distance for Lyapunov divergence measurement).

### Ergodic Systems (`ergodic_systems/`)

Framework for defining ergodic dynamical systems and computing KS entropy numerically. Organized into three subdirectories:

**`systems/` — Dynamical system definitions:**
- `ergodic_system.py` — ABC with abstract methods (`iterate`, `generate_trajectory`, `sample_initial_state`) and optional methods (`jacobian`, `symbolize`, `analytical_ks_entropy`)
- `bernoulli_shift.py` — `BernoulliShift` subclass (`is_symbolic=True`, `analytical_ks_entropy()` = H(p))
- `logistic_map.py` — `LogisticMap` subclass (continuous, `symbolize()` via binary partition at 0.5, `analytical_ks_entropy()` = 1.0 bit for r=4 via Pesin's identity)

**`entropy/` — KS entropy computation methods:**
- `block_counting.py` — `shannon_entropy(dist)`, `block_entropy_estimates(system, ...)` → `(ks, H_k, h_rate, h_diff)`, `plot_entropy_convergence(...)`, `symbolize_timeseries(series, n_bins, method)` — discretize continuous values into integer symbols

**`results/` — Plots and output files**

**`sims/` — Simulation scripts:**
- `bernoulli_sim.py` — trajectory plots (fair vs biased, 3 seeds) + entropy rate validation H(k)/k ≈ H(p)
- `logistic_sim.py` — trajectory plots (3 initial conditions) + entropy rate and conditional entropy validation

**Design choices:**
- Block distributions stored as dicts keyed by tuples
- All randomness via `numpy.random.Generator` with explicit seeds
- `ErgodicSystem` subclasses only implement what they need (optional methods not abstract)
- Imports use package-relative paths (`from systems import BernoulliShift`, `from entropy import block_entropy_estimates`)

## Research context

The long-term goal is to find the KS entropy that maximizes a meritocracy/social-mobility metric. The Sugarscape model is the **Phase 1 baseline** — standard model, no agent memory. The Bernoulli shift experiment validates the theoretical relationship between entropy and convergence rates. Phase 2 will add explicit agent memory (history-weighted movement decisions) as a tunable parameter directly linked to KS entropy. See `PHILOSOPHY.md` for the ergodic theory framework and `README.md` for full project description.
