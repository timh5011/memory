# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

```
memory/
├── agent_based_models/
│   ├── sugarscape/          # Agent-based model of resource competition
│   │   ├── scripts/
│   │   │   ├── run_single.py
│   │   │   ├── run_entropy_distribution.py  # Approach 1: distribution-state entropy
│   │   │   ├── run_entropy_agents.py        # Approach 2: agent-trajectory entropy
│   │   │   └── run_entropy_lyapunov.py      # Approach 3: Lyapunov exponent via perturbation
│   │   ├── sim/
│   │   │   ├── agents.py
│   │   │   ├── config.py
│   │   │   ├── grid.py
│   │   │   ├── metrics.py
│   │   │   └── model.py
│   │   └── results/
│   └── minority_game/       # Minority Game (El Farol Bar Problem)
│       ├── scripts/
│       │   ├── run_single.py        # Single run diagnostic (4-panel plot)
│       │   ├── run_entropy.py       # KS entropy analysis at M∈{3,6,9}
│       │   └── run_sweep.py         # Sweep M: phase transition + entropy analysis
│       ├── sim/
│       │   ├── agents.py            # MinorityGameAgent with S strategy tables
│       │   ├── config.py            # MinorityGameConfig (N, M, S, α=2^M/N)
│       │   ├── metrics.py           # volatility, efficiency, predictability
│       │   └── model.py             # MinorityGameModel (simultaneous moves, no grid)
│       └── results/
├── ergodic_systems/
│   ├── systems/                     # dynamical system definitions
│   │   ├── ergodic_system.py        # ABC: ErgodicSystem base class
│   │   ├── bernoulli_shift.py       # BernoulliShift
│   │   └── logistic_map.py          # LogisticMap
│   ├── entropy/                     # KS entropy computation methods
│   │   ├── block_counting.py        # block entropy estimation and plotting
│   │   └── lyapunov.py              # Lyapunov exponent estimation (perturbation + Jacobian)
│   ├── sims/                        # simulation scripts
│   │   ├── bernoulli_sim.py
│   │   ├── logistic_sim.py
│   │   └── lyapunov_sim.py          # Lyapunov validation on logistic map
│   └── results/                     # plots and output
├── ml/                          # KS entropy of neural network training dynamics
│   ├── data.py                  # Synthetic spiral dataset generator
│   ├── model.py                 # MLP definition + param flattening
│   ├── lyapunov.py              # Benettin algorithm with Hessian-vector products
│   ├── train.py                 # Training loop with Lyapunov tracking
│   ├── experiment.py            # LR sweep runner (multiple seeds)
│   ├── analyze.py               # Analysis and plotting
│   ├── PLAN.md                  # Full theoretical spec
│   └── results/                 # Plots and experiment data
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

# Minority Game: single run diagnostic (500 steps) → agent_based_models/minority_game/results/single_run.png
cd agent_based_models/minority_game && python scripts/run_single.py

# Minority Game: KS entropy at M∈{3,6,9} (5000 steps) → results/entropy_analysis.png
cd agent_based_models/minority_game && python scripts/run_entropy.py

# Minority Game: sweep M=2..12, 20 seeds → results/sweep_phase_transition.png, sweep_entropy.png
cd agent_based_models/minority_game && python scripts/run_sweep.py

# Bernoulli shift → ergodic_systems/results/bernoulli_trajectories.png, bernoulli_entropy_rate.png
cd ergodic_systems && python sims/bernoulli_sim.py

# Logistic map → ergodic_systems/results/logistic_trajectories.png, logistic_entropy_rate.png
cd ergodic_systems && python sims/logistic_sim.py

# Lyapunov exponent (logistic map) → ergodic_systems/results/logistic_lyapunov.png
cd ergodic_systems && python sims/lyapunov_sim.py

# ML: single training run (quick test) → prints loss, KS entropy
cd ml && python train.py

# ML: full LR sweep experiment (20 LRs × 5 seeds × 500 steps) → ml/results/experiment_results.pkl
cd ml && python experiment.py

# ML: analyze results → ml/results/ks_vs_convergence.png, ks_vs_loss.png, etc.
cd ml && python analyze.py
```

There is no test suite and no linter configured. The project uses standard `anaconda3` Python 3.11. Dependencies: `mesa==3.3.1`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `torch`.

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

### Agent-Based Models — Minority Game (`agent_based_models/minority_game/`)

The Minority Game (El Farol Bar Problem): N agents simultaneously choose 0 or 1, the minority side wins. Memory length M is the key parameter controlling the phase transition at α_c ≈ 0.34 where α = 2^M / N.

**Data flow:**

```
MinorityGameConfig  →  MinorityGameModel  →  mesa.DataCollector
       ↑                      ↑                       ↓
    config.py              model.py           pandas DataFrame
                               ↑
                  MinorityGameAgent (agents.py)
                  S strategy tables, cumulative scores
```

**Key design choices:**
- **No Mesa grid** — agents exist only in `model.agents`, no spatial structure needed
- **Simultaneous moves** — all agents choose before outcome is revealed (model orchestrates via `choose()` then `update_scores()`)
- Agents have no `step()` method; the model calls `choose(history_tuple)` and `update_scores(history_tuple, winning_action)` directly
- Each agent holds S strategy tables (dict mapping M-length binary tuples → 0/1) with cumulative scores; best-scoring strategy is used (random tie-breaking)
- `model.history` is a `deque(maxlen=M)` of recent winning actions; `model.outcomes` stores full outcome sequence for entropy analysis
- Binary outcome sequence is the natural symbol sequence — no symbolization/binning needed

**Metrics (`metrics.py`):** `volatility()` (σ²/N), `efficiency()` (1 - σ²/(N/4)), `predictability()` (⟨A²⟩/N - N/4)

### Ergodic Systems (`ergodic_systems/`)

Framework for defining ergodic dynamical systems and computing KS entropy numerically. Organized into three subdirectories:

**`systems/` — Dynamical system definitions:**
- `ergodic_system.py` — ABC with abstract methods (`iterate`, `generate_trajectory`, `sample_initial_state`) and optional methods (`jacobian`, `metric`, `perturb`, `symbolize`, `analytical_ks_entropy`)
- `bernoulli_shift.py` — `BernoulliShift` subclass (`is_symbolic=True`, `analytical_ks_entropy()` = H(p))
- `logistic_map.py` — `LogisticMap` subclass (continuous, `symbolize()` via binary partition at 0.5, `analytical_ks_entropy()` = 1.0 bit for r=4 via Pesin's identity, `metric()`/`perturb()` for Lyapunov estimation)

**`entropy/` — KS entropy computation methods:**
- `block_counting.py` — `shannon_entropy(dist)`, `block_entropy_estimates(system, ...)` → `(ks, H_k, h_rate, h_diff)`, `plot_entropy_convergence(...)`, `symbolize_timeseries(series, n_bins, method)` — discretize continuous values into integer symbols
- `lyapunov.py` — `lyapunov_perturbation(system, ...)` (Benettin's algorithm), `lyapunov_jacobian(system, ...)`, `plot_lyapunov_convergence(...)` — works on any `ErgodicSystem` with the required optional methods

**`results/` — Plots and output files**

**`sims/` — Simulation scripts:**
- `bernoulli_sim.py` — trajectory plots (fair vs biased, 3 seeds) + entropy rate validation H(k)/k ≈ H(p)
- `logistic_sim.py` — trajectory plots (3 initial conditions) + entropy rate and conditional entropy validation
- `lyapunov_sim.py` — Lyapunov exponent validation on logistic map (perturbation + Jacobian methods vs analytical ln(2))

**Design choices:**
- Block distributions stored as dicts keyed by tuples
- All randomness via `numpy.random.Generator` with explicit seeds
- `ErgodicSystem` subclasses only implement what they need (optional methods not abstract)
- Imports use package-relative paths (`from systems import BernoulliShift`, `from entropy import block_entropy_estimates`)

### ML — Neural Network Training Dynamics (`ml/`)

Treats gradient descent as a dynamical system on weight space. Measures KS entropy via Lyapunov exponents (Pesin's identity) to find whether there's an optimal "forgetting rate" for learning.

**Data flow:**

```
TrainingConfig → run_training() → TrainingResult
                      ↑                  ↓
              MLP + LyapunovTracker    loss curves, Lyapunov spectra,
              make_spirals()           KS entropy timeseries
```

**Key design choices:**
- Full-batch gradient descent (autonomous dynamics, clean Lyapunov computation)
- Manual SGD (no optimizer) — weights updated via `p -= lr * p.grad`
- Lyapunov exponents via modified Benettin: tangent vectors propagated as `v <- v - lr * (H @ v)` where H is the Hessian
- Hessian-vector products via double-backward trick (never forms full Hessian)
- Tangent vectors stored as d×k matrix, reorthonormalized via QR every R steps
- MLP with `flat_params()` / `load_flat_params()` for parameter flattening
- All computation on CPU (MacBook target, ~4500 parameters)
- Primary control variable: learning rate (log-spaced sweep 1e-4 to 1.0)

**Key files:**
- `lyapunov.py` — `hessian_vector_product()` and `LyapunovTracker` class (core algorithmic contribution)
- `train.py` — `TrainingConfig` / `TrainingResult` dataclasses, `run_training()` loop
- `experiment.py` — `run_experiment()` LR sweep with multiple seeds
- `analyze.py` — 5 plots: KS vs convergence, KS vs loss, KS timeseries, Lyapunov spectrum, loss curves


