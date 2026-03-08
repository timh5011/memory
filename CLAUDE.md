# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

```
memory/
├── agent_based_models/
│   └── sugarscape/          # Agent-based model of resource competition
│       ├── scripts/
│       │   └── run_single.py
│       ├── sim/
│       │   ├── agents.py
│       │   ├── config.py
│       │   ├── grid.py
│       │   ├── metrics.py
│       │   └── model.py
│       └── results/
├── ergodic_systems/
│   ├── ergodic_system.py        # ABC: ErgodicSystem base class
│   ├── bernoulli_shift.py       # BernoulliShift subclass + utility functions
│   ├── ks_entropy.py            # block entropy estimation and plotting
│   └── sim.py                   # demo: entropy rate validation for Bernoulli shifts
├── PHILOSOPHY.md
└── README.md
```

## Commands

```bash
# Sugarscape: single simulation (500 steps) → agent_based_models/sugarscape/results/single_run.png
cd agent_based_models/sugarscape && python scripts/run_single.py

# Ergodic systems: entropy rate validation → ergodic_systems/entropy_rate.png
cd ergodic_systems && python sim.py
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
- The grid is `MultiGrid` (multiple agents per cell allowed). Agents do not restrict movement to unoccupied cells — they compete for sugar in random activation order.

**Metrics (`metrics.py`):** `gini()`, `social_mobility_index()`, `approximate_ks_entropy()`.

### Ergodic Systems (`ergodic_systems/`)

Framework for defining ergodic dynamical systems and computing KS entropy numerically.

**`ergodic_system.py` — Base class:**
- ABC with abstract methods: `iterate(state)`, `generate_trajectory(initial_state, n_steps, seed)`, `sample_initial_state(seed)`
- Optional methods (raise `NotImplementedError` by default): `jacobian()`, `symbolize()`, `analytical_ks_entropy()`

**`bernoulli_shift.py` — BernoulliShift subclass + utilities:**
- `BernoulliShift(probs)` — i.i.d. process, `is_symbolic=True`, implements `analytical_ks_entropy()` as H(p)
- Module-level: `make_distribution`, `generate_sequence`, `shift`, `empirical_block_distribution`, `true_block_distribution`

**`ks_entropy.py` — Entropy computation:**
- `shannon_entropy(dist)` — H = -sum q log2(q) over a distribution dict
- `block_entropy_estimates(system, n_steps, k_max, seed)` — returns `(ks, H_k, h_rate, h_diff)`
- `plot_entropy_convergence(...)` — H(k)/k vs k visualization

**`sim.py` — Demo/validation:**
- Compares fair vs biased Bernoulli shifts, validates H(k)/k ≈ H(p) for all k

**Design choices:**
- Block distributions stored as dicts keyed by tuples
- All randomness via `numpy.random.Generator` with explicit seeds
- `ErgodicSystem` subclasses only implement what they need (optional methods not abstract)

## Research context

The long-term goal is to find the KS entropy that maximizes a meritocracy/social-mobility metric. The Sugarscape model is the **Phase 1 baseline** — standard model, no agent memory. The Bernoulli shift experiment validates the theoretical relationship between entropy and convergence rates. Phase 2 will add explicit agent memory (history-weighted movement decisions) as a tunable parameter directly linked to KS entropy. See `PHILOSOPHY.md` for the ergodic theory framework and `README.md` for full project description.
