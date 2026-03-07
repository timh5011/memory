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
│   └── bernoulli_shift/     # Block distribution convergence experiment
│       ├── scripts/
│       │   └── run_convergence.py
│       ├── sim/
│       │   ├── shift.py
│       │   └── entropy.py
│       └── results/
├── PHILOSOPHY.md
└── README.md
```

## Commands

```bash
# Sugarscape: single simulation (500 steps) → agent_based_models/sugarscape/results/single_run.png
cd agent_based_models/sugarscape && python scripts/run_single.py

# Bernoulli shift: block convergence → ergodic_systems/bernoulli_shift/results/block_convergence.png
cd ergodic_systems/bernoulli_shift && python scripts/run_convergence.py
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

### Ergodic Systems — Bernoulli Shift (`ergodic_systems/bernoulli_shift/`)

A minimal dynamical systems experiment (no agents) demonstrating how KS entropy controls the rate at which empirical block distributions converge to the true product measure.

**Core modules:**
- `sim/shift.py` — sequence generation (`generate_sequence`), empirical/true block distributions, KL divergence
- `sim/entropy.py` — `shannon_entropy(distribution)` computes H(p) = KS entropy h for a Bernoulli shift

**Design choices:**
- Alphabet: arbitrary discrete size (2+), distribution passed as a numpy array
- Block distributions stored as dicts keyed by tuples
- KL divergence sums only over empirically observed blocks (Q̂(b) > 0)
- All randomness via `numpy.random.Generator` with explicit seeds

## Research context

The long-term goal is to find the KS entropy that maximizes a meritocracy/social-mobility metric. The Sugarscape model is the **Phase 1 baseline** — standard model, no agent memory. The Bernoulli shift experiment validates the theoretical relationship between entropy and convergence rates. Phase 2 will add explicit agent memory (history-weighted movement decisions) as a tunable parameter directly linked to KS entropy. See `PHILOSOPHY.md` for the ergodic theory framework and `README.md` for full project description.
