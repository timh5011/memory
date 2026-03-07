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
│   └── bernoulli_shift/     # Entropy rate convergence experiments
│       ├── bernoulli_shift.py   # core: distribution, sequence generation, shift, block distributions, KL divergence
│       ├── sim.py               # Bernoulli: empirical entropy rate H(k-block)/k vs k
│       └── markov_sim.py        # Markov chain: entropy rate convergence showing memory effects
├── PHILOSOPHY.md
└── README.md
```

## Commands

```bash
# Sugarscape: single simulation (500 steps) → agent_based_models/sugarscape/results/single_run.png
cd agent_based_models/sugarscape && python scripts/run_single.py

# Bernoulli shift: entropy rate vs block length → ergodic_systems/bernoulli_shift/entropy_rate.png
cd ergodic_systems/bernoulli_shift && python sim.py

# Markov chain: entropy rate convergence showing memory effects → markov_entropy_rate.png
cd ergodic_systems/bernoulli_shift && python markov_sim.py
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

Two experiments demonstrating KS entropy and the role of memory in dynamical systems.

**Core module (`bernoulli_shift.py`):**
- `make_distribution(probs)` — validates and returns a probability vector
- `generate_sequence(p, length, seed)` — i.i.d. samples from p (the Bernoulli process)
- `shift(seq)` — left-shift map: drops first symbol
- `true_block_distribution(p, k)` — product measure over all length-k blocks via `itertools.product`
- `empirical_block_distribution(seq, k)` — sliding window counts normalized to frequencies
- `kl_divergence(empirical, true)` — KL(Q̂ || P), summing only over observed blocks

**`sim.py` — Bernoulli entropy rate:**
- Shows `H(empirical k-block) / k` vs k for fair vs biased coin
- Both curves are flat (already at H(p) for all k) — because i.i.d. symbols have no memory

**`markov_sim.py` — Markov chain entropy rate:**
- Same experiment on two Markov chains: low memory (near-independent) vs high memory (sticky)
- High-memory chain shows `H(k-block)/k` starting high at k=1 and converging downward to true entropy rate
- This convergence is invisible in Bernoulli but emerges when symbols have temporal correlations

**Design choices:**
- Block distributions stored as dicts keyed by tuples
- KL divergence sums only over empirically observed blocks (Q̂(b) > 0)
- All randomness via `numpy.random.Generator` with explicit seeds

## Research context

The long-term goal is to find the KS entropy that maximizes a meritocracy/social-mobility metric. The Sugarscape model is the **Phase 1 baseline** — standard model, no agent memory. The Bernoulli shift experiment validates the theoretical relationship between entropy and convergence rates. Phase 2 will add explicit agent memory (history-weighted movement decisions) as a tunable parameter directly linked to KS entropy. See `PHILOSOPHY.md` for the ergodic theory framework and `README.md` for full project description.
