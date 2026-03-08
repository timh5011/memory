# Dynamical System Memory

Memory is simultaneously one of the most empowering and crippling qualities that humanity possesses. Our memories enable us to learn, adapt, and celebrate old traditions. Without a strong memory, we would quickly forget the lessons we have learned and never be able to advance. However, too strong a memory can act as an inertial force, holding us back from change. We get stuck in bad habits and fall victim to past trauma. How many world religions preach forgiveness in some way or another? How many times have we heard that the secret to success and happiness is staying present? There is a fine line between a healthy respect for the past and becoming overly traditional.

I aim to answer the questions: how much memory is optimal for growth? How far back into a system's history must we go until the past no longer has significant influence on the present? These questions lead me to learning about ergodic theory. We must first be able to quantify a system's memory.

## Repository Structure

This project contains two categories of computational experiments:

- **Agent-Based Models** (`agent_based_models/`) — simulations of interacting agents that produce emergent social phenomena
- **Ergodic Systems** (`ergodic_systems/`) — direct experiments on dynamical systems from ergodic theory

See `PHILOSOPHY.md` for the full theoretical framework connecting KS entropy, mixing, and Bernoulli shifts.

---

# Agent-Based Models

## Sugarscape (`agent_based_models/sugarscape/`)

An implementation of the Sugarscape agent-based model from Epstein & Axtell's *Growing Artificial Societies* (1996). Sugarscape is one of the foundational models in computational social science, demonstrating how simple individual rules produce complex collective phenomena — most notably, realistic wealth inequality emerges without anyone designing it in.

### The Model

#### Environment

A 50x50 toroidal grid represents the landscape. Each cell holds some amount of sugar (0-4 units), distributed as two Gaussian peaks — think of them as two "mountains" of resources. After agents harvest sugar each step, it regrows at a fixed rate up to each cell's natural capacity.

#### Agents

250 agents inhabit the grid. Each is born with randomly drawn traits:

- **Vision** (1-6) — how far the agent can see in each cardinal direction
- **Metabolism** (1-4) — how much sugar the agent burns each step just to survive
- **Max age** (60-100) — natural lifespan

Agents start with a random sugar endowment (5-25 units). When an agent dies — either from starvation (sugar drops below zero) or old age — a new agent with fresh random traits is placed at a random empty cell, keeping the population constant at 250.

#### Behavior

Each step, agents act in random order. An agent:

1. Surveys all cells within its vision range in the four cardinal directions
2. Moves to whichever visible cell has the most sugar (breaking ties by proximity)
3. Harvests all sugar on that cell
4. Pays its metabolism cost

That's it. No cooperation, no communication, no strategy beyond "move toward the most sugar I can see." The striking result is that these minimal rules are enough to generate rich, realistic economic dynamics.

### What the Model Measures

**Gini Coefficient** — quantifies wealth inequality (0 = perfect equality, 1 = perfect inequality). Typically stabilizes around 0.49-0.51, demonstrating that inequality emerges naturally from heterogeneous abilities interacting with spatially structured resources.

**Mean Sugar (Wealth)** — average sugar held by agents over time. Spikes early then settles into a fluctuating steady state as competition and metabolism balance out regrowth.

**Wealth Distribution** — characteristically right-skewed, strikingly similar to real-world wealth distributions.

**Population** — held constant at 250 via replacement. Turnover rate reflects environmental harshness.

### Why It Matters

Sugarscape demonstrates a core principle of complex systems: **macro-level patterns need not be designed or intended — they can emerge from micro-level rules.** No agent is trying to create inequality. No central planner is distributing resources unfairly. Yet substantial inequality reliably emerges from nothing more than agents with different abilities foraging on a landscape.

### KS Entropy Estimation

Two complementary approaches estimate the KS entropy of the Sugarscape dynamics, providing groundwork for Phase 2 (tunable agent memory).

**Approach 1: Wealth Distribution State** (`scripts/run_entropy_distribution.py`)

At each time step, all 250 agents' sugar values are binned into a histogram (5 bins: 0–10, 10–25, 25–50, 50–100, 100+). This histogram tuple is the symbol for that step. Block counting on the resulting sequence estimates how unpredictable the *macro-level* wealth distribution is from step to step.

Result: The conditional entropy h(k) = H(k) − H(k−1) drops to ~0 after k=2. The aggregate distribution shape is nearly deterministic given the previous step — the economy's macro state has very low entropy rate.

**Approach 2: Individual Agent Wealth Trajectories** (`scripts/run_entropy_agents.py`)

Each agent's sugar is recorded at every step of its lifetime. All trajectories are discretized into 8 symbols using globally-computed quantile bins, and block statistics are pooled across ~24,000 agent lifetimes.

Result: H(k)/k is still declining at k=10 (~0.52 bits), with conditional entropy ~0.19 bits. Individual wealth is genuinely unpredictable — there is real entropy in agent-level dynamics, much more than in the aggregate distribution.

### Running

```bash
cd agent_based_models/sugarscape
python scripts/run_single.py                # → results/single_run.png
python scripts/run_entropy_distribution.py   # → results/distribution_entropy.png
python scripts/run_entropy_agents.py         # → results/agent_entropy.png
```

**Configuration** (in `sim/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `grid_width` / `grid_height` | 50 | Grid dimensions |
| `alpha` | 1.0 | Sugar regrowth per step per cell |
| `n_agents` | 250 | Number of agents |
| `max_vision` | 6 | Upper bound of vision distribution |
| `n_steps` | 500 | Simulation length |
| `seed` | 42 | Random seed for reproducibility |

---

# Ergodic Systems

## Framework (`ergodic_systems/`)

A general-purpose framework for defining ergodic dynamical systems and computing their KS entropy numerically.

### Directory Structure

```
ergodic_systems/
├── systems/                  # dynamical system definitions
│   ├── ergodic_system.py     # ABC: ErgodicSystem base class
│   ├── bernoulli_shift.py    # BernoulliShift
│   └── logistic_map.py       # LogisticMap
├── entropy/                  # KS entropy computation methods
│   └── block_counting.py     # block entropy estimation and plotting
├── sims/                     # simulation/validation scripts
│   ├── bernoulli_sim.py
│   └── logistic_sim.py
└── results/                  # plots and output
```

**`systems/`** — Each system type gets its own module. `ErgodicSystem` ABC defines the interface (`iterate`, `generate_trajectory`, `sample_initial_state`, plus optional `jacobian`, `symbolize`, `analytical_ks_entropy`). Current implementations: `BernoulliShift` (i.i.d. symbolic process) and `LogisticMap` (continuous, x → r*x*(1-x)).

**`entropy/`** — Each KS entropy estimation method gets its own module. `block_counting.py` implements the box-counting approach: compute block entropies H(k) from empirical symbol frequencies, then estimate the entropy rate as H(k)/k. Also provides `symbolize_timeseries()` to discretize continuous time series into integer symbols (quantile or uniform binning).

**`sims/`** — Simulation scripts, one per system. Each produces trajectory visualizations and entropy rate validation plots.

**`results/`** — All plots and output files.

### Running

```bash
cd ergodic_systems
python sims/bernoulli_sim.py    # → results/bernoulli_trajectories.png, bernoulli_entropy_rate.png
python sims/logistic_sim.py     # → results/logistic_trajectories.png, logistic_entropy_rate.png
```

### Results

**Bernoulli shift** — Trajectory plots show fair and biased coin sequences from 3 different seeds. Fair coin H(k)/k = 1.0 bits, biased coin H(k)/k = 0.469 bits, both flat for all k. No memory means instant convergence.

**Logistic map (r=4)** — Trajectory plots show chaotic evolution from 3 initial conditions. H(k)/k converges to 1.0 bits (= log2(2), matching the Lyapunov exponent ln(2) via Pesin's identity). The conditional entropy H(k)-H(k-1) drops below the analytical value at large k due to finite-sample underestimation — a known limitation of the box-counting method.

