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

### Running

```bash
cd agent_based_models/sugarscape
python scripts/run_single.py    # → results/single_run.png
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

### Architecture

**`ergodic_system.py`** — Abstract base class for measure-preserving ergodic systems (X, B, mu, T). Defines the interface:
- `iterate(state)` — apply the map T once
- `generate_trajectory(initial_state, n_steps, seed)` — produce a trajectory from the invariant measure
- `sample_initial_state(seed)` — sample from mu
- Optional: `jacobian()`, `symbolize()`, `analytical_ks_entropy()`

**`bernoulli_shift.py`** — The Bernoulli shift as a concrete `ErgodicSystem` subclass, plus module-level utility functions:
- `BernoulliShift(probs)` — i.i.d. process on a finite alphabet with distribution p
- `make_distribution(probs)` — validate a probability vector
- `generate_sequence(p, length, seed)` — sample i.i.d. symbols
- `shift(seq)` — left-shift map
- `empirical_block_distribution(seq, k)` — sliding window block frequencies
- `true_block_distribution(p, k)` — exact product measure over k-blocks

**`ks_entropy.py`** — Entropy computation via the box-counting (block entropy) method:
- `shannon_entropy(dist)` — H = -sum q log2(q) over a distribution dict
- `block_entropy_estimates(system, n_steps, k_max, seed)` — returns block entropies H(k), entropy rates H(k)/k, and differences H(k) - H(k-1)
- `plot_entropy_convergence(...)` — visualize H(k)/k vs k with optional analytical reference line

**`sim.py`** — Demo/validation script comparing fair coin ([0.5, 0.5]) and biased coin ([0.9, 0.1]) Bernoulli shifts. Both produce flat H(k)/k curves because i.i.d. processes have no memory — the entropy rate equals H(p) for all block lengths k.

### Running

```bash
cd ergodic_systems
python sim.py    # → entropy_rate.png
```

### Results

- **Fair coin**: H(k)/k = 1.0 bits for all k
- **Biased coin**: H(k)/k = 0.469 bits for all k
- Both flat — Bernoulli processes have zero memory, so convergence to the KS entropy is instant. This validates the framework; systems with temporal correlations (e.g. Markov chains) show non-trivial convergence from above.

