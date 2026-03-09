# Dynamical System Memory

Memory is simultaneously one of the most empowering and crippling qualities that humanity possesses. Our memories enable us to learn, adapt, and celebrate old traditions. Without a strong memory, we would quickly forget the lessons we have learned and never be able to advance. However, too strong a memory can act as an inertial force, holding us back from change. We get stuck in bad habits and fall victim to past trauma. How many world religions preach forgiveness in some way or another? How many times have we heard that the secret to success and happiness is staying present? There is a fine line between a healthy respect for the past and becoming overly traditional.

I aim to answer the questions: how much memory is optimal for growth? How far back into a system's history must we go until the past no longer has significant influence on the present? These questions lead me to learning about ergodic theory. We must first be able to quantify a system's memory.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Agent-Based Models](#agent-based-models)
  - [Sugarscape](#sugarscape-agent_based_modelssugarscape)
    - [The Model](#the-model)
    - [What the Model Measures](#what-the-model-measures)
    - [Why It Matters](#why-it-matters)
    - [KS Entropy Estimation](#ks-entropy-estimation)
  - [Minority Game](#minority-game-agent_based_modelsminority_game)
    - [The Model](#the-model-1)
    - [The Phase Transition](#the-phase-transition)
    - [KS Entropy Analysis](#ks-entropy-analysis)
    - [Entropy vs Efficiency](#entropy-vs-efficiency)
- [Neural Network Training Dynamics](#neural-network-training-dynamics-ml)
  - [The Dynamical System](#the-dynamical-system)
  - [Lyapunov Exponents via Hessian-Vector Products](#lyapunov-exponents-via-hessian-vector-products)
  - [Experimental Design](#experimental-design)
- [Ergodic Systems](#ergodic-systems)
  - [Framework](#framework-ergodic_systems)
  - [Results](#results)

## Repository Structure

This project contains two categories of computational experiments:

- **Agent-Based Models** (`agent_based_models/`) — simulations of interacting agents that produce emergent social phenomena
- **Neural Network Training Dynamics** (`ml/`) — treating gradient descent as a dynamical system and measuring its KS entropy via Lyapunov exponents
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

Three approaches estimate the KS entropy of the Sugarscape dynamics, providing groundwork for Phase 2 (tunable agent memory). The first two use block counting on symbolic sequences; the third uses Lyapunov exponent estimation via perturbation experiments.

**Approach 1: Wealth Distribution State** (`scripts/run_entropy_distribution.py`)

We take the entire wealth distribution to be the state of the system. This is a macroscopic description of the system. We can view this state as a state vector or a state distribution. At each time step, all 250 agents' sugar values are binned into a histogram (5 bins: 0–10, 10–25, 25–50, 50–100, 100+). This histogram tuple is the symbol for that step. Block counting on the resulting sequence estimates how unpredictable the *macro-level* wealth distribution is from step to step.

Result: H(k)/k starts at ~12.2 bits (k=1) and decreases as 12.2/k — the total block entropy H(k) barely grows past k=1, meaning the conditional entropy h(k) = H(k) − H(k−1) drops to ~0 after k=2. The aggregate distribution shape is nearly deterministic given the previous step — the economy's macro state has very low entropy rate.

**Approach 2: Individual Agent Wealth Trajectories** (`scripts/run_entropy_agents.py`)

We look at individual agents' wealth as the system's state. We consider the trajectory of an agents' wealth over time. This is a more microscopic description of the system - there could be some issues with well-definedness of the system's state; are we considering all agents or just one? Each agent's sugar is recorded at every step of its lifetime. All trajectories are discretized into 8 symbols using globally-computed quantile bins, and block statistics are pooled across ~24,000 agent lifetimes.

Result: H(k)/k is still declining at k=10 (~0.52 bits), with conditional entropy ~0.19 bits. Individual wealth is genuinely unpredictable — there is real entropy in agent-level dynamics, much more than in the aggregate distribution.

**Approach 3: Lyapunov Exponent** (`scripts/run_entropy_lyapunov.py`)

Rather than symbolizing the dynamics, this approach directly measures sensitivity to initial conditions. At regular intervals along a baseline simulation, the model is cloned and a small perturbation is applied (δ=1 sugar to one agent). Both copies are run forward and the Wasserstein-1 distance between their wealth distributions is tracked over time. The exponential growth rate of this divergence estimates the largest Lyapunov exponent, which by Pesin's identity bounds the KS entropy from below.

The Wasserstein-1 (earth mover's) distance was chosen as the metric on wealth distributions over alternatives: L2 norm on binned histograms doesn't respect the ordinal structure of wealth (shifting agents between adjacent bins costs the same as distant bins), and KL divergence is not a true metric (asymmetric, unbounded). Wasserstein-1 measures the minimum total wealth that would need to be redistributed to make two economies identical.

Result: λ ≈ 0.063 nats/step (0.091 bits/step), fitted over the initial growth phase (t=1..30) before saturation. The divergence curves show clear exponential growth followed by saturation — a tiny δ=1 perturbation (d(0) = 0.004) grows to d(100) ≈ 4.3. The positive Lyapunov exponent confirms the Sugarscape economy is chaotic: small differences in individual wealth compound into macroscopically different outcomes.

### Running

```bash
cd agent_based_models/sugarscape
python scripts/run_single.py                # → results/single_run.png
python scripts/run_entropy_distribution.py   # → results/distribution_entropy.png
python scripts/run_entropy_agents.py         # → results/agent_entropy.png
python scripts/run_entropy_lyapunov.py       # → results/lyapunov_entropy.png
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

## Minority Game (`agent_based_models/minority_game/`)

An implementation of the Minority Game, also known as the El Farol Bar Problem (Arthur 1994, Challet & Zhang 1997). N agents simultaneously choose between two options each round; those on the minority side win. Unlike Sugarscape, the Minority Game has a single parameter — memory length M — that directly controls the system's KS entropy, and a clear success metric (efficiency) that measures how well agents coordinate. This makes it ideally suited for answering the project's central question: *what level of system memory optimizes collective outcomes?*

### The Model

#### Setup

301 agents (odd, to guarantee a minority exists) play a repeated binary choice game. There is no spatial structure — agents interact only through the aggregate outcome. Each agent holds S=2 strategy tables, randomly assigned at initialization. A strategy table maps every possible M-length binary history pattern to an action (0 or 1), so each table has 2^M entries.

#### Game Mechanics

Each round:

1. All agents observe the same shared history — the last M winning outcomes
2. Each agent consults their highest-scoring strategy table to choose 0 or 1
3. All choices are collected simultaneously (no agent sees others' choices)
4. The minority side wins
5. Every strategy table (not just the one used) gets +1 if it would have picked the winning side, -1 otherwise
6. The winning action is appended to the shared history

The key insight is that agents don't learn individually — they accumulate evidence about which of their fixed strategies performs best. Strategies that correctly anticipate crowd behavior rise in score; strategies that follow the crowd fall.

#### The Complexity Ratio α

The single most important parameter is α = 2^M / N, the ratio of possible history patterns to the number of agents. This ratio controls the phase transition:

- **α < α_c ≈ 0.34 (crowded phase)**: Too many agents share too few strategies. Agents crowd onto the same patterns, creating herding behavior. Volatility exceeds the random baseline — agents do *worse* than coin flips.
- **α ≈ α_c (critical point)**: Optimal coordination. Volatility reaches its minimum and efficiency peaks.
- **α > α_c (uncrowded phase)**: Agents effectively act independently. Volatility approaches the random coin-flip baseline (σ²/N = 1/4).

### The Phase Transition

The sweep across M=2..12 (with N=301 fixed) reveals the classic Minority Game phase transition (`results/sweep_phase_transition.png`):

| M | α = 2^M/N | Volatility σ²/N | Efficiency | Predictability |
|---|---|---|---|---|
| 2 | 0.013 | 3.72 | -13.9 | high (exploitable) |
| 5 | 0.106 | 0.77 | -2.1 | moderate |
| 7 | 0.425 | 0.14 | 0.46 | ≈ 0 (efficient) |
| 8 | 0.851 | 0.16 | 0.35 | ≈ 0 |
| 12 | 13.6 | 0.24 | 0.06 | ≈ 0 |

At low α (short memory, crowded phase), agents are anti-coordinated so badly that volatility is 15x the random baseline and predictability is high — the market is exploitable. Near the critical point (M=7), efficiency peaks and predictability drops to zero — agents achieve genuine coordination, doing better than random. At high α (long memory), agents behave nearly independently and efficiency drops back toward zero.

### KS Entropy Analysis

The KS entropy of the binary outcome sequence is estimated using block counting (the same `empirical_block_distribution` and `shannon_entropy` functions from the ergodic systems framework). The outcome sequence is already binary, so no symbolization is needed.

At three representative memory lengths (M=3 crowded, M=6 near-critical, M=9 uncrowded):

- **M=3 (crowded)**: Conditional entropy converges to ~0.31 bits — highly predictable outcomes due to herding
- **M=6 (near-critical)**: Conditional entropy ~0.59 bits — moderate unpredictability
- **M=9 (uncrowded)**: Conditional entropy ~0.57 bits — near-random but with residual structure

The crowded phase has the lowest entropy because the herding dynamics create strong temporal correlations in the outcome sequence. As α increases through the critical point, the outcome becomes less predictable.

### Entropy vs Efficiency

The entropy analysis (`results/sweep_entropy.png`) plots KS entropy against each of the standard metrics across the full range of memory lengths. Each point in the scatter plots represents one simulation (M value, seed), colored by M, with the mean trajectory overlaid.

The entropy-efficiency plot reveals that **efficiency is maximized at intermediate entropy** — not at the lowest entropy (crowded phase, where herding destroys coordination) and not at the highest entropy (uncrowded phase, where agents act randomly). The optimal collective outcome occurs near the phase transition, where the system retains enough memory to coordinate but not so much that agents crowd onto identical strategies. The entropy-volatility and entropy-predictability plots show the same relationship from complementary angles.

This is a direct computational answer to the project's motivating question: *how much memory is optimal?* In the Minority Game, the answer is clear — the critical point α_c ≈ 0.34 represents the optimal balance between remembering too much and too little.

### Running

```bash
cd agent_based_models/minority_game
python scripts/run_single.py     # → results/single_run.png
python scripts/run_entropy.py    # → results/entropy_analysis.png
python scripts/run_sweep.py      # → results/sweep_phase_transition.png, sweep_entropy.png
```

**Configuration** (in `sim/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `n_agents` | 301 | Number of agents (must be odd) |
| `memory_length` | 6 | M — history bits agents observe |
| `n_strategies` | 2 | S — strategy tables per agent |
| `n_steps` | 500 | Simulation length |
| `seed` | 42 | Random seed for reproducibility |
| `alpha` (property) | 2^M/N | Complexity ratio controlling the phase transition |

---

# Neural Network Training Dynamics (`ml/`)

This experiment extends the project's central question — *how much forgetting is optimal for a system to make progress?* — to machine learning. We treat the gradient descent training process of a neural network as a discrete-time dynamical system on weight space and measure its KS entropy via Lyapunov exponents. The goal is to determine whether there is an optimal "forgetting rate" (KS entropy) that produces the best learning outcomes.

### The Dynamical System

The state of the system at time $t$ is the full weight vector $\theta_t \in \mathbb{R}^d$. The dynamics are defined by the gradient descent update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Each training step is one iteration of the map. We use full-batch gradient descent (autonomous dynamics) on a synthetic 3-class spiral dataset, with a 2-layer MLP (~4,500 parameters). The spiral dataset provides a nontrivial classification problem that is fast to train and easy to visualize.

### Lyapunov Exponents via Hessian-Vector Products

The Jacobian of one gradient descent step is $J = I - \eta H$, where $H$ is the Hessian of the loss. A positive Lyapunov exponent means nearby weight configurations diverge (the system "forgets" its initial conditions); a negative exponent means they converge (the system "remembers").

We compute the top-$k$ Lyapunov exponents using a modified Benettin algorithm. The key trick: rather than forming the full $d \times d$ Hessian, we propagate tangent vectors using Hessian-vector products ($H \cdot v$) via PyTorch's double-backward trick. This costs roughly one additional backward pass per tangent vector per step. Every $R$ steps, the tangent vectors are reorthonormalized via QR decomposition, and the diagonal of $R$ accumulates the stretching rates.

KS entropy is then estimated via Pesin's formula: $h_{KS} = \sum_{\lambda_i > 0} \lambda_i$

### Experimental Design

**Control variable:** Learning rate $\eta$, swept log-uniformly from $10^{-4}$ to $10.0$ (25 values), covering three regimes:
- **Too slow** (small $\eta$): nearly no progress, low KS entropy
- **Sweet spot** (moderate $\eta$): fast convergence, intermediate KS entropy
- **Divergent** (large $\eta$): loss increases or oscillates above random chance, high KS entropy

**Hypothesis:** There is an optimal KS entropy that minimizes convergence time — too low means slow exploration, too high means chaotic dynamics that prevent settling.

**Convergence step** is defined as the first training step at which a run's training loss drops to or below the 25th percentile of all final training losses across the entire sweep. This is a relative threshold: it asks "when did this run get as good as the better runs eventually got?" Runs that never reach the target within 500 steps are recorded as not converged and plotted at 500.

**Pipeline:**
1. `data.py` — generate spiral dataset
2. `model.py` — MLP with parameter flattening utilities
3. `lyapunov.py` — `LyapunovTracker` with Hessian-vector products
4. `train.py` — training loop with integrated Lyapunov tracking
5. `experiment.py` — sweep learning rates × random seeds
6. `analyze.py` — generate analysis plots

**Plots produced (`results/`):**
- `ks_vs_convergence.png` — two-panel plot: left shows KS entropy vs convergence step; right shows learning rate vs convergence step with a mean line tracing the U-shape. The right panel is the cleaner view of the primary result because KS entropy is not monotonically ordered by LR in the divergent regime.
- `ks_vs_loss.png` — KS entropy vs final train and test loss
- `ks_timeseries.png` — KS entropy evolution during training for representative LRs
- `lyapunov_spectrum.png` — top-k Lyapunov exponents vs learning rate (mean ± std)
- `loss_curves.png` — training loss vs step for different LRs

### Running

```bash
cd ml
python train.py          # Quick single-run test
python experiment.py     # Full sweep (~2-3 min on MacBook)
python analyze.py        # Generate all plots → results/
```

See `ml/PLAN.md` for the full theoretical specification.

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
│   ├── block_counting.py     # block entropy estimation and plotting
│   └── lyapunov.py           # Lyapunov exponent estimation
├── sims/                     # simulation/validation scripts
│   ├── bernoulli_sim.py
│   ├── logistic_sim.py
│   └── lyapunov_sim.py
└── results/                  # plots and output
```

**`systems/`** — Each system type gets its own module. `ErgodicSystem` ABC defines the interface (`iterate`, `generate_trajectory`, `sample_initial_state`, plus optional `jacobian`, `metric`, `perturb`, `symbolize`, `analytical_ks_entropy`). Current implementations: `BernoulliShift` (i.i.d. symbolic process) and `LogisticMap` (continuous, x → r*x*(1-x), implements `metric`/`perturb` for Lyapunov estimation).

**`entropy/`** — Each KS entropy estimation method gets its own module. `block_counting.py` implements the box-counting approach: compute block entropies H(k) from empirical symbol frequencies, then estimate the entropy rate as H(k)/k. Also provides `symbolize_timeseries()` to discretize continuous time series into integer symbols (quantile or uniform binning). `lyapunov.py` estimates the largest Lyapunov exponent via two methods: perturbation with renormalization (Benettin's algorithm, requires `metric`/`perturb`) and Jacobian averaging (requires `jacobian`). Both work generically on any `ErgodicSystem` subclass that implements the required methods.

**`sims/`** — Simulation scripts, one per system. Each produces trajectory visualizations and entropy rate validation plots.

**`results/`** — All plots and output files.

### Running

```bash
cd ergodic_systems
python sims/bernoulli_sim.py    # → results/bernoulli_trajectories.png, bernoulli_entropy_rate.png
python sims/logistic_sim.py     # → results/logistic_trajectories.png, logistic_entropy_rate.png
python sims/lyapunov_sim.py     # → results/logistic_lyapunov.png
```

### Results

**Bernoulli shift** — Trajectory plots show fair and biased coin sequences from 3 different seeds. Fair coin H(k)/k = 1.0 bits, biased coin H(k)/k = 0.469 bits, both flat for all k. No memory means instant convergence.

**Logistic map (r=4)** — Trajectory plots show chaotic evolution from 3 initial conditions. H(k)/k converges to 1.0 bits (= log2(2), matching the Lyapunov exponent ln(2) via Pesin's identity). The conditional entropy H(k)-H(k-1) drops below the analytical value at large k due to finite-sample underestimation — a known limitation of the box-counting method.

**Lyapunov exponent (logistic map)** — Both the perturbation method (Benettin's algorithm) and the Jacobian method converge to λ = ln(2) ≈ 0.6931 nats/step = 1.0 bits/step, matching the analytical value exactly. This validates Pesin's identity (h_KS = λ for smooth 1D maps) and confirms consistency between the block-counting and Lyapunov approaches. The Bernoulli shift is excluded from Lyapunov estimation since it is an i.i.d. symbolic process with no continuous state to perturb.
