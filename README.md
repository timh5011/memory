# Dynamical System Memory

Memory is simultaneously one of the most empowering and crippling qualities that humanity possesses. Our memories enable us to learn, adapt, and celebrate old traditions. Without a strong memory, we would quickly forget the lessons we have learned and never be able to advance. However, too strong a memory can act as an inertial force, holding us back from change. We get stuck in bad habits and fall victim to past trauma. How many world religions preach forgiveness in some way or another? How many times have we heard that the secret to success and happiness is staying present? There is a fine line between a healthy respect for the past and becoming overly traditional.

I aim to answer the questions: how much memory is optimal for growth? How far back into a system's history must we go until the past no longer has significant influence on the present? These questions lead me to learning about ergodic theory, whoch I apply the toosl of to machine learning and agent based models from game theory. We must first be able to quantify a system's memory.

## Table of Contents

- [Basic](#basic) — `basic/`
  - [Ergodic Systems](#ergodic-systems) — `basic/ergodic_systems/`
  - [Agent-Based Models](#agent-based-models) — `basic/agent_based_models/`
    - [Sugarscape](#sugarscape) — `basic/agent_based_models/sugarscape/`
    - [Minority Game](#minority-game) — `basic/agent_based_models/minority_game/`
  - [Neural Network Training Dynamics](#neural-network-training-dynamics) — `basic/ml/`
- [ML Training as an Ergodic System](#ml-training-as-an-ergodic-system) — `ml_ergodic/`
- [LLM Agent-Based Models](#llm-agent-based-models) — `llm_abm/`
  - [LLM Minority Game](#llm-minority-game) — `llm_abm/minority_game/`
  - [Polis](#polis) — `llm_abm/society/`

Every experiment below quantifies a system's memory as its **Kolmogorov–Sinai (KS) entropy** — the rate at which the system generates new information and forgets its past — and asks how that quantity relates to the system's success. See `doc/PHILOSOPHY.md` for the theoretical framework and `basic/ergodic_systems/THEORY.md` for the entropy-estimation methods.

## Basic

`basic/` — the foundational, non-LLM experiments, plus the shared KS-entropy tooling that every other track reuses.

### Ergodic Systems

`basic/ergodic_systems/`

The theoretical and computational foundation for everything else. A small framework that defines ergodic dynamical systems (the Bernoulli shift, the logistic map) behind a common `ErgodicSystem` interface, plus two independent ways to estimate KS entropy: block counting on symbol sequences, and the largest Lyapunov exponent via Benettin's algorithm (equal to the entropy for these maps by Pesin's identity). It answers the prerequisite question — *can we measure a system's memory at all?* — by recovering known analytical entropies (H(p) for a biased coin, ln 2 for the logistic map) to validate the tools every other experiment reuses. Relies on: ergodic theory, symbolic dynamics, Lyapunov exponents.

### Agent-Based Models

`basic/agent_based_models/` — simulations of many interacting agents whose emergent collective behavior is the object of study.

#### Sugarscape

`basic/agent_based_models/sugarscape/`

A classic agent-based model of resource competition (Epstein & Axtell 1996): agents with heterogeneous vision and metabolism forage sugar on a grid, and realistic wealth inequality emerges from nothing but local foraging rules. Here it is a testbed for *how much memory lives in an emergent economy* — the KS entropy of its dynamics is estimated three ways (macro wealth-distribution entropy, pooled individual-trajectory entropy, and a Lyapunov exponent measured by perturbing one agent and tracking Wasserstein divergence), showing the aggregate economy is nearly deterministic while individual trajectories are genuinely chaotic. Relies on: agent-based modeling (Mesa), ergodic theory, block-counting and Lyapunov entropy.

#### Minority Game

`basic/agent_based_models/minority_game/`

The El Farol Bar problem (Arthur 1994; Challet & Zhang 1997): an odd number of agents repeatedly choose between two options and the minority side wins, each agent selecting among fixed strategy tables keyed on the last M outcomes. It is the project's cleanest instrument because a single parameter — the memory length M — tunes KS entropy directly, while efficiency gives an unambiguous success metric. Sweeping M reproduces the game's phase transition at α = 2^M/N ≈ 0.34 and shows that *collective efficiency is maximized at intermediate entropy* — neither the low-entropy herding phase nor the high-entropy random phase — a direct computational answer to "how much memory is optimal?" Relies on: game theory, agent-based modeling, block-counting KS entropy.

### Neural Network Training Dynamics

`basic/ml/`

Treats gradient descent as a discrete-time dynamical system on weight space and asks whether there is an optimal "forgetting rate" for learning. Training a small MLP on a spiral dataset, it computes the top Lyapunov exponents of the weight dynamics with a modified Benettin algorithm using Hessian-vector products (avoiding the full Hessian via PyTorch's double-backward), then estimates KS entropy by Pesin's identity. Sweeping the learning rate reveals a U-shaped relationship between entropy and convergence speed: too little chaos explores too slowly, too much prevents the network from settling. Relies on: machine learning (PyTorch), dynamical-systems theory, Lyapunov/Pesin entropy. See `basic/ml/PLAN.md`.

## ML Training as an Ergodic System

`ml_ergodic/`

Wraps that same gradient-descent training as a first-class `ErgodicSystem`, so the generic block-counting and Benettin tools from `basic/ergodic_systems/` apply to training trajectories directly. It complements the Pesin-route estimate in `basic/ml/` with a symbolic-route estimate from the loss signal, and adds empirical checks of the ergodicity assumption itself (Birkhoff time-averages across seeds, autocorrelation decay as a mixing proxy). Relies on: machine learning, ergodic theory, block-counting entropy. See `ml_ergodic/README.md`.

## LLM Agent-Based Models

`llm_abm/` — agent-based models whose agents are LLMs reasoning in natural language (via a CAMEL-AI custom loop) rather than lookup tables. A free rule-based mock backend runs each full pipeline at zero cost before any paid API call. See `llm_abm/README.md`.

### LLM Minority Game

`llm_abm/minority_game/`

The Minority Game again, but the fixed strategy tables are replaced by LLM agents that reason in natural language. The tunable memory parameter becomes the *memory window* — how many past rounds are rendered into each agent's prompt — kept exact by externalizing all memory into the prompt (every model call is stateless and single-turn). The free mock backend runs the entire pipeline (simulation, transcripts, metrics, KS entropy, memory sweeps) at zero cost, and the classical game supplies a calibrated baseline. Relies on: LLM agents (CAMEL-AI), game theory, block-counting KS entropy.

### Polis

`llm_abm/society/`

The project's most ambitious experiment: a minimally complete society built from first principles rather than a well-posed game. LLM agents have heterogeneous identities (occupation, class of origin, temperament, households, friendships, political faction) and personal *value weights* over four fulfillment dimensions (prosperity, belonging, standing, security), so each agent succeeds by its own lights, while a deterministic world engine applies every consequence of their chosen actions.

It carries the project's philosophy — *how forgiving should a society be? how much does the past hold people back?* — into two literal, sweepable memory knobs: an individual one (the prompt's memory window) and a societal one (how fast reputation and relationships decay). Ergodic analysis is then applied to the resulting state trajectories, including a **mobility entropy rate** that expresses social mobility itself as an entropy — 0 bits is a frozen hierarchy, full reshuffling is maximal — computed per sub-population equivalence class. Relies on: LLM agents (CAMEL-AI), ergodic theory, agent-based modeling. See `llm_abm/society/README.md`.
