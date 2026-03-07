# Sugarscape

An implementation of the Sugarscape agent-based model from Epstein & Axtell's *Growing Artificial Societies* (1996). Sugarscape is one of the foundational models in computational social science, demonstrating how simple individual rules produce complex collective phenomena — most notably, realistic wealth inequality emerges without anyone designing it in.

## The Model

### Environment

A 50x50 toroidal grid represents the landscape. Each cell holds some amount of sugar (0-4 units), distributed as two Gaussian peaks — think of them as two "mountains" of resources. After agents harvest sugar each step, it regrows at a fixed rate up to each cell's natural capacity.

### Agents

250 agents inhabit the grid. Each is born with randomly drawn traits:

- **Vision** (1-6) — how far the agent can see in each cardinal direction
- **Metabolism** (1-4) — how much sugar the agent burns each step just to survive
- **Max age** (60-100) — natural lifespan

Agents start with a random sugar endowment (5-25 units). When an agent dies — either from starvation (sugar drops below zero) or old age — a new agent with fresh random traits is placed at a random empty cell, keeping the population constant at 250.

### Behavior

Each step, agents act in random order. An agent:

1. Surveys all cells within its vision range in the four cardinal directions
2. Moves to whichever visible cell has the most sugar (breaking ties by proximity)
3. Harvests all sugar on that cell
4. Pays its metabolism cost

That's it. No cooperation, no communication, no strategy beyond "move toward the most sugar I can see." The striking result is that these minimal rules are enough to generate rich, realistic economic dynamics.

## What the Model Measures

### Gini Coefficient

The Gini coefficient quantifies wealth inequality on a scale from 0 (perfect equality — everyone has the same sugar) to 1 (perfect inequality — one agent has everything). In a typical Sugarscape run, the Gini starts low when agents have similar random endowments, then rises and stabilizes around 0.49-0.51. This emergent inequality is one of the model's most important results: you don't need greed, exploitation, or institutional failure to produce a highly unequal wealth distribution. It arises naturally from heterogeneous abilities (vision, metabolism) interacting with a spatially structured resource.

### Mean Sugar (Wealth)

The average sugar held by agents over time reveals the system's macroeconomic trajectory. It typically spikes early as agents spread across the grid and find sugar, then settles into a fluctuating steady state as competition and metabolism balance out regrowth. The steady-state level reflects the carrying capacity of the landscape given the population's traits.

### Wealth Distribution

The histogram of agent wealth at any given step is characteristically right-skewed: many agents hold modest amounts of sugar while a long tail of wealthy agents extends far to the right. This shape is strikingly similar to real-world wealth distributions, which is remarkable given the model's simplicity.

### Population

With the replacement rule active, population stays constant at 250. The turnover rate — how frequently agents die and are replaced — reflects how harsh the environment is. Higher metabolism or lower sugar regrowth means more deaths and faster population turnover.

## Why It Matters

Sugarscape demonstrates a core principle of complex systems: **macro-level patterns need not be designed or intended — they can emerge from micro-level rules.** No agent is trying to create inequality. No central planner is distributing resources unfairly. Yet substantial inequality reliably emerges from nothing more than agents with different abilities foraging on a landscape. This makes Sugarscape a powerful pedagogical tool for understanding how simple, decentralized interactions can produce the large-scale social patterns we observe in the real world.

## Repository Structure

```
sugarscape/
├── sim/
│   ├── __init__.py       # Package exports
│   ├── config.py         # SugarscapeConfig dataclass
│   ├── grid.py           # Sugar landscape initialization
│   ├── agents.py         # SugarAgent class
│   ├── model.py          # SugarscapeModel class
│   └── metrics.py        # Gini coefficient
├── scripts/
│   └── run_single.py     # Single simulation + diagnostic plots
└── results/              # Generated outputs (PNG)
```

## Setup and Usage

**Dependencies:** Python 3.11, `mesa==3.3.1`, `numpy`, `pandas`, `matplotlib`

```bash
pip install mesa numpy pandas matplotlib
```

**Run a simulation:**

```bash
cd sugarscape
python scripts/run_single.py
```

Outputs a 4-panel diagnostic plot to `results/single_run.png` showing population, mean sugar, Gini coefficient over time, and the final wealth distribution histogram.

**Configuration** (in `sim/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `grid_width` / `grid_height` | 50 | Grid dimensions |
| `alpha` | 1.0 | Sugar regrowth per step per cell |
| `n_agents` | 250 | Number of agents |
| `max_vision` | 6 | Upper bound of vision distribution |
| `n_steps` | 500 | Simulation length |
| `seed` | 42 | Random seed for reproducibility |
