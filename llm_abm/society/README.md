# Polis: A Minimally Complete LLM Society

Polis is the project's most ambitious experiment: instead of a well-posed
game with a known phase structure, it is an attempt to define a **minimally
complete society from first principles** — identities, values, relationships,
politics, an economy — populate it with LLM agents, and apply ergodic theory
to its state trajectories. It deliberately trades the analytical cleanliness
of the Minority Game for **scope**: the questions in `doc/PHILOSOPHY.md`
(How forgiving should a society be? How much does the past hold people back?
What memory level optimizes progress?) are asked of a whole small
civilization rather than a single coordination game.

Built on the CAMEL-custom-loop pattern via the existing
`llm_abm/sim/backends.py` abstraction: the free `SocietyMockBackend` runs the
entire pipeline at zero cost; live LLM runs plug in later behind the same
guardrails plus a hard API-call budget cap.

## The Axioms (what "minimally complete" means here)

A society, minimally, has people who:

1. **Are different from each other** — occupation, class of origin,
   temperament (ambition, gregariousness, progressivism), and political
   faction (Hearth League vs Meridian Society).
2. **Are embedded in relationships** — households (family, whose ties never
   fully decay) and friendships (which do), forming a social graph that the
   agents' own choices reshape.
3. **Want different things** ⭐ — each agent has **value weights**: a
   normalized vector over four fulfillment dimensions saying what *this*
   person cares about. Someone weighted 55% toward belonging should — and,
   via the system prompt, does — live differently from someone weighted 55%
   toward prosperity. This is the "someone cares more about family than
   money" requirement made into the core success metric.
4. **Succeed by their own lights** — fulfillment
   `F_i = Σ_d w_id · state_id` over four dimensions:
   **prosperity** (wealth), **belonging** (tie strengths), **standing**
   (public reputation), **security** (reserves + buffer). Society-level
   success = mean F (welfare), Gini of F (inequality), and **value
   alignment** (are people doing well at what *they* value, beyond generic
   success?).
5. **Act under scarcity** — one action per season (WORK, SOCIALIZE, HELP,
   HOST_GATHERING, VENTURE, ADVOCATE, REST), a cost of living, hardship when
   wealth runs out. The LLM decides; the deterministic world engine
   (`world.py`) applies every consequence. No physics is delegated to the
   language model.
6. **Live in a society that remembers** — standing and ties decay each
   season. The decay rates are the *society's* memory.

## The Two Memory Knobs

This is the experiment. Both are exact, sweepable parameters:

| Knob | Meaning | Philosophy |
|---|---|---|
| `memory_window` w | Seasons of events in each agent's prompt (the prompt IS the agent's memory — same externalized-memory principle as the MG pipeline) | *How much does an individual's past condition their present?* |
| `reputation_decay`, `tie_decay` | How fast standing and relationships fade | *How forgiving is the institution? Do grudges and glory persist?* |

The central plots: KS entropy and success metrics as functions of each knob
(`scripts/society_sweep.py --knob memory` / `--knob forgiveness`).

## Ergodic Analysis (`analysis/society_entropy.py`)

Four views, extending the sugarscape playbook:

1. **Trajectory entropy** — block entropy of individual fulfillment series
   (quantile-symbolized, pooled), computable **per equivalence class**.
2. **Macro entropy** — block entropy of the whole fulfillment distribution's
   season-to-season evolution.
3. **Mobility entropy** ⭐ — per-season fulfillment-rank quintiles → empirical
   transition matrix → Markov entropy rate. **Social mobility expressed
   literally as an entropy rate**: 0 bits = frozen hierarchy (the society
   never forgets your station), log₂5 ≈ 2.32 bits = station reshuffled every
   season. A meritocracy score in the project's observable-only sense.
4. **Equivalence classes** — sub-populations by class of origin, faction,
   dominant value, and temperament. The pointed question: *do the lower
   classes live lower-entropy (more stuck) trajectories, and does turning up
   the forgiveness knob change that?*

## What the Mock Baseline Already Shows

The validated mock run (N=24, 120 seasons, seed 42) produces: a plausible
action economy (~69% WORK), welfare ≈ 0.36, fulfillment Gini ≈ 0.17, mobility
entropy ≈ 1.22 bits/season with visible bottom-quintile stickiness ("stuck at
the bottom" appears in the mock already), and upper-class initial advantage
eroding over ~30 seasons. The mock is weakly memory-sensitive by design
(reciprocity toward remembered helpers, venture hot-hand, gathering
imitation), so both knobs produce real variation even at zero cost — in the
micro-sweep, more individual memory already lowers mobility entropy (memory
entrenches hierarchy). These mock numbers are the hand-coded-rationality
baseline that live LLM societies get compared against.

## Running

```bash
cd llm_abm

# Free — validated:
python scripts/society_run_mock.py                    # → results/mock_society_run.png
python scripts/society_sweep.py --knob memory         # → results/society_sweep_memory_mock.png
python scripts/society_sweep.py --knob forgiveness    # → results/society_sweep_forgiveness_mock.png
python scripts/society_estimate_tokens.py             # usage estimate for live runs

# Live (costs money; camel-ai + API key; start small):
python scripts/society_sweep.py --knob forgiveness --backend camel --live \
    --agents 12 --steps 40 --seeds 1 --values 0.0,0.1,0.5
```

## Budget

Default config (N=24, 120 seasons) ≈ 2,880 calls/run at ~600 input tokens per
call (≈2M input tokens/run) — single-digit dollars per run with a small
model; a full two-knob sweep stays far under the $100 ceiling. Guardrails, in
depth: paid backends must be constructed explicitly with
`allow_api_calls=True`; sweeps require `--live`; and `max_api_calls`
(default 5,000) hard-stops any live run mid-flight, **keeping all partial
records and transcripts**. Run `society_estimate_tokens.py` before every live
run.

## Files

```
society/
├── config.py       # SocietyConfig — the two memory knobs + budget cap
├── identity.py     # who people are: roles, classes, temperament, households,
│                   #   friendships, factions, VALUE WEIGHTS (Dirichlet, temperament-biased)
├── world.py        # deterministic engine: action resolution, economy, decay
│                   #   (societal memory), fulfillment computation
├── prompts.py      # system prompt (identity + values in words), observation
│                   #   rendering, strict JSON action parsing (fallback: REST)
├── agents.py       # bounded event memory → prompt; memory-signal extraction
├── mock_policy.py  # free in-character stand-in: value-weighted action scoring
└── model.py        # season loop, BudgetGuard, records for ergodic analysis
```

## Known Limitations (by design, for now)

- **The engine's magnitudes are stipulated**, not calibrated: wage tables,
  tie deltas, decay constants are plausible-but-arbitrary. The *comparative*
  results (entropy/welfare as functions of the knobs) are the point, not the
  absolute numbers.
- **Population-level statistics only** — with N=24 and stochastic LLMs,
  individual trajectories are illustrative, never evidence (per the tool
  report's reliability guidance). Multiple seeds per knob value are mandatory
  for live conclusions.
- **Short runs strain block counting** — hence 5-bin alphabets, k ≤ 4,
  pooling within sub-populations, and the mobility entropy (k=1 transitions)
  as the most data-robust of the entropy measures.
- **No birth/death or role change yet.** Natural extensions: generational
  turnover (inheritance!), occupational mobility, exogenous shocks (drought
  seasons), and interviewing agents post-run about their lives.
