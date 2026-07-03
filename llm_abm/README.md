# LLM Agent-Based Models: KS Entropy of an LLM Society

This experiment carries the project's central question — *how much memory is
optimal for a system to make progress?* — into agent-based models whose agents
are **LLMs instead of lookup tables**. It is the natural next step after the
classical Minority Game (`basic/agent_based_models/minority_game/`): same
game, same metrics, same KS entropy machinery, but the strategy tables are
replaced by language-model reasoning, and the memory parameter becomes
something much closer to the project's philosophical motivation — *how much of
its own history a mind is shown before it acts*.

The tooling choices follow `doc/tool_report_camel_oasis_mirofish.md`. Of the
three frameworks surveyed there (CAMEL, OASIS, MiroFish), this experiment
targets **CAMEL with a custom simulation loop** — the report's own
recommendation for research simulations of a population of agents in a shared
environment. OASIS/MiroFish are locked into a social-media action space, which
doesn't fit a controlled game-theoretic environment with a clean success
metric; CAMEL is the general-purpose layer underneath them.

## The Experiment

**Environment: the Minority Game, again — deliberately.** N agents (odd)
simultaneously choose 0 or 1 each round; the minority side wins. We already
know this game inside out from the classical version: it has a clear success
metric (efficiency), a binary outcome sequence that is directly a symbol
sequence for block-counting KS entropy, and a well-understood
memory-controlled phase structure. Reusing it means every LLM result has a
classical baseline to compare against.

**The control variable: the memory window `w`.** Each round, every agent
receives a fresh single-turn prompt containing exactly the last `w` rounds of
history (winning options, its own choices, its win count). `w` is the LLM
analog of the classical memory length M. The sweep over `w`, plotting KS
entropy of the outcome sequence against collective efficiency, is the LLM
edition of the classical "money plot" (`sweep_entropy.png`).

**Key design decision — memory is external and researcher-controlled.** We do
NOT use the LLM framework's internal conversation memory (CAMEL's
`ChatHistoryMemory`, etc.). Every backend call is stateless and single-turn;
the prompt *is* the agent's entire memory. This makes the memory depth an
exact, tunable experimental parameter rather than an opaque property of a
framework, makes mock and LLM backends perfectly interchangeable, and makes
every agent decision fully reproducible from the transcript log.

**Persona heterogeneity.** Agents are assigned distinct persona lines
(contrarian, trend-follower, statistician, ...) cycled across the population.
This is the LLM analog of the random strategy tables in the classical game:
identical agents given identical prompts would herd maximally. Set
`persona_diversity=False` to study exactly that failure mode.

## Architecture

```
llm_abm/
├── sim/
│   ├── config.py     # LLMMinorityGameConfig — w (memory_window) is the key parameter
│   ├── prompts.py    # system/observation templates + robust 0/1 response parsing
│   ├── backends.py   # AgentBackend protocol: MockBackend (free) / CamelBackend ($)
│   ├── agents.py     # LLMAgent: bounded deque memory → prompt → backend → action
│   ├── model.py      # round loop, transcripts (JSONL), run records (JSON)
│   └── metrics.py    # volatility / efficiency / predictability (same defs as classical MG)
├── analysis/
│   └── ks_entropy.py # block entropy of outcome sequences (reuses basic/ergodic_systems)
├── scripts/
│   ├── run_mock_single.py   # free single-run diagnostic (4-panel plot)
│   ├── run_sweep_memory.py  # THE experiment: sweep w → entropy vs efficiency
│   └── estimate_tokens.py   # usage estimate BEFORE any live run
└── results/                 # plots; results/runs/ holds JSON records + JSONL transcripts
```

The backend abstraction is the load-bearing piece:

- **`MockBackend`** — a mixture of simple behavioral policies standing in for
  LLM reasoning. Free, local, no network. The entire pipeline (simulation,
  parsing, retries, transcripts, metrics, entropy analysis, sweep, plots) runs
  and is validated against it. It occasionally returns verbose text on purpose
  so the response parser and retry path stay exercised.
- **`CamelBackend`** — wraps a CAMEL `ChatAgent` for live runs. Requires
  `pip install camel-ai` and an API key. **Untested until camel-ai is
  installed** — verify the `ModelFactory`/`ChatAgent` calls against the
  installed version before the first live run.

### Money guardrails

Nothing in this directory can spend money by accident:

1. `CamelBackend` refuses to construct without `allow_api_calls=True`.
2. The sweep script refuses paid backends without the `--live` flag.
3. `build_backend()` only auto-builds the mock; paid backends must be an
   explicit line of code in the caller.
4. `scripts/estimate_tokens.py` renders the real prompts and reports call and
   token counts for any planned configuration before you commit to it. (It
   reports tokens, not dollars — multiply by current prices yourself so the
   numbers can't go stale.)

## Running

```bash
cd llm_abm

# Free — validated, runs today:
python scripts/run_mock_single.py                 # → results/mock_single_run.png
python scripts/run_sweep_memory.py                # → results/sweep_memory_mock.png
python scripts/estimate_tokens.py --agents 9 --steps 30 --window 4

# Live (costs money; requires camel-ai + API key; start tiny):
python scripts/run_sweep_memory.py --backend camel --live \
    --agents 9 --steps 30 --seeds 1 --windows 0,2,4
```

**Configuration** (`sim/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `n_agents` | 51 | Number of agents (must be odd) |
| `memory_window` | 4 | w — rounds of history in each agent's prompt |
| `n_steps` | 300 | Rounds per run |
| `backend` | "mock" | "mock" is free; LLM backends passed explicitly |
| `model_name` | "claude-sonnet-5" | Model for live runs (via CAMEL) |
| `temperature` | 0.7 | Stochasticity of live agents |
| `persona_diversity` | True | Cycle distinct personas across agents |
| `transcript_path` | None | JSONL log of every prompt/response |

Mock baseline from the validation run (N=51, w=4, 300 rounds): efficiency
≈ 0.37, entropy rate ≈ 0.54 bits/round — same order as the classical game
near criticality, which is what makes the comparison meaningful.

## Analysis Path

The winning-option sequence is already binary, so KS entropy estimation is
identical to the classical Minority Game: block counting via
`empirical_block_distribution` / `shannon_entropy` from
`basic/ergodic_systems`, no symbolization step. `analysis/ks_entropy.py`
wraps this for saved runs, so analysis is fully decoupled from simulation —
you can re-analyze old transcripts without re-running (or re-paying for)
anything.

## Caveats (from the tool report — they apply here)

- **Stochasticity**: LLM outputs are stochastic; individual trajectories are
  not reproducible. Population-level statistics are the reliable quantities —
  hence multiple seeds per window in the sweep, with cross-seed variance
  treated as meaningful.
- **Prompt sensitivity**: agent behavior is fully determined by the prompt;
  small wording changes are a form of measurement uncertainty. Prompts live in
  one file (`sim/prompts.py`) and are logged verbatim in transcripts so every
  run is auditable against the exact wording that produced it.
- **No rationality ground truth**: LLM agents approximate strategic reasoning;
  they are not game-theoretically rational. The classical MG results are the
  calibrated baseline the LLM society gets compared to.
- **Cost scales as N × steps × seeds**: every agent-round is an API call.
  This is why the mock-first workflow exists, and why `estimate_tokens.py`
  should precede every live run.

## Roadmap

1. **Now (free)**: mock sweeps to finalize experiment design, plots, and
   statistics.
2. **First live run (small)**: N=9, 30 rounds, one seed, cheapest adequate
   model — a sanity check that LLM agents produce a nontrivial outcome
   sequence, plus a check of parse-failure rates.
3. **The real sweep**: w ∈ {0..8}, several seeds, N and steps chosen from the
   token estimate. Money plot: entropy rate vs efficiency, colored by w,
   side-by-side with the classical `sweep_entropy.png`.
4. **Extensions**: `persona_diversity=False` (herding limit); temperature as a
   second entropy knob; semantic memory (CAMEL `VectorDBMemory`) vs recency
   memory — does *what kind* of memory matter, or only how much?; an async
   round loop for cheap parallel live runs.
