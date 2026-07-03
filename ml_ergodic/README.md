# ML Training as an Ergodic System

This experiment treats neural network training as a **measure-preserving-ish
dynamical system and studies it with the project's ergodic toolkit directly**.
It complements `basic/ml/`, which measures the KS entropy of training via
Lyapunov exponents (Hessian-vector products + Pesin's identity). Here we take
the opposite, trajectory-level route: wrap gradient descent as an
`ErgodicSystem` and apply the same symbolic block-counting and generic
Benettin machinery that was validated on the Bernoulli shift and logistic map
in `basic/ergodic_systems/`.

## The Dynamical System

`system.py` defines `GradientDescentSystem(ErgodicSystem)`:

- **State**: the flattened weight vector θ ∈ R^d (d ≈ 4,500 for the default
  MLP from `basic/ml/model.py`).
- **Map T**: one full-batch gradient descent step,
  θ → θ − lr·∇L(θ), on the spiral dataset from `basic/ml/data.py`.
  Full-batch makes the map autonomous and deterministic: all randomness lives
  in `sample_initial_state()` (the PyTorch init distribution stands in for the
  measure μ).
- **Symbolization**: a scalar observable of the weights (training loss by
  default; weight norm and test accuracy also implemented), quantile-binned
  into an 8-symbol alphabet via the shared `symbolize_timeseries`. Because
  this partition coarse-grains weight space through a single observable, the
  resulting entropy rate is a **lower bound** on h_KS (h(T,P) ≤ h_KS — see
  `basic/ergodic_systems/THEORY.md` on generating partitions).
- **`metric()` / `perturb()`**: Euclidean distance and random-direction
  perturbation in weight space, so the generic Benettin estimator
  (`entropy/lyapunov.py: lyapunov_perturbation`) works on training dynamics
  unchanged.

## Why Both Routes?

`basic/ml` and this directory measure the same object two independent ways:

| | `basic/ml` (Pesin route) | `ml_ergodic` (symbolic route) |
|---|---|---|
| Method | Top-k Lyapunov spectrum via Hessian-vector products | Block entropy of a symbolized observable; Benettin perturbation for λ_max |
| Needs | Autodiff double-backward, tangent vectors, QR | Only the ability to run the map and observe it |
| Gives | h_KS estimate via Σλ⁺ (full spectrum) | Lower bound h(T,P); independent λ_max cross-check |
| Analog in repo | — | Exactly how the logistic map was validated: block counting and Lyapunov agreeing on 1 bit/step |

On the logistic map the two routes agree (Pesin's identity, validated in
`basic/ergodic_systems`). Whether they agree on training dynamics — and where
the coarse-grained symbolic bound sits relative to the Pesin sum — is itself
an experimental question this infrastructure can now ask.

## Scripts

```bash
cd ml_ergodic

# Entropy rate of the loss signal vs learning rate (block counting)
python scripts/run_block_entropy.py          # → results/block_entropy_vs_lr.png

# λ_max vs learning rate via generic Benettin perturbation
python scripts/run_lyapunov_perturbation.py  # → results/lyapunov_perturbation.png

# Is training ergodic? Birkhoff time-averages across seeds + correlation decay
python scripts/run_ergodicity.py             # → results/ergodicity_check.png
```

(None of these have been run yet — the system was smoke-tested with a 15-step
trajectory: deterministic, finite, loss decreasing, symbolization and
perturbation interfaces working.)

**`run_block_entropy.py`** — for each lr, generate a weight trajectory,
symbolize the post-transient loss series, compute H(k)/k and h(k). Expected:
small lr → near-deterministic monotone descent → h ≈ 0; large lr →
oscillatory/chaotic loss → h > 0. The interesting region is between, at the
learning rates that actually train well.

**`run_lyapunov_perturbation.py`** — the Benettin cross-check of
`basic/ml`'s HVP-based exponents, through a completely different numerical
route (perturb weights by δ, iterate both copies, measure divergence,
renormalize). Agreement between the two is what makes either credible.

**`run_ergodicity.py`** — the script that takes the title of this directory
seriously. `basic/ml/PLAN.md` §3 argues training is *probably not* ergodic
(multiple basins of attraction). This tests that claim empirically:

1. **Birkhoff check**: running time-averages of the loss for several random
   initializations at the same lr. Ergodicity ⇒ all seeds converge to the same
   value (time average = ensemble average). Seed-dependent limits = broken
   ergodicity. The question: does large lr (more chaotic dynamics) shrink the
   spread — i.e., does chaos restore ergodicity?
2. **Mixing proxy**: autocorrelation decay of post-transient loss
   fluctuations. Fast decay is the numerical signature of mixing — the
   "forgetting" that PHILOSOPHY.md formalizes as
   μ(T⁻ⁿA ∩ B) → μ(A)μ(B).

## Relation to the Broader Project

The project's motivating question is *how much forgetting is optimal for
progress*. In `basic/ml` the answer is sought through the KS-entropy-vs-
convergence sweep. This directory asks the prior question those results
implicitly lean on: **in what sense is training an ergodic system at all?**
If ergodicity is broken at good learning rates but restored at chaotic ones,
that's a sharp, checkable statement connecting the "edge of chaos" folklore to
the measure-theoretic framing in `doc/PHILOSOPHY.md` — and it disciplines how
much weight the entropy numbers from either route can bear.
