"""Ergodic analysis of Polis society records.

Four views of the society's dynamics, mirroring (and extending) the
sugarscape analyses:

1. trajectory_entropy — block entropy of individual fulfillment trajectories,
   poolable over any subset of agents (equivalence classes).
2. macro_entropy — block entropy of the whole fulfillment DISTRIBUTION's
   season-to-season evolution (the macro state).
3. mobility — quantile-rank transition matrix and its Markov entropy rate:
   social mobility expressed literally as an entropy rate (a "meritocracy
   score" in the project's sense).
4. partitions — equivalence classes of the population (class of origin,
   faction, dominant value, temperament) for per-subpopulation entropy.

Welfare metrics (mean fulfillment, Gini, value-alignment) give the success
axis to plot entropy against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "basic" / "ergodic_systems"))

from entropy.block_counting import symbolize_timeseries, shannon_entropy, \
    empirical_block_distribution  # noqa: E402


def load_record(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------- #
# 1. Individual trajectory entropy (poolable over equivalence classes)
# --------------------------------------------------------------------- #

def pooled_block_entropy(seqs: list[np.ndarray], k: int) -> float:
    """Shannon entropy of length-k blocks with counts pooled across sequences."""
    counts: dict[tuple, int] = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            block = tuple(seq[i:i + k])
            counts[block] = counts.get(block, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return shannon_entropy({b: c / total for b, c in counts.items()})


def trajectory_entropy(F: np.ndarray, agent_ids=None, n_bins: int = 5,
                       k_max: int = 4, burn_in: int = 0):
    """Pooled block entropy of fulfillment trajectories F (T, N).

    Bin edges are computed globally over the selected agents so all
    trajectories share one alphabet. Few bins and modest k on purpose: LLM
    runs are short, and 5^4 blocks is already ambitious for T ~ 100.

    Returns (ks, H_k, h_rate, h_diff).
    """
    F = np.asarray(F)[burn_in:]
    if agent_ids is None:
        agent_ids = list(range(F.shape[1]))
    values = F[:, agent_ids]
    _, edges = symbolize_timeseries(values.ravel(), n_bins=n_bins,
                                    method="quantile")
    seqs = [symbolize_timeseries(values[:, j], bin_edges=edges)[0]
            for j in range(values.shape[1])]

    ks = list(range(1, k_max + 1))
    H_k = [pooled_block_entropy(seqs, k) for k in ks]
    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]
    return ks, H_k, h_rate, h_diff


# --------------------------------------------------------------------- #
# 2. Macro (distribution-state) entropy
# --------------------------------------------------------------------- #

def macro_entropy(F: np.ndarray, n_bins: int = 5, k_max: int = 4,
                  burn_in: int = 0):
    """Block entropy of the fulfillment distribution's evolution.

    Each season is symbolized as the histogram tuple of the population's
    fulfillment over globally fixed bins (sugarscape Approach 1 analog).
    """
    F = np.asarray(F)[burn_in:]
    _, edges = symbolize_timeseries(F.ravel(), n_bins=n_bins, method="quantile")
    symbols = [tuple(np.histogram(F[t], bins=edges)[0]) for t in range(len(F))]

    ks = list(range(1, k_max + 1))
    H_k = [shannon_entropy(empirical_block_distribution(symbols, k)) for k in ks]
    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]
    return ks, H_k, h_rate, h_diff


# --------------------------------------------------------------------- #
# 3. Mobility: rank-transition matrix and its entropy rate
# --------------------------------------------------------------------- #

def rank_classes(F: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """(T, N) array of per-season quantile rank classes (0 = poorest)."""
    F = np.asarray(F)
    T, N = F.shape
    classes = np.empty((T, N), dtype=int)
    for t in range(T):
        ranks = np.argsort(np.argsort(F[t]))
        classes[t] = (ranks * n_classes) // N
    return classes


def mobility(F: np.ndarray, n_classes: int = 5, burn_in: int = 0):
    """Empirical rank-transition matrix P and its Markov entropy rate.

    h = Σ_i π_i Σ_j −P_ij log2 P_ij  (bits/season), with π the empirical
    occupancy. h = 0 means a frozen hierarchy (perfect memory of station);
    h = log2(n_classes) means station is reshuffled every season (no memory).
    This is social mobility measured literally as an entropy rate.
    """
    classes = rank_classes(np.asarray(F)[burn_in:], n_classes)
    T = len(classes)
    counts = np.zeros((n_classes, n_classes))
    for t in range(T - 1):
        for i_from, i_to in zip(classes[t], classes[t + 1]):
            counts[i_from, i_to] += 1

    row_sums = counts.sum(axis=1)
    pi = row_sums / row_sums.sum()
    h = 0.0
    P = np.zeros_like(counts)
    for i in range(n_classes):
        if row_sums[i] == 0:
            continue
        P[i] = counts[i] / row_sums[i]
        row_h = -sum(p * np.log2(p) for p in P[i] if p > 0)
        h += pi[i] * row_h
    return P, float(h)


# --------------------------------------------------------------------- #
# 4. Equivalence-class partitions of the population
# --------------------------------------------------------------------- #

def partitions(identities: list[dict]) -> dict[str, dict[str, list[int]]]:
    """Sub-populations by class of origin, faction, dominant value, ambition."""
    parts: dict[str, dict[str, list[int]]] = {
        "social_class": {}, "faction": {}, "dominant_value": {}, "ambition": {},
    }
    for ident in identities:
        i = ident["agent_id"]
        parts["social_class"].setdefault(ident["social_class"], []).append(i)
        parts["faction"].setdefault(ident["faction"], []).append(i)
        parts["dominant_value"].setdefault(ident["dominant_value"], []).append(i)
        amb = "ambitious" if ident["temperament"]["ambition"] > 0 else "content"
        parts["ambition"].setdefault(amb, []).append(i)
    return parts


# --------------------------------------------------------------------- #
# Welfare / success metrics
# --------------------------------------------------------------------- #

def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def welfare_metrics(record: dict, window: int = 20) -> dict:
    """Success metrics over the final `window` seasons.

    - welfare: mean fulfillment (people succeeding by their OWN values)
    - fulfillment_gini: inequality of fulfillment
    - value_alignment: mean of (F_i − unweighted mean of i's dimensions).
      Positive = the society lets people do well at what THEY care about,
      beyond generic across-the-board success.
    """
    F = np.asarray(record["fulfillment"])[-window:]
    dims = np.asarray(record["dims"])[-window:]        # (w, N, 4)
    unweighted = dims.mean(axis=2)                     # (w, N)
    return {
        "welfare": float(F.mean()),
        "fulfillment_gini": float(gini(F.mean(axis=0))),
        "value_alignment": float((F - unweighted).mean()),
    }
