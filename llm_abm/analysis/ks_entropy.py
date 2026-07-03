"""KS entropy estimation for LLM Minority Game outcome sequences.

The winning-option sequence is already a binary symbol sequence, so block
counting applies directly with no symbolization step — exactly as in the
classical Minority Game analysis. Reuses the block-counting implementation
from basic/ergodic_systems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow importing the shared entropy toolkit
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "basic" / "ergodic_systems"))

from entropy.block_counting import empirical_block_distribution, shannon_entropy  # noqa: E402


def block_entropy_analysis(outcomes, k_max: int = 10, burn_in: int = 0):
    """Block entropy estimates from a binary outcome sequence.

    Returns (ks, H_k, h_rate, h_diff) where:
        ks:     [1, ..., k_max]
        H_k:    block entropy H(k) for each k
        h_rate: H(k)/k for each k
        h_diff: conditional entropy H(k) - H(k-1), with H(0) = 0
    """
    seq = np.asarray(outcomes)[burn_in:]
    ks = list(range(1, k_max + 1))
    H_k = []
    for k in ks:
        dist = empirical_block_distribution(seq, k)
        H_k.append(shannon_entropy(dist))
    h_rate = [H_k[i] / ks[i] for i in range(len(ks))]
    h_diff = [H_k[0]] + [H_k[i] - H_k[i - 1] for i in range(1, len(ks))]
    return ks, H_k, h_rate, h_diff


def entropy_rate(outcomes, k_max: int = 10, burn_in: int = 0) -> float:
    """Convenience: the converged conditional entropy h(k_max) in bits/round."""
    _, _, _, h_diff = block_entropy_analysis(outcomes, k_max=k_max, burn_in=burn_in)
    return float(h_diff[-1])
