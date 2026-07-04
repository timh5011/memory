"""Metrics for the LLM Minority Game.

Identical definitions to basic/agent_based_models/minority_game/sim/metrics.py
(duplicated here so llm_abm/minority_game stays a self-contained package), so results
are directly comparable with the classical Minority Game sweep.
"""

import numpy as np


def volatility(attendance_series: np.ndarray, N: int) -> float:
    """σ²/N — the standard metric from Challet & Zhang."""
    return float(np.var(attendance_series) / N)


def efficiency(attendance_series: np.ndarray, N: int) -> float:
    """1 - σ²/(N/4) — normalized so 0.0 = random coin flip, 1.0 = perfect coordination."""
    sigma2 = np.var(attendance_series)
    return float(1.0 - sigma2 / (N / 4))


def predictability(attendance_series: np.ndarray, N: int) -> float:
    """⟨A²⟩/N - N/4 — H > 0 means the game is exploitable."""
    return float(np.mean(attendance_series ** 2) / N - N / 4)
