"""Outcome metrics for the Sugarscape model."""

import numpy as np


def gini(wealth_array: np.ndarray) -> float:
    """Standard Gini coefficient for a wealth distribution."""
    arr = np.asarray(wealth_array, dtype=float)
    arr = arr[arr >= 0]  # ignore negative (shouldn't happen but guard anyway)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(arr)
    return (2 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * arr.sum()) / (
        n * arr.sum()
    )
