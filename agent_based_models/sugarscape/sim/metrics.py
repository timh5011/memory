"""Outcome metrics for the Sugarscape model."""

import numpy as np


def wasserstein_1d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Wasserstein-1 (earth mover's) distance between two 1D empirical distributions.

    For equal-sized samples, W1 = mean of |sorted_a[i] - sorted_b[i]|.
    This is the optimal transport distance on the real line.

    Chosen over alternatives:
    - L2 norm on histograms: doesn't respect the ordinal structure of wealth
      (shifting agents between adjacent bins costs the same as distant bins).
    - KL divergence: not a true metric (asymmetric, unbounded), undefined when
      one distribution has zero mass in a bin the other doesn't.

    Wasserstein-1 has a clear physical interpretation: the minimum total wealth
    that would need to be redistributed to make two economies identical.
    """
    a = np.sort(np.asarray(values_a, dtype=float))
    b = np.sort(np.asarray(values_b, dtype=float))
    if len(a) != len(b):
        raise ValueError(
            f"Both samples must have the same size, got {len(a)} and {len(b)}"
        )
    return float(np.mean(np.abs(a - b)))


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
