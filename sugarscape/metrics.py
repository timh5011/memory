"""Outcome metrics for the Sugarscape model."""

import numpy as np
from scipy.stats import spearmanr


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


def social_mobility_index(
    initial_ranks: np.ndarray, final_ranks: np.ndarray
) -> float:
    """
    Spearman rank correlation between initial and final wealth ranks.
    Returns 1.0 = perfect persistence (no mobility), 0.0 = no correlation.
    """
    if len(initial_ranks) < 3:
        return float("nan")
    corr, _ = spearmanr(initial_ranks, final_ranks)
    return float(corr)


def approximate_ks_entropy(
    trajectory: np.ndarray, embedding_dim: int = 3, tau: int = 1
) -> float:
    """
    Estimate KS entropy from a scalar time series using the K2 correlation entropy
    (Grassberger-Procaccia method). Returns K2 in nats per step, or np.nan if unreliable.
    """
    traj = np.asarray(trajectory, dtype=float)
    n = len(traj)

    # Need enough points to embed and compute statistics
    min_length = 100
    if n < min_length:
        return float("nan")

    # Normalize trajectory to [0, 1]
    traj_range = traj.max() - traj.min()
    if traj_range == 0:
        return 0.0  # constant series → perfectly deterministic → zero entropy
    traj_norm = (traj - traj.min()) / traj_range

    def embed(series: np.ndarray, dim: int, lag: int) -> np.ndarray:
        """Time-delay embedding."""
        n_pts = len(series) - (dim - 1) * lag
        return np.array(
            [series[i : i + dim * lag : lag] for i in range(n_pts)]
        )

    def correlation_integral(embedded: np.ndarray, r: float) -> float:
        """Fraction of point pairs within distance r (Chebyshev norm)."""
        n_pts = len(embedded)
        if n_pts < 2:
            return 0.0
        count = 0
        for i in range(n_pts - 1):
            diffs = np.max(np.abs(embedded[i + 1 :] - embedded[i]), axis=1)
            count += np.sum(diffs < r)
        total_pairs = n_pts * (n_pts - 1) / 2
        return count / total_pairs if total_pairs > 0 else 0.0

    # Use a range of r values between 5th and 50th percentile of pairwise distances
    embedded_m = embed(traj_norm, embedding_dim, tau)
    embedded_m1 = embed(traj_norm, embedding_dim + 1, tau)

    # Sample pairwise distances to pick r range (subsample for speed)
    sample_size = min(200, len(embedded_m))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(embedded_m), size=sample_size, replace=False)
    sample = embedded_m[idx]
    dists = []
    for i in range(len(sample) - 1):
        d = np.max(np.abs(sample[i + 1 :] - sample[i]), axis=1)
        dists.extend(d.tolist())

    if not dists:
        return float("nan")

    dists = np.array(dists)
    r_values = np.percentile(dists[dists > 0], [10, 20, 30, 40, 50]) if np.any(dists > 0) else None
    if r_values is None or len(r_values) == 0:
        return float("nan")

    k2_estimates = []
    for r in r_values:
        c_m = correlation_integral(embedded_m[:sample_size], r)
        c_m1 = correlation_integral(embedded_m1[:sample_size], r)
        if c_m > 0 and c_m1 > 0:
            k2 = -np.log(c_m1 / c_m)
            if np.isfinite(k2):
                k2_estimates.append(k2)

    if not k2_estimates:
        return float("nan")

    return float(np.median(k2_estimates))
