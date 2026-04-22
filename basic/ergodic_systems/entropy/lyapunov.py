"""Lyapunov exponent estimation for ergodic dynamical systems.

Two methods:
  1. Perturbation (Benettin's algorithm): perturb state, track divergence,
     renormalize periodically. Works for any system with metric() and perturb().
  2. Jacobian: average ln|J| along a trajectory. Works for any system with
     jacobian(). More efficient when available.

Both estimate the largest Lyapunov exponent, which by Pesin's identity equals
the KS entropy for 1D maps with smooth invariant measures.
"""

import numpy as np
import matplotlib.pyplot as plt


def lyapunov_perturbation(system, n_steps=100_000, delta=1e-8,
                          renorm_interval=1, seed=42):
    """Estimate the largest Lyapunov exponent via perturbation and renormalization.

    Benettin's algorithm:
      1. Start with state x and perturbed state x' = perturb(x, delta)
      2. Iterate both for renorm_interval steps
      3. Measure d = metric(x, x')
      4. Accumulate ln(d / delta)
      5. Renormalize x' back to distance delta from x
      6. Repeat

    Parameters
    ----------
    system : ErgodicSystem
        Must implement iterate(), metric(), perturb(), sample_initial_state().
    n_steps : int
        Total iteration steps.
    delta : float
        Perturbation magnitude.
    renorm_interval : int
        Steps between renormalizations.
    seed : int
        Random seed.

    Returns
    -------
    lyap_nats : float
        Lyapunov exponent in nats/step.
    lyap_bits : float
        Lyapunov exponent in bits/step.
    running_estimate : np.ndarray
        Running average of λ after each renormalization.
    """
    rng = np.random.default_rng(seed)

    x = system.sample_initial_state(seed=int(rng.integers(0, 2**31)))
    x_prime = system.perturb(x, delta, rng)

    log_stretches = []
    n_renorms = n_steps // renorm_interval

    for _ in range(n_renorms):
        # Iterate both states
        for _ in range(renorm_interval):
            x = system.iterate(x)
            x_prime = system.iterate(x_prime)

        # Measure divergence
        d = system.metric(x, x_prime)
        if d == 0:
            # Trajectories collapsed — re-perturb
            x_prime = system.perturb(x, delta, rng)
            continue

        log_stretches.append(np.log(d / delta))

        # Renormalize: rescale x' back to distance delta from x
        scale = delta / d
        if system.dimension == 1:
            x_prime = x + (x_prime - x) * scale
        else:
            diff = np.asarray(x_prime) - np.asarray(x)
            x_prime = np.asarray(x) + diff * scale

    log_stretches = np.array(log_stretches)
    cumsum = np.cumsum(log_stretches)
    counts = np.arange(1, len(log_stretches) + 1)
    running_estimate = cumsum / (counts * renorm_interval)

    lyap_nats = float(running_estimate[-1]) if len(running_estimate) > 0 else 0.0
    lyap_bits = lyap_nats / np.log(2)

    return lyap_nats, lyap_bits, running_estimate


def lyapunov_jacobian(system, n_steps=100_000, seed=42):
    """Estimate the largest Lyapunov exponent via the Jacobian.

    For 1D maps: λ = (1/N) Σ ln|f'(x_i)|

    Parameters
    ----------
    system : ErgodicSystem
        Must implement iterate(), jacobian(), sample_initial_state().
    n_steps : int
        Trajectory length.
    seed : int
        Random seed.

    Returns
    -------
    lyap_nats : float
        Lyapunov exponent in nats/step.
    lyap_bits : float
        Lyapunov exponent in bits/step.
    running_estimate : np.ndarray
        Running average of λ at each step.
    """
    x = system.sample_initial_state(seed=seed)

    log_derivs = np.empty(n_steps)
    for i in range(n_steps):
        j = system.jacobian(x)
        abs_j = abs(j) if system.dimension == 1 else np.abs(np.linalg.det(j))
        log_derivs[i] = np.log(abs_j) if abs_j > 0 else -np.inf
        x = system.iterate(x)

    cumsum = np.cumsum(log_derivs)
    running_estimate = cumsum / np.arange(1, n_steps + 1)

    lyap_nats = float(running_estimate[-1])
    lyap_bits = lyap_nats / np.log(2)

    return lyap_nats, lyap_bits, running_estimate


def plot_lyapunov_convergence(running_estimate, analytical_nats=None,
                              method_label="", ax=None, save_path=None):
    """Plot the running Lyapunov exponent estimate vs iteration count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.arange(1, len(running_estimate) + 1)
    running_bits = running_estimate / np.log(2)

    ax.plot(steps, running_bits, linewidth=0.8,
            label=f'{method_label} estimate' if method_label else 'Estimate')

    if analytical_nats is not None:
        analytical_bits = analytical_nats / np.log(2)
        ax.axhline(analytical_bits, linestyle='--', color='coral', alpha=0.7,
                   label=f'Analytical = {analytical_bits:.4f} bits')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('λ (bits/step)')
    ax.set_title('Lyapunov Exponent Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax
