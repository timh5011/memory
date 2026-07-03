"""GradientDescentSystem: neural network training as an ErgodicSystem.

Wraps full-batch gradient descent on the spiral-classification MLP from
basic/ml as a subclass of the ErgodicSystem ABC from basic/ergodic_systems.
This makes training dynamics a first-class citizen of the ergodic framework:

- state:  the flattened weight vector theta in R^d
- map T:  one gradient descent step, theta -> theta - lr * grad L(theta)
- mu:     the initialization distribution (PyTorch default init) stands in
          for an invariant measure — see the ergodicity caveat in README.md
- symbolize: a scalar observable of the weights (training loss by default),
          quantile-binned into a finite alphabet

Because metric() and perturb() are implemented, the generic Benettin
perturbation estimator (entropy/lyapunov.py: lyapunov_perturbation) works on
this system directly — an independent cross-check of the Hessian-vector
Pesin estimates in basic/ml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "basic" / "ml"))
sys.path.insert(0, str(_ROOT / "basic" / "ergodic_systems"))

from data import make_spirals                          # noqa: E402  (basic/ml)
from model import MLP                                  # noqa: E402  (basic/ml)
from systems.ergodic_system import ErgodicSystem       # noqa: E402
from entropy.block_counting import symbolize_timeseries  # noqa: E402


class GradientDescentSystem(ErgodicSystem):
    """Full-batch gradient descent on a fixed dataset as a deterministic map.

    The map is autonomous (same dataset every step), so given an initial
    weight vector the trajectory is fully deterministic — all randomness
    lives in sample_initial_state().
    """

    def __init__(self, lr: float = 0.05, hidden_dims=None, n_classes: int = 3,
                 noise: float = 0.5, data_seed: int = 42, n_bins: int = 8,
                 observable: str = "loss"):
        self.lr = lr
        self.n_bins = n_bins
        self.observable = observable

        X_train, y_train, X_test, y_test = make_spirals(
            n_classes=n_classes, noise=noise, seed=data_seed
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self._net = MLP(input_dim=2, hidden_dims=hidden_dims,
                        output_dim=n_classes)
        super().__init__(dimension=self._net.param_count(), is_symbolic=False)

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #

    def _loss_and_grad(self, state: np.ndarray):
        """Loss value and flattened gradient at a weight vector."""
        flat = torch.tensor(state, dtype=torch.float32)
        self._net.load_flat_params(flat)
        self._net.zero_grad()
        loss = F.nll_loss(self._net(self.X_train), self.y_train)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in self._net.parameters()])
        return loss.item(), grad.numpy()

    def iterate(self, state):
        """One gradient descent step."""
        state = np.asarray(state, dtype=np.float64)
        _, grad = self._loss_and_grad(state)
        return state - self.lr * grad

    def sample_initial_state(self, seed=None):
        """Fresh PyTorch-default initialization, flattened."""
        if seed is not None:
            torch.manual_seed(seed)
        linear_dims = [m.out_features for m in self._net.net
                       if isinstance(m, torch.nn.Linear)]
        net = MLP(input_dim=self.X_train.shape[1],
                  hidden_dims=linear_dims[:-1],
                  output_dim=linear_dims[-1])
        return net.flat_params().numpy().astype(np.float64)

    def generate_trajectory(self, initial_state=None, n_steps=1000, seed=None):
        """Weight-space trajectory, shape (n_steps, d), float32.

        Truncates with a warning if the loss diverges (large lr).
        """
        if initial_state is None:
            initial_state = self.sample_initial_state(seed=seed)
        state = np.asarray(initial_state, dtype=np.float64)

        traj = np.empty((n_steps, self.dimension), dtype=np.float32)
        for t in range(n_steps):
            traj[t] = state
            state = self.iterate(state)
            if not np.all(np.isfinite(state)):
                print(f"  [warn] diverged at step {t + 1} (lr={self.lr}); "
                      f"truncating trajectory")
                return traj[: t + 1]
        return traj

    # ------------------------------------------------------------------ #
    # Observables and symbolization
    # ------------------------------------------------------------------ #

    def loss_series(self, trajectory) -> np.ndarray:
        """Training loss at every state along a trajectory."""
        losses = np.empty(len(trajectory))
        with torch.no_grad():
            for t, state in enumerate(trajectory):
                self._net.load_flat_params(torch.tensor(state, dtype=torch.float32))
                losses[t] = F.nll_loss(self._net(self.X_train), self.y_train).item()
        return losses

    def weight_norm_series(self, trajectory) -> np.ndarray:
        """L2 norm of the weight vector along a trajectory."""
        return np.linalg.norm(np.asarray(trajectory, dtype=np.float64), axis=1)

    def test_accuracy_series(self, trajectory) -> np.ndarray:
        """Test accuracy at every state along a trajectory."""
        accs = np.empty(len(trajectory))
        with torch.no_grad():
            for t, state in enumerate(trajectory):
                self._net.load_flat_params(torch.tensor(state, dtype=torch.float32))
                pred = self._net(self.X_test).argmax(dim=1)
                accs[t] = (pred == self.y_test).float().mean().item()
        return accs

    def observable_series(self, trajectory) -> np.ndarray:
        if self.observable == "loss":
            return self.loss_series(trajectory)
        if self.observable == "weight_norm":
            return self.weight_norm_series(trajectory)
        if self.observable == "test_accuracy":
            return self.test_accuracy_series(trajectory)
        raise ValueError(f"Unknown observable: {self.observable!r}")

    def symbolize(self, trajectory, partition=None):
        """Observable series -> quantile-binned integer symbols.

        The partition is over the observable's range, not weight space
        directly — a coarse-graining, so the resulting entropy rate is a
        lower bound on the entropy rate of the full dynamics (h(T,P) <= h_KS).
        """
        series = self.observable_series(trajectory)
        symbols, _ = symbolize_timeseries(series, n_bins=self.n_bins,
                                          method="quantile")
        return symbols

    # ------------------------------------------------------------------ #
    # Lyapunov perturbation interface
    # ------------------------------------------------------------------ #

    def metric(self, state_a, state_b):
        """Euclidean distance in weight space."""
        return float(np.linalg.norm(np.asarray(state_a) - np.asarray(state_b)))

    def perturb(self, state, delta, rng):
        """Nearby state at distance delta in a random direction."""
        direction = rng.standard_normal(self.dimension)
        direction /= np.linalg.norm(direction)
        return np.asarray(state, dtype=np.float64) + delta * direction
