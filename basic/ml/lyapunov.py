"""Lyapunov exponent computation for neural network training dynamics.

Uses modified Benettin's algorithm with Hessian-vector products to compute
the top-k Lyapunov exponents of the training trajectory without forming
the full Hessian.
"""

import torch
import numpy as np


def hessian_vector_product(loss_fn, params, v):
    """Compute Hessian-vector product H @ v without forming H.

    Uses the double-backward trick: H @ v = d/dp [grad(L) . v]

    Parameters
    ----------
    loss_fn : callable
        params -> scalar loss. Must be differentiable twice.
    params : list of torch.Tensor
        Model parameters (requires_grad=True).
    v : list of torch.Tensor
        Tangent vector, same structure as params.

    Returns
    -------
    hvp : list of torch.Tensor
        Hessian-vector product, same structure as params.
    """
    # First backward: get gradients
    grads = torch.autograd.grad(loss_fn(), params, create_graph=True)

    # Dot product of gradients with v
    gv = sum((g * vi).sum() for g, vi in zip(grads, v))

    # Second backward: differentiate the dot product
    hvp = torch.autograd.grad(gv, params)

    return [h.detach() for h in hvp]


class LyapunovTracker:
    """Track top-k Lyapunov exponents during training via Benettin's algorithm.

    The Jacobian of one GD step is J = I - lr * H, where H is the Hessian.
    Propagating a tangent vector v through one step: v <- v - lr * (H @ v).
    """

    def __init__(self, model, k=10, reorth_interval=1):
        """
        Parameters
        ----------
        model : MLP
            The neural network (used to get parameter shapes).
        k : int
            Number of Lyapunov exponents to track.
        reorth_interval : int
            Steps between QR reorthonormalization.
        """
        self.k = k
        self.reorth_interval = reorth_interval
        self.n_params = model.param_count()
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]

        # Initialize tangent vectors (d x k) and orthonormalize
        Q = torch.randn(self.n_params, k)
        Q, _ = torch.linalg.qr(Q)
        self.tangent_vectors = Q

        # Accumulators
        self.log_R_diag = torch.zeros(k)
        self.step_count = 0
        self.reorth_count = 0

        # History for windowed estimates
        self.log_R_history = []  # list of (step, log|R_ii|) tuples

    def _flat_to_param_list(self, flat):
        """Split flat vector into list of tensors matching parameter shapes."""
        parts = []
        offset = 0
        for shape, size in zip(self.param_shapes, self.param_sizes):
            parts.append(flat[offset:offset + size].reshape(shape))
            offset += size
        return parts

    def _param_list_to_flat(self, param_list):
        """Concatenate parameter list into flat vector."""
        return torch.cat([p.reshape(-1) for p in param_list])

    def step(self, loss_fn, model, lr):
        """Propagate tangent vectors through one training step.

        Parameters
        ----------
        loss_fn : callable
            () -> scalar loss (must be differentiable w.r.t. model params).
        model : MLP
            Current model (parameters must have requires_grad=True).
        lr : float
            Learning rate.
        """
        params = list(model.parameters())

        # Propagate each tangent vector: v_i <- v_i - lr * (H @ v_i)
        new_tangents = []
        for i in range(self.k):
            vi_flat = self.tangent_vectors[:, i]
            vi_params = self._flat_to_param_list(vi_flat)

            hvp = hessian_vector_product(loss_fn, params, vi_params)
            hvp_flat = self._param_list_to_flat(hvp)

            new_tangents.append(vi_flat - lr * hvp_flat)

        self.tangent_vectors = torch.stack(new_tangents, dim=1)
        self.step_count += 1

        # QR reorthonormalization
        if self.step_count % self.reorth_interval == 0:
            Q, R = torch.linalg.qr(self.tangent_vectors)
            log_r_diag = torch.log(torch.abs(torch.diag(R)).clamp(min=1e-30))
            self.log_R_diag += log_r_diag
            self.log_R_history.append((self.step_count, log_r_diag.clone()))
            self.tangent_vectors = Q
            self.reorth_count += 1

    def get_exponents(self):
        """Return current Lyapunov exponent estimates.

        Returns
        -------
        exponents : np.ndarray
            Array of k Lyapunov exponents (nats per step).
        """
        if self.step_count == 0:
            return np.zeros(self.k)
        return (self.log_R_diag / self.step_count).numpy()

    def get_ks_entropy(self):
        """Return KS entropy estimate (sum of positive exponents)."""
        exponents = self.get_exponents()
        return float(np.sum(exponents[exponents > 0]))

    def get_windowed_exponents(self, window_size=50):
        """Compute Lyapunov exponents in sliding windows.

        Parameters
        ----------
        window_size : int
            Number of reorthonormalization events per window.

        Returns
        -------
        steps : np.ndarray
            Step numbers at window centers.
        exponents : np.ndarray
            Shape (n_windows, k) array of exponent estimates.
        ks_entropy : np.ndarray
            Shape (n_windows,) KS entropy at each window.
        """
        if len(self.log_R_history) < window_size:
            return np.array([]), np.array([]).reshape(0, self.k), np.array([])

        n_windows = len(self.log_R_history) - window_size + 1
        steps = np.zeros(n_windows)
        exponents = np.zeros((n_windows, self.k))

        for w in range(n_windows):
            window = self.log_R_history[w:w + window_size]
            step_start = window[0][0]
            step_end = window[-1][0]
            steps[w] = (step_start + step_end) / 2

            log_sum = torch.stack([lr for _, lr in window]).sum(dim=0)
            n_steps = step_end - step_start + self.reorth_interval
            exponents[w] = (log_sum / n_steps).numpy()

        ks_entropy = np.sum(np.maximum(exponents, 0), axis=1)

        return steps, exponents, ks_entropy
