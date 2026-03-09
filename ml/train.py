"""Training loop with integrated Lyapunov exponent tracking."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from data import make_spirals
from model import MLP
from lyapunov import LyapunovTracker


@dataclass
class TrainingConfig:
    lr: float = 0.05
    n_steps: int = 1000
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    n_classes: int = 3
    noise: float = 0.5
    seed: int = 42
    k_exponents: int = 10
    reorth_interval: int = 1
    window_size: int = 50
    convergence_threshold: float = 0.1
    eval_interval: int = 20


@dataclass
class TrainingResult:
    loss_curve: list
    test_loss_curve: list
    test_accuracy_curve: list
    lyapunov_exponents: np.ndarray  # final exponents (k,)
    lyapunov_steps: np.ndarray  # windowed step positions
    lyapunov_timeseries: np.ndarray  # windowed exponents (n_windows, k)
    ks_entropy_timeseries: np.ndarray  # windowed KS entropy (n_windows,)
    ks_entropy_final: float
    convergence_step: Optional[int]
    final_train_loss: float
    final_test_loss: float
    final_test_accuracy: float
    config: TrainingConfig


def run_training(config: TrainingConfig) -> TrainingResult:
    """Run a single training experiment with Lyapunov tracking.

    Parameters
    ----------
    config : TrainingConfig
        All hyperparameters.

    Returns
    -------
    TrainingResult
        Complete results including loss curves, Lyapunov exponents, etc.
    """
    torch.manual_seed(config.seed)

    # Data
    X_train, y_train, X_test, y_test = make_spirals(
        n_classes=config.n_classes, noise=config.noise, seed=config.seed
    )

    # Model
    model = MLP(input_dim=2, hidden_dims=config.hidden_dims,
                output_dim=config.n_classes)

    # Lyapunov tracker
    tracker = LyapunovTracker(model, k=config.k_exponents,
                              reorth_interval=config.reorth_interval)

    # Training state
    loss_curve = []
    test_loss_curve = []
    test_accuracy_curve = []
    convergence_step = None

    for step in range(config.n_steps):
        # Define loss function (closure for HVP)
        def loss_fn():
            return F.nll_loss(model(X_train), y_train)

        # Compute loss and gradients for the weight update
        loss = loss_fn()
        loss_val = loss.item()
        loss_curve.append(loss_val)

        # Check convergence
        if convergence_step is None and loss_val < config.convergence_threshold:
            convergence_step = step

        # Compute gradients
        loss.backward()

        # Propagate tangent vectors (before weight update, using current Hessian)
        tracker.step(loss_fn, model, config.lr)

        # Manual SGD update
        with torch.no_grad():
            for p in model.parameters():
                p -= config.lr * p.grad
                p.grad.zero_()

        # Periodic evaluation
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            with torch.no_grad():
                test_out = model(X_test)
                test_loss = F.nll_loss(test_out, y_test).item()
                test_acc = (test_out.argmax(dim=1) == y_test).float().mean().item()
                test_loss_curve.append(test_loss)
                test_accuracy_curve.append(test_acc)

    # Final metrics
    with torch.no_grad():
        final_train_loss = F.nll_loss(model(X_train), y_train).item()
        test_out = model(X_test)
        final_test_loss = F.nll_loss(test_out, y_test).item()
        final_test_accuracy = (test_out.argmax(dim=1) == y_test).float().mean().item()

    # Lyapunov results
    lyap_exponents = tracker.get_exponents()
    ks_entropy = tracker.get_ks_entropy()
    lyap_steps, lyap_ts, ks_ts = tracker.get_windowed_exponents(config.window_size)

    return TrainingResult(
        loss_curve=loss_curve,
        test_loss_curve=test_loss_curve,
        test_accuracy_curve=test_accuracy_curve,
        lyapunov_exponents=lyap_exponents,
        lyapunov_steps=lyap_steps,
        lyapunov_timeseries=lyap_ts,
        ks_entropy_timeseries=ks_ts,
        ks_entropy_final=ks_entropy,
        convergence_step=convergence_step,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        final_test_accuracy=final_test_accuracy,
        config=config,
    )


if __name__ == "__main__":
    config = TrainingConfig(lr=0.05, n_steps=100, k_exponents=5,
                            reorth_interval=1, window_size=20)
    print(f"Running training: {config.n_steps} steps, lr={config.lr}, "
          f"k={config.k_exponents}")

    result = run_training(config)

    print(f"\nFinal train loss: {result.final_train_loss:.4f}")
    print(f"Final test loss:  {result.final_test_loss:.4f}")
    print(f"Final test acc:   {result.final_test_accuracy:.4f}")
    print(f"Convergence step: {result.convergence_step}")
    print(f"KS entropy:       {result.ks_entropy_final:.6f}")
    print(f"Top Lyapunov exponents: {result.lyapunov_exponents}")
