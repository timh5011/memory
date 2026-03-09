# PLAN: Measuring the KS Entropy of Neural Network Training Dynamics

## Motivation and Context

This experiment is part of a broader project studying the role of **memory** in dynamical systems. The guiding question is: *how much forgetting is optimal for a system to make progress?* Too much memory creates inertia; too little memory means lessons are lost. KS (Kolmogorov-Sinai) entropy quantifies the rate at which a dynamical system generates new information — equivalently, the rate at which it forgets its initial conditions. We aim to measure this quantity for the training dynamics of a neural network and study how it relates to learning performance.

This experiment is a standalone investigation: we set aside the social-systems framing from the broader project and instead treat a neural network's training process as a dynamical system in its own right. The question becomes: *is there an optimal rate of information generation (KS entropy) during training that leads to the best learning outcomes?*

---

## 1. Defining the Dynamical System

### State Space

The **state** of the system at time $t$ is the full parameter (weight) vector of the neural network:

$$\theta_t \in \mathbb{R}^d$$

where $d$ is the total number of trainable parameters.

### The Map

The **dynamics** are defined by the gradient descent update rule. For full-batch gradient descent:

$$\theta_{t+1} = T(\theta_t) = \theta_t - \eta \nabla L(\theta_t)$$

where $\eta$ is the learning rate and $L$ is the loss function computed over the training data.

For stochastic gradient descent (SGD), the map becomes:

$$\theta_{t+1} = T_t(\theta_t) = \theta_t - \eta \nabla L_{\mathcal{B}_t}(\theta_t)$$

where $\mathcal{B}_t$ is the mini-batch at step $t$. Note that in the SGD case, the map is **non-autonomous** — it changes at each step because the mini-batch changes. This is an important theoretical consideration (see Section 3).

Each training step corresponds to one iteration of the map. This gives us a discrete-time dynamical system on the parameter space $\mathbb{R}^d$.

---

## 2. What We Are Measuring: KS Entropy via Lyapunov Exponents

### KS Entropy and Pesin's Formula

The KS entropy of a dynamical system measures the rate at which it generates Shannon information over time. For a smooth system, Pesin's entropy formula relates KS entropy to the positive Lyapunov exponents:

$$h_\mu(T) = \sum_{\lambda_i > 0} \lambda_i$$

where $\lambda_i$ are the Lyapunov exponents of the system. We will compute the positive Lyapunov exponents of the training trajectory and sum them to obtain an estimate of KS entropy.

### Lyapunov Exponents of the Training Trajectory

The Lyapunov exponents measure how quickly nearby trajectories in weight space diverge or converge. For our system, the Jacobian of the map at step $t$ is:

$$J_t = \frac{\partial T(\theta_t)}{\partial \theta} = I - \eta H_t$$

where $H_t = \nabla^2 L(\theta_t)$ is the Hessian of the loss at the current weights.

A Lyapunov exponent $\lambda_i$ tells us the average exponential rate at which perturbations in the $i$-th direction grow or shrink over time. Positive exponents correspond to directions in which nearby weight configurations diverge (the system is "forgetting" — chaotic behavior). Negative exponents correspond to convergent directions (the system is "remembering" — stable behavior).

### Computational Method: Benettin's Algorithm (Modified)

We do **not** need the full Lyapunov spectrum. We only need the **positive** exponents to compute KS entropy via Pesin's formula. We will compute the top $k$ exponents using the following algorithm:

1. Initialize $k$ orthonormal tangent vectors $\{v_1, ..., v_k\}$ in $\mathbb{R}^d$ (randomly initialized, then orthonormalized).
2. At each training step $t$:
   - a. Update the weights normally: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$
   - b. Propagate each tangent vector forward: $v_i \leftarrow J_t \cdot v_i = v_i - \eta (H_t \cdot v_i)$
   - c. **Crucially**, compute $H_t \cdot v_i$ using **Hessian-vector products** (PyTorch's `torch.autograd.functional.hvp` or equivalent). This avoids ever forming or storing the full $d \times d$ Hessian matrix.
3. Every $R$ steps (the reorthonormalization interval, e.g., $R = 1$ or $R = 5$):
   - a. Perform QR decomposition on the matrix of tangent vectors: $[v_1 | ... | v_k] = Q \cdot R$
   - b. Accumulate the running Lyapunov exponent estimates: $\Lambda_i \leftarrow \Lambda_i + \log |R_{ii}|$
   - c. Replace the tangent vectors with the orthonormalized columns of $Q$.
4. After $N$ total training steps, the Lyapunov exponents are: $\lambda_i = \Lambda_i / N$
5. The estimated KS entropy is: $h = \sum_{\lambda_i > 0} \lambda_i$

**Key implementation detail**: The Hessian-vector product $H_t \cdot v_i$ can be computed in PyTorch at roughly the cost of one additional backward pass per tangent vector. This is the critical trick that makes this tractable — we never form the full Hessian, which would be $d \times d$.

### Finite-Time Interpretation

Since we are measuring over a finite training run (not an infinite-time limit), what we obtain are **finite-time Lyapunov exponents** and a **finite-time KS entropy estimate**. This is appropriate and meaningful for our purposes — we are interested in the transient training dynamics, not long-run equilibrium behavior. The finite-time KS entropy tells us: *during this training run, how rapidly were nearby weight configurations being pulled apart?*

We can also compute these in a **windowed** fashion: compute the Lyapunov exponents over sliding windows of, say, 50-100 training steps. This gives us a time-resolved picture of how the KS entropy evolves during training (e.g., it might be high early in training and decay as the network converges).

---

## 3. Considerations Regarding Ergodicity

### Is this system ergodic?

Rigorously, **probably not** in general. The loss landscape typically has many local minima and saddle points, creating multiple basins of attraction. A training trajectory captured in one basin will never visit others — this is exactly the kind of invariant decomposition that violates ergodicity.

### Why this does not invalidate the experiment

Several considerations mitigate this concern:

1. **We use finite-time Lyapunov exponents**, which do not require ergodicity. They measure local divergence rates along the actual trajectory, regardless of global phase-space properties.

2. **SGD noise helps**. Mini-batch stochasticity acts as a thermal perturbation that can enhance mixing within and potentially between basins. Recent literature has shown that SGD trajectories can exhibit mixing-like behavior, and that even deterministic GD with large learning rates can produce ergodic and mixing dynamics.

3. **We are studying the transient**, not the equilibrium. We care about the training process (the journey through weight space), not the long-run stationary behavior (which, for a converging network, is trivially a fixed point with zero KS entropy).

4. **Path-dependence is a feature, not a bug**. Without ergodicity, our KS entropy estimates will depend on the initialization. We handle this by running multiple random initializations and studying the distribution of outcomes. If the results are consistent across initializations, that is itself informative.

### Practical implication for implementation

Start with **full-batch gradient descent** for clean, autonomous dynamics. The map $T$ is the same at every step, making the Lyapunov exponent computation cleaner. Later, we can switch to SGD to study the effect of stochasticity. For SGD, the non-autonomous nature of the map (different mini-batch each step) means the Jacobian $J_t$ genuinely varies, but the Benettin algorithm still works — it just computes Lyapunov exponents of the non-autonomous trajectory.

---

## 4. Network Architecture and Dataset

### Architecture: Multi-Layer Perceptron (MLP)

Use a simple fully-connected feedforward network:

- **Input layer**: sized to match the dataset features
- **Hidden layers**: 2-3 layers, 32-64 neurons each
- **Output layer**: sized to match the number of classes
- **Activation**: ReLU (for hidden layers), softmax or log-softmax (for output)
- **Loss function**: cross-entropy

This gives roughly **2,000-5,000 trainable parameters** — small enough for Lyapunov computation to be tractable, large enough for nontrivial training dynamics.

We use a simple MLP (no convolutions, no recurrence, no weight sharing) so that the dynamics are as clean as possible. We want to isolate the relationship between KS entropy and learning without architectural complications.

### Dataset: Synthetic 2D Classification

Start with a **synthetic dataset** — spirals, concentric circles, or similar 2D classification problems. This has several advantages:

- Very fast to train (seconds per run)
- We can control the difficulty of the task (e.g., number of spirals, noise level)
- The low input dimension keeps the network small
- Easy to visualize the decision boundary for sanity checks

Suggested starting configuration:
- **Spiral dataset**: 2 or 3 interleaved spirals with moderate noise
- **Training set**: ~1000 points
- **Test set**: ~500 points

Later, we can scale up to MNIST or Fashion-MNIST if the method works.

### Computational Budget

Target: the entire experiment (all learning rates, all initializations) should run on a MacBook Air within approximately **30 minutes**. This constrains our choices:

- Network: ~2,000-5,000 parameters
- Number of tangent vectors $k$: start with 10-20 (we only need enough to capture the positive exponents)
- Training steps per run: ~500-2000 (sufficient for a small network on a toy problem)
- Number of learning rates to sweep: ~15-25 values
- Number of random initializations per learning rate: ~5-10
- Reorthonormalization interval $R$: every 1-5 steps

---

## 5. Experimental Design

### Control Variable: Learning Rate

The learning rate $\eta$ is the primary control variable. It directly scales the map $T$ and governs the magnitude of each weight update. Larger $\eta$ leads to bigger steps, more chaotic dynamics, and (we hypothesize) higher KS entropy. The learning rate sweep should cover:

- **Very small** $\eta$: nearly gradient flow, very stable, low KS entropy, slow convergence
- **Moderate** $\eta$: the "sweet spot" where training works well
- **Large** $\eta$: chaotic or divergent dynamics, high KS entropy, potentially fast early progress but poor convergence
- **Very large** $\eta$: divergent, the network fails to learn

Suggested sweep: logarithmic spacing from $10^{-4}$ to $10^{0}$ (or until divergence), e.g., 20 values.

### Future Control Variables (not implemented in v1, but design for extensibility)

- **Batch size**: Controls stochasticity. Smaller batches → more noise → potentially more mixing.
- **Momentum**: Adds explicit memory to the update rule ($\theta_{t+1}$ depends on $\theta_t$ and $\theta_{t-1}$). Directly relevant to the "memory" theme.
- **Weight decay**: Regularization that pulls weights toward zero. Affects the geometry of the trajectory and is known to push networks toward the edge of chaos.

The code should be structured so that these can be added as additional sweep parameters later without refactoring.

### Metrics to Record (for each training run)

1. **Convergence rate**: Number of training steps to reach a threshold training loss (e.g., 95% of optimal, or a fixed loss value). This is the **primary** metric. It directly measures how the system's "memory" affects its speed of learning.

2. **Final training loss**: The loss value at the end of training. Confirms the network learned (or didn't).

3. **Final test loss / test accuracy**: Measures generalization. This is a **secondary** metric — it tells us whether KS entropy during training predicts the quality of the solution the network finds (e.g., flat vs sharp minima).

4. **Lyapunov exponent time series**: The top $k$ Lyapunov exponents computed in a windowed fashion over the course of training. This gives us a time-resolved view (not just a single number per run).

5. **KS entropy time series**: Sum of positive Lyapunov exponents at each window.

6. **Training loss curve**: Full loss-vs-step trajectory for visualization.

---

## 6. Analysis Plan

### Primary Analysis: KS Entropy vs. Convergence Rate

- For each learning rate, compute the average KS entropy over the training run (or over the first N steps) and the convergence rate.
- Plot KS entropy vs. convergence rate (scatter plot across learning rates).
- **Hypothesis**: There is an optimal KS entropy that minimizes convergence time. Too low → slow exploration. Too high → chaotic, fails to converge.

### Secondary Analysis: KS Entropy vs. Final Loss

- Plot KS entropy vs. final training loss and vs. final test loss.
- **Hypothesis**: There may be a "sweet spot" of KS entropy that produces the best generalization (lowest test loss), potentially different from the point of fastest convergence.

### Time-Resolved Analysis: How KS Entropy Evolves During Training

- Plot the KS entropy time series for different learning rates.
- **Expected behavior**: KS entropy may be high early (exploration phase) and decrease as the network converges to a minimum (exploitation phase). The rate of this decrease likely depends on the learning rate.

### Robustness Across Initializations

- For each learning rate, run multiple random initializations.
- Report mean and standard deviation of all metrics.
- This addresses the lack of ergodicity: if results are consistent across initializations, the findings are robust despite path-dependence.

---

## 7. Code Structure

Implement in **Python** using **PyTorch**. Organize as follows:

```
project/
├── README.md
├── requirements.txt          # torch, numpy, matplotlib, scipy
├── data.py                   # Synthetic dataset generation (spirals, circles, etc.)
├── model.py                  # MLP definition
├── lyapunov.py               # Core Lyapunov exponent / KS entropy computation
│                             #   - Hessian-vector product wrapper
│                             #   - Benettin algorithm (tangent vector propagation + QR)
│                             #   - Windowed Lyapunov exponent computation
│                             #   - KS entropy from positive exponents (Pesin's formula)
├── train.py                  # Training loop with integrated Lyapunov tracking
│                             #   - Runs a single training experiment for given hyperparams
│                             #   - Returns: loss curve, Lyapunov time series, KS entropy,
│                             #             convergence rate, final train/test metrics
├── experiment.py             # Experiment runner
│                             #   - Sweeps over learning rates
│                             #   - Multiple initializations per learning rate
│                             #   - Saves all results to disk
├── analyze.py                # Analysis and plotting
│                             #   - KS entropy vs. convergence rate
│                             #   - KS entropy vs. final train/test loss
│                             #   - KS entropy time series for different learning rates
│                             #   - Summary statistics across initializations
└── results/                  # Output directory for plots and data
```

### Implementation Notes

- **Hessian-vector products**: Use `torch.autograd.functional.hvp` or the double-backward trick. For a function `loss(params)`, computing `H @ v` costs approximately one additional forward + backward pass. This is the performance bottleneck, so keep $k$ (number of tangent vectors) as small as practical.

- **QR decomposition**: Use `torch.linalg.qr`. The tangent vectors are stored as a $d \times k$ matrix where $d$ is the number of parameters and $k$ is the number of Lyapunov exponents we track.

- **Numerical stability**: The tangent vectors can grow or shrink enormously over many steps. The QR reorthonormalization step (every $R$ steps) prevents this. Use $R = 1$ initially for maximum stability; increase later for performance if needed.

- **Full-batch GD first**: Start with full-batch gradient descent (batch size = full training set). This gives autonomous dynamics and clean Lyapunov exponents. Add SGD as a second phase.

- **Parameter flattening**: PyTorch stores parameters across multiple tensors. Flatten all parameters into a single vector $\theta \in \mathbb{R}^d$ for the Lyapunov computation. The tangent vectors live in the same $\mathbb{R}^d$.

- **Convergence criterion**: Define convergence as reaching a training loss below some threshold (e.g., 0.1, or 95% accuracy). If the threshold is never reached within the maximum number of steps, record the run as "did not converge" with the step count set to the maximum.

- **Reproducibility**: Set random seeds for both PyTorch and NumPy. Record all hyperparameters and seeds with results.

---

## 8. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State of dynamical system | Weight vector $\theta_t \in \mathbb{R}^d$ | Most natural definition; weights fully specify the network |
| Map / dynamics | Gradient descent update rule | Each training step = one iteration |
| KS entropy method | Pesin's formula via finite-time Lyapunov exponents | Avoids computing KS entropy directly from partitions; tractable via Hessian-vector products |
| Lyapunov algorithm | Modified Benettin (top $k$ only, using Hessian-vector products) | Avoids forming the full Hessian; scales to thousands of parameters |
| Network architecture | MLP, 2-3 hidden layers, 32-64 neurons | Small enough for Lyapunov computation; complex enough for nontrivial dynamics |
| Dataset | Synthetic 2D classification (spirals) | Fast, controllable difficulty, easy to visualize |
| Primary control variable | Learning rate $\eta$ | Most direct influence on dynamics; cleanest single variable |
| Primary metric | Convergence rate (steps to threshold loss) | Directly measures "how much forgetting helps learning" |
| Secondary metrics | Final training loss, final test loss | Training loss confirms learning; test loss captures generalization |
| Gradient mode (v1) | Full-batch gradient descent | Autonomous dynamics; cleaner Lyapunov computation |
| Ergodicity handling | Finite-time exponents + multiple initializations | Does not require ergodicity; robustness checked empirically |
| Computational target | ~30 min total on MacBook Air | Constrains network size, sweep resolution, and number of seeds |
