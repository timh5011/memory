The goal of this directory is to define a general ergodic system class that I can use to define specific ergodic systems like bernoulli shifts and Logistic maps, etc...

I then want the capability to compute the Ks entropy of each system numerically. This should be a file (or multiple for different appraoches) that computes the KS entropy of any ergodic system you give it. This is a summary of possible methods. I am interested in them all at the moment, and may rule some out later:

Method 1: Pesin's Formula (via Lyapunov Exponents)
Pesin's identity states that for smooth systems with an SRB measure:
hKS=∑λi>0λih_{KS} = \sum_{\lambda_i > 0} \lambda_ihKS​=λi​>0∑​λi​
i.e., the KS entropy equals the sum of all positive Lyapunov exponents.
How to compute it from trajectory data:

Evolve your system from many initial conditions.
Track the evolution of small perturbations (the tangent map / Jacobian) along each trajectory.
Use standard algorithms (Benettin et al., 1980) — periodically re-orthonormalize a set of tangent vectors via QR decomposition, accumulating the log of the stretching factors.
Average over time and over initial conditions (ergodicity means these should agree).

Pros: This is the gold standard for smooth systems. It's well-understood, widely implemented, and gives you the full Lyapunov spectrum as a bonus.
Cons: You need access to the Jacobian (or a way to evaluate the map on nearby points for finite-difference approximation of the tangent map). It's not purely "data-driven" — you need the equations of motion, not just sampled trajectories. For high-dimensional systems, computing the full spectrum is expensive: O(d²) per time step for d-dimensional systems because you're evolving d tangent vectors.
Verdict: If you have the dynamical equations and can compute or approximate the Jacobian, this is the most reliable method. It's not as expensive as people sometimes think for moderate dimensions.
Method 2: Box-Counting / Partition-Based Entropy
This follows the definition of KS entropy more directly:

Partition your state space into boxes of size ε.
For each trajectory, record the symbolic sequence of which box it visits at each time step.
Compute the block entropy H(n) = −Σ p(s₁,...,sₙ) log p(s₁,...,sₙ) for symbol blocks of length n.
The KS entropy is h = lim(n→∞) lim(ε→0) [H(n) − H(n−1)].

Pros: Conceptually direct; works from trajectory data alone without needing equations.
Cons: This is the most expensive method in practice. You need enormous amounts of data because the number of possible n-length symbol sequences grows exponentially. The double limit (n→∞, ε→0) is treacherous numerically. Curse of dimensionality makes this essentially impractical for d > 3 or so.
Verdict: Rarely used in practice for quantitative computation. Useful pedagogically and for very low-dimensional maps.
Method 3: Correlation Integral / Grassberger-Procaccia Style Approaches
Grassberger and Procaccia (1983) showed you can estimate the Rényi entropy of order 2 (h₂), which is a lower bound on KS entropy, using correlation integrals:

From your trajectory data, form delay vectors (embedding).
Compute the correlation integral C(ε, m) = fraction of pairs of m-length trajectory segments that stay within distance ε of each other.
h₂ = lim(ε→0) lim(m→∞) [ln C(ε, m) − ln C(ε, m+1)].

Pros: Works from time series data alone (no equations needed). Much more data-efficient than the full partition method. Well-suited to your setup of sampled trajectories.
Cons: Gives h₂ ≤ h_KS, not h_KS itself (though they're often close, and equal for many standard systems). Sensitive to embedding parameters, noise, and finite-data effects. Still struggles in high dimensions.
Method 4: Compression-Based Estimates
For symbolic dynamics or discretized trajectories, you can use data compression algorithms (Lempel-Ziv complexity, etc.) to estimate entropy rates. The asymptotic compression ratio of a symbolic sequence converges to the entropy rate.
Pros: Very cheap computationally. No parameters to tune. Works on any symbolic sequence.
Cons: Convergence can be slow. Only applies after you've already symbolized your dynamics (which reintroduces partition-dependence). Gives an approximation, not a precise value.
Method 5: Nearest-Neighbor / Kraskov-Type Estimators
Adapted from the mutual information literature, these methods estimate entropy rates by looking at nearest-neighbor distances in reconstructed state spaces. They avoid explicit partitioning.
Pros: More data-efficient than box-counting. Adaptive resolution.
Cons: Still affected by dimensionality. Less established specifically for dynamical entropy than the methods above.