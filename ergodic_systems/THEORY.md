## Computing KS (Kolmogorov-Sinai) Entropy of Ergodic Systems

### What KS Entropy Measures
The KS entropy (or metric entropy) quantifies the rate of information production in a dynamical system -- essentially how quickly nearby trajectories diverge in a measure-theoretic sense. For ergodic systems, several equivalent characterizations give rise to different computational strategies.

### Method 1: Pesin's Formula (via Lyapunov Exponents)
Pesin's identity states that for smooth systems with an SRB measure:

$$h_{KS} = \sum_{\lambda_i > 0} \lambda_i$$

i.e., the KS entropy equals the sum of all positive Lyapunov exponents.

**How to compute it from trajectory data:**

1. Evolve your system from many initial conditions.
2. Track the evolution of small perturbations (the tangent map / Jacobian) along each trajectory.
3. Use standard algorithms (Benettin et al., 1980) -- periodically re-orthonormalize a set of tangent vectors via QR decomposition, accumulating the log of the stretching factors.
4. Average over time and over initial conditions (ergodicity means these should agree).

**Pros:** This is the gold standard for smooth systems. It's well-understood, widely implemented, and gives you the full Lyapunov spectrum as a bonus.

**Cons:** You need access to the Jacobian (or a way to evaluate the map on nearby points for finite-difference approximation of the tangent map). It's not purely "data-driven" -- you need the equations of motion, not just sampled trajectories. For high-dimensional systems, computing the full spectrum is expensive: O(d^2) per time step for d-dimensional systems because you're evolving d tangent vectors.

**Verdict:** If you have the dynamical equations and can compute or approximate the Jacobian, this is the most reliable method. It's not as expensive as people sometimes think for moderate dimensions.

### Method 2: Box-Counting / Partition-Based Entropy
This follows the definition of KS entropy more directly:

1. Partition your state space into boxes of size epsilon.
2. For each trajectory, record the symbolic sequence of which box it visits at each time step.
3. Compute the block entropy $H(n) = -\sum p(s_1, \ldots, s_n) \log p(s_1, \ldots, s_n)$ for symbol blocks of length n.
4. The KS entropy is $h = \lim_{n \to \infty} \lim_{\varepsilon \to 0} [H(n) - H(n-1)]$.

**Pros:** Conceptually direct; works from trajectory data alone without needing equations.

**Cons:** This is the most expensive method in practice. You need enormous amounts of data because the number of possible n-length symbol sequences grows exponentially. The double limit (n -> infinity, epsilon -> 0) is treacherous numerically. Curse of dimensionality makes this essentially impractical for d > 3 or so.

**Verdict:** Rarely used in practice for quantitative computation. Useful pedagogically and for very low-dimensional maps.

### Method 3: Correlation Integral / Grassberger-Procaccia Style Approaches
Grassberger and Procaccia (1983) showed you can estimate the Renyi entropy of order 2 ($h_2$), which is a lower bound on KS entropy, using correlation integrals:

1. From your trajectory data, form delay vectors (embedding).
2. Compute the correlation integral $C(\varepsilon, m)$ = fraction of pairs of m-length trajectory segments that stay within distance epsilon of each other.
3. $h_2 = \lim_{\varepsilon \to 0} \lim_{m \to \infty} [\ln C(\varepsilon, m) - \ln C(\varepsilon, m+1)]$.

**Pros:** Works from time series data alone (no equations needed). Much more data-efficient than the full partition method. Well-suited to your setup of sampled trajectories.

**Cons:** Gives $h_2 \leq h_{KS}$, not $h_{KS}$ itself (though they're often close, and equal for many standard systems). Sensitive to embedding parameters, noise, and finite-data effects. Still struggles in high dimensions.

### Method 4: Compression-Based Estimates
For symbolic dynamics or discretized trajectories, you can use data compression algorithms (Lempel-Ziv complexity, etc.) to estimate entropy rates. The asymptotic compression ratio of a symbolic sequence converges to the entropy rate.

**Pros:** Very cheap computationally. No parameters to tune. Works on any symbolic sequence.

**Cons:** Convergence can be slow. Only applies after you've already symbolized your dynamics (which reintroduces partition-dependence). Gives an approximation, not a precise value.

### Method 5: Nearest-Neighbor / Kraskov-Type Estimators
Adapted from the mutual information literature, these methods estimate entropy rates by looking at nearest-neighbor distances in reconstructed state spaces. They avoid explicit partitioning.

**Pros:** More data-efficient than box-counting. Adaptive resolution.

**Cons:** Still affected by dimensionality. Less established specifically for dynamical entropy than the methods above.

---

## Relation Between KS Entropy and Entropy Rate

### The Core Connection
The KS entropy *is* the entropy rate of the dynamical system, in a specific sense. But the devil is in the details of what "entropy rate" means.

Given a partition P of the state space, you get a symbolic sequence by recording which partition element the trajectory visits at each time step. This symbolic process has a well-defined Shannon entropy rate:

$$h(T, P) = \lim_{n \to \infty} \frac{1}{n} H(X_0, X_1, \ldots, X_{n-1})$$

or equivalently the conditional form:

$$h(T, P) = \lim_{n \to \infty} H(X_n \mid X_0, \ldots, X_{n-1})$$

The KS entropy is then the supremum over all finite measurable partitions:

$$h_{KS}(T) = \sup_P \, h(T, P)$$

So KS entropy is the maximum entropy rate you can extract from the system by any finite observation scheme. A coarse partition will miss information and underestimate; the KS entropy captures the intrinsic rate.

### Why the Supremum Matters
For any particular partition, $h(T, P) \leq h_{KS}$. The partition might lump together distinguishable states, losing information. The supremum recovers all of it.

The key theoretical shortcut is the concept of a **generating partition** -- a partition P such that the iterated refinements $T^{-1}P \vee T^{-2}P \vee \cdots$ separate all points (up to measure zero). For a generating partition, $h(T, P) = h_{KS}$ exactly. The Krieger generator theorem guarantees these exist for finite-entropy ergodic systems.

This is why symbolic dynamics is so powerful for certain systems: if you can find a generating partition (like the standard one for hyperbolic toral automorphisms, or Markov partitions for Axiom A systems), the entropy rate of the resulting symbol sequence *is* the KS entropy -- no supremum needed.

### Where Confusion Arises
In information theory, "entropy rate" usually refers to a specific stationary stochastic process. You have one process, one entropy rate.

In dynamical systems, the system is fixed but you have a choice of observation (partition). Different partitions give different symbolic processes with different entropy rates. KS entropy optimizes over this choice.

So the relationship is:

1. Every partition gives you a stochastic process whose entropy rate is $h(T, P)$.
2. KS entropy = sup over partitions of these entropy rates.
3. If you happen to use a generating partition, you get equality.

### Practical Implications
This distinction is exactly why the partition-based numerical methods are hard. You'd need to either find a generating partition (which is analytically tractable for only a few systems) or take the supremum over partitions (computationally intractable). In practice people refine the partition and watch if the estimate has converged, but there's no guarantee you've reached the supremum.

This is also why Pesin's formula is so appealing -- it sidesteps the partition problem entirely. The Lyapunov exponents are intrinsic to the system, with no partition choice involved. You get $h_{KS}$ directly (assuming the Pesin conditions hold: smooth system, SRB/physical measure).

### A Useful Mental Model
Think of it this way: a dynamical system is producing information at a fixed intrinsic rate ($h_{KS}$). Any observation scheme (partition) captures at most that rate. A generating partition captures all of it. A coarse partition is like a lossy channel -- some information is lost. The KS entropy is the channel capacity of the system, in a loose analogy.
