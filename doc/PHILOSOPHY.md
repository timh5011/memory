# Project Overview

*Tim Healy — January 2025*

---

## Project Philosophical Introduction: Memory

Memory is simultaneously one of the most empowering and crippling qualities that humanity possesses. Our memories enable us to learn, adapt, and celebrate old traditions. Without a strong memory, we would quickly forget the lessons we have learned and never be able to advance. However, too strong a memory can act as an inertial force, holding us back from change. We get stuck in bad habits and fall victim to past trauma. How many world religions preach forgiveness in some way or another? How many times have we heard that the secret to success and happiness is staying present? There is a fine line between a healthy respect for the past and becoming overly traditional.

I aim to answer the questions: how much memory is optimal for growth? How far back into a system's history must we go until the past no longer has significant influence on the present? These questions lead me to learning about ergodic theory. We must first be able to quantify a system's memory.

---

## Project Deliverable Overview

I am looking for a personal project to do that aims to answer the questions above. I am hoping that you can help me design a project. I want to run experiments with the following framework: I would like to simulate rational actors (with an Agent Based model) or get real world data (ideally timeseries economic data) that somehow indicates social mobility within some system (or both simulation and real world data). I would like there to be parameters on the different types of systems I am measuring that adjust the social mobility or inertia for an individual operating within the system. That is, I would like to give the system a "meritocracy score" that is a measure of how much that system is a true meritocracy. Then I would run the experiment to learn the optimal KS entropy to optimize meritocracy, and thus answer the question: how forgiving should a society be for the goal of progress?

There are a few philosophical assumptions, or axioms, I am making. First of all, I am not making subjective determinations about fairness grounded in any particular moral framework. I adopt the view of physicists, that the only thing that matters is observables. Thus, we define the best outcome to be the outcome that yields the best societal result. We must define what this means first. Imagine that a person makes a small mistake. How severely should they be punished? I take the view that the best measurable result is the one that reforms this person the most on average, and thus contributes the most net positive to society. A society with zero forgiveness for mistakes is not going to be one that progresses. But a society without any consequences for mistakes at all is also not going to progress.

We need to define metrics that capture what progress means for a society and how well the society is doing. We also need to identify different systems with different levels of "inertia" in a society. For example, for most people I would assume that a democracy and free market economy has more social mobility than a dictatorship.

---

I now want to introduce another agent based model into the directory. I initially choise sugarscape because it was described to me as the "hello world" of agent based modelling. it was a good choice. I would like my next abm to also be fairly standard and simple, but I would like it to be better suited to calculating the KS entropy of and doing analysis on how the KS entropy impacts some success parameter. before we choose one, lets look at what was good and bad about sugarscape. initially, I see that sugarscape was good because it was simple and it had the "wealth" variable for agents, as well as the gini coeff and wealth distributions for the entire population. we also could have run more experiments where we played with agent parameters (vision, etc) and see how those impacted the KS entropy and wealth of the agents. we didnt do this. maybe we will later, but not now. the thing that was not god about sugarscape was there was no true success metric or social wellbeing metric. there were also not clear parameters of the entire system that we could tune that would impact KS entropy. please expand on this frame of though to determine what was god and bad about sugarscaoe for the sake of our KS entropy analysis 

---

## Sugarscape Retrospective and Next Model Selection

### What Sugarscape Did Well

Sugarscape provided a clean continuous observable for entropy analysis — each agent's wealth (sugar) is a single scalar that changes every step, making symbolization straightforward. Simple behavioral rules (move-to-best-sugar) produced emergent complexity: realistic wealth distributions with Gini coefficients around 0.5, right-skewed wealth, and natural population turnover. The model supported multi-scale entropy analysis: macro-level (wealth distribution state), micro-level (individual agent trajectories), and perturbation-based (Lyapunov exponent via Wasserstein-1 divergence). Spatial structure on a toroidal grid created natural locality and competition.

### What Sugarscape Did Poorly for KS Entropy Research

**No clear success metric.** The Gini coefficient measures inequality but isn't normative — it's descriptive. We can say "inequality emerged" but can't ask "what KS entropy maximizes social welfare?" because we never defined welfare. We need a model with a clear, defensible outcome metric that we can plot *against* KS entropy.

**No parameter that directly tunes KS entropy.** The tunable parameters (alpha, vision, metabolism ranges) affect dynamics indirectly and in tangled ways. There's no single knob that maps cleanly to "how much memory/predictability does the system have." We need a sweep variable for the x-axis of an "entropy vs. success" plot.

**No strategic agent-agent interaction.** Sugarscape agents forage independently on a shared resource. There's no cooperation, defection, communication, or game-theoretic tension. This means there's no mechanism by which one agent's *memory of other agents* matters.

**Macro-level entropy was trivially low.** The wealth distribution is nearly deterministic step-to-step (~0 bits conditional entropy after k=2). The interesting dynamics are all at the individual level, but individual trajectories are hard to connect to a collective outcome.

**Agent heterogeneity confounds pooled entropy estimates.** Agents have different vision/metabolism, so pooling their trajectories mixes fundamentally different stochastic processes. The entropy estimate is an average over agent types, not the entropy of a single well-defined process.

### Candidate Models Evaluated

Three models were considered as the next ABM:

**Minority Game / El Farol Bar Problem.** N agents choose one of two options each round; minority wins. Each agent holds S strategies (lookup tables mapping the last M rounds to an action). Memory length M is the central parameter, controlling α = 2^M / N, which drives a sharp phase transition at α_c ≈ 0.34. Success metric: volatility (variance of attendance) — low volatility = efficient resource utilization. Cleanest parameter→entropy mapping. Simplest implementation (~100 lines). No spatial structure.

**Spatial Iterated Prisoner's Dilemma.** Agents on a 2D grid play cooperate/defect with neighbors. Payoffs accumulate; agents imitate high-payoff neighbors. Mutation rate μ tunes stochasticity. Success metric: cooperation rate. Canonical and well-studied, but memory→entropy mapping is indirect and there's no sharp critical point.

**Public Goods Game with Punishment.** Agents contribute to a shared pool (multiplied and split equally) and can pay to punish low contributors. Multiple candidate parameters for entropy tuning (mutation rate, punishment cost, memory of past defectors), creating ambiguity. Success metrics can conflict: high cooperation ≠ high welfare (punishment is costly). Most complex implementation (~250 lines).

### Selection: Minority Game

The Minority Game was selected because it directly addresses every Sugarscape weakness:

- **Clear success metric:** volatility/efficiency — how well agents collectively utilize a shared resource
- **Single parameter that directly tunes entropy:** memory length M drives a sharp phase transition between chaotic (high entropy, high volatility) and ordered (low entropy, low volatility) regimes
- **Strategic interaction:** agents counter-predict each other; memory of past outcomes directly determines strategy selection
- **Memory IS the model's central concept** — not a bolt-on parameter

The main tradeoff is no spatial structure, but this isolates the memory→entropy→success relationship without spatial confounds.

The key experiment will be sweeping M, computing KS entropy at each value, and plotting entropy against efficiency — producing the "money plot" that answers: *what level of system memory optimizes collective outcomes?*

---

## Appendix: Ergodic Theory

### Mixing

Ergodic theory provides a way to describe the average behavior of a dynamical system over time, without worrying about specific details of its evolution. Before getting into the math, we should consider conceptually what "memory" even means for a system. If a person (or an entire population) remembers something from their past, then that thing has the ability to influence their present; it still *matters* to some degree. On the other hand, if something has been completely forgotten — that is, if there are no remnants of it, no surviving trace — then it no longer has any influence on the present state of reality.

Ergodic theory formalizes this idea. A measure-preserving dynamical system $(X, \mathcal{B}, \mu, T)$ is said to be *strongly mixing* if for all $A, B \in \mathcal{B}$, we have

$$\lim_{n\to\infty}\mu(T^{-n}A\cap B) = \mu(A)\mu(B).$$

The set $T^{-n}A \cap B$ is the set of points currently in $B$ which were in the set $A$ $n$ iterations of $T$ earlier. Therefore, this equality simply states that as time progresses, any statistical dependence between $A$ and $B$ — which may have existed in the system's history — is lost. Thus, information about the initial conditions of the system is forgotten. Since this holds for all sets $A$ and $B$, we can think of $T$ as "mixing" the sets around the phase space $X$, resulting in a loss of the initial structure.

### Ergodicity and the Birkhoff Ergodic Theorem

Ergodic theory relies on a more general definition than mixing: ergodicity. A measure-preserving dynamical system $(X, \mathcal{B}, \mu, T)$ is *ergodic* if:

$$\text{If } A \in \mathcal{B} \text{ such that } T^{-1}A = A, \text{ then } \mu(A) = 0 \text{ or } \mu(A) = 1.$$

In other words, the phase space of the system cannot be decomposed into smaller subsets which are invariant under the system; that is, there is no region of the phase space which is only ever mapped into itself. This property is similar to, but weaker than, that of mixing.

The Birkhoff Ergodic Theorem is an important result in ergodic theory. It also justifies a key assumption which statistical mechanics relies on. We consider some observable $f$ of the system which is a $\mu$-integrable function. Firstly, the Birkhoff Ergodic Theorem guarantees that the time average of $f$ over a measure-preserving dynamical system exists. That is,

$$\lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} f(T^k(x))$$

exists for $\mu$-almost every $x \in X$.

Secondly, if the system is ergodic, then the ensemble average of $f$ over the system is equal to its time average over the system. That is,

$$\lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} f(T^k(x)) = \int_X f \, d\mu,$$

for $\mu$-almost every $x \in X$.

### Kolmogorov-Sinai Entropy

We can now begin to quantify a dynamical system's memory. The KS (Kolmogorov-Sinai) Entropy of a measure-preserving dynamical system $(X, \mathcal{B}, \mu, T)$ is defined as follows:

For a finite measurable partition $\mathcal{P} = \{P_1, P_2, \dots, P_k\} \subset \mathcal{B}$, define the entropy of $\mathcal{P}$ with respect to $\mu$ by

$$H_\mu(\mathcal{P}) = -\sum_{i=1}^k \mu(P_i) \log \mu(P_i).$$

This is the Shannon-information entropy of the partition $\mathcal{P}$ with respect to the measure $\mu$.

The entropy of the partition under $T$ is defined as:

$$h_\mu(T, \mathcal{P}) = \lim_{n \to \infty} \frac{1}{n} H_\mu \left( \bigvee_{i=0}^{n-1} T^{-i} \mathcal{P} \right),$$

where $\bigvee_{i=0}^{n-1} T^{-i} \mathcal{P}$ is the refinement of the partition $\mathcal{P}$ under the preimages of $T$ over $n$ steps. Conceptually, the refinement of the partition captures a similar idea to the concept of microstates in the thermodynamic definition of entropy. Following this rough analogy, the partition splits the phase space into equivalence classes, which are analogous to macrostates.

The Kolmogorov-Sinai entropy of the system is defined

$$h_\mu(T) = \sup_{\mathcal{P}} h_\mu(T, \mathcal{P}),$$

where the supremum is taken over all finite measurable partitions $\mathcal{P}$ of $X$.

The KS entropy of a dynamical system describes the rate at which the system generates Shannon information over time. If a dynamical system has positive KS entropy, it is chaotic and thus has a short memory. One way to understand this is that the present state of the system depends on newly generated information — information which was not contained in the system's past states. Thus the present depends more weakly on the past than it would in an invertible system, which would not generate information over time.

### Lyapunov Exponents and Pesin's Entropy Formula

While KS entropy contains information about the global complexity of a dynamical system, we can study the system locally with Lyapunov Exponents. These describe how quickly nearby trajectories of the dynamical system diverge or converge.

Here, we assume $(X, \mathcal{B}, \mu, T)$ to be a smooth, measure-preserving dynamical system. Starting with some $x \in X$ as an initial condition, we consider a small perturbation, and let $\delta_0$ be the vector of initial separation between two trajectories. The separation between these trajectories often grows like

$$\|\delta(t)\| \approx e^{\lambda t}\|\delta_0\|.$$

We call $\lambda$ a Lyapunov Exponent.

Lyapunov Exponents measure the average exponential rate of divergence or convergence of nearby trajectories in a dynamical system. These are defined as

$$\lambda(x, v) = \lim_{n \to \infty} \frac{1}{n} \log \| D_xT^n(v) \|,$$

where $D_xT^n$ is the Jacobian (derivative) of the dynamical system after $n$ iterations, and $v$ is a vector in the tangent space at $x \in X$. Notice that if $v$ is an eigenvector of $D_xT^n$, then the norm $\|D_xT^n(v)\|$ will simply scale $v$ by the corresponding eigenvalue. Thus, the eigenvalues of the Jacobian can give us insight into the Lyapunov Exponents.

Pesin's Entropy Formula bridges our study of the system's local chaotic behavior with its global complexity. This formula expresses KS entropy in terms of the spectrum of positive Lyapunov Exponents, stating that

$$h_{\mu}(T) = \int_X \sum_{\lambda_i(x) > 0} \lambda_i(x) \, d\mu(x).$$

Here, the directional dependence of Lyapunov exponents is implicitly assumed to be given by the eigendirections of the Jacobian matrix $D_xT^n$.

Taking a step back, Lyapunov Exponents can help us answer one of our previous questions: How far back into a system's history must we go until the past no longer has significant influence on the present?

### Measure-Theoretic Conjugacy of Dynamical Systems

It is possible that apparently different systems follow the same underlying dynamics. If this is the case, we say that these systems are *conjugate*. Formally, two measure-preserving dynamical systems $(X, \mathcal{B}_X, \mu, T)$ and $(Y, \mathcal{B}_Y, \nu, S)$ are conjugate if there exists a measurable bijection $\phi: X \to Y$ such that

1. $\phi$ preserves the measure: $\nu = \phi_{*}\mu$, where $\phi_{*}\mu(A) := \mu(\phi^{-1}(A))$ for all $A \in \mathcal{B}_Y$, and
2. $\phi$ preserves the dynamics: $\phi \circ T = S \circ \phi$, for $\mu$-almost every $x \in X$.

Measure-theoretically conjugate systems are equivalent to each other from a measure-theoretic perspective, meaning that their long-term statistical behavior is identical. This notion plays a similar role to conjugacy and, more generally, the concept of an isomorphism, in algebra; we are trying to ignore the way that we "label" the system and only focus on its fundamental structure. Thus, it is a natural result that a system's KS entropy is invariant under conjugacy.

### Bernoulli Shifts

A Bernoulli shift is a stochastic process that can be used to model the chaotic part of a dynamical system. This is used in ergodic theory as well as in a related field called symbolic dynamics, which models dynamical systems as symbolic sequences.

Given an *alphabet* of finite size $\mathcal{A} = \{1, \dots, n\}$, we call bi-infinite sequences $x: \mathbb{Z} \to \mathcal{A}$, denoted $(x_i)_{i \in \mathbb{Z}}$, *words*. We let $\Omega_n := \mathcal{A}^\mathbb{Z}$ be the set of all words. The *Bernoulli product measure* $\mu^\mathbb{Z}$ is defined

$$\mu^\mathbb{Z}(x_{i_1} = a_{i_1}, \dots, x_{i_k} = a_{i_k}) = \prod_{j=1}^k \mu(a_{i_j}),$$

with $\mu(a)$ being the probability assigned to symbol $a \in \mathcal{A}$. The random variables $\{\pi_i\}_{i \in \mathbb{Z}}$, where $\pi_i(x) = x_i$, are independent and identically distributed (i.i.d.) with distribution $\mu$.

We then define the shift map $\sigma$ by $(\sigma x)_i = x_{i+1}$. This simply shifts every symbol in the sequence one index to the left.

The Bernoulli shift $(\Omega_n, \mu^\mathbb{Z}, \sigma)$ is a measure-preserving dynamical system.

We can consider the KS entropy of a Bernoulli shift. Since the KS entropy of a system describes how much Shannon information is generated by the system over time, and because each position of the sequence's value is independent of all others, the KS entropy of the shift is simply the Shannon entropy of the distribution $\mu$. That is,

$$h_\mu(\sigma) = -\sum_{i=1}^n p_i \log p_i.$$

Bernoulli shifts satisfy the Markov property which says that the future state only directly depends on the present state, not on the path which led to that current state. This is why Markov processes are called memoryless. However, this does not necessarily mean that these processes completely forget the initial conditions, because the probability of reaching that present state in the first place was determined by that past. However, the next step is no longer *directly* dependent on that past. Therefore, these processes tend to forget their history faster.

### The Sinai Factor Theorem and the Ornstein Isomorphism Theorem

These next two theorems tell us that Bernoulli shifts play a role in describing any dynamical system with positive KS entropy, and allow us to characterize these systems.

The Sinai Factor Theorem states that any measure-preserving ergodic dynamical system can be decomposed into a Bernoulli shift of equal entropy to that of the system, and a trivial (invertible) component. An equivalent statement is that if the system has KS entropy $h$, then a Bernoulli shift with KS entropy less than or equal to $h$ is a *factor* of the system. Formally, if $(X, \mathcal{B}, \mu, T)$ is a measure-preserving ergodic dynamical system, then there exists a measurable, measure-preserving *factor map*

$$\pi: X \to \Omega_n,$$

such that $\pi \circ T = \sigma \circ \pi$, which projects the chaotic part of the system onto a Bernoulli shift. The trivial part of the system is $\ker \pi$.

In the context of a physical system, any symmetries of the system are contained in the trivial part. For example, we know that time-translational symmetry leads to conservation of energy. Since energy is constant, no new information is generated here as the system evolves over time. Thus, this part of the system would not be represented in the Bernoulli shift factor.

This means that in order to study the memory of some dynamical system, we can simply study the memory of a Bernoulli shift of equal KS entropy. This may allow us to employ tools of symbolic dynamics.

Lastly, the Ornstein Isomorphism Theorem states that any two Bernoulli shifts with the same KS entropy are isomorphic.
