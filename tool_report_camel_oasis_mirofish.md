# A Comprehensive Report on CAMEL AI, OASIS, and MiroFish

---

## Table of Contents

1. [CAMEL AI](#1-camel-ai)
2. [OASIS](#2-oasis)
3. [MiroFish](#3-mirofish)
4. [Comparative Summary](#4-comparative-summary)

---

## 1. CAMEL AI

### 1.1 What It Is

CAMEL (Communicative Agents for "Mind" Exploration of Large Language Model Society) is an open-source Python framework for building multi-agent systems powered by large language models. It was introduced in a 2023 NeurIPS paper and is maintained by CAMEL-AI.org, a research collective with institutional affiliations spanning Oxford, KAUST, CMU, Stanford, and others. It is best understood as the **foundational infrastructure layer** on which more specialized simulation tools (including OASIS, described below) are built.

CAMEL's stated research mission is to find the "scaling laws of agents" — that is, to understand how multi-agent LLM systems behave at increasing scale, and what emergent properties arise from that scale. In practice, however, the framework is general-purpose enough to serve as the backbone for anything from research simulations to production automation workflows.

### 1.2 Purpose

CAMEL is a **toolkit**, not an application. It does not simulate any domain by itself. Instead, it provides:

- A clean abstraction for a single LLM-powered agent (the `ChatAgent`)
- Composable patterns for connecting multiple agents
- A memory system for agents to maintain context across interactions
- A tool-calling interface for agents to interact with the external world
- Support for dozens of LLM backends (OpenAI, Anthropic, Gemini, Mistral, local models via Ollama or vLLM, and many others)

The researcher or developer brings their own domain logic, environment, and objectives; CAMEL handles the scaffolding.

### 1.3 Foundational Mechanism: The ChatAgent

The atomic unit of the entire CAMEL framework is the `ChatAgent`. Every other component in the stack is either a wrapper around, or a coordination layer on top of, this class.

A `ChatAgent` is initialized with:

- A **system message**: a natural language description of the agent's role, personality, constraints, and objectives. This is injected into the LLM context at the start of every interaction and is the primary mechanism by which agent behavior is conditioned.
- A **model backend**: the LLM that powers the agent's reasoning (e.g., GPT-4o-mini, Claude Sonnet, Llama-3).
- A **memory module**: determines how much of its conversation history the agent can access at each step.
- A **tool list**: Python functions the agent is permitted to call.

At each simulation step, the researcher calls `agent.step(observation)`, where `observation` is a natural language description of the agent's current state. The LLM processes the system message plus the observation plus its memory window, and returns either a natural language response or a structured tool call. The tool call is then executed deterministically in Python, and the result is logged.

This is the entire agent loop. It is simple by design and intentionally domain-agnostic.

### 1.4 How Agents Are Conditioned to Behave

Agent behavior in CAMEL is governed almost entirely through **prompt engineering**. There is no separate reward signal, no policy gradient, no hardcoded behavioral rule. The agent does what an LLM conditioned on its system prompt would do when presented with a given observation.

The system message serves several behavioral functions simultaneously:

- **Role definition**: "You are a rational economic actor in a simulated economy."
- **Constraint specification**: "You may only take actions from the following list..."
- **Personality and heterogeneity**: "You are risk-averse, have a preference for short-term gains, and are distrustful of authority."
- **Goal specification**: "Your objective is to maximize your accumulated wealth over time."
- **Contextual grounding**: "You live in a world where the following laws apply..."

Memory adds a second conditioning layer. By default, CAMEL uses `ChatHistoryMemory`, which maintains a sliding window of recent exchanges. The `message_window_size` parameter directly controls how many past turns are available to the agent when making its current decision — making it an explicit, tunable parameter governing agent memory depth. Longer windows mean the agent's current behavior is conditioned on a longer history of past observations and actions.

Three memory implementations are available:

- **ChatHistoryMemory**: Recency-based sliding window. Simple and fast.
- **VectorDBMemory**: Semantic retrieval. The agent can access past memories that are conceptually relevant to the current observation, regardless of recency.
- **LongtermAgentMemory**: Hybrid of both, combining recency and semantic relevance.

### 1.5 Multi-Agent Patterns

Beyond single agents, CAMEL provides several coordination patterns:

**Role-Playing**: The original CAMEL contribution. Two agents are assigned complementary roles (e.g., "AI assistant" and "human user") and engage in multi-turn dialogue toward a shared task. The framework manages turn-taking and ensures each agent stays in character.

**Workforce**: A hierarchical pattern where a supervisor agent decomposes tasks and delegates to specialist worker agents. The supervisor monitors progress and integrates results.

**Agent Societies (custom)**: The researcher builds their own coordination logic — for instance, a simulation loop that iterates over a population of agents, presents each with the current world state, collects their actions, and updates the environment deterministically.

For research simulations of the kind described in this project (a population of agents operating in a shared economy/legal system), the third pattern — custom simulation loop — is the most appropriate.

### 1.6 Strengths

- **Generality**: No domain assumptions. Supports any environment the researcher can describe in natural language and Python.
- **Backend flexibility**: One API supports 20+ LLM providers. Switching from GPT-4o-mini to Claude Sonnet is a one-line change.
- **Structured outputs**: Agents can return Pydantic-validated JSON objects, making it easy to parse actions deterministically.
- **Mature tool-calling**: Clean interface for defining custom Python functions as agent tools. This is the primary mechanism for connecting agent decisions to simulation state updates.
- **Memory modularity**: Three interchangeable memory backends with configurable window sizes, supporting both recency and semantic retrieval.
- **Active development**: The framework is under rapid development with strong institutional support and a large contributor community.

### 1.7 Limitations and Reliability Concerns

- **Prompt sensitivity**: Agent behavior is fully determined by the system prompt. Small wording changes can produce meaningfully different behavioral patterns, introducing a form of measurement uncertainty. There is no separate optimization process ensuring behavioral consistency.
- **No ground truth for rationality**: CAMEL agents are not rational in the game-theoretic sense. They approximate rationality through LLM reasoning, which is shaped by training data, hallucination tendencies, and model-specific biases. An agent may make decisions inconsistent with its stated objective if the LLM's priors are sufficiently strong in the opposite direction.
- **Context window limits**: The memory window is bounded by the LLM's context window. At high step counts or with large observations, older history is necessarily dropped, regardless of its relevance.
- **Stochasticity**: LLM outputs are stochastic. Two runs of the same simulation with the same parameters will not produce identical trajectories. Temperature controls the degree of stochasticity but cannot eliminate it.
- **Cost at scale**: Every agent step requires at least one API call. Large populations or long simulations incur significant cost.
- **Not a physics engine**: CAMEL does not enforce logical consistency or conservation laws. An agent can, in principle, output an action that violates the rules of the simulated world; the researcher must implement validation logic to catch this.

---

## 2. OASIS

### 2.1 What It Is

OASIS (Open Agent Social Interaction Simulations) is a **domain-specific social simulation framework** built on top of CAMEL. It was introduced in a NeurIPS 2024 workshop paper by researchers from CAMEL-AI, Shanghai AI Lab, Oxford, KAUST, Imperial College London, and other institutions. It is designed to simulate large-scale social media environments — specifically platforms resembling Twitter/X and Reddit — with populations of up to one million LLM-powered agents.

OASIS is not a general-purpose simulation framework. It is explicitly and deeply opinionated about its domain. Its architecture, database schema, action space, agent prompts, and recommendation systems are all designed around social media dynamics. This specificity is both its strength (it replicates real social phenomena with validated fidelity) and its limitation (it cannot easily be repurposed for non-social-media domains without substantial modification).

### 2.2 Purpose

OASIS exists to address a specific gap in the prior literature on agent-based social simulation. Previous LLM-based ABMs were each purpose-built for a single phenomenon (e.g., misinformation spread, opinion polarization) and could not be reused across research questions. OASIS provides a generalizable platform for studying any social media phenomenon, at realistic scale, with agents whose behavior is grounded in LLM reasoning rather than hand-coded rules.

Its primary research applications are: information spreading dynamics, group polarization, herd effects, content moderation policy evaluation, recommendation algorithm effects, and opinion formation.

### 2.3 Architecture

OASIS is built on five core components:

**1. Platform (Environment Server)**
A SQLite database that serves as the shared world state. It stores all posts, comments, likes, follower relationships, and engagement metrics. Every agent reads from and writes to this database. It is the "ground truth" of the simulation — the single source of facts about what has happened in the simulated social network at any given time.

**2. Recommendation System (RecSys)**
Determines what content appears in each agent's feed at each timestep. Two implementations are provided: an interest-based system (content matched to agent profile preferences) and a hot-score system (content ranked by engagement metrics, similar to Reddit's actual algorithm). This component is architecturally significant: it controls the information each agent can see, which directly shapes their decisions. Researchers can swap recommendation algorithms to study how content routing affects social dynamics.

**3. Time Engine**
Controls the progression of simulated time and determines which agents are "active" at each step. Agents can be given schedules (simulating variable activity patterns across a user population). The Time Engine activates the appropriate subset of agents at each timestep and coordinates the simulation loop.

**4. Agent Module (SocialAgent)**
Each agent is an instance of `SocialAgent`, which inherits from CAMEL's `ChatAgent`. Agents have a unique profile (name, demographic attributes, personality description, interest profile) and can perform 23 distinct actions including: create post, create comment, like post, dislike post, follow user, mute user, search posts, view trends, and do nothing.

**5. Agent Graph**
Represents the social network structure — who follows whom, what the initial relationship topology looks like. This is initialized from real social network data or synthetic profiles before the simulation begins and evolves dynamically as agents follow/unfollow during the simulation.

### 2.4 How Agents Are Conditioned to Behave

Agent conditioning in OASIS operates through a layered system prompt that is generated from the agent's `UserInfo` object. For a Reddit-style agent, the system prompt contains:

- **Platform context**: "You are a Reddit user. You will be shown posts and must choose actions."
- **Self-description**: Name, age, gender, MBTI personality type, country of origin, and a free-text personality profile.
- **Behavioral instruction**: "Your actions should be consistent with your self-description and personality."
- **Response format**: "Please perform actions by tool calling."

At each active timestep, the agent receives an observation consisting of the current recommended posts and comments in its feed, plus the available action types. The LLM reasons about its persona, the content, and the available actions, and selects an action via structured tool calling.

The agent's memory in OASIS is its conversation history within the simulation — the sequence of past observations and actions it has taken during the current run. By default, agents use CAMEL's `ChatHistoryMemory` with a finite window. There is no persistent memory across separate simulation runs.

### 2.5 What OASIS Has Validated

The OASIS paper demonstrates empirical replication of several real-world social phenomena, which establishes some confidence in the framework's behavioral validity:

- **Information spreading**: Simulated message propagation patterns match real-world trends in scale, depth, and reach.
- **Opinion polarization**: In a 196-agent Twitter simulation discussing a social psychology topic, opinions became more extreme over time — consistent with real social media dynamics. An "uncensored" model condition produced even stronger polarization.
- **Herd effects**: In Reddit simulations with pre-upvoted or pre-downvoted posts, agents showed stronger susceptibility to herd behavior than real humans, suggesting LLMs may amplify social conformity effects.
- **Scale effects**: Larger agent populations produce more enhanced group dynamics and more diverse expressed opinions.

### 2.6 Strengths

- **Validated at scale**: The only LLM-based ABM that has been validated against real social media data at population scales comparable to real platforms.
- **Realistic information environment**: The recommendation system and dynamic social graph produce an information environment that mirrors real platform mechanics.
- **Out-of-the-box replication**: If your research question involves social media dynamics (information spread, polarization, herd behavior), OASIS gives you a working environment immediately.
- **Installable via pip**: `pip install camel-oasis`. Relatively low barrier to entry for the standard use case.
- **Async architecture**: The simulation loop is asynchronous, enabling efficient concurrent API calls to the LLM backend.

### 2.7 Limitations and Reliability Concerns

- **Domain lock-in**: The architecture is hardcoded around social media. The database schema stores posts, likes, and follower relationships. The action space is social media actions. The agent prompts frame agents as social media users. Repurposing OASIS for a non-social-media simulation would require rewriting the platform layer, the action space, the recommendation system, and the agent prompt templates — at which point you are essentially building on CAMEL directly rather than OASIS.
- **Agent rationality is social, not economic**: OASIS agents are conditioned to behave as social media users. Their decision-making reflects the behavioral patterns of social media engagement (like, comment, follow) rather than rational economic or legal reasoning. This makes them poor models of agents operating in an economic or legal system.
- **LLM conformity bias**: The herd effect finding above suggests that LLM agents may be systematically more conformist than real humans, which could confound results in studies of collective behavior.
- **No ground truth for individual agent behavior**: While population-level dynamics have been validated, individual agent decisions are not independently validated against real user behavior. Individual trajectories should be treated as illustrative, not empirically calibrated.
- **Memory is ephemeral**: Agents do not remember past simulation runs. Each run starts fresh.

---

## 3. MiroFish

### 3.1 What It Is

MiroFish is a **prediction-oriented multi-agent simulation application** developed by a team incubated by Shanda Group (a Chinese technology conglomerate). It is open-source (AGPL-3.0 license), actively developed, and has accumulated substantial community traction (~46k GitHub stars as of early 2026). It is built on top of OASIS (and therefore on CAMEL) as its simulation engine.

MiroFish is best understood as a complete, opinionated **product** built on top of the OASIS simulation infrastructure, rather than a framework or toolkit. It includes a full web frontend, a backend API server, a graph-based memory system, and a report generation agent. Users interact with it through a web interface, not through Python code.

### 3.2 Purpose

MiroFish's stated purpose is **prediction**: given some "seed material" describing a real-world situation (a news report, a policy document, a financial signal, or even a fictional narrative), it constructs a high-fidelity simulated world populated by LLM agents, lets those agents interact and evolve, and returns a prediction report about how the real-world situation will unfold.

The tagline — "predicting anything" — reflects an ambitious scope. Demonstrated use cases include: public opinion simulation (predicting how a university controversy will unfold), literary prediction (extrapolating the lost ending of a classical novel), and planned extensions to financial forecasting and political analysis.

The framing is explicitly a "digital sandbox" or "parallel world" in which real-world dynamics can be rehearsed before they unfold. The target user is a decision-maker who wants low-risk pre-testing of policies, communications strategies, or scenarios.

### 3.3 Architecture and Workflow

MiroFish adds several layers on top of the OASIS simulation engine:

**1. Seed Ingestion and Graph Building (GraphRAG)**
The user uploads seed material in natural language (a document, report, or story). MiroFish uses a Retrieval-Augmented Generation (RAG) pipeline with a knowledge graph backend to extract entities, relationships, and narrative structure from the seed. This graph becomes the "world model" from which the simulation is initialized.

**2. Persona Generation**
From the extracted entities, MiroFish automatically generates agent profiles — individual personalities, backgrounds, relationships, and behavioral tendencies — grounded in the seed material. A simulation about a university controversy would generate agents representing students, administrators, faculty, and media figures, each with characteristics derived from the source text.

**3. Simulation Engine (OASIS)**
The core simulation runs on OASIS's infrastructure: agents interact in a shared environment, with the Time Engine coordinating activity and the Recommendation System managing information flow. MiroFish adds "dual-platform parallel simulation" — running two simultaneous instances and comparing results to improve robustness.

**4. Temporal Memory (Zep Cloud)**
Unlike base OASIS (which uses only in-context conversation history), MiroFish integrates Zep Cloud for persistent long-term memory. Agents have both individual memory (their own history) and collective memory (shared knowledge about what has happened in the simulation world). This memory is updated dynamically as the simulation progresses, enabling agents to reference earlier events in later decisions.

**5. ReportAgent**
After the simulation completes, a dedicated ReportAgent — itself an LLM agent with a rich tool suite — queries the simulation environment and generates a structured prediction report. The user can also directly interview any agent in the simulated world via the web interface.

### 3.4 How Agents Are Conditioned to Behave

MiroFish agent conditioning operates at two levels:

**Persona level**: Each agent's system prompt is generated from the seed material and the persona extraction pipeline. Agents are given names, backgrounds, relationships to other agents, and motivations derived from the real-world context. A simulation about a corporate merger might include agents representing executives, employees, shareholders, and journalists, each with prompts reflecting their actual roles and likely interests.

**Memory level**: Agents are conditioned not just by their initial persona but by the evolving history of the simulation, accessed through Zep Cloud. At each timestep, an agent's effective context includes: their base persona, their individual history, and the relevant subset of collective simulation history retrieved from the memory store. This produces agents whose behavior evolves with the simulation rather than remaining frozen at initialization.

**Platform constraint**: Despite the richer conditioning, agents still operate within OASIS's social media action space. They post, comment, like, and follow — even in simulations that are ostensibly about non-social-media phenomena. The "civilization" being simulated is, at the implementation level, always a social network.

### 3.5 Strengths

- **End-to-end product**: The full pipeline from seed material to prediction report is automated and accessible through a web UI, with no programming required from the user.
- **Rich memory architecture**: The Zep Cloud integration gives agents persistent, evolving memory that base OASIS lacks, producing more coherent long-run behavior.
- **Grounded agent personas**: Personas derived from real seed material may better capture the specific dynamics of a real-world situation than generic synthetic profiles.
- **High community interest**: The ~46k GitHub star count suggests substantial practitioner interest and active development.
- **Demonstrated validation**: The documented use cases (Wuhan University opinion simulation, Dream of the Red Chamber literary prediction) demonstrate that the system can produce internally coherent, qualitatively plausible simulations.

### 3.6 Limitations and Reliability Concerns

- **Built on OASIS's social media constraints**: All the domain lock-in problems of OASIS apply equally to MiroFish. The simulation action space is social media interaction regardless of what the seed material describes. A financial scenario is simulated through the lens of agents posting and reacting to posts, not through agents executing trades or making investment decisions.
- **Prediction validity is unverified**: MiroFish's core claim — that its simulated dynamics predict real-world outcomes — has not been independently evaluated against ground truth. The demonstrated use cases are illustrative, not validated predictions. There is no published evidence of prospective prediction accuracy.
- **GraphRAG quality is critical and opaque**: The quality of the simulation depends heavily on how well the GraphRAG pipeline extracts meaningful structure from the seed material. This extraction process is not transparent to the user and may introduce errors or biases that propagate through the simulation.
- **Zep Cloud dependency**: Persistent memory requires a paid cloud service, introducing an external dependency and ongoing cost.
- **Primary language is Chinese**: The documentation and community are predominantly Chinese-language, which may create friction for English-speaking researchers.
- **Not designed for parameter sweeps**: MiroFish is built around the use case of simulating one specific scenario end-to-end. It does not provide infrastructure for the kind of systematic parameter variation and comparative analysis that a research simulation requires.

---

## 4. Comparative Summary

### 4.1 Positioning

| Dimension | CAMEL | OASIS | MiroFish |
|---|---|---|---|
| Type | General-purpose framework | Domain-specific simulation platform | End-to-end prediction application |
| Domain | None (agnostic) | Social media (Twitter/Reddit) | Social media (via OASIS) |
| Interface | Python API | Python API | Web UI |
| User | Developer / researcher | Researcher | Decision-maker / practitioner |
| Built on | Native (foundational) | CAMEL | OASIS + CAMEL |
| Validated phenomena | None (framework) | Polarization, herd effects, info spread | Qualitative plausibility only |

### 4.2 The Agent Conditioning Stack

All three tools use the same underlying conditioning mechanism: **system prompt + conversation history + tool calling**. The differences are in how rich and structured the system prompt is, and how the memory is managed:

- **CAMEL**: System prompt is entirely researcher-defined. Memory is a simple sliding window of conversation history (or vector DB for semantic retrieval). Maximum flexibility, minimum prescription.
- **OASIS**: System prompt is generated from a demographic/personality profile template. Memory is CAMEL's conversation history. Agents are told they are social media users. Platform enforces the action space.
- **MiroFish**: System prompt is generated from seed material via GraphRAG. Memory is persistent and evolving via Zep Cloud, combining individual and collective history. Agents still ultimately operate as social media users within OASIS's action space.

### 4.3 Shared Reliability Assumptions

All three tools share a set of foundational assumptions that determine where they are reliable and where they are not:

**They assume LLM reasoning approximates rational agency.** This is the central bet. LLMs make decisions by pattern-matching on their training data, not by optimizing a utility function. In domains well-represented in training data (social media interaction, common economic intuitions, familiar social roles), this approximation may be reasonable. In exotic or highly technical domains (complex legal reasoning, specialist financial instruments, novel institutional structures), it may break down significantly.

**They assume prompts faithfully condition behavior.** There is no guarantee that an agent told to be "risk-averse" will consistently behave in a risk-averse manner across all situations. LLMs have strong prior patterns from training that may override prompt instructions in edge cases. Behavioral consistency within a simulation should be empirically verified, not assumed.

**They produce stochastic outputs.** Simulations are not reproducible at the individual trajectory level without fixing a random seed and temperature to zero, and even then, API-level nondeterminism may persist.

**Population-level statistics are more reliable than individual trajectories.** Aggregate measures (mean behavior, distribution shapes, group-level dynamics) are less sensitive to individual LLM stochasticity than single-agent trajectory analysis. Research designs relying on entropy of individual trajectories should account for this by running multiple independent simulation instances and treating the variance across runs as a meaningful quantity.

**They have not been calibrated against economic or legal ground truth.** OASIS has been validated against social media phenomena. Neither OASIS nor CAMEL nor MiroFish has been validated as a model of economic or legal behavior. Researchers using these tools for economic or institutional simulation are working in an unvalidated regime and should treat results as generating hypotheses rather than confirming them.

---

*Report compiled April 2026. All framework details reflect current documentation and may change as active development continues.*
