# World models under the functional taxonomy: separating simulation from generation in embodied AI

**A review of the Dreamer lineage and its alternatives**

*Fei-Fei Li and the World Labs team (2026) have proposed a functional taxonomy that classifies world models into Renderers, Simulators, and Planners. Applying this taxonomy as a critical lens reveals that the majority of systems currently marketed as "world models" cover only one of the three functions. The Dreamer series, by contrast, has maintained a dual commitment to Simulator and Planner since its inception, and its latest iteration, R2-Dreamer, demonstrates that a Simulator can operate effectively without a Renderer — a result that validates a core prediction of the taxonomy.*

---

## The POMDP loop and its three projections

The term "world model" has become one of the most consequential and most diluted concepts in contemporary artificial intelligence. Computer vision, robotics, reinforcement learning, and generative video each claim to build world models, and each means something substantially different. A diffusion model that generates visually stunning but physically impossible flames, a language model that improvises a playable game from a prompt, and a physics engine that faithfully simulates combustion all travel under the same banner.

This ambiguity is not new. The Greek schools could never agree on what the world was made of — fire, water, or indivisible atoms — because "world" was never a single thing. It was always a placeholder for whatever totality a given thinker needed to reason about. The field of AI has inherited the same problem at exactly the moment when precision matters most.

A clarifying framework arrived in June 2026, when World Labs published "A Functional Taxonomy of World Models." The taxonomy roots itself in a diagram that has existed for decades: the partially observable Markov decision process (POMDP), the canonical representation of how an agent interacts with an environment. In this loop, an agent takes actions, which affect the state of the world. The agent never sees the state directly. What reaches the agent are observations — photons on a retina, sensor readings, pixels in a video frame. New observations inform new actions, and the cycle continues.

Within this loop, three distinct functional roles have been conflated under the single term "world model." A **Renderer** outputs observations — pixels meant for human eyes — and its contract is visual fidelity. A **Simulator** outputs state: a geometrically, physically, or dynamically faithful representation that both humans and computer programs can compute on and interact with. A **Planner** outputs actions: given an observation and a goal, it decides what the agent should do next. The three functions share the same underlying knowledge — geometry, physics, dynamics — but they project that knowledge differently.

The same knowledge that allows a model to render a cup from any angle should, in principle, allow it to simulate what happens when the cup is pushed and to plan a hand that picks the cup up. The convergence of these three functions into a single architecture is the defining open problem in world model research today.

---

## What a world model is for

The POMDP loop gives us a precise way to separate genuine world models from systems that merely borrow the name. Several of the most prominent systems that call themselves world models turn out, under this lens, to be Renderers — powerful and commercially significant, but operating on a fundamentally different contract.

OpenAI's Sora, released in February 2024, was introduced under the title "Video generation models as world simulators." Sora is a diffusion transformer that operates on spacetime patches of video, trained at massive scale on internet data. It can generate a minute of high-fidelity video from text prompts, and it exhibits emergent properties such as three-dimensional consistency and object permanence. These properties are real and important. But Sora has no explicit representation of state. Its contract is visual, not structural. OpenAI's own technical report acknowledges that Sora "does not accurately model the physics of many basic interactions, like glass shattering." A model that cannot predict the consequences of a collision is not simulating the world; it is rendering a plausible version of it.

Google DeepMind's Genie, also released in 2024, takes a step closer to interactivity. Genie is an 11-billion-parameter model trained from unlabelled internet video, and it generates action-controllable virtual worlds. Its three-component architecture — a spatiotemporal video tokenizer, an autoregressive dynamics model, and a learned latent action model — allows users to influence frame generation on a frame-by-frame basis. Genie's own authors described it as a "foundation world model." Under the functional taxonomy, Genie is best understood as an action-conditioned Renderer. Its "state" is encoded implicitly in video tokens, not surfaced as geometry that a physics engine or a robot controller could consume.

At the other end of the spectrum, systems such as DeepMind's GATO (2022) represent the Planner function in isolation. GATO is a generalist agent trained via behavioural cloning across hundreds of tasks. It outputs actions. It contains no internal Simulator — no model of how the world will respond to those actions. GATO is a policy, not a world model.

The Renderers and the standalone Planners are both genuine and valuable contributions. But they are not world models in the sense that the POMDP tradition — the tradition that gave the term its technical meaning — intended. The vision of a world model, from Kenneth Craik's 1943 proposal that minds run "small-scale models" of reality, through the recurrent neural network world models of the 1990s, to the modern Dreamer series, has always been about a model that can predict the consequences of actions and plan accordingly. That requires both Simulator and Planner.

---

## The Dreamer series: a case study in Simulator–Planner alignment

The Dreamer series, developed by Danijar Hafner and colleagues at Google DeepMind, is the only family of systems that has maintained a strict dual commitment to the Simulator and Planner functions across four generations of architectural evolution.

### DreamerV1: learning behaviours by latent imagination

DreamerV1 (Hafner et al., 2019, ICLR 2020) established the paradigm that has defined the series. The core architectural insight was the Recurrent State-Space Model (RSSM), a hybrid of deterministic and stochastic state representations. The deterministic path, implemented as a gated recurrent unit, carries long-term temporal structure. The stochastic path, sampled from a learned posterior, captures the environment's irreducible randomness and multimodality. Together, they produce a latent state that can be rolled forward in imagination without decoding to pixels.

The Planner component operates entirely within this latent space. An actor network generates actions in the imagined trajectory; a critic network estimates state values; gradients propagate back along the imagined path to improve the policy. On twenty continuous control tasks, DreamerV1 exceeded prior model-based and model-free methods in data efficiency and final performance.

### DreamerV2: the discrete transition

DreamerV2 (Hafner et al., 2020, ICLR 2021) made a single architectural change that proved decisive: it replaced the continuous Gaussian latent variables of DreamerV1 with categorical discrete variables — 32 categorical variables with 32 categories each, yielding 1,024 discrete combinations. The motivation was that physical structure is often discrete: object types, event boundaries, game state transitions. Discrete representations proved substantially better at capturing these structures.

The result was the first world-model-based agent to achieve human-level performance across the 55-game Atari benchmark, matching the best model-free methods with substantially less environment interaction. This result carried an important implication: the quality of the Simulator sets an upper bound on the performance of the Planner.

### DreamerV3: general algorithm, single configuration

DreamerV3 (Hafner et al., 2023) demonstrated that a single hyperparameter configuration could master more than 150 tasks spanning Atari games, continuous control benchmarks, dexterous manipulation, and Minecraft — tasks with reward scales ranging over five orders of magnitude and observation spaces from pixels to proprioception.

Three techniques made this possible. A symlog transform compresses rewards and returns into a bounded logarithmic space, removing the need for task-specific reward normalization. Quantile regression replaces mean-squared error in the critic, making no assumption about the distribution of target values. A bidirectional variant of RMSNorm keeps activations stable across widely varying input statistics.

The Minecraft diamond achievement captured the field's attention. Minecraft is an open-world game with sparse rewards and a deep skill tree: the agent must learn to chop trees, craft planks, build a crafting table, craft a wooden pickaxe, mine stone, craft a stone pickaxe, mine iron, smelt iron, craft an iron pickaxe, and finally mine diamond. No reinforcement learning algorithm had previously completed this chain from scratch. DreamerV3 succeeded by learning a rich world model in the early stages and then planning in latent imagination across the long skill horizon.

Under the functional taxonomy, DreamerV3 scores strongly on both the Simulator axis (the RSSM predicts state transitions, rewards, and episode continuations in latent space) and the Planner axis (actor-critic learning operates entirely within imagined trajectories). The Renderer axis is present but subsidiary: the image decoder exists only to supervise the representation, not to produce outputs for human consumption.

### R2-Dreamer: a Simulator without a Renderer

The latest iteration, R2-Dreamer (Morihira et al., 2026, ICLR 2026), tests a hypothesis that the functional taxonomy makes explicit: the Simulator can stand alone.

A weakness of the DreamerV3 design is that the decoder must reconstruct all pixels, including vast task-irrelevant regions such as sky and background texture, wasting parameters and computation. R2-Dreamer removes the decoder entirely. In its place, it introduces a self-supervised redundancy-reduction objective inspired by Barlow Twins: the cross-correlation matrix of the latent representation is driven toward the identity matrix, with off-diagonal elements forced toward zero (eliminating redundant dimensions) and diagonal elements forced toward one (preserving informative structure). Crucially, this regularizer is internal — it requires no data augmentation, which distinguishes R2-Dreamer from prior decoder-free approaches.

The results validate the taxonomy's prediction. On DeepMind Control Suite and Meta-World, R2-Dreamer matches DreamerV3 and TD-MPC2 while training 1.59 times faster. On the DMC-Subtle benchmark, where task-relevant objects are tiny and easily lost in background reconstruction, R2-Dreamer substantially outperforms all baselines. The implicit structure learned in the latent space is sufficient to support planning without pixel-level supervision.

This finding matters beyond the Dreamer series. It suggests that the Simulator is not merely a bridge between Renderer and Planner but a viable standalone module that can be optimized independently. It also lends weight to the World Labs claim that a unified world model is achievable: if the shared geometric and physical knowledge that underlies all three functions can be learned without rendering, then the path to a single model that can render, simulate, and plan is clearer.

---

## The partial coverage of current alternatives

The Dreamer series stands nearly alone in its dual coverage of Simulator and Planner. Every other system we surveyed covers one function well and the others partially or not at all.

World Labs' own Marble (2025) is the only commercial system that bridges Renderer and Simulator. It takes multimodal prompts — text, image, video, or spatial sketch — and generates explorable three-dimensional environments, outputting Gaussian splats for visual exploration alongside collision meshes that a physics engine can operate on. This dual output covers two of the three functions, and Marble is commercially deployed. It does not, however, plan, and its authors are explicit about this boundary.

PointWorld (Huang, Fei-Fei et al., 2026) occupies the Simulator function with unusual clarity. It takes one or a few RGB-D images and a sequence of robot action commands, and it forecasts per-pixel three-dimensional displacements in response to those actions. By representing actions as three-dimensional point flows rather than embodiment-specific joint positions, it generalizes across robotic morphologies — a single pretrained checkpoint enables a Franka arm to perform rigid-body pushing, deformable and articulated object manipulation, and tool use in real-world settings, all from a single image and without any demonstrations. PointWorld is a pure Simulator. It has no Renderer, and its Planner is an external model predictive control loop. Its success reinforces the independence of the Simulator function.

NVIDIA's Cosmos platform, positioned as a world foundation model, targets the Simulator role for physical AI, generating physics-aware video for autonomous vehicle and robotics training. It is the most ambitious commercial effort to build a general-purpose Simulator, but it faces the same challenge as all generative simulators: the sim-to-real gap. AI-generated geometry can appear correct while containing self-intersections or incorrect scale that produces nonsensical physics. Cosmos's outputs are improving rapidly but have not yet replaced traditional physics engines for high-stakes applications.

Systems that claim the world model label but function as Renderers alone — Sora, Genie, RTFM (World Labs, 2025), GameNGen — are commercially mature and technologically impressive, but the functional taxonomy gives the field a precise vocabulary for what they do and what they do not do. They produce observations, not state. They are not substitutes for Simulators in robotics, architecture, or scientific simulation.

---

## The bottlenecks that remain

Applying the functional taxonomy across the field reveals an asymmetric data landscape. Renderers are awash in internet video. Planners have access to large offline datasets from robot teleoperation and game playthroughs. Simulators face a fundamental scarcity: three-dimensional data with explicit geometry, material properties, and physical annotations is orders of magnitude rarer than the pixels that feed Renderers. This data gap is the single most important bottleneck in world model research.

The sim-to-real gap compounds the problem. A Simulator trained on synthetic or AI-generated geometry must contend with the discrepancy between simulation and reality. R2-Dreamer's results are impressive in simulation, but the method has not yet been validated on real robots. PointWorld has demonstrated real-world transfer, but its task horizon remains short.

A deeper architectural challenge is the tension between the three functions within a single model. A Renderer optimized for visual beauty may sacrifice the precision that a Simulator or Planner requires. A Planner optimized for task success may learn representations that are not visually interpretable. Reconciling these tensions inside a single architecture is the open problem that World Labs has identified, and it is the problem that Marble and R2-Dreamer approach from opposite directions — Marble from the Simulator–Renderer boundary, R2-Dreamer from the Simulator–Planner boundary.

---

## The convergence ahead

The most important pattern in the field today is that the three categories are beginning to collapse into one another. Marble already outputs Gaussian splats and collision meshes from a single model, dissolving the Renderer–Simulator boundary. A small but growing body of work from robotics laboratories has demonstrated that a pretrained video Renderer can serve as the backbone for joint world-and-action prediction, suggesting a bridge between Renderer and Planner. R2-Dreamer has shown that the Simulator can be optimized independently of the Renderer.

The logical endpoint is a unified world model: one foundation model that can render photorealistic views, produce physically accurate structure, and plan action sequences, switching between output modalities depending on what the downstream application needs. The shared insight across all of these converging threads is that the knowledge required to render a world, simulate its dynamics, and act within it is substantially the same knowledge.

That convergence has been the field's quiet assumption since the late 1980s, when recurrent neural network world models first appeared. It is now being tested at scale by multiple independent research programmes. The Dreamer series has shown what a Simulator–Planner system can do. R2-Dreamer has shown that the Simulator can operate without a Renderer. Marble has shown that the Renderer–Simulator boundary is fluid. The remaining step — a single system that spans all three functions with equal fidelity — will require breakthroughs in three-dimensional data acquisition, architectures that can balance visual and structural objectives, and evaluation benchmarks that measure all three capabilities jointly.

The term "world model" has been diluted by overuse. The functional taxonomy gives the field a tool to reclaim its precision. The convergence that is now underway gives the field a reason to want it back.

---

**Terminology Ledger (locked)**

| Canonical term | Definition | First-use form |
|---|---|---|
| World model | A learned model of an environment's dynamics within the POMDP loop | Partially Observable Markov Decision Process (POMDP) |
| Functional taxonomy | Three-category classification: Renderer, Simulator, Planner | — |
| RSSM | Recurrent State-Space Model | Recurrent State-Space Model (RSSM) |
| Symlog | Sign-symmetric logarithmic transform | — |
| R2-Dreamer | Redundancy-Reduced Dreamer | — |

**Assumptions or missing inputs:**

1. The World Labs taxonomy (June 2026) is treated as the canonical framework. The full blog post is available at https://worldlabs.ai/blog/taxonomy-of-world-models.
2. Technical details of DreamerV1/V2/V3 and R2-Dreamer are drawn from the published arXiv papers cited. No unreleased results are referenced.
3. Claims about Sora's limitations cite OpenAI's own technical report.

**Claim-evidence map**

| Claim | Evidence | Status |
|---|---|---|
| Most self-described world models cover only one function | Sora, Genie, GATO, GameNGen analysed against taxonomy | Supported by paper descriptions and author statements |
| Dreamer series is the only family covering both Simulator and Planner | V1 (ICLR 2020), V2 (ICLR 2021), V3 (2023) | Supported by published architectures |
| R2-Dreamer operates without decoder and matches V3 performance | ICLR 2026 paper (arXiv:2603.18202) | Supported by published results |
| Simulator can be optimized independently of Renderer | R2-Dreamer results, PointWorld results | Supported |
| Three functions are converging | Marble, robotics video-renderer works, R2-Dreamer | Trend-level evidence, partial |

**Why this structure**

1. **POMDP-first opening.** The Nature audience needs the conceptual framework before the taxonomy. The POMDP loop bridges reinforcement learning, robotics, and generative AI into one language.
2. **Negative examples before the Dreamer series.** Placing Sora, Genie, and GATO first establishes what a world model is not, making the Dreamer series' contribution sharper.
3. **Architecture-to-evidence ordering within the Dreamer section.** Each generation is presented as a motivated architectural decision followed by the evidence it produced, so the non-specialist reader understands why each change mattered.
4. **R2-Dreamer as the pivot.** The most recent result is positioned not as an incremental improvement but as a test of the taxonomy's core prediction, linking back to the framework and forward to the convergence narrative.
5. **Bounded conclusion.** The closing section states the convergence trend honestly, without promising that the unified model is imminent.

**To redirect me:** Name the paragraph or claim that is off and I will revise only that, keeping the rest.
