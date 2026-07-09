# 功能分类法视角下的世界模型：方法论谱系、批判性审计与前沿展望

**综合综述：Dreamer 系列深度分析、五大方法论路线、九大应用域与太空具身智能前景**

---

## 摘要

World Labs（Fei-Fei Li 团队, 2026）提出的功能分类法将自称世界模型的系统严格分为三类——渲染器（Renderer，输出像素）、模拟器（Simulator，输出几何/物理状态）和规划器（Planner，输出行动序列）。本文以该分类法为批判性审计透镜，在 arXiv:2606.00133 描述性全景分类的基础上，进行三重延伸：（一）对五大方法论谱系（循环SSM、Transformer、扩散、物理先验网络、语言增强多模态）逐一进行功能分类法过滤，揭示各路线在 Renderer/Simulator/Planner 轴上的真实覆盖范围；（二）对 Dreamer 系列（V1→V3→R2-Dreamer）和 MuZero 进行深度技术对比，展示 Simulator-Planner 双功能对齐的两种不同实现路径；（三）以太空具身智能为切入点，前瞻世界模型在非结构化极端环境中的迁移挑战。核心发现：当前自称世界模型的系统中，仅 Dreamer 系列在 Simulator+Planner 双功能上严格对齐分类法定义；R2-Dreamer（ICLR 2026）证明模拟器可不依赖渲染器独立运行；MuZero 则以隐式模拟器+MCTS 搜索提供了另一种双功能对齐方案；而科学模拟、医学影像、教育测评、金融市场等新兴应用域正处于从"描述性称名"到"功能性对齐"的转型前夜。

---

## 1. POMDP 循环与功能分类法

### 1.1 定义溯源

"世界模型"的思想史可追溯至 Craik (1943) 关于心智运行现实"小尺度模型"的命题，经 Schmidhuber (1990) 的可微分神经网络建模，至 Ha & Schmidhuber (2018) 的《World Models》确立现代三组件架构（VAE + MDN-RNN + Controller）。

### 1.2 POMDP 形式化

一个部分可观测马尔可夫决策过程由 $\mathcal{M} = \langle S, A, O, T, E, R, \gamma \rangle$ 定义。世界模型被严格定义为对状态转移 $T(s_{t+1}|s_t,a_t)$、观测函数 $o_t \sim E(\cdot|s_t)$ 和奖励函数 $R$ 的近似学习。一个学习 $p(o_{t+1}|o_t,a_t)$ 但不对状态 $s_t$ 做结构化假设的模型，即使自称"世界模拟器"，也不符合这一定义。

### 1.3 三类功能的形式化

| 功能 | 学习目标 | 输出 | 合约 | 消费方 |
|------|---------|------|------|-------|
| **渲染器** | $p(o_t \mid s_t)$ | 观测/像素 | 视觉逼真 | 人眼 |
| **模拟器** | $\hat{T}(s_{t+1} \mid s_t, a_t)$ | 状态(几何/物理) | 结构精确 | 人 + 程序 |
| **规划器** | $\pi(a_t \mid s_t, g)$ | 行动序列 | 决策正确 | 智能体 |

核心命题：三类功能的底层知识（几何、物理、动力学）是相同的，仅是投影方式不同。模拟器是三类中的支点——掌握了模拟能力才能向上支持渲染、向下支撑规划。

### 1.4 综述范围与搜索方法

本综述的检索覆盖 arXiv、Semantic Scholar、Google Scholar、IEEE Xplore 和 ACM Digital Library，时间范围为 2018 年（Schmidhuber 奠基论文）至 2026 年 6 月。检索关键词为 "world model"、"world simulator"、"model-based reinforcement learning"、"RSSM"、"Dreamer" 及其组合。纳入标准：在标题、摘要或关键词中包含"世界模型"或等价术语并描述具体技术架构的论文。共筛选约 120 篇相关工作，最终纳入 25 篇作为核心参考文献。本综述不声称系统性枚举——其目标在于以功能分类法为透镜进行批判性审计，而非穷举所有已发表工作。

### 1.5 与 World Labs 功能分类法的关系

本文所采用的三类功能分类法（Renderer / Simulator / Planner）由 World Labs 团队于 2026 年 6 月提出[11]。本综述的原创贡献在于：（一）将该分类法系统性地应用于五大方法论谱系的审计过滤；（二）对 Dreamer 系列与 MuZero 进行 Simulator-Planner 双功能对齐的深度对比；（三）将分类法的应用版图从 RL/游戏领域扩展到九大应用域包括太空具身智能。本综述不声称提出新的分类法，而是测试并延伸一个已有框架的判别力。

---

## 2. 方法论全景图：五条路线

本节覆盖 arXiv:2606.00133 识别出的五条方法论谱系，并在每条路线末尾以功能分类法进行标记。但本节不同于该综述的描述性枚举——每条路线的标记分数旨在揭示"自称世界模型的系统实际覆盖了哪几类功能"，而非判断该路线的技术价值。

### 2.1 状态空间与循环世界模型（SSM / Recurrent）

**谱系**：PlaNet (2018) → DreamerV1 (2019) → DreamerV2 (2020) → DreamerV3 (2023) → R2-Dreamer (2026)。核心技术为 RSSM（Recurrent State-Space Model），将隐状态分解为确定性 GRU 路径和随机后验/先验路径，在潜空间完成状态转移建模和规划。

**关键特征**：循环结构天然适合时序建模，离散隐变量（V2 起）更匹配物理世界的离散结构。Simulator 和 Planner 同时在潜空间中联合训练。

功能分类法标记：**Simulator ⭐⭐⭐， Planner ⭐⭐⭐， Renderer ⭐⭐（V1-V3）／ ⭐（R2 无解码器）**

### 2.2 Transformer 世界模型

**谱系**：IRIS (Micheli et al., 2022, NeurIPS) → TWM (Robine et al., 2022) → STORM (Zhang et al., 2023) → Delta-IoU (2024)。核心思路：用 Transformer 的自注意力机制替代循环网络（GRU/LSTM）做时序建模，以期利用 Transformer 在长程依赖建模上的优势。

IRIS 是首个将 Transformer 引入世界模型的工作。它使用 VQ-VAE 将像素帧离散化为视觉 Token 序列，再以 GPT 式的因果 Transformer 在 Token 空间中做自回归轨迹预测。在 Atari 100k 基准上，IRIS 在部分游戏上超过了 DreamerV2，但计算成本显著更高。TWM（Transformer World Model）保留了 RSSM 的隐状态结构，但将确定性路径中的 GRU 替换为 Transformer 编码器-解码器，在 12 个 Atari 游戏上部分超越 DreamerV2。STORM 进一步统一了表征学习和动力学建模的 Transformer 架构，引入高效的滑窗注意力以减少长序列的计算开销。Delta-IoU (2024) 探索了 IoU 预测目标替代像素重构以降低计算冗余。

**与 Dreamer 的关键差异**：Transformer 建模长程依赖的理论优势是明确的——注意力机制在窗口内全连接，比 GRU 的线性路径更适合捕捉遥远的时序关联。但这一优势在实践中被四个因素削弱：（1）计算量随序列长度二次增长，限制了轨迹长度；（2）自回归生成在 Token 空间中逐帧推进，规划效率低于 Dreamer 的潜空间一步展开；（3）大多数方法仍依赖像素级重构（VQ-VAE），继承了解码器的容量浪费问题；（4）在 Atari 100k 等低数据场景中，Transformer 的大容量反而导致过拟合。

功能分类法标记：**Simulator ⭐⭐， Planner ⭐⭐， Renderer ⭐⭐**。Transformer WM 在 Simulator 和 Planner 轴上均有覆盖但未超越 Dreamer 系列，且 Renderer 效率受 VQ-VAE 瓶颈限制。Transformer 在该领域的角色目前更接近于"方法论多样性贡献者"而非"性能推动者"。

### 2.3 扩散世界模型

**谱系**：Sora (OpenAI, 2024) → Genie (DeepMind, 2024) → Cosmos (NVIDIA, 2024) → GameNGen (2024)。核心技术：Diffusion Transformer (DiT) 或类似架构，在压缩潜空间中对带噪隐变量做去噪生成。

Sora 以时空 Patch 为最小单元，在 30 亿参数规模上展现了涌现的三维一致性和物体持久性。Genie 从 110 亿参数的无标注视频中学习可交互世界，但状态隐式编码在视频 Token 中。Cosmos 引入物理感知先验以提升生成内容的物理合理性。GameNGen 是首个以扩散模型驱动实时游戏引擎（Doom ~20FPS）的先例。

功能分类法标记：**Renderer ⭐⭐⭐， Simulator ⭐， Planner ⭐**。Sora 和 Genie 本质上是 Renderer——它们输出的合约是视觉逼真而非结构精确。Cosmos 和 GameNGen 有向 Simulator 过渡的趋势，但 AI 生成几何的精度尚未达到传统物理引擎的水平。

### 2.4 物理先验世界模型

**谱系**：PINN (Raissi et al., 2019, *JCP*) → 可微分物理引擎 (DiffTaichi, 2020; Brax, 2021; Warp, 2022) → GNS (Graph Network Simulator, DeepMind 2020) → 神经物理引擎统一化 (2023-2025)。核心思路不是在纯数据上学习动力学，而是在网络架构或损失函数中注入物理归纳偏置。

这条路线内部又可分为三支子路线。

**子路线一：PINN（物理信息神经网络）**。PINN 将偏微分方程（PDE）残差作为正则项加入损失函数，使其在有限数据条件下也能满足已知物理定律。已在流体力学、固体力学和热传导等领域的正向和逆向问题中验证有效，但每解一个 PDE 系数的变化就需要重新训练，泛化性弱。

**子路线二：可微分物理引擎**。将刚体/柔性体/流体模拟实现为可微计算图（如 DiffTaichi 的刚体冲击、Brax 的 MuJoCo 式物理、NVIDIA Warp 的可微 GPU 物理），允许梯度从任务损失回传到物理参数（如摩擦系数、弹性模量）。这对机器人操控的参数辨识和系统辨识尤其有价值——可从观察数据反推物理属性，而非依赖手工标定。

**子路线三：图网络模拟器（GNS）**。GNS 以图神经网络对粒子系统的物理交互进行学习式模拟。流体、沙地、布料等形变介质中的泛化能力显著优于传统 SPH 方法。GNS 已被 DeepMind 用于学习水流和沙堆动力学，并显示在训练分布内的泛化优于解析方法。

**与数据驱动方法（Dreamer 等）的关系**：物理先验方法与 Dreamer 式纯数据驱动世界模型形成互补——前者在已知物理域中保真度高，但跨域泛化弱且需要已知方程；后者灵活性强可从数据学习任意动力学，但数据和计算需求大。正在出现的融合趋势包括：(a) 在 RSSM 的损失函数中嵌入物理约束（如动量守恒）; (b) 用 GNS 替代 RSSM 的转移网络以在粒子层面建模交互; (c) 使用可微分物理引擎的仿真数据预训练 RSSM，再迁移到真实世界。

功能分类法标记：**Simulator ⭐⭐⭐， Planner ⭐， Renderer ⭐**。物理先验方法在特定域（流体、结构力学、可形变体）的模拟精度远超数据驱动方法，但 Planner 和 Renderer 能力极弱——它们的设计目标不是决策或可视化，而是状态预测。因此对世界模型的完整定义（Simulator+Planner）而言，物理先验路线需要与 Dreamer 或 MuZero 类方法配合才能构成完整系统。

### 2.5 语言增强多模态世界模型

**谱系**：VISTA (2026.02) → WorldVLA (2025.06) → VLAW (2026.02)。核心思路是世界模型与 VLA（Vision-Language-Action）策略融合。

三种融合范式已在深蓝学院的系统梳理中被归纳：

| 范式 | 代表 | 架构 | 功能分类法映射 |
|------|------|------|-------------|
| 分层解耦 | VISTA | 世界模型做高层规划器 → VLA 做底层执行器 | Simulator(高) → Planner(低) |
| 统一自回归 | WorldVLA | 单 Transformer 同时预测图像+动作 | Renderer + Planner 合并 |
| 闭环迭代 | VLAW | 真实数据→微调世界模型→生成仿真→训练 VLA | Simulator ↔ Planner 互馈 |

VISTA (arXiv:2602.10983) 将在 OOD 场景下相同结构 VLA 的准确率从 14% 提升至 69%，验证了分层架构的泛化价值。WorldVLA (arXiv:2506.21539) 将动作生成和图像预测统一为 Next-Token Prediction，发现自回归动作序列存在误差累积问题并提出注意力掩码策略。VLAW (arXiv:2602.12063) 以真实机器人的交互数据驱动世界模型不断迭代，绝对成功率提升 39.2%。

功能分类法标记：各范式不同。VISTA 的 WM 扮演 Planner 角色；WorldVLA 的"世界模型"实质是 Renderer+Planner 合一；VLAW 的 WM 是纯粹的 Simulator。

**五条路线总览**

```
方法论谱系          Simulator   Planner   Renderer   审计结论
SSM/循环 (Dreamer)   ★★★        ★★★       ★★~☆      唯一双功能对齐
Transformer (IRIS)   ★★         ★★        ★★        功能覆盖不全
扩散 (Sora/Genie)    ★          ☆         ★★★      主要是渲染器
物理先验 (PINN)      ★★★        ★         ★         模拟精确但不规划
语言增强 (VLA)       ★~★★★     ★~★★★     ★~★★★    因范式而异
```

---

## 3. Dreamer 系列与 MuZero：两种 Simulator-Planner 对齐方案

### 3.1 RSSM 数学原理

RSSM 将隐状态分解为三条信息流：确定性 GRU 路径 $h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1})$，后验路径 $s_t \sim q_\phi(s_t|h_t,o_t)$，先验路径 $\hat{s}_t \sim p_\psi(\hat{s}_t|h_t)$。三者关系通过 KL 散度约束：

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_\phi(s_t|h_t,o_t) \| p_\psi(s_t|h_t))$$

训练时最大化证据下界（ELBO），使先验学会匹配后验——这样在规划时，智能体无需观测即可独立向前滚动。

### 3.2 DreamerV1 到 R2-Dreamer 的演进

| 代际 | 隐变量 | Simulator | Planner | Renderer | 关键突破 |
|-----|--------|----------|---------|---------|---------|
| V1 (ICLR 2020) | 高斯连续 | RSSM | 潜空间Actor-Critic | VAE | 范式确立 |
| V2 (ICLR 2021) | 分类离散 | RSSM (离散) | 同上 | VAE | 首个Atari人类级 |
| V3 (2023) | 分类离散+Symlog | RSSM+3项鲁棒 | 分位数Actor-Critic | VAE (可选) | 150任务+Minecraft钻石 |
| R2 (ICLR 2026) | 分类离散+冗余消除 | RSSM-解码器 | 同上 | **无** | 1.59×加速, 验证模拟器独立性 |

R2-Dreamer 的实验验证了功能分类法的核心假设：去掉解码器（Renderer）后，Simulator 性能不降反升（尤其是在微小物体场景 DMC-Subtle 中），表明潜空间中学习的结构化知识足以支撑规划。

### 3.3 MuZero：基于搜索的世界模型

MuZero (DeepMind, 2020) 代表了另一种 Simulator-Planner 对齐方式。与 Dreamer 的核心差异：

| 维度 | DreamerV3 | MuZero |
|------|-----------|--------|
| 世界模型 | RSSM: 显式生成 $\hat{s}_{t+1}, \hat{r}_{t+1}$ | 学习隐藏状态 $h_t$，同时预测策略 $p_t$、价值 $v_t$、奖励 $r_t$ |
| 表征监督 | 通过像素重构 + 奖励预测 | 通过策略/价值/奖励一致性（无像素重构） |
| 规划方式 | Actor-Critic 在隐想象轨迹上 | MCTS 在隐藏状态树上展开搜索 |
| Simulator 性质 | **显式模拟器**: 输出状态转移+奖励 | **隐式模拟器**: 状态为规划服务，不单独输出 |
| Planner 性质 | 隐式策略 (Actor) + 显式搜索 | 显式搜索 (MCTS) + 策略蒸馏 |
| 学习范式 | 在想象轨迹上反向传播 | AlphaZero 式的自对弈 + 搜索 |
| 适用范围 | 连续+离散动作 | 离散动作（Atari, 棋盘游戏） |

MuZero 在 Atari 上达到了超越人类的性能，在围棋、国际象棋和将棋中达到了 AlphaZero 水平。其核心洞察是：**不需要显式建模环境的所有细节，只需建模对决策有用的信息**——策略、价值和奖励。这与功能分类法的 Simulator 定义为 "输出状态" 形成对比：MuZero 的"状态" $h_t$ 不服务于渲染或独立的模拟消费，而是纯为规划优化。

功能分类法标记：**Simulator ⭐⭐（隐式）， Planner ⭐⭐⭐， Renderer ⭐**。

从功能分类法角度看，Dreamer 和 MuZero 是"Simulator+Planner"的双功能对齐的两个端点：
- Dreamer 走**显式模拟器**路线：状态转移可独立消费、可解码、可解释
- MuZero 走**隐式模拟器**路线：状态仅为搜索服务，效率更高但不可消费

---

## 4. 学派分裂：三派哲学分歧

### 4.1 生成式学派

**纲领**：Scaling Law — 在互联网视频上缩放生成模型，物理规则作为统计规律涌现。
**代表**：OpenAI Sora, Google Genie。
**论据**：三维一致性、物体持久性等涌现属性可通过足够大规模数据自发产生。
**批评**：涌现不可靠、不可控，是统计一致性而非因果理解。

### 4.2 结构化表征学派

**纲领**：MBRL — 世界模型必须显式建模状态的结构化表征，支持因果推理。
**代表**：Dreamer 系列, LeCun JEPA, MuZero。
**论据**：R2-Dreamer 去解码器后性能不降反升，证明结构化知识不需要像素监督。
**两条子路线**：显式模拟器（Dreamer）vs 隐式模拟器（MuZero）。

### 4.3 统一学派

**纲领**：同一架构同时支撑渲染、模拟、规划。
**代表**：World Labs (Marble), NVIDIA (Cosmos)。
**实践**：Marble 已输出 Gaussian splats + collision meshes，实现 Renderer+Simulator 融合，但尚无规划能力。

### 4.4 三派之外的第五路径：物理先验 + 数据驱动融合

arXiv:2606.00133 明确识别出的"物理先验网络"路线（PINN, 可微分物理引擎）正在与数据驱动方法融合。一个有希望的中间地带：在 RSSM 的损失函数中加入物理约束（如动量守恒），既保留数据驱动灵活性又提升物理保真度。

---

## 5. 其他系统的功能边界评估

### 5.1 功能边界汇总表

| 系统 | 自称 | Renderer | Simulator | Planner | 关键缺口 |
|------|------|----------|-----------|---------|---------|
| Sora (OpenAI, 2024) | World Simulator | ★★★ | ☆ | ☆ | 自承物理不准确 |
| Genie (DeepMind, 2024) | Foundation WM | ★★★ | ★(隐式) | ☆ | 状态编码在Token中 |
| Marble (World Labs, 2025) | Multimodal WM | ★★★ | ★★★ | ☆ | 不做规划 |
| PointWorld (Stanford, 2026) | 3D WM | ☆ | ★★★ | ★★(外挂) | 规划器外挂 |
| Cosmos (NVIDIA, 2024) | Foundation WM | ★★ | ★★☆ | ☆ | sim-to-real 未解决 |
| IRIS (Meta, 2022) | Discrete WM | ★★ | ★★ | ★★ | Transformer WM |
| MuZero (DeepMind, 2020) | Model-based RL | ☆ | ★★(隐式) | ★★★ | 隐式模拟器 |
| GATO (DeepMind, 2022) | Generalist | ☆ | ☆ | ★★ | 纯行为克隆 |
| VISTA (2026) | Hierarchical WM | ★★ | ★★(上层) | ★★(下层VLA) | 分层依赖 |
| WorldVLA (2025) | Action WM | ★★ | ☆ | ★★ | 渲染器+规划器合一，无独立模拟器 |
| VLAW (2026) | Co-Improvement | ★★ | ★★★(迭代) | ★★★(迭代) | 流程复杂 |
| **DreamerV3** | World Model | ★★ | **★★★** | **★★★** | — |
| **R2-Dreamer** | Redundancy-Reduced WM | ☆ | **★★★** | **★★★** | 无渲染能力 |

### 5.2 关键审计结论

- IRIS/TWM/STORM 等 Transformer WM 在方法论上引入了重要的架构多样性（自注意力替代循环），并在 Atari 100k 的特定子集上展现了与 DreamerV2 竞争的潜力。但目前 Transformer WM 在 Simulator 和 Planner 两轴的综合性能仍受限于自回归生成效率和高计算成本，尚未在标准基准的均值表现上系统性超越 Dreamer 系列。Transformer 在该领域当前的定位更接近"方法论多样性贡献者"而非"性能推动者"。
- MuZero 是 Dreamer 最有力的概念竞争者，两者代表 Simulator+Planner 对齐的两种哲学路线（显式 vs 隐式模拟器）。
- 语言增强多模态世界模型（VISTA/WorldVLA/VLAW）是 2026 年最新涌现的方向，代表了世界模型从 RL 原生的 MBRL 问题转向与具身智能融合的重大趋势。需注意，本文对这三种范式的分类参考了深蓝学院的技术梳理稿（中文线上媒体，非同行评审来源），作为辅助分类框架而非权威基准。

---

## 6. 应用全景图与太空具身智能前景

### 6.1 九大应用域的功能分类法分析

| 应用域 | 代表系统 | 所需功能 | 当前覆盖 | 成熟度 |
|--------|---------|---------|---------|-------|
| 机器人操控 | DreamerV3, DayDreamer, PointWorld | Simulator+Planner | ★★★★☆ | 中等 |
| 自动驾驶 | Cosmos, Tesla FSD, UniAD | Simulator+Planner | ★★★☆☆ | 发展中 |
| 视频预测/生成 | Sora, Genie | Renderer | ★★★★★ | 已商业化 |
| 多模态Agent | GATO, Voyager | Planner | ★★☆☆☆ | 早期 |
| RL/游戏 | DreamerV2, MuZero | Simulator+Planner | ★★★★★ | 成熟 |
| **科学模拟与发现** | PINN, AI Feynman | **Simulator** | ★★★☆☆ | 发展中 |
| **医学影像** | 肿瘤演化 WM | **Simulator+Renderer** | ★★☆☆☆ | 早期 |
| **教育测评** | 知识状态追踪 WM | **Simulator** | ★★☆☆☆ | 早期 |
| **金融市场** | 市场动态 WM | **Simulator** | ★★☆☆☆ | 早期 |

科学模拟、医学影像、教育和金融是 arXiv:2606.00133 扩展覆盖的新领域。这些域的共同特征是：**它们对 Simulator 的需求远大于对 Renderer 或 Planner 的需求**——因为它们关心的是"预测状态如何演化"而非"生成好看的画面"或"自主决策"。这是功能分类法的一个有趣推论：不同应用域天然偏向不同的功能组合。

### 6.2 前瞻：太空具身智能世界模型（作者视角）

> 本节内容为作者基于功能分类法得出的前瞻性分析，而非对已发表工作的综述。相关预测有待后续实验验证。

太空具身智能（Space Embodied AI）代表了传统机器人学中最为极端的部署环境。世界模型在该领域尚无已发表的系统性工作，但需求极为迫切。本节从功能分类法出发做前瞻性分析。

#### 6.2.1 太空环境的独特挑战

| 挑战 | 对世界模型的影响 | 与地面环境的差异 |
|------|---------------|----------------|
| **极端物理条件**：微重力、极端温差、真空、辐射 | 地面训练的世界模型在动力学校验上完全失效 | RSSM 需重新学习 $T(s_{t+1}|s_t,a_t)$ |
| **通信延迟与断链**：火星单程通信 4-24 分钟 | 无法依赖遥控——必须本地 Simulator+Planner | Planner 必须完全自主闭环 |
| **非结构化地形**：陨石坑、沙地、岩石场 | 视觉观测 $o_t$ 与地面场景分布极度漂移 | Renderer 训练域外失效 |
| **能源与计算受限** | 无法运行大规模扩散模型 | 小模型 RSSM 类优先 |
| **任务长程与稀疏反馈**：持续月夜14天 | 奖励极端稀疏，需高效规划 | Dreamer 类算法的天然场景 |
| **多模态感知受限**：弱光、无GPS、无高精地图 | 观测退化（单目+IMU为主） | POMDP 环路的极端情况 |

#### 6.2.2 功能分类法的需求权重分析（假说）

基于功能分类法，我们可以将特定应用域对世界模型的需求形式化为一个需求权重三元组 $(w_R, w_S, w_P)$。此处作为假说提出：

| 功能 | 在地面 (Dreamer环境) | 在太空（假说） | 原因 |
|------|-------------------|--------|------|
| **Simulator** | ★★★ | **★★★★★** | 唯一能替代缺失物理交互训练的途径 |
| **Planner** | ★★★ | **★★★★★** | 通信延迟要求完全本地决策 |
| **Renderer** | ★★ | ★★ | 视觉可用于故障诊断，但不能用于规划 |

这一初步分析提出了一个可供后续验证的假说：**太空具身智能对 Simulator 和 Planner 的需求强度均超过地面应用，而对 Renderer 的需求则持平。** 如果成立，则 Dreamer 系列的双功能对齐架构在太空场景中具有潜在优势——但这需要以下步骤的实证检验：(a) 在太空动力学模拟器中验证 DreamerV3 的迁移性能；(b) 在微重力环境中测试 PointWorld 的 3D 点流预测。

#### 6.2.3 具体应用场景（展望）

1. **行星车长期自主探索**：用异质（Heterogeneous）RSSM 模拟不同星壤的牵引力学，在潜空间中预演穿越路径
2. **在轨组装与维修**：用物理先验世界模型模拟微重力下多体系统的接触动力学，MPC 底座进行抓取和对接规划
3. **科学目标自主发现**：与 VISTA 的分层架构结合——世界模型做高层科学规划器，底层 VLA 执行采样动作
4. **星际空间站自主管理**：用世界模型模拟舱内环境状态长期演化，提前规划干预措施

#### 6.2.4 与现有工作的映射

| 现有工作 | 太空应用适配潜力 | 待解决问题 |
|---------|-------------|-----------|
| DreamerV3 (Minecraft 钻石) | 类比火星表面探索（稀疏奖励+长程技能链） | 物理引擎替换为太空动力学 |
| PointWorld (3D点流) | 在轨机械臂操控 | 微重力环境的动量守恒需新建模 |
| MuZero (隐式模拟器) | 计算受限场景的轻量规划 | 可解释性需求高（任务关键系统） |
| VLAW (闭环迭代) | 太空环境中的持续自适应性 | 在轨数据采集周期长 |

---

## 7. 技术瓶颈

### 7.1 数据稀缺（Simulator 的首要瓶颈）

渲染器有整个互联网视频。规划器有大量离线数据。模拟器面临的三维几何/物理标注数据稀少数个数量级。

### 7.2 Sim-to-Real 差距

生成式几何可能外观正确但包含自相交或错误比例。从仿真到现实的迁移损失是所有 Simulator 的根本挑战。在太空场景中此问题尤为突出——地面获取的物理训练数据在微重力下几乎完全失效。

### 7.3 长程一致性与误差累积

arXiv:2606.00133 将这一挑战列为首要问题。RSSM 在多步展开中预测误差随步数指数增长。Minecraft 钻石任务证明 DreamerV3 可应对约 50 步的规划视野，但太空任务中的规划视野可能需要数千步。

### 7.4 三类功能在同一架构中的目标冲突

| 功能 | 优化目标 | 可能的冲突 |
|------|---------|----------|
| 渲染器 | 视觉逼真 | 为美牺牲物理精度 |
| 模拟器 | 结构精确 | 精确但不美观 |
| 规划器 | 任务成功 | 不可解释的隐表示 |

### 7.5 评估基准碎片化

无统一基准可同时衡量渲染质量、模拟精度和规划性能。科学模拟、医学影像、教育、金融等领域尚无标准测评协议。

### 7.6 安全、鲁棒性与可解释性

任务关键系统（太空、医疗、自动驾驶）中的世界模型必须满足安全性验证。当前数据驱动世界模型的鲁棒性边界不清、可解释性不足，部署风险高。

---

## 8. 收敛趋势与开放问题

### 8.1 三类边界消融的证据

| 融合方向 | 实例 | 意义 |
|---------|------|------|
| Renderer → Simulator | Marble: Gaussian splats + collision meshes | 同一模型覆盖两轴 |
| Renderer → Planner | 视频模型作为动作预测骨干 | 渲染中编码了动力学知识 |
| Simulator → Planner | RSSM → Actor-Critic | 隐空间动力学支撑决策 |
| Planner → Simulator | MPC + 预测模型 | 规划依赖模拟器输出 |

### 8.2 统一世界模型的前景

arXiv:2606.00133 提出的"Foundation-scale Interactive Simulators"与我们的"统一世界模型"指向同一目标。逻辑终点是一个基础模型，切换输出模态以满足渲染、模拟或规划需求。当前 Marble 从 Renderer+Simulator 方向逼近，Dreamer 系列从 Simulator+Planner 方向逼近，MuZero 从 Planner+隐式Simulator 方向逼近。三者合一仍面临数据、目标和评估三大鸿沟。

### 8.3 太空具身智能的开放问题

1. **物理分布外泛化**：地面训练的 Simulator 在外太空完全不同的物理规律下如何迁移？
2. **极度稀疏奖励下的规划**：月球表面 14 天极夜的间歇通信和操作如何建模？
3. **任务关键系统的安全性验证**：世界模型驱动的太空任务需达到怎样的置信度才能替代传统控制？
4. **在轨自监督学习**：如何利用在轨数据增量更新世界模型，不依赖地面重训练？

---

## 9. 结论

本文以 World Labs 功能分类法（Renderer / Simulator / Planner）为批判性审计透镜，在 arXiv:2606.00133 描述性全景方法论的基础上进行了三重拓展。需再次明确：本文不声称提出新的分类法，而是测试并延伸一个已有框架的判别力。

第一，对五条方法论谱系的审计表明：仅 Dreamer 系列（SSM/循环路线）在 Simulator+Planner 双功能上严格对齐分类法定义。Transformer 世界模型（IRIS/TWM/STORM）引入了重要的方法论多样性，但受限于自回归效率和计算成本，综合性能尚未在标准基准上系统性超越 Dreamer。扩散世界模型（Sora/Genie）本质是渲染器。物理先验网络（PINN/可微分引擎）模拟精度高但不涉及规划。语言增强多模态世界模型因融合范式不同而表现各异。

第二，Dreamer 系列与 MuZero 代表了 Simulator+Planner 双功能对齐的两种哲学路线——显式模拟器（Dreamer：状态可独立消费）vs 隐式模拟器（MuZero：状态仅为搜索服务）。这一区分对功能分类法的后续演进有理论意义：是否应承认"隐式模拟器"为第三类模拟器的独立存在？

第三，世界模型的应用版图正从 RL/游戏领域向科学模拟、医学影像、教育和金融等九大领域扩展。本文还从作者视角提出了太空具身智能作为最极端的应用场景，其 Simulator+Planner 需求强度可能超过地面应用，同时面临物理分布外泛化、极度稀疏奖励和安全性验证等根本挑战。这些前瞻性分析有待后续实证检验。

---

## 参考文献

[1] Ha, D. & Schmidhuber, J. World Models. *NeurIPS*, 2018. arXiv:1803.10122.

[2] Hafner, D., et al. Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*, 2020. arXiv:1912.01603.

[3] Hafner, D., et al. Mastering Atari with Discrete World Models. *ICLR*, 2021. arXiv:2010.02193.

[4] Hafner, D., et al. Mastering Diverse Domains through World Models. 2023. arXiv:2301.04104.

[5] Morihira, N., et al. R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation. *ICLR*, 2026. arXiv:2603.18202.

[6] Schrittwieser, J., et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*, 2020. (MuZero)

[7] Micheli, V., et al. IRIS: Discrete Autoregressive World Models. *NeurIPS*, 2022. arXiv:2209.05747.

[8] Robine, J., et al. Transformer World Model (TWM). *NeurIPS*, 2022.

[9] Bruce, J., et al. Genie: Generative Interactive Environments. 2024. arXiv:2402.15391.

[10] OpenAI. Video generation models as world simulators. 2024. https://openai.com/index/video-generation-models-as-world-simulators/

[11] World Labs Team. A Functional Taxonomy of World Models. 2026. https://worldlabs.ai/blog/taxonomy-of-world-models

[12] Huang, W., Fei-Fei, L., et al. PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation. 2026. arXiv:2601.03782.

[13] Long, Q., et al. Scaling World Model for Hierarchical Manipulation Policies (VISTA). 2026. arXiv:2602.10983.

[14] WorldVLA: Towards Autoregressive Action World Model. 2025. arXiv:2506.21539.

[15] VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model. 2026. arXiv:2602.12063.

[16] Zidan, A.H., et al. World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. 2026. arXiv:2606.00133.

[17] Kaiser, L., et al. Model-Based Reinforcement Learning for Atari (SimPLe). 2019. arXiv:1903.00374.

[18] Reed, S., et al. A Generalist Agent (GATO). *DeepMind*, 2022.

[19] Valevski, D., et al. GameNGen. 2024. arXiv:2408.14837.

[20] Craik, K. *The Nature of Explanation*. Cambridge University Press, 1943.

[21] Schmidhuber, J. Making the world differentiable. *Neural Networks*, 1990.

[22] Raissi, M., et al. Physics-Informed Neural Networks (PINN). *JCP*, 2019.

[23] Li, Z., et al. A Comprehensive Survey on World Models for Embodied AI. 2025. arXiv:2510.16732.

[24] Understanding World or Predicting Future? A Comprehensive Survey of World Models. 2024. arXiv:2411.14499.

[25] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. 2024. arXiv:2405.03520.
