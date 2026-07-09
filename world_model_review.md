# 功能分类法视角下的世界模型：从生成到模拟的边界勘定

**Dreamer 系列及其替代方案综述**

---

## 摘要

World Labs（Fei-Fei Li 团队, 2026）提出了世界模型的功能分类法，将自称世界模型的系统严格分为三类：渲染器（Renderer，输出像素）、模拟器（Simulator，输出几何/物理状态）和规划器（Planner，输出行动序列）。以此分类法为分析透镜可以发现，当前市面上大多数自称"世界模型"的系统实际上只覆盖三类功能中的一种。相比之下，Dreamer 系列从其诞生起就同时承担了模拟器和规划器的功能，其最新迭代 R2-Dreamer 进一步证明模拟器可以在不需要渲染器的条件下有效运行——这一结果验证了功能分类法的一个核心预测。本文从 POMDP 第一性原理出发，深入分析 Dreamer 系列四代架构（V1→V2→V3→R2-Dreamer）的数学原理、技术跃迁与设计哲学，系统对比 Sora、Genie、Marble、PointWorld、GATO 等其他典型项目的功能边界，并揭示领域内 Renderer→Simulator→Planner 三类功能正在加速收敛的根本趋势。

---

## 1. POMDP 循环及其三种投影

### 1.1 概念溯源

"世界模型"一词的思想史可追溯至 Kenneth Craik 在 1943 年的命题：心智通过运行现实的"小尺度模型"来推理[16]。这一概念在 1980 年代末被引入神经网络领域——Schmidhuber 在 1990 年的《Making the World Differentiable》[17]中首次提出了通过可微分神经网络学习世界模型的框架。Ha 和 Schmidhuber 在 2018 年的奠基论文《World Models》[1]中正式确立了现代世界模型的三组件架构（VAE + MDN-RNN + Controller），首次在"梦境"中训练策略并转移回真实环境。

然而，"世界模型"已成为当代人工智能中最重要但也最被滥用的术语之一。计算机视觉、机器人学、强化学习和生成式 AI 各自声称在构建世界模型，各执一词。一个扩散模型可以生成视觉惊艳但物理上不可能的火焰，一个大语言模型可以从提示中即兴编出可玩的游戏，一个物理引擎可以忠实模拟燃烧——它们都被冠以同一名称。

### 1.2 POMDP 的形式化定义

部分可观测马尔可夫决策过程（POMDP）是理解世界模型技术含义的数学基础。一个 POMDP 由以下元组定义：

$$\mathcal{M} = \langle S, A, O, T, E, R, \gamma \rangle$$

其中：
- $s_t \in S$：不可直接观测的真实世界状态
- $a_t \in A$：智能体采取的行动
- $o_t \in O$：智能体接收到的观测，服从 $o_t \sim E(\cdot | s_t)$
- $T(s_{t+1} | s_t, a_t)$：状态转移概率
- $R(r_t | s_t, a_t)$：奖励函数
- $\gamma \in [0,1)$：折扣因子

智能体的目标：学习策略 $\pi(a_t | o_{\leq t}, a_{< t})$ 以最大化期望累积折扣回报 $\mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t]$。

在这个框架下，"世界模型"被严格定义为对**状态转移函数 $T$**、**观测函数 $E$** 和**奖励函数 $R$** 的近似学习。一个学习 $p(o_{t+1}|o_t,a_t)$ 但不对状态做任何结构化假设的模型，即使被称为"世界模拟器"，也不符合这一定义。

### 1.3 三类功能的严格形式化

World Labs 的功能分类法[8]在这一 POMDP 框架内识别出三种已被混淆的功能角色：

**渲染器（Renderer）** 学习观测模型 $p(o_t | s_t)$ 或直接的条件分布 $p(o_{t+1} | o_t, a_t)$。其输出是观测序列，合约是视觉逼真。数学上等价于带条件的视频生成模型。渲染器无需也不试图推断背后的状态 $s_t$。

**模拟器（Simulator）** 学习状态转移函数 $\hat{T}(s_{t+1} | s_t, a_t)$。其输出是状态（或状态表征），合约是结构精确——几何、物理、动力学必须在可计算层面上保持一致。模拟器同时服务两个消费端：人类（架构师、设计师）需要精度；程序（RL 智能体、机器人控制器）需要可交互的虚拟训练场。

**规划器（Planner）** 学习策略 $\pi(a_t | s_t, g)$，其中 $g$ 为目标。其输出是行动序列，合约是决策正确。规划器是渲染器的对偶：渲染器以行动为输入产生观测，规划器以观测为输入产生行动，闭合感知-行动循环。

三类功能的底层知识是相同的——几何、物理、动力学——只是投影方式不同。能使模型从任意角度渲染一个杯子的知识，原则上也应使其能模拟杯子被推后的物理因果链，并规划手去捡起杯子。

---

## 2. 渲染器的陷阱：被误称为世界模型的生成系统

POMDP 框架提供了一个精确的判据来分离真正的世界模型与仅借用其名的系统。本节分析三个最典型的案例。

### 2.1 Sora：视觉合约而非结构合约

OpenAI 的 Sora (2024) 以"Video generation models as world simulators"为题发布[7]。其技术核心：将视频压缩为潜空间 → 分解为时空 Patch (spacetime patches) → 用 Diffusion Transformer 生成。Sora 在视觉质量上达到了前所未有的水平，并展现出三维一致性、物体持久性等涌现属性。

然而，Sora 没有输出任何可以被物理引擎消费的结构化表示。它的合约 100% 是视觉的。OpenAI 自承"does not accurately model the physics of many basic interactions, like glass shattering"[7]。一个无法预测碰撞后果的模型不是在模拟世界——它只是在渲染一个合理的版本。Sora 在功能分类法中是一个输入条件化的渲染器，学习的是 $p(o_t | \text{text})$，而非 $p(s_{t+1} | s_t, a_t)$。

### 2.2 Genie：行动条件化的渲染器

Google DeepMind 的 Genie (2024)[6] 以 110 亿参数成为当时最大的"基础世界模型"。其三组件架构——时空视频分词器、自回归动力学模型、学习的潜在动作模型——支持用户逐帧控制生成内容。

Genie 核心创新是**无监督学习潜在动作空间**——从无标注互联网视频中自动发现"动作"的编码。但从分类法看，它的"状态"始终隐式编码在视频 Token 中，未以几何形式呈现。Genie 的功能合约是 action-conditioned rendering：给定一个潜在动作编码，生成下一帧。它没有输出可被物理引擎消费的结构化状态。Genie 是渲染器，不是模拟器。

### 2.3 GATO：孤立的规划器

DeepMind 的 GATO (2022) 代表了另一个极端。GATO 是一个通过行为克隆在数百个任务上训练的通用智能体[12]，输出行动序列。它不包含内部模拟器。GATO 是一个策略（policy），不是世界模型。

渲染器和孤立规划器都是真实且有价值的贡献。但它们不符合 POMDP 传统赋予"世界模型"的技术含义。

---

## 3. Dreamer 系列：唯一严格的 Simulator-Planner 双功能对齐

Dreamer 系列由 Danijar Hafner 及其合作者在 Google DeepMind 从 2019 年发展至今，是唯一在四代架构中始终保持对模拟器和规划器双重承诺的系统家族。

### 3.1 RSSM：循环状态空间模型的数学原理

Dreamer 系列架构的核心是 RSSM（Recurrent State-Space Model），它解决了部分可观测环境中隐状态学习的根本矛盾：环境同时包含确定性元素（需长期记忆）和随机性元素（需概率建模）。

RSSM 将隐状态分解为三条信息流：

```
确定性路径:  h_t = f_θ(h_{t-1}, s_{t-1}, a_{t-1})      # GRU 确定性更新
后验路径:    s_t ∼ q_φ(s_t | h_t, o_t)                   # 从观测推断
先验路径:    ŝ_t ∼ p_ψ(ŝ_t | h_t)                         # 不依赖观测的预测
```

其中 $f_θ$ 是门控循环单元（GRU），$q_φ$ 是后验编码器（从当前观测提取信息修正信念），$p_ψ$ 是先验预测器（仅靠历史预测未来）。RSSM 的关键设计选择是：确定性 GRU 的输出 $h_t$ 既作为后验 $q_φ$ 的条件，也作为先验 $p_ψ$ 的条件。这使得先验和后验的差异仅在于是否引入了当前观测 $o_t$ 的信息。训练时最小化两者之间的 KL 散度：

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_φ(s_t | h_t, o_t) \| p_ψ(s_t | h_t))$$

当先验能够准确预测后验时，智能体就可以放弃观测、仅靠先验在"梦境"中规划。

### 3.2 DreamerV1：潜在想象的范式确立

DreamerV1[2]（ICLR 2020）是第一代在潜在空间中同时完成世界模型学习和行为优化的系统。其训练循环分为三步：

**第一步：世界模型学习。** 在真实环境中收集轨迹 $\{(o_t, a_t, r_t)\}_{t=1}^T$，通过最大化证据下界（ELBO）训练 RSSM + 奖励预测器：

$$\mathcal{L}_{\text{WM}} = \sum_{t} \left[ -\ln p(o_t | s_t) - \ln p(r_t | s_t) + \beta \cdot D_{\text{KL}}(q_φ(s_t|h_t,o_t) \| p_ψ(s_t|h_t)) \right]$$

编码器从观测 $o_t$ 编码到 $s_t$，解码器重构 $\hat{o}_t$ 和 $\hat{r}_t$，RSSM 在潜空间做时序预测。

**第二步：行为学习。** 在 RSSM 生成的想象轨迹上执行 Actor-Critic：

$$\text{Actor: } \max_π \mathbb{E}_{p_ψ, π} \left[ \sum_{τ=t}^{t+H} \gamma^{τ-t} V(s_τ) \right]$$

$$\text{Critic: } \min_V \mathbb{E}_{p_ψ, π} \left[ \left( V(s_t) - \mathbb{E}_{p_ψ}[R_t + \gamma V(s_{t+1})] \right)^2 \right]$$

其中 $H$ 为想象视野（通常 15 步），$R_t$ 为预测奖励。

**第三步：环境交互。** 将学到的 Actor 策略 $π(a_t|s_t)$ 部署到真实环境，收集新数据。

在二十个连续控制任务上，DreamerV1 在数据效率和最终性能上超越了当时的无模型方法。

### 3.3 DreamerV2：离散化跃迁与其深远后果

DreamerV2[3]（ICLR 2021）的核心更改只有一个，但被证明具有决定意义：将 RSSM 的隐变量从连续高斯替换为**分类离散分布**：

$$V1: s_t \sim \mathcal{N}(\mu_t, \sigma_t)$$
$$V2: s_t \sim \prod_{k=1}^{K} \text{Cat}(\pi_t^k)$$

其中 $K=32$ 个分类变量，每个 $|\mathcal{C}|=32$ 个类别，共 $32^{32}$ 种理论组合，常用 $32 \times 32 = 1024$ 个 logits。

离散化的动机——物理世界的本质结构是离散的：物体的类型是离散的，事件边界是离散的，游戏状态切换是离散的。连续高斯分布假设隐变量属于各向同性的凸区域，但真实世界的状态空间可能具有复杂的拓扑结构。

离散化的实际收益触发了领域级突破：
- 在 55 个 Atari 游戏中达到人类水平均值，首个实现此成就的基于世界模型的智能体
- 超越了当时最先进的无模型方法 IQN 和 Rainbow
- 同一套代码可直接应用于连续控制任务（人形机器人行走）

更重要的一阶推论是：**模拟器的质量设定了规划器性能的上限**。DreamerV2 的成功本质上是因为离散潜空间更精确地捕捉了 Atari 游戏中离散状态转换的结构。

### 3.4 DreamerV3：通用算法的工程实现

DreamerV3[4]（2023）证明单一固定超参数配置可驾驭 150+ 任务——横跨 Atari、DM Control、灵巧操作（Dexterous Manipulation）和 Minecraft，其中奖励量级跨越五个数量级（0/1 二元到上万的连续值）。

三项鲁棒性技术在其通用性能中起关键作用：

**Symlog 变换：** 将奖励和回报按符号压缩到对数空间：$\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$。逆变换为 symexp。这使得网络输出层的尺度对不同量级的输入输出自动适应，消除了任务特定的奖励归一化。

**分位数回归（Quantile Regression）：** 在 Critic 中用分位数损失替代均方误差：

$$\mathcal{L}_{\text{QR}}(\tau, \hat{\tau}) = |\tau - \mathbb{I}(\hat{\tau} < \tau)| \cdot |\hat{\tau} - \tau| / \kappa(\tau)$$

分位数回归不做任何目标值分布假设，天然适应多模态和高偏态回报。

**双向 RMSNorm：** 对每个隐藏层同时施加 RMS 归一化和逆变换，使激活值在不同任务、不同数据统计量下保持稳定。

**Minecraft 钻石成就：** Minecraft 的稀疏奖励和深层技能树（砍树→木板→木棍→木镐→石镐→铁→钻石）此前无 RL 算法完成。DreamerV3 在约 100M 环境步后首次成功，证明了世界模型在长程规划中的价值。

在功能分类法下，DreamerV3 的双轴评分：
- Simulator ⭐⭐⭐：RSSM 预测状态转移 + 奖励 + 回合终止
- Planner ⭐⭐⭐：Actor-Critic 完全在想象轨迹中运行
- Renderer ⭐⭐：解码器存在但仅用于监督训练，非主要功能输出

### 3.5 R2-Dreamer：模拟器独立性的决定性检验

R2-Dreamer[5]（Morihira et al., ICLR 2026）检验了功能分类法的一个核心假设：模拟器是否可以脱离渲染器独立存在。

DreamerV3 的一个被广泛讨论的弱点是：解码器重构所有像素，包括大量任务无关区域（天空、背景纹理），浪费参数和计算。但去解码器面临的核心挑战是：没有像素重构的监督信号，什么是正确的隐空间表征？

R2-Dreamer 的解决方案是冗余消除（Redundancy Reduction）：

$$\mathcal{L}_{\text{RR}} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2$$

其中 $C$ 是不同数据增强视角下隐表征的互相关矩阵。第一项迫使对角元素接近 1（保留信息），第二项迫使非对角元素接近 0（消除冗余）。灵感来自 Barlow Twins 自监督学习。

关键差异：R2-Dreamer 不需要外部的数据增强——旧有去解码器方法（如 TD-MPC2）严重依赖数据增强的强度，限制了通用性。R2-Dreamer 的正则化器是内部的。

实验结果验证了假设：

| 指标 | DreamerV3 | R2-Dreamer | 差距 |
|------|-----------|-----------|------|
| 训练速度 | 1× | **1.59×** | +59% |
| DMControl 均值 | 基准 | 持平 | — |
| Meta-World 成功率 | 基准 | 持平或略优 | — |
| DMC-Subtle（微小物体） | 低 | **大幅超越** | 显著 |

DMC-Subtle 的结果尤其值得注意：微小物体场景中，解码器的像素重构浪费了大量容量在背景上。R2-Dreamer 的冗余消除直接跳过这一无关信息，从未知中提取任务相关结构。

这一发现有三个层次的含义：
1. **实践层面**：去解码器架构可以训练 1.59× 更快，性能持平或更好
2. **理论层面**：隐空间可以学到足够支撑规划的结构化知识，无需像素级监督
3. **分类法层面**：模拟器是独立于渲染器的可行模块，不是两类功能之间的被动桥梁

### 3.6 Dreamer 家族四代定量对比

| 维度 | DreamerV1 (2019) | DreamerV2 (2020) | DreamerV3 (2023) | R2-Dreamer (2026) |
|------|-----------------|-----------------|-----------------|-------------------|
| **隐变量类型** | 高斯连续 | 分类离散 (K=32, |C|=32) | 分类离散 + Symlog | 分类离散 + 冗余消除 |
| **Simulator** | RSSM (连续) | RSSM (离散) | RSSM + 三项鲁棒 | RSSM - 解码器 |
| **Planner** | Actor-Critic (潜空间) | Actor-Critic (潜空间) | Actor-Critic (分位数) | Actor-Critic (分位数) |
| **解码器** | 有 | 有 | 有 | **无** |
| **任务数** | 20 | 55 + 连续控制 | **150+** | DMC + Meta-World |
| **Minecraft 钻石** | ✗ | ✗ | **✓** | ✗ (未测试) |
| **最高会议** | ICLR 2020 | ICLR 2021 | — | **ICLR 2026** |
| **训练速度** | 1× | ~1× | ~1× | **1.59×** |
| **开源** | ✗ | TensorFlow | ✓ (JAX) | ✓ (PyTorch) |

---

## 4. 学派分裂：三种世界模型哲学

除了功能分类法提供的横向划分，该领域还存在三个深层的**纵向学派**分歧。它们关注的不是"输出什么"，而是"世界模型应该是什么"。

### 4.1 生成式学派（Scaling Law 纲领）

代表：OpenAI Sora、Google Genie、Meta Video Joint Embedding

主张：在互联网规模的视频数据上缩放生成模型，物理规则会作为统计规律**涌现**。Sora 团队的论文标题本身就表达了这一立场——"Video generation models as world simulators"。

核心论据：三维一致性、物体持久性、动物行为等涌现属性可以通过足够大的模型和足够多的数据自发产生，无需显式物理建模。

核心批评：涌现不可靠、不可控。Sora 可以生成一只行走的北极熊，但无法预测将一个杯子推下桌子的后果。涌现属性是**统计一致性**而非**因果理解**。

### 4.2 结构化表征学派（MBRL 纲领）

代表：Dreamer 系列、SimPLe、PILCO、LeCun JEPA

主张：世界模型必须显式建模状态的结构化表征——要么是 RSSM 式的潜在动力学，要么是 JEPA 式的抽象预测空间。LeCun 在其客观函数（Objective-Driven AI）框架中明确指出，纯粹在像素空间预测未来浪费计算，应该在抽象表征空间中预测。

核心论据：R2-Dreamer 的结果直接支持了这一立场——去掉像素重构后模拟器性能不降反升，证明任务相关的结构化知识不需要像素级监督来学习。

### 4.3 统一学派（World Labs 纲领）

代表：World Labs（Marble）、NVIDIA（Cosmos）

主张：渲染、模拟、规划共享同一知识基础，应统一在同一架构中。Marble 输出 Gaussian splats（视觉）+ collision meshes（物理），是 Renderer+Simulator 统一的首个商业实例。World Labs 团队明确将"three categories collapsing into one another"定义为领域的定义性趋势。

### 4.4 三派分歧的实质

三个学派的根本分歧不在于技术路线，而在于对"世界模型"的核心功能定位：

- 生成式学派：世界模型 ≈ 逼真的世界视频生成器
- 结构化表征学派：世界模型 ≈ 可用于控制的因果动力学模型
- 统一学派：世界模型 ≈ 支持渲染+模拟+决策的通用空间智能引擎

功能分类法的价值在于：它为这三个看似不可通约的纲领提供了统一的比较维度——每个学派的系统都可以在 Renderer/Simulator/Planner 三个轴上定位，从而暴露它们"宣称了什么"和"提供了什么"之间的差异。

---

## 5. 其他典型项目的功能边界

除 Dreamer 系列外，当前领域内值得关注的系统可分为三类。

### 5.1 Marble：唯一商业化的 Renderer+Simulator 融合

World Labs 的 Marble (2025)[10]接收文本/图像/视频/空间草图等多模态输入，生成可探索的三维环境，同时输出 Gaussian splats（视觉消费）和 collision meshes（物理引擎消费）。这是当前唯一在单一模型中同时覆盖渲染器和模拟器的商业系统。但它不做规划，其作者对此边界有明确声明。

### 5.2 PointWorld：纯粹模拟器的形态泛化

PointWorld[9]（Huang, Fei-Fei Li et al., 2026）将机器人动作和世界状态统一表示为**三维点流**：

给定 RGB-D 图像和动作序列 $(a_1, ..., a_T)$，预测每个像素在三维空间中的逐帧位移 $\{\Delta p_{ij}^t\}$。这一表示天然跨机械臂形态泛化——单臂 Franka 和双臂人形机器人的动作都统一为三维点位移。

实验结果：单个预训练检查点使真实 Franka 机械臂从单张图片零样本执行刚体推动、可变形物体操作和工具使用。PointWorld 是纯 Simulator，规划由外挂 MPC 完成。

### 5.3 NVIDIA Cosmos 与 sim-to-real 困境

NVIDIA Cosmos 定位为世界基础模型（World Foundation Model），为自动驾驶和机器人训练生成物理感知视频。它是构建通用模拟器最有雄心的商业努力。但面临所有生成式模拟器的共同困境：AI 生成的几何可能外观正确但包含自相交或错误比例，导致物理引擎产生无意义的结果。Cosmos 的输出正在改进但尚未在高风险应用中取代传统引擎。

### 5.4 功能边界汇总

| 系统 | 自称 | Renderer | Simulator | Planner | 关键缺口 |
|------|------|----------|-----------|---------|---------|
| Sora | World Simulator | ★★★ | ☆ | ☆ | 无状态输出，自承物理不准确 |
| Genie | Foundation WM | ★★★ | ☆ | ☆ | 状态隐式编码在 Token 中 |
| Marble | Multimodal WM | ★★★ | ★★★ | ☆ | 不做规划 |
| PointWorld | 3D WM | ☆ | ★★★ | ★★(外挂) | 规划器外挂非内生 |
| Cosmos | Foundation WM | ★★ | ★★☆ | ☆ | sim-to-real 仍受限 |
| GATO | Generalist | ☆ | ☆ | ★★ | 无世界模型组件，纯行为克隆 |
| **DreamerV3** | World Model | ★★ | **★★★** | **★★★** | — |
| **R2-Dreamer** | Redundancy-Reduced WM | ☆ | **★★★** | **★★★** | 无渲染能力 |

---

## 6. 持续存在的技术瓶颈

### 6.1 Simulator 的数据稀缺

渲染器有整个互联网视频作为训练数据。规划器有来自遥操作和游戏记录的大量离线数据集。模拟器面临的却是根本性的稀缺：带显式几何、材质属性和物理标注的三维数据比供给渲染器的像素稀少数个数量级。这是目前领域内最重要的单一瓶颈。

潜在突破路径：
- 游戏引擎自动标注（Minecraft/Isaac Sim 可大规模生成带完整状态记录的数据）
- 从渲染器蒸馏模拟器（利用海量视频训练隐式物理先验 → 迁移到结构化模拟器）
- 自监督几何学习（PointWorld 和 R2-Dreamer 已经展示了不需要人工标注的路径）

### 6.2 Sim-to-real 差距

从仿真到现实的迁移损失是所有基于模拟器的系统必须面对的问题。当前格局：

| 方法 | 模拟性能 | 真实世界性能 | 差距 |
|------|---------|------------|------|
| DreamerV3 (Minecraft) | ★★★★★ | 游戏本身是确定数字环境 | 无 |
| R2-Dreamer | ★★★★☆ | 未在真实机器人验证 | 未知 |
| PointWorld | ★★★★☆ | ★★★★☆ (短任务) | 任务视野受限 |
| Cosmos | ★★★☆ | ★★☆ | 生成式几何缺陷 |

### 6.3 三类功能同一架构中的目标冲突

| 功能 | 优化目标 | 成功标准 | 可能的冲突 |
|------|---------|---------|----------|
| 渲染器 | 视觉逼真 | FID/CLIP | 为视觉美牺牲物理精度 |
| 模拟器 | 结构精确 | 物理守恒偏差 | 需要精确但可能不美观的表示 |
| 规划器 | 任务成功 | 成功率 | 学到了不具可解释性的隐表示 |

R2-Dreamer 和 PointWorld 分别通过去渲染器和去渲染器+外挂规划器绕过了这一冲突。但它们也证明了在单一架构中同时优化三者的困难。

### 6.4 评估基准缺失

领域内尚无统一的评估框架可以同时衡量：
- 渲染质量（视觉逼真度 + 物理合理性）
- 模拟精度（几何正确率 + 物理守恒偏差 + 跨状态泛化）
- 规划性能（从模拟到真实的零样本迁移成功率）

每项子能力都在被独立测量，但联合评估的缺失使得系统之间的公平比较几乎不可能。

---

## 7. 收敛趋势与开放问题

### 7.1 三类边界消融的证据

| 融合方向 | 示例 | 意义 |
|---------|------|------|
| Renderer → Simulator | Marble: 同模型输出 splats + collision meshes | 视觉和几何可共用同一表征 |
| Renderer → Planner | 预训练视频模型做动作预测骨干 | 渲染器中编码了部分动力学知识 |
| Simulator → Planner | DreamerV3 RSSM → Actor-Critic | 隐空间动力学可直接支撑决策 |
| Planner → Simulator | PointWorld MPC + 点流模型 | 规划需要模拟器提供预测 |

### 7.2 统一世界模型的架构挑战

逻辑终点的"统一世界模型"需要同时满足：

$$f_{\text{unified}}: (\text{prompt}, a_t) \mapsto (o_t, s_t, a_{t+1})$$

其中 prompt 可以是文本、图像、视频或草图，$o_t$ 是像素输出（渲染），$s_t$ 是几何/物理输出（模拟），$a_{t+1}$ 是行动输出（规划）。当前最接近的是：
- Marble：实现了 prompt → (splats, meshes)，缺规划输出
- DreamerV3：实现了 $a_t$ → 隐 $s_{t+1}$ → 隐规划，缺可视渲染输出
- PointWorld：实现了 (RGB-D, $a_t$) → 3D 点位移，缺规划内生和渲染

### 7.3 开放问题

1. **Simulator 的可扩展性假设**：渲染器的 scaling law 已在 Sora 等模型中验证。模拟器是否也能在数据规模增长时展现出类似的幂律改进？目前没有确定性证据。

2. **物理先验 vs 端到端学习**：是否需要在架构中注入物理归纳偏置（如动量守恒、因果结构）？RSSM 走的是最小先验路线，DreamerV3 在 Minecraft 中成功但许多物理交互场景仍失败。

3. **统一模型的损失函数设计**：渲染损失（像素级 MSE/感知损失）、模拟损失（状态预测准确率）、规划损失（策略梯度）在同一架构中的权重和交互仍未解决。

4. **从仿真到迁移到理解**：在模拟中训练的策略能否迁移到真实物理世界？当前 PointWorld 展示了短任务迁移，但长任务和复杂物理交互的迁移仍是开放挑战。

---

## 8. 结论

本文以 2026 年 World Labs 功能分类法为分析框架，对世界模型领域进行了系统性过滤和深入分析。

**第一，大多数自称世界模型的系统实际上只覆盖三类功能中的一类。** Sora 和 Genie 是渲染器——输出像素而非状态，具有视觉合约而非结构合约。GATO 是孤立的规划器——输出行动而不预测世界。这些系统都是重要且有价值的技术贡献，但它们不是 POMDP 传统意义上的世界模型。

**第二，Dreamer 系列是唯一同时覆盖模拟器和规划器的系统家族。** 四代架构中，RSSM 持续扮演模拟器角色（隐空间中的状态转移预测），Actor-Critic 持续扮演规划器角色（在想象轨迹中决策）。R2-Dreamer（ICLR 2026）进一步证明：去除解码器后模拟器训练加快 59%，且在微小物体场景中表现更好，验证了分类法的核心预测——模拟器可以不依赖渲染器独立存在。

**第三，三类功能的收敛是领域的主导趋势。** Marble（Renderer+Simulator）和 Dreamer 系列（Simulator+Planner）从两端逼近全功能统一。但三者合一的系统仍面临数据稀缺、sim-to-real 差距、评估基准缺失和架构内目标冲突等根本挑战。解决这些挑战——而非继续稀释术语——是世界模型研究的下一个关键步骤。

---

## 参考文献

[1] Ha, D. & Schmidhuber, J. World Models. *NeurIPS*, 2018. arXiv:1803.10122.

[2] Hafner, D., et al. Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*, 2020. arXiv:1912.01603.

[3] Hafner, D., et al. Mastering Atari with Discrete World Models. *ICLR*, 2021. arXiv:2010.02193.

[4] Hafner, D., et al. Mastering Diverse Domains through World Models. 2023. arXiv:2301.04104.

[5] Morihira, N., et al. R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation. *ICLR*, 2026. arXiv:2603.18202.

[6] Bruce, J., et al. Genie: Generative Interactive Environments. 2024. arXiv:2402.15391.

[7] OpenAI. Video generation models as world simulators. 2024. https://openai.com/index/video-generation-models-as-world-simulators/

[8] World Labs Team. A Functional Taxonomy of World Models: Renderers, Simulators, Planners, and the Loop That Connects Them. 2026. https://worldlabs.ai/blog/taxonomy-of-world-models

[9] Huang, W., Fei-Fei, L., et al. PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation. 2026. arXiv:2601.03782.

[10] World Labs. Marble: A Multimodal World Model. 2025. https://worldlabs.ai/blog/marble-world-model

[11] Park, J.S., et al. Generative Agents: Interactive Simulacra of Human Behavior. *UIST*, 2023. arXiv:2304.03442.

[12] Reed, S., et al. A Generalist Agent (GATO). *DeepMind*, 2022.

[13] Kaiser, L., et al. Model-Based Reinforcement Learning for Atari. 2019. arXiv:1903.00374.

[14] Valevski, D., et al. GameNGen. 2024. arXiv:2408.14837.

[15] Wang, L., et al. A Survey on Large Language Model based Autonomous Agents. 2025. arXiv:2308.11432.

[16] Craik, K. *The Nature of Explanation*. Cambridge University Press, 1943.

[17] Schmidhuber, J. Making the world differentiable: On using fully recurrent self-supervised neural networks for dynamic robot control. *Neural Networks*, 1990.
