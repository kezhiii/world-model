# 世界模型（World Model）：功能分类法视角下的系统性综述

> **摘要**：世界模型旨在为AI系统构建可预测、可想象、可规划的"内心世界"，是通向空间智能和通用具身智能的关键路径。然而近年来"世界模型"概念被严重滥用——视频生成、游戏模拟、智能体框架等不同技术路线都自冠此名，导致领域边界模糊、评估标准混乱。本文以 World Labs（Fei-Fei Li团队, 2026）提出的功能分类法为分析框架，对现有自称世界模型的工作进行系统性过滤与归类。该分类法基于部分可观测马尔可夫决策过程（POMDP）循环，将世界模型严格分为三类功能：渲染器（Renderer，输出像素）、模拟器（Simulator，输出几何/物理状态）、规划器（Planner，输出行动序列）。在此框架下，本文深入分析了 Dreamer 系列（DreamerV1→V3→R2-Dreamer）的架构演化与技术原理，并对其余典型项目进行对比评估。核心结论：当前仅 Dreamer 系列在 Simulator + Planner 双功能上完整对齐分类法定义；R2-Dreamer（ICLR 2026）进一步证明 Simulator 可不依赖 Renderer 独立存在。文章最后分析了领域瓶颈与统一世界模型的未来方向。

> **关键词**：世界模型；功能分类法；Dreamer；RSSM；MBRL；空间智能

---

## 1. 引言

### 1.1 背景与动机

语言模型赋予机器对概念、词汇和推理的了不起的掌控力，但物理世界——无论虚拟或现实——运行在不同的基座上。语言模型学习文本的统计结构，而世界模型学习空间和时间的统计结构：光线如何在表面落下，物体如何响应力的作用，物理定律如何支配变化（World Labs, 2026）。

这一分野使得"世界模型"成为当下AI领域最重要但也最被滥用的术语之一。计算机视觉、机器人学、强化学习和生成式AI各自声称在构建世界模型，各执一词。一个视频模型可以生成华丽但物理上不可能的火焰，一个大语言模型可以即兴编出一个可交互的游戏，一个物理引擎可以忠实模拟燃烧——它们都叫同一个名字。

古希腊人从未就世界由什么构成达成共识——火、水还是不可分的原子——因为"世界"从来不是一个单一事物。AI继承了同样的问题，恰在领域最需要精确性的时刻。

### 1.2 评估框架：功能分类法

本文采用 World Labs 于 2026 年 6 月 3 日发表的《A Functional Taxonomy of World Models》作为核心分析框架。该分类法基于强化学习中经典的**部分可观测马尔可夫决策过程（POMDP）**循环：

```
智能体 → 行动 → 影响世界状态 → 产生观测 → 反馈给智能体
```

三类功能分别对应循环中的不同输出环节：

| 类别 | 输出 | 合约 | 核心质量 | 消费方 |
|------|------|------|---------|-------|
| **渲染器** | 观测/像素 | 视觉逼真 | 外观真实性 | 人眼 |
| **模拟器** | 状态（几何/物理） | 结构精确 | 物理守恒 | 人 + 程序 |
| **规划器** | 行动序列 | 决策正确 | 任务完成率 | 智能体 |

关键论断：**模拟器是三类中的"支点"**。掌握了模拟能力，才能向上为渲染器提供视觉外观，向下为规划器提供行动后果预测。仅掌握渲染或规划其一，不构成真正的世界模型。

### 1.3 综述范围与方法

本文对 2018-2026 年间自称"世界模型"的代表性工作进行系统检索与过滤。排除标准：（1）不包含 POMDP 循环中至少一个环节的显式建模；（2）仅作为 Agent 系统的外围组件而非核心引擎；（3）纯粹的语言推理或工具调用能力。最终纳入分析的工作聚焦于 Dreamer 系列、Sora、Genie、Marble、PointWorld、NVIDIA Cosmos、GameNGen、GATO 等典型项目。

---

## 2. 功能分类法核心框架

### 2.1 分类法的起源：从 Craik 到 POMDP

"世界模型"一词的思想史可追溯至 Kenneth Craik 在 1943 年提出的命题：心智通过运行现实的"小尺度模型"来推理。这一概念在1980年代末被引入神经网络学派（Schmidhuber, 1990; 1991），并在 Sutton & Barto 的经典教材中以 POMDP 的形式固化。

POMDP 框架的核心洞察：智能体永远无法直接观测到世界的真实"状态"——它只能接收到经过感官过滤的"观测"。 状态是世界的底层真实：每一个物体的位置、每一个速度、每一个属性，原则上完整但任何内部智能体都不可见。观测是智能体对这一真实的部分视角。行动是智能体的回应。

这一循环——智能体→行动→状态→观测→智能体——给了"世界模型"一词严格的技术含义。今天被称为世界模型的不同事物，本质上是这一循环的不同**投影**。

### 2.2 三类功能的严格形式化

设 POMDP 定义为元组 $(S, A, O, T, E, R)$：

- $s_t \in S$：不可直接观测的真实状态
- $a_t \in A$：智能体采取的行动
- $o_t \in O$：智能体接收的观测，服从 $p(o_t|s_t)$
- $T(s_{t+1}|s_t,a_t)$：状态转移函数
- $R(r_t|s_t,a_t)$：奖励函数

在此框架下，三类世界模型的功能定义如下：

**渲染器（Renderer）**：学习 $p(o_t|s_t)$ 或 $p(o_{t+1}|o_t,a_t)$。输出观测。合约是**视觉逼真**——输出必须看起来像真实的观测，但不需要内部结构一致。数学上等价于带条件的视频生成模型。

**模拟器（Simulator）**：学习 $T(s_{t+1}|s_t,a_t)$，即状态转移函数。输出状态。合约是**结构精确**——几何、物理、动力学必须在可计算层面上保持一致。模拟器有两个消费端：人类（建筑师、设计师、游戏开发者）需要精度；程序（RL智能体、机器人控制器、自动驾驶系统）需要可交互的虚拟训练场。

**规划器（Planner）**：学习 $\pi(s_t) \mapsto a_t$，即策略函数。输出行动序列。合约是**决策正确**——在给定目标和观测约束下，找到使期望累积奖励最大化的行动。这是渲染器的对偶：渲染器输入行动输出观测，规划器输入观测输出行动，闭合感知-行动循环。

### 2.3 三类功能的关系与收敛趋势

三类功能共享相同的底层知识——几何、物理、动力学——只是以不同的方式投射出来。一个真正理解世界的模型应当能同时做到：**渲染**一个杯子（从任何角度），**模拟**杯子被推后发生什么，并**规划**手去捡起杯子。

最值得注意的趋势是三类边界正在消融：

- **Renderer → Simulator**：Marble 同模型输出 Gaussian splats（视觉）和 collision meshes（物理）
- **Renderer → Planner**：预训练视频渲染器作为世界+行动联合预测骨干
- **Simulator → 双向**：DreamerV3 内部 RSSM 是 Simulator，Actor-Critic 是 Planner

逻辑终点是一个**统一世界模型**：一个基础模型可以切换输出模态，按下游需求输出渲染视图、模拟结构或行动序列。

---

## 3. Dreamer 系列：最符合定义的世界模型

### 3.1 家族谱系总览

Dreamer 系列由 Danijar Hafner 及其合作者（Google DeepMind）从 2019 年发展至今，横跨四代：

| 代际 | 年份 | 论文 | 会议 | 隐变量类型 | 任务覆盖 | 关键突破 |
|-----|------|------|------|-----------|---------|---------|
| DreamerV1 | 2019 | *Dream to Control* | ICLR 2020 | 连续高斯 | 20个连续控制 | 潜空间Actor-Critic |
| DreamerV2 | 2020 | *Mastering Atari with Discrete World Models* | ICLR 2021 | 分类离散 | 55 Atari + 控制 | 首个Atari人类级世界模型 |
| DreamerV3 | 2023 | *Mastering Diverse Domains through World Models* | — | 分类离散+鲁棒技术 | **150+** | Minecraft钻石首发 |
| R2-Dreamer | 2026 | *Redundancy-Reduced World Models w/o Decoders* | **ICLR 2026** | 分类离散+冗余消除 | DMC + Meta-World | 去解码器，1.59×加速 |

### 3.2 RSSM：循环状态空间模型

RSSM（Recurrent State-Space Model）是 Dreamer 系列世界模型的核心架构。它解决了部分可观测环境中隐状态学习的两个核心矛盾：

1. **随机性与确定性**：环境中既有随机事件也需要确定性的记忆
2. **长期依赖与短期更新**：需要同时捕捉长期时间结构和短期观测变化

RSSM 架构的关键设计：

```
确定路径:        h_t = f(h_{t-1}, s_{t-1}, a_{t-1})    # GRU
随机路径:   s_t ∼ p(s_t | h_t, o_t)                     # 后验编码器
预测路径:  ŝ_t ∼ p(ŝ_t | h_t)                           # 先验预测器
```

其中 $h_t$ 是确定性隐状态（GRU的隐藏层），$s_t$ 是随机隐状态（从后验采样）。两个路径的信息流设计是 RSSM 的核心：

- **确定性路径** 负责长期记忆和时间结构建模
- **随机路径** 负责捕捉环境中的随机性和多模态

### 3.3 DreamerV1 → V2 的跃迁：离散化

DreamerV2 最关键的技术抉择是从**连续高斯隐变量**转向**分类离散隐变量**：

$$V1:\ s_t \sim \mathcal{N}(\mu_t, \sigma_t)$$
$$V2:\ s_t \sim \text{Cat}(\text{softmax}(\cdot))$$

动机：物理世界的许多本质结构是离散的——物体的类型、事件的分界、游戏状态的切换。离散表示天然更适合捕捉这些结构。具体而言，DreamerV2 使用 32 个分类变量，每个有 32 个类别，共 1024 种离散组合。

离散化的实际收益：DreamerV2 成为首个在 Atari 55 个游戏中达到人类水平的基于世界模型的智能体，超越当时最先进的无模型方法 IQN 和 Rainbow。这一结论验证了 Simulator 的质量直接决定 Planner 性能的上限。

### 3.4 DreamerV3：通用算法的三项工程突破

DreamerV3 以**单一固定超参数设置**在 150+ 任务上超越或打平专用方法。实现这一前所未有的通用性的三项鲁棒性技术：

**(1) Symlog 变换**
奖励和回报的量级在不同任务中差异巨大（从 0/1 二元到上万）。Symlog 变换 `symlog(x) = sign(x)·ln(|x|+1)` 将任意范围的值紧凑映射到近似对数空间，使网络无需先验知晓奖励量级。

**(2) 分位数回归（Quantile Regression）**
取代均方误差来训练 Critic。分位数回归对目标值分布不做任何假设，天然适应多模态和高偏态的回报分布。

**(3) 双向归一化**
层归一化的双向变体：在 RMSNorm 基础上增加逆变换，使激活值在训练中保持在合理范围内，不受输入剧烈变化的影响。

**Minecraft 钻石成就**：
Minecraft 是 RL 的"果酱挑战"——稀疏奖励、开放世界、长程多级技能树（砍树→木板→木棍→木镐→石镐→铁→钻石），此前无 RL 算法从零完成。DreamerV3 在 100M 环境步数后首次达成。

### 3.5 R2-Dreamer（ICLR 2026）：去解码器的冗余消除

**痛点**：DreamerV3 用解码器重构完整像素帧来监督世界模型训练，但大量像素是任务无关的（天空、背景、纹理），浪费容量和计算。

**R2 的解决思路**：

```
DreamerV3:   o_t → 编码器 → s_t → 解码器 → 重构 o_t (大量冗余)
R2-Dreamer:  o_t → 编码器 → s_t → 自监督冗余消除 → 无需解码器
```

**技术核心**：
- 丢弃全部解码器参数
- 借鉴 Barlow Twins 的冗余消除目标：计算隐表示维度的互相关矩阵，迫使非对角元素接近 0（消除冗余），对角元素接近 1（保留信息）
- 不需要任何数据增强——旧有去解码器方法（如 TD-MPC2）依赖外部数据增强，限制了通用性

**性能**：

| 指标 | DreamerV3 | R2-Dreamer | 提升 |
|------|-----------|-----------|------|
| 训练速度 | 1× | **1.59×** | +59% |
| Meta-World | 基准 | **持平或略优** | — |
| DMC-Subtle | 不适应微小物体 | **大幅超越** | 显著 |

**对功能分类法的验证意义**：
R2-Dreamer 直接验证了分类法的一个核心预设——**Simulator 可以不依赖 Renderer 独立存在**。丢弃解码器不仅不损害任务表现，反而在微小物体场景中更好，因为它不再浪费容量在冗余像素上。这也暗示了"三位能力共享同一知识基础"的可能性：隐空间中学习的结构化表示足够支持规划，不需要像素级渲染作为中间监督。

### 3.6 Dreamer 系列与功能分类法的对应

```
Dreamer 系列架构 = Simulator(RSSM) + Planner(Actor-Critic) + Renderer(可选)

DreamerV1/V2/V3:   完整 RSSM（状态转移+奖励预测）+ Actor-Critic
                    ├─ Simulator: ⭐⭐⭐
                    ├─ Planner: ⭐⭐⭐
                    └─ Renderer: ⭐⭐ (V1/V2/V3 均有解码器)

R2-Dreamer:        RSSM (去解码器) + Actor-Critic
                    ├─ Simulator: ⭐⭐⭐ (更专注)
                    ├─ Planner: ⭐⭐⭐ (不变)
                    └─ Renderer: ⭐ (去除)
```

Dreamer 系列是当前唯一在学术层面上同时严格对齐 **Simulator** 和 **Planner** 两条功能线的世界模型。

---

## 4. 其他典型项目评估与过滤

### 4.1 渲染器主导的项目

#### Sora（OpenAI, 2024）

| 维度 | 评估 |
|------|------|
| 自称 | "Video generation models as world simulators" |
| 实际功能 | **Renderer ⭐⭐⭐**：文本/图像→视频的 Diffusion Transformer |
| Simulator | **0**：无显式状态表示 |
| Planner | **0**：不输出行动 |
| 关键证据 | OpenAI 自承 "does not accurately model the physics of many basic interactions, like glass shattering" |
| 结论 | 视觉逼真但无结构合约。不能用于设计建筑或训练机器人。 |

Sora 的核心架构：将视频压缩为潜空间→分解为时空 Patch（spacetime patches）→用 Diffusion Transformer 生成。在功能分类法中，Sora 是一个**输入条件化的渲染器**——它学习的是 $p(o_t|\text{text})$，而非 $p(s_{t+1}|s_t,a_t)$。涌现的"3D一致性"和"物体永久性"是统计数据特征而非显式建模。

#### Genie（Google DeepMind, 2024）

| 维度 | 评估 |
|------|------|
| 自称 | "Foundation world model"（11B参数） |
| 实际功能 | **Renderer ⭐⭐⭐**：action-conditioned 帧生成 |
| Simulator | **1**：隐式编码潜在动作模型 |
| Planner | **0**：不输出行动 |

Genie 的三组件：时空视频分词器 + 自回归动力学模型 + 潜在动作模型。关键创新是**无监督学习潜在动作空间**——从无标注互联网视频中自动发现"动作"的存在。但从功能分类法看，它的"状态"始终是隐式的、编码在视频token中的，没有输出可供物理引擎或RL智能体消费的结构化状态表示。

#### GameNGen（Google, 2024）

扩散模型驱动的实时 Doom 模拟器（~20FPS）。交互是基于像素层面的——它模拟的是"玩家看到的画面"而非"游戏引擎的变量"（血量、弹药、敌人位置）。功能分类法定位：介于 **Renderer（主导）**与 **Simulator（雏形）**之间。

#### RTFM（World Labs, 2025）

实时交互帧生成模型。World Labs 自承属于 Renderer 类别。输入用户操作，实时生成交互视频帧。与 Genie 的异同：两者都是 action-conditioned renderer，但 RTFM 明确没有几何/物理输出。

### 4.2 模拟器主导的项目

#### Marble（World Labs, 2025）

| 维度 | 评估 |
|------|------|
| 自称 | "Multimodal world model" |
| 实际功能 | **Renderer ⭐⭐⭐ + Simulator ⭐⭐⭐** |
| Planner | **0**：不规划行动 |

Marble 是当前唯一在商业层面同时覆盖 Renderer 和 Simulator 的模型。输入文本/图像/视频/空间草图，输出两种格式：
- Gaussian splats → 视觉探索（消费方：人眼）
- Collision meshes → 物理引擎可以操作的结构化几何（消费方：程序）

**定位**：Renderer 和 Simulator 的桥梁。但它明确不涉足 Planner——不自称做决策、不输出行动序列。这与 Dreamer 系列的 Simulator+Planner 合一形成互补。

#### PointWorld（Stanford / Fei-Fei Lab, 2026）

| 维度 | 评估 |
|------|------|
| 原文标题 | "Scaling 3D World Models for In-The-Wild Robotic Manipulation" |
| 实际功能 | **Simulator ⭐⭐⭐**：3D点流预测 |
| Renderer | **0**：不输出像素 |
| Planner | **2**：可集成MPC，但非内生 |

PointWorld 的核心创新：将机器人的动作空间统一为**3D点流（point flows）**——给定RGB-D图像和动作序列，预测每个像素在3D空间中的位移。这一表示打破了不同机械臂的关节差异，实现跨形态泛化。

功能分类法定位：纯 Simulator。它输出的是几何状态变化（3D点位移），不是像素（Renderer）也不是动作（Planner——规划由外挂MPC完成）。

#### NVIDIA Cosmos（2024-2025）

定位为"世界基础模型平台"，主打物理感知视频生成。Cosmos 的目标是 Simulator——但它面临的是所有 Simulator 的共同痛点：sim-to-real gap。NVIDIA 的估值锚点（万亿美元级工厂/仓库/供应链数字孪生市场）基于 Simulator 的精确合约，但 Cosmos 的生成精度仍达不到替代传统物理引擎的水平。

### 4.3 规划器主导的项目

#### GATO（DeepMind, 2022）

GATO 是一个通用智能体——一个模型玩 Atari、控制机器人、对话聊天。但从世界模型的角度评估：

| 维度 | 评估 |
|------|------|
| 实际功能 | **Planner ⭐⭐**：多任务行为克隆 |
| Simulator | **0**：无世界模型组件 |

GATO 使用行为克隆（behavior cloning）跨任务学习策略，核心是**模仿**而非**理解**。它没有内部的 Simulator 来想象未来——这使其严格不属于"世界模型"范畴。称它为"通用智能体"是准确的，称它为"世界模型"则属于概念滥用。

#### Voyager（Microsoft, 2023）

基于 GPT-4 的 Minecraft 自动探索 Agent。核心组件是技能库 + 自动课程 + 迭代提示。没有世界模型组件——GPT-4 提供的高层常识不起到 Simulator 的作用。属于**Planner（任务规划层面）**但不应归类为世界模型。

### 4.4 被过滤的项目总结

| 项目 | 自称 | 实际功能 | 过滤原因 |
|------|------|---------|---------|
| Generative Agents (Stanford, 2023) | 智能体小镇 | 社交LLM模拟 | 无物理/状态建模，LLM+记忆系统 |
| Toolformer (Meta, 2023) | 工具使用LM | API调用代理 | 语言模型工具扩展 |
| Quiet-STaR (Stanford, 2024) | 思考型LM | 推理增强 | 语言模型内部推理 |
| LLM-Agent综述 (2023) | 综述 | 文献汇编 | 非模型 |

---

## 5. 当前技术瓶颈

### 5.1 按功能类别的瓶颈分布

| 瓶颈 | Renderer | Simulator | Planner | 严重程度 |
|------|----------|-----------|---------|---------|
| 物理幻觉 | ★★★★★ | ★★★ | ★★ | 高 |
| 长期一致性崩溃 | ★★★★★ | ★★★ | ★★★ | 高 |
| 3D/物理数据稀缺 | ★★ | **★★★★★** | ★★★★ | **极高** |
| Sim-to-real gap | ★ | ★★★★★ | ★★★★★ | **极高** |
| 生成式几何不可靠 | ★★ | ★★★★ | ★★★ | 中 |
| 多物理场成本 | ★ | ★★★★★ | ★★★★ | 高 |
| 探索-利用困境 | ★ | ★★★ | ★★★★★ | 高 |
| 评估基准匮乏 | ★★★ | ★★★★ | ★★★★ | 中 |

### 5.2 核心瓶颈详解

**（1）Simulator 的数据瓶颈**——三类中最严重的问题

Renderer 拥有整个互联网视频作为训练数据。Planner 也有海量的离线数据（机器人遥操作、游戏回放）。但 Simulator 需要的**带显式几何/物理标注的3D数据**极为稀缺。一个物体需要精细的3D网格、材质属性、物理响应——这些都远超互联网视频可提供的。

**（2）Sim-to-real gap**

这是 Simulator 和 Planner 共用的难题。仿真中的物理近似总是导致现实中的性能下降。DreamerV3 在 Minecraft 中工作是因为游戏环境本身是确定的数字环境；但在真实世界中，摩擦力、柔性变形、传感器噪声等因素都会使 Simulator 的预测偏离现实。R2-Dreamer 在模拟器上展示了优异性能，但尚未在真实机器人上验证。

**（3）生成式几何不可靠**

当 Simulator 本身是生成模型时（如 Marble 的碰撞网格），它可能产生看似合理但实际有缺陷的几何——自相交、错误比例、不真实的连通性。这些缺陷对视觉消费可能无伤大雅，但对物理引擎是致命的。

**（4）统一模型的优化冲突**

追求视觉美（Renderer 目标）可能牺牲 Planner 需要的决策精度。同一套参数如果要同时服务三套消费端，目标函数的设计成为未解决的关键问题。R2-Dreamer 通过丢弃 Renderer 绕过了这个冲突——但它也因此失去了 Render 能力。

### 5.3 学派分歧

- **生成式学派**（Sora/Genie）：主张 scaling law 驱动，大规模视频数据训练→物理规则自然涌现
- **结构化表征学派**（Dreamer/RSSM）：主张显式建模隐空间动力学结构，强调因果推理与规划模拟
- **混合学派**（Marble/World Labs）：主张同模型覆盖 Simulator+Renderer，逐步纳入 Planner 能力

---

## 6. 未来研究方向

### 6.1 统一世界模型

功能分类法指出的逻辑终点。一个基础世界模型应做到：

```
多模态输入（文本/图像/视频/3D）
           ↓
  统一世界模型（Foundation World Model）
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
 渲染器  模拟器  规划器
(像素) (状态)  (动作)
```

当前最接近这一目标的是 Marble（Renderer+Simulator）和 DreamerV3（Simulator+Planner），但尚无三者合一。

### 6.2 Simulator 的 scaling law

能否找到 Simulator 的"互联网视频时刻"——即某种数据源能规模化提供带显式几何/物理标注的训练信号？可能的候选包括：
- 游戏引擎的渲染管线（自动输出完整状态+像素+动作）
- 工业/建筑领域的 BIM 数据
- 合成数据引擎（如 NVIDIA Isaac Sim）

### 6.3 从 Renderer 蒸馏 Simulator

利用海量互联网视频训练的 Renderer 已经"学会"了丰富的物理先验（即使不完美）。反向蒸馏——用 Renderer 的知识监督 Simulator 学习——是一个前沿方向。GameNGen 的实践经验表明：Renderer 学到的隐式物理知识可以迁移到 Simulator。

### 6.4 跨实体泛化（Cross-Embodiment）

PointWorld 已验证 3D 点流表示可使世界模型跨单臂/双臂机器人工作。扩展这一方向：让世界模型不受具体物理形态限制地泛化，是一个重要趋势。

### 6.5 评估基准

当前世界模型领域最迫切的缺失之一：统一的评估基准集。需要覆盖：
- Renderer 质量：FID/CLIP score + 物理合理性（物理落地得分）
- Simulator 精度：几何准确率 + 物理守恒偏差
- Planner 性能：任务成功率 + 零样本泛化能力
- 统一模型：前三者的联合评分

---

## 7. 结论

本文以 World Labs 功能分类法为框架，对世界模型领域进行了系统性过滤、分析与综述。核心发现：

1. **"世界模型"概念确实被严重滥用**。许多自冠此名的项目实际上只覆盖了 POMDP 循环中的单个环节（Sora/Genie=Renderer，GATO=Planner），不构成完整的"世界模型"。

2. **Dreamer 系列是当前最严格符合定义的工作**。DreamerV1→V3→R2-Dreamer 持续进化，在 Simulator（RSSM）和 Planner（Actor-Critic）双线上深度对齐分类法。R2-Dreamer（ICLR 2026）进一步验证了 Simulator 可不依赖 Renderer 的独立价值。

3. **Simulator 是三类功能的支点，也是最薄弱的环节**。数据稀缺、sim-to-real gap、生成几何不可靠等问题使 Simulator 的技术成熟度远低于 Renderer。R2-Dreamer 的 1.59×加速方案证明了聚焦 Simulator 本身是有效的方向。

4. **统一世界模型是逻辑终点，但路径未明**。在单一架构中同时实现渲染-模拟-规划三者，面临目标函数冲突、数据不均衡、评估缺失等根本挑战。World Labs 的 Marble 和 DeepMind 的 DreamerV3 分别从两端逼近，但跨过全线仍需基础性突破。

---

## 参考文献

[1] Ha, D., & Schmidhuber, J. (2018). World Models. *NeurIPS*. arXiv:1803.10122.

[2] Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. arXiv:1912.01603.

[3] Hafner, D., et al. (2021). Mastering Atari with Discrete World Models. *ICLR 2021*. arXiv:2010.02193.

[4] Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104.

[5] Morihira, N., et al. (2026). R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation. *ICLR 2026*. arXiv:2603.18202.

[6] Bruce, J., et al. (2024). Genie: Generative Interactive Environments. arXiv:2402.15391.

[7] OpenAI (2024). Video generation models as world simulators. https://openai.com/index/video-generation-models-as-world-simulators/

[8] World Labs Team (2026). A Functional Taxonomy of World Models: Renderers, Simulators, Planners, and the Loop That Connects Them. https://worldlabs.ai/blog/taxonomy-of-world-models

[9] Huang, W., Fei-Fei, L., et al. (2026). PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation. arXiv:2601.03782.

[10] World Labs (2025). Marble: A Multimodal World Model. https://worldlabs.ai/blog/marble-world-model

[11] Park, J.S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. arXiv:2304.03442.

[12] Reed, S., et al. (2022). A Generalist Agent. *GATO*. DeepMind.

[13] Kaiser, L., et al. (2019). Model-Based Reinforcement Learning for Atari. arXiv:1903.00374.

[14] Valevski, D., et al. (2024). GameNGen. arXiv:2408.14837.

[15] Wang, L., et al. (2023-2025). A Survey on Large Language Model based Autonomous Agents. arXiv:2308.11432.

[16] Craik, K. (1943). The Nature of Explanation. Cambridge University Press.

[17] Schmidhuber, J. (1990). Making the world differentiable. *Neural Networks*.

[18] Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction. 2nd ed. MIT Press.
