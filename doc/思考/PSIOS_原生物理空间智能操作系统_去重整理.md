# Physics-Native Space Intelligence Operating System（PSIOS）

## 第一章 问题陈述与范式定义

### 1.1 从自动驾驶汽车到太空航天器

以 Tesla FSD、Waymo、Apollo 等为代表的自动驾驶技术，集环境感知、世界建模、行为预测、路径规划与车辆控制于一体，通常采用"感知—预测—规划—控制（PPPC）"架构。这套经过十余年发展的体系，已成为当前最成熟的具身智能系统之一。

一个自然的问题是：能否直接将自动驾驶汽车的技术体系迁移到航天器，实现卫星自动驾驶？

从表面看确实相似——两者都需要自主定位、导航、决策和控制，都在动态环境下保证运行安全。近年来已有大量研究借鉴世界模型、强化学习、端到端神经网络等技术开展航天器自主导航。

但这种直接迁移存在根本性局限。**自动驾驶汽车面对的是经验驱动的开放世界；航天器运行的是物理驱动的确定性世界。** 两类系统智能的产生基础完全不同。

### 1.2 Physics Unknown 与 Physics Beyond

自动驾驶汽车的交通环境高度不确定——行人不可预测、驾驶行为随机、道路施工天气变化难以统一建模。因此它必须大规模依赖数据驱动学习，学的是"世界本身"。

航天器的空间环境截然不同。轨道运动严格遵循经典力学与轨道动力学，状态演化满足确定性方程：

| 动力学因素 | 数学描述 |
|-----------|---------|
| 两体动力学 | $\ddot{\mathbf{r}} = -\frac{\mu}{r^3}\mathbf{r}$ |
| 地球非球形引力 | J2/J3 摄动项 |
| 大气阻力 | $a_{drag} = -\frac{1}{2}\rho v^2 \frac{C_D A}{m}$ |
| 太阳辐射压 | $a_{srp} = -P_{sun} \frac{C_R A}{m}$ |
| 第三天体摄动 | 日/月引力 |

这些模型经过几十年工程验证，构成了现代航天任务设计的基础。

因此，AI 没有必要重新学习已被物理规律准确描述的世界。真正需要智能学习的是动力学模型无法完整描述的部分：未知扰动、任务意图理解、多目标决策、风险评估、自主任务规划、多航天器协同。

**自动驾驶汽车学习的是 Physics Unknown（未知物理），太空自动驾驶学习的是 Physics Beyond（物理之外的语义和不确定性）。AI 不应该替代物理模型，而应该扩展物理模型。**

### 1.3 两条路线的困境

当前航天智能研究存在两类技术路线，彼此割裂：

**模型驱动方法** 完全建立在轨道动力学基础上，通过最优控制、MPC、轨迹优化等技术完成自主导航。优势在于可解释性、安全性和工程可靠性，但缺乏自主学习能力——面对复杂未知环境适应能力有限。

**数据驱动方法** 利用神经网络、强化学习等方法直接学习轨迹规划或控制策略。但存在两个根本问题：
1. 神经网络往往重新学习已经明确的轨道动力学规律。例如，大量网络需要通过训练学习轨道传播过程，而该过程本可由高精度动力学模型直接计算。这增加训练成本，降低可解释性。
2. 纯神经网络难以保证航天系统的安全性与可验证性。轨道控制属于 Safety-Critical System，任何控制策略都必须满足推进能力、燃料约束和任务约束。纯数据驱动模型缺乏显式物理约束，决策过程不透明。

**模型驱动强调物理规律却缺乏学习能力，数据驱动强调学习能力却忽略了物理规律——这种割裂正是当前太空智能系统发展的核心瓶颈。**

### 1.4 Physics-Native Intelligence：定义

**Physics-Native Intelligence** 不是传统意义上的 Physics-Informed Learning（在损失函数中加入动力学约束），而是将轨道动力学作为整个智能系统的**原生组成部分（Native Component）**。

对比两种范式：

| 维度 | Physics-Informed Learning | Physics-Native Intelligence |
|------|--------------------------|---------------------------|
| 物理知识的位置 | 损失函数、正则项、网络结构设计中 | 系统运行时的持续组件 |
| 物理模型的角色 | 训练阶段的约束条件 | 运行时持续参与推理 |
| 训练完成后 | 物理模型退出运行 | 物理模型持续运行 |
| 神经网络学什么 | 动力学 + 未知因素 | 仅学未知因素 |
| 系统架构 | AI 为主，物理为辅 | 物理为基，AI 增强 |

在该理念下：
- **物理模型**负责描述世界如何演化（Evolution）——基于已知动力学方程
- **神经网络**负责理解世界意味着什么（Cognition）——基于数据学习
- **智能决策**负责确定下一步应该做什么（Decision）——基于物理约束下的搜索与评估

三者共同参与系统推理。未来的太空自动驾驶不应建立在"Physics + AI"的简单叠加之上，而应建立在物理模型与智能认知深度融合的统一体系之上。

### 1.5 智能的产生机制

在 Physics-Native Intelligence 中，智能不是某个模型输出的结果，而是持续推理过程的产物。系统运行的每一步都遵循以下闭环：

```
┌─────────────────────────────────────────────┐
│                                             │
│  (1) 动力学传播  →  (2) 环境认知与风险分析   │
│         │                   │               │
│         ▼                   ▼               │
│   获得未来物理状态    理解当前环境语义        │
│         │                   │               │
│         └───────┬───────────┘               │
│                 ▼                           │
│         (3) 候选行为生成                      │
│                 │                           │
│                 ▼                           │
│         (4) 物理一致性验证                    │
│                 │                           │
│            ┌────┴────┐                      │
│            ▼         ▼                      │
│         通过      不通过 → 重新优化            │
│            │                                │
│            ▼                                │
│         (5) 行为执行 → 状态更新 → 回到(1)      │
│                                             │
└─────────────────────────────────────────────┘
```

整个过程循环运行。**智能不是一次计算结果，而是一种持续运行的能力。** 系统不会因为一次推理结束而停止，而是在整个任务生命周期内持续进行状态传播、环境认知、策略优化和行为验证。

### 1.6 四项基本原则

这四项原则源自 Physics-Native Intelligence 的核心定义，贯穿后续所有模块设计：

**第一，物理优先（Physics First）**。所有状态演化必须遵循客观物理规律。任何学习算法均不能突破动力学约束。物理模型在所有推理环节中处于优先级最高的位置——它首先确定"什么是可能的"，然后神经网络再在可能空间中寻找"什么是最优的"。

**第二，职责分离（Responsibility Separation）**。物理模型负责世界演化（How the world changes），神经网络负责环境认知和经验学习（What the world means）。两者职责明确，不相互替代。这条原则的关键在于：神经网络不应该学习已经由物理定律描述的内容，从而避免 Data-driven 路线中常见的重复学习问题。

**第三，持续推理（Continuous Reasoning）**。智能来源于持续循环的推理过程，不是单次模型推断。系统不会因为某个任务完成而停止思考，而是始终维护对世界的最新认知。这条原则直接对应于 1.5 节的智能产生机制。

**第四，可扩展（Extensibility）**。系统能够不断增加新的智能能力（如新任务类型、新感知模态、新控制策略），而无需改变底层运行机制。这要求系统以能力单元而非功能模块组织，新能力以插件形式加入。

### 1.7 为什么称为"操作系统"？

本文标题中的"Operating System"并非修饰。PSIOS 之所以是一个操作系统，因为它满足操作系统的核心定义：

| OS 特征 | 传统 OS | PSIOS |
|---------|---------|-------|
| 资源管理 | 管理 CPU、内存、I/O | 管理推进剂、电源、通信、计算资源 |
| 进程调度 | 管理和调度进程 | 管理和调度 Runtime（能力单元） |
| 安全机制 | 用户态/内核态分离 | Physics-First 约束验证（安全边界不能突破） |
| 驱动层 | 硬件驱动 | GNC 底层执行接口 |
| 文件系统 | 数据持久化 | USS（Unified State Space）统一状态持久化 |
| 持续运行 | 操作系统不关机 | 智能引擎在整个任务周期内不停止 |

在这个意义上，**PNIE（Physics-Native Intelligence Engine）是 PSIOS 的内核（Kernel）**。它负责管理所有 Runtime 的生命周期、维护 USS、调度能力执行，以及保证物理安全性。PNIE 之于 PSIOS，相当于 Linux Kernel 之于 Linux Distribution。

PSIOS 还包括：任务接口层（Mission Interface）、遥测遥信子系统、人机交互终端、以及部署在地面端的数字孪生仿真环境。

---

## 第二章 PNIE 总体架构

### 2.1 PNIE 的系统定位

PNIE 是 PSIOS 的核心智能引擎。它不是替代传统 GNC 系统，而是在已有导航、制导与控制基础上，为航天器提供持续运行的自主智能能力。

传统航天软件按功能划分为独立模块——轨道确定、轨迹规划、姿态控制、任务管理等。每个模块完成固定功能，通过预定义接口交换数据。这种设计能满足确定任务需求，但当任务复杂度不断增加时，模块间容易形成大量耦合（例如：增加自主交会能力需要修改轨道模块、规划模块和控制模块的接口，并需要新增大量点对点通信路径）。

PNIE 则围绕**智能能力的产生机制**构建。导航、交会、补给等只是不同的能力表现，它们共享同一套运行框架——都依赖物理认知、环境理解、行为规划和资源管理。增加新能力只需注册新的 Capability，而不修改已有模块。

从系统层次来看：

```
┌─────────────────────────┐
│    Mission Interface     │  ← 地面/星上任务指令
├─────────────────────────┤
│         PNIE            │  ← 智能引擎（本文核心）
│  ┌─────────────────────┐│
│  │  Runtime Management ││
│  │  Behavior Planning  ││
│  │  State Maintenance  ││
│  └─────────────────────┘│
├─────────────────────────┤
│   GNC (Guidance / Nav   │  ← 底层执行
│        / Control)       │
└─────────────────────────┘
```

PNIE 位于任务系统与飞行控制系统之间：上层提出任务目标（"接近目标卫星"、"完成推进剂补给"），下层执行姿态控制、轨道机动和机构驱动，PNIE 负责将任务目标转化为满足物理约束的自主行为序列。

### 2.2 为什么采用 Runtime 架构

传统软件模块（Module）适用于流程明确、功能边界清晰的系统：输入→计算→输出→结束。但持续智能系统完全不同——航天器在整个飞行过程中需要**持续**进行轨道传播、环境认知、风险分析、任务评估和行为调整。这些能力并不会因为某个任务完成而停止。

PNIE 采用 **Runtime** 作为基本运行单元，因为 Runtime 具备以下关键属性：

**（1）持续性（Persistence）**。Runtime 不是一次计算完成即退出的函数，而是在整个任务过程中不断更新内部状态。例如，负责轨道传播的 Runtime 需要持续更新未来轨道预测，而不是 "调用一次算完就结束"。

**（2）独立状态管理（Independent State）**。每个 Runtime 拥有自己的状态空间和更新机制。Physics Runtime 维护物理世界的数字化表示，Neural Runtime 维护对环境的认知状态。它们独立演化但共享状态。

**（3）松耦合通信（Loose Coupling）**。Runtime 之间不直接调用彼此的内部实现，而是通过统一的 USS（Unified State Space）交换信息。这种设计避免了模块间 N×N 的点对点耦合。

**（4）异步运行（Asynchronous Execution）**。不同 Runtime 具有不同的计算周期——轨道传播可能每秒更新，认知推理可能每十秒更新，行为规划可能仅在环境变化时触发。Runtime 架构天然支持这种异构时间尺度。

### 2.3 Runtime 的统一定义

每种 Runtime 具备四项基本属性：

- **状态（State）**：Runtime 维护自身负责的信息集合。这些状态不是临时变量，而是在整个系统运行过程中持续存在并演化。
- **行为（Behavior）**：Runtime 根据当前状态执行相应计算。包括状态更新、预测分析、风险评估或能力生成。
- **接口（Interface）**：Runtime 不直接访问其他 Runtime 内部数据，而是通过 USS 读取输入并写回输出。
- **生命周期（Lifecycle）**：Runtime 从系统启动开始运行，在整个任务过程中持续更新，直到任务结束或系统关闭。

这一统一定义适用于 PNIE 中所有运行单元，是整个系统设计的一致规范。

### 2.4 四类 Runtime 及其职责

PNIE 由四类核心 Runtime 组成：

| Runtime | 核心问题 | 维护的状态 | 典型实现 |
|---------|---------|-----------|---------|
| **Physics Runtime** | 世界如何演化？ | 轨道、姿态、资源、约束 | 轨道传播器、姿态动力学模型、资源消耗模型 |
| **Neural Runtime** | 世界意味着什么？ | 环境认知、目标意图、风险评估 | Transformer、GNN、持续学习模块 |
| **Behavior Runtime** | 下一步做什么？ | 当前行为、任务阶段、行为图 | 行为推理器、规划器、Graph Engine |
| **Memory Runtime** | 过去教会我们什么？ | 任务历史、经验模型、目标档案 | 向量数据库、知识图谱、经验回放 |

四类 Runtime 没有主从关系，而是在统一运行机制下协同工作——这类似于一个多进程操作系统中各进程的关系。

### 2.5 协同运行机制

各 Runtime 不是按固定顺序依次调用，而是围绕 USS 持续独立运行：

```
                         Unified State Space (USS)
                    ┌──────────────────────────────────┐
                    │  Physical │ Cognitive │ Mission  │
                    │  State    │ State     │ State    │
                    │           │           │          │
                    │  Resource │ Behavior  │ Memory   │
                    │  State    │ State     │ Context  │
                    └──────────────────────────────────┘
                         ▲  ▲  ▲  ▲  │  │  │  │
                         │  │  │  │  ▼  ▼  ▼  ▼
                    ┌────┴──┴──┴──┴──────────────┐
                    │   Unified Space             │
                    │   Representation (USR)      │
                    └─────────────────────────────┘
                         ▲       ▲       ▲       ▲
                         │       │       │       │
                    ┌────┴──┐┌───┴───┐┌──┴──┐┌───┴───┐
                    │Physics││Neural ││Behav││Memory │
                    │  RT   ││  RT   ││ RT  ││  RT   │
                    └───────┘└───────┘└─────┘└───────┘
```

运行流程如下：

1. **系统启动**：所有 Runtime 同时进入运行状态
2. **Physics Runtime**：根据观测信息和动力学模型更新物理状态，持续预测未来演化
3. **Neural Runtime**：读取最新物理状态，对环境、目标和任务进行认知分析，形成新的认知信息
4. **Behavior Runtime**：综合物理状态和认知状态，依据任务目标生成或调整当前行为
5. **Memory Runtime**：记录过程信息，持续积累长期知识，并为其他 Runtime 提供历史经验检索

随着新的观测数据到来，整个系统立即进入下一轮状态更新，形成连续、自适应的智能闭环。

### 2.6 PNIE 需要多少计算资源？

这是一个经常被问到的工程问题。PNIE 的计算需求取决于 Runtime 的更新频率：

| Runtime | 典型更新周期 | 计算需求（估算） |
|---------|------------|----------------|
| Physics Runtime | 1-10 Hz（轨道传播） | 低（解析/数值积分，可在 ARM M7 上运行） |
| Neural Runtime | 0.1-1 Hz（认知推理） | 中等（Transformer 推理，需 NPU/低功耗 GPU） |
| Behavior Runtime | 事件驱动（行为切换时） | 低（图遍历 + 约束检查） |
| Memory Runtime | 0.01-0.1 Hz（经验积累） | 低至中等（向量检索） |

星载计算平台（如 Xilinx Versal、NVIDIA Jetson Orin 的 Rad-Hard 版本）已能够支持上述负载。关键不是算力绝对值，而是 **Physics Runtime 承担了计算量最大的轨道传播工作**（这部分在传统方法中本就存在），Neural Runtime 仅处理认知推理，因此相比纯神经网络方案大幅度降低了计算需求。

---

## 第三章 Physics Runtime——物理世界的持续运行层

### 3.1 设计目标

Physics Runtime 是 PNIE 中唯一负责维护客观物理世界的运行时组件。它的职责不是生成智能，而是保证整个智能系统始终建立在真实物理规律之上。

在传统航天软件中，动力学模型是按需调用的算法工具：给定当前状态和传播时间 → 计算未来轨道 → 返回结果 → 算法结束。这种模式可表示为简单的 `Input → Propagate → Output`，但是在 PNIE 中，物理模型不再是工具，而是一个持续运行的 Runtime：
- 始终维护航天器及空间环境的物理状态
- 随时间不断更新
- 为其他 Runtime 提供一致、可信的物理基础
- 保证所有智能行为满足动力学和工程约束

### 3.2 从轨道传播器到物理运行时

传统轨道传播器只负责"从现在算未来"。PNIE 的 Physics Runtime 维护的是整个 **Physical State Space（物理状态空间）**：

- 不只是位置和速度，而是包括姿态、资源、约束和环境
- 不是 "算一次就结束"，而是 "持续演化"
- 不是 "被动被调用"，而是 "主动维护"

这相当于将航天器的**数字物理世界（Digital Physics Layer）** 从离散的计算工具提升为持续运行的基础设施。

### 3.3 Physics Runtime 的五大职责

**（1）动力学传播（Dynamics Propagation）**

根据任务需求采用不同精度模型。Physics Runtime 不限制传播模型的具体实现，但要求所有传播结果具有统一数据接口：
- 二体模型：基础轨道传播
- 摄动模型：含 J2/J3、大气阻力
- 高精度模型：完整力模型 + 数值积分
- 姿态动力学：欧拉方程 + 四元数积分

这意味着无论未来采用数值积分、解析传播还是 ML 辅助传播，对外提供的状态表示保持一致——这是可扩展原则的具体体现。

**（2）约束维护（Constraint Maintenance）**

任何航天器运行都受到大量约束。传统控制系统在规划阶段单独计算这些约束；PNIE 要求 Physics Runtime 持续维护所有约束状态，其他 Runtime 可直接读取：

- 推进剂约束：剩余量、最小保持量
- 推力约束：最大推力、最小脉冲、推力方向限制
- 通信可见性约束：地面站过境窗口
- 热控约束：太阳角、散热面朝向
- 姿态机动约束：角速度限制、禁止指向方向
- 电源约束：电池荷电状态、太阳能输入功率

**（3）可达域分析（Reachability Analysis）**

系统不仅需要知道"现在在哪里"，更需要知道"未来能够到哪里"。Physics Runtime 持续维护：
- 可到达轨道集合（给定推进剂约束）
- 最小燃料轨迹
- 最小时间轨迹
- 安全轨迹空间（不含碰撞风险区域）

**（4）资源演化（Resource Evolution）**

航天器资源状态同样属于物理世界的一部分，具有明确的物理规律：
- 推进剂消耗：$m(t) = m_0 - \int_0^t \dot{m}(\tau) d\tau$
- 电池荷电状态：$\text{SoC}(t) = \text{SoC}_0 + \int_0^t (P_{solar}(\tau) - P_{load}(\tau)) d\tau / C_{battery}$
- 热平衡：热容、辐射散热、日照加热
- 执行机构寿命：累计工作时间、启停次数

**（5）物理一致性验证（Physics Consistency Verification）**

这是 Physics Runtime 最关键的职责之一。对于任何 Behavior Runtime 输出的候选行为，Physics Runtime 必须完成：

| 验证维度 | 具体检查 |
|---------|---------|
| 动力学一致性 | 推算的轨迹是否符合轨道力学？ |
| 工程约束 | 所需 ΔV 是否在推进系统能力内？ |
| 安全边界 | 是否满足与目标的最小安全距离？ |
| 资源可行性 | 推进剂、电力是否支持完成该行为序列？ |

只有通过全部验证的行为，才能进入底层 GNC 执行。Physics Runtime 同时承担了整个 PNIE 的**安全守护（Safety Guardian）** 角色。

### 3.4 Physical State 的数据组织

| 状态类别 | 主要内容 | 更新频率 |
|---------|---------|---------|
| Orbit State | 位置矢量 $\mathbf{r}$、速度矢量 $\mathbf{v}$、轨道根数 $(a, e, i, \Omega, \omega, \theta)$ | 1-10 Hz |
| Attitude State | 姿态四元数 $\mathbf{q}$、角速度 $\mathbf{\omega}$、姿态模式 | 1-10 Hz |
| Dynamic Environment | 引力场、大气密度、太阳辐射压、第三天体位置 | 0.1-1 Hz |
| Resource State | 推进剂余量、电池 SoC、热状态、执行机构状态 | 1 Hz |
| Constraint State | 可见性窗口、碰撞预测、姿态禁区、推进能力边界 | 按需更新 |

### 3.5 与 Neural Runtime 的关系

Physics Runtime 和 Neural Runtime 具有严格边界：Physics Runtime 不学习、不预测任务语义、不输出行为策略；Neural Runtime 不负责轨道传播、不维护资源状态、不修改动力学模型。二者唯一的联系是通过 USS 共享状态：Physics Runtime 提供客观事实，Neural Runtime 在事实基础上形成认知。

---

## 第四章 Neural Runtime——空间智能的认知运行层

### 4.1 设计目标

Neural Runtime 是 PNIE 中负责智能认知的持续运行层。在 Physics-Native 架构中，所有能够由物理模型准确描述的内容均由 Physics Runtime 负责维护；Neural Runtime 关注的是物理模型无法直接表达的信息。

它与传统 AI 方法的根本区别在于：**传统 AI 方法将神经网络作为 Physics Runtime 的替代者（学动力学、学轨迹、学控制）；PNIE 将神经网络定位为 Physics Runtime 的增强者（在已有物理认知基础上，形成高层理解）。**

### 4.2 Neural Runtime 学什么（以及不学什么）

这是区分 Physics-Native 架构与传统 AI-for-Aerospace 的关键：

| 传统方法的神经网络学什么 | PNIE 中的 Neural Runtime 学什么 |
|-------------------------|-------------------------------|
| 轨道动力学（学习 $s_{t+1} = f(s_t, a_t)$） | ✅ 由 Physics Runtime 负责，**不学** |
| 控制策略（学习 $\pi(a|s)$ 直接输出控制量） | ✅ 由 Behavior Runtime 规划 + GNC 执行，**不学** |
| 完整轨迹（学习整条运动路径） | ✅ 由 Physics Runtime 传播 + Planner 优化，**不学** |
| 目标行为意图 | ⭐ 学——这是 Neural Runtime 的核心任务 |
| 任务语义理解 | ⭐ 学——当前处于任务的什么阶段？ |
| 环境风险评估 | ⭐ 学——是否存在碰撞/通信中断/推进剂不足风险？ |
| 多目标协同决策 | ⭐ 学——多个航天器之间如何协调？ |
| 长期经验积累 | ⭐ 学——哪些策略在哪些场景下有效？ |

**Neural Runtime 的输出不是轨迹，不是控制量，而是认知信息（Cognitive Information）。**

### 4.3 从状态预测到认知建模：一个具体例子

考虑这样一个场景：

> **两颗卫星具有几乎相同的相对轨道状态：距离 5 km、接近速度 0.5 m/s、位置夹角 15 度。**

从纯动力学角度看，这两个场景**完全相同**。但从任务角度看却截然不同：

- **场景 A**：目标卫星正在执行自主交会任务，按预定计划主动接近——系统应该保持当前交会策略
- **场景 B**：目标卫星轨道发生异常漂移，正在无控逼近——系统应该立即进入避碰模式

仅靠轨道状态（位置 + 速度）无法区分这两种情况。区分它们需要：
- 历史轨迹分析（目标是否在持续主动接近？）
- 通信信号分析（目标是否在发送交会信号？）
- 异常检测（目标运动是否偏离历史模式？）

**这就是 Neural Runtime 的职责——它不关注"状态是什么"，而关注"状态意味着什么"。**

### 4.4 Space Cognitive State

Neural Runtime 持续维护 Space Cognitive State，描述系统对当前环境的理解：

| 认知类别 | 具体内容 | 来源 |
|---------|---------|------|
| Environment Understanding | 目标识别、空间碎片分布、电磁环境 | 传感器融合 + 学习 |
| Mission Understanding | 任务阶段判定、完成度估计、优先级 | 任务模型 + 学习 |
| Intent Estimation | 目标行为意图分类、未来动作预测 | 时序模型 + 学习 |
| Risk Assessment | 碰撞概率、资源风险、任务风险 | 概率模型 + 学习 |
| Experience Context | 相似历史场景检索、策略有效性评价 | Memory Runtime 提供 |

一个具体的运作示例：Physics Runtime 告诉系统"目标未来十分钟的位置坐标是 XXX"；Neural Runtime 判断"该目标正在主动接近，交会意图明确，当前风险等级低，建议继续当前交会策略"。

### 4.5 持续学习机制

Neural Runtime 不采用"地面训练 + 星上部署"的固定模式，而采用持续学习机制。更新来自三个方面：

**任务反馈**：系统根据任务执行结果评估当前认知模型是否准确，并增量更新。例如，若预测目标会保持当前轨道而实际上目标发生了机动，则更新意图预测模型。

**环境变化**：新的目标类型、新的轨道环境、新的空间事件都应促使系统更新认知模型。这是星上在线学习的能力。

**多智能体协同**：当多航天器组成网络时，Neural Runtime 可以共享认知经验——例如，一个航天器遇到的碎片云信息可以帮助其他航天器提前规划规避路径。

需要强调的是，持续学习不会改变 Physics Runtime 中的动力学模型（物理规律不变），仅更新认知模型（对环境的理解在变）。

---

## 第五章 Memory Runtime——长期经验积累与知识复用

### 5.1 为什么需要独立的 Memory Runtime？

前文定义了四种 Runtime，其中 Physics、Neural、Behavior 都有明确对应的章节，但 Memory Runtime 的设计动机需要特别说明。

Neural Runtime 虽然具备经验学习的能力，但它主要关注**当前任务周期内**的认知推理。Memory Runtime 解决的是一个不同层次的问题：**跨越多个任务、多个轨道周期的长期知识积累和复用**。

具体来说：

| 问题 | Neural Runtime 的回答 | Memory Runtime 的回答 |
|------|----------------------|---------------------|
| 当前目标的意图是什么？ | 分析当前观测序列 → 判断 | 检索历史上类似目标的行为模式 → 提供先验 |
| 当前轨道区域有什么风险？ | 分析当前环境数据 → 评估 | 检索该区域的历史事件记录 → 补充 |
| 当前策略效果如何？ | 监测执行结果 → 反馈 | 检索类似场景下各种策略的历史表现 → 提供参考 |

**Memory Runtime 是 PNIE 的"长期记忆"，而 Neural Runtime 的 Experience Context 是"工作记忆"。**

### 5.2 Memory Runtime 维护的知识类型

| 知识类别 | 内容 | 存储形式 | 更新方式 |
|---------|------|---------|---------|
| 任务历史 | 已完成任务的完整轨迹、决策点、结果 | 结构化时序数据 | 每个任务结束时写入 |
| 目标档案 | 已知目标的轨道特征、行为模式、交互记录 | 键值数据库 | 每次交互后更新 |
| 策略知识 | 不同场景下各种策略的效果评价 | 知识图谱 | 策略执行后增量更新 |
| 环境知识 | 不同轨道区域的特征（碎片密度、通信质量） | 空间索引 | 每次经过时更新观测 |
| 经验案例 | 具有代表性的成功/失败案例及其完整上下文 | 向量嵌入 | 定期筛选入库 |

### 5.3 经验检索机制

当其他 Runtime 需要历史知识支持决策时，Memory Runtime 提供两种检索方式：

**（1）上下文相似检索**：给定当前状态（轨道、目标类型、任务阶段），检索历史上最相似的场景及其处理方式。技术上可采用向量嵌入 + 相似度搜索实现。

**（2）结构化查询**：针对特定问题精确查询，如"在 500 km 太阳同步轨道上，过去 30 天内发生过多少次碎片近距离交会事件？"

检索结果以结构化上下文的形式注入 USS，供 Neural Runtime 和 Behavior Runtime 使用。

### 5.4 与其他 Runtime 的关系

```
Memory Runtime ──→ USS.Cognitive State (Experience Context)
                ──→ Neural Runtime (提供历史案例辅助认知推理)
                ──→ Behavior Runtime (提供历史策略效果辅助决策)

Neural Runtime ──→ Memory Runtime (写入新的认知经验和案例)
Behavior Runtime ──→ Memory Runtime (写入任务执行结果)
```

Memory Runtime 持续积累、从不删除（除非知识被证伪或过时），是 PNIE 实现"越飞越聪明"的关键机制。

---

## 第六章 Unified State Space 与 Unified Space Representation

### 6.1 设计动机

传统航天控制系统将状态定义为动力学状态向量（位置、速度、姿态），足以支撑轨道控制、姿态控制等传统任务。数据驱动方法引入了神经网络隐藏状态，但缺乏明确物理意义，难以与动力学状态建立统一关系。

PNIE 面临的是一个更根本的问题：**四个 Runtime 各自维护不同层次的状态，如果状态之间无法统一表达，整个系统就无法形成一致的推理。**

答案是将两种设计思想融合为一个统一的系统：

**Unified State Space（USS）** 定义了智能系统需要维护的**抽象状态空间**——回答"系统有哪些状态"的问题。

**Unified Space Representation（USR）** 是 USS 的**运行时实现**——回答"状态如何在软件中组织、更新和访问"的问题。

### 6.2 USS：统一状态空间的定义

设系统在时刻 $t$ 的统一状态表示为：

$$S(t) = \{ S_p,\ S_c,\ S_m,\ S_r \}$$

其中：
- $S_p$：Physical State
- $S_c$：Cognitive State
- $S_m$：Mission State
- $S_r$：Resource State

USS 不要求四类状态采用相同数据类型，只要求它们能在统一框架下组织、更新和访问。

### 6.3 四类状态的详细组成

| 状态类别 | 子项 | 由谁维护 |
|---------|------|---------|
| **Physical State** | 轨道状态、姿态状态、动力学环境、物理约束 | Physics Runtime |
| **Cognitive State** | 环境认知、意图推断、风险评估、经验上下文 | Neural Runtime + Memory Runtime |
| **Mission State** | 任务目标、当前行为、进度、优先级 | Behavior Runtime |
| **Resource State** | 推进剂、电源、热控、通信、计算资源 | Physics Runtime |

### 6.4 四类状态之间的动态耦合

USS 不是四类状态的简单拼接，而是一个**动态耦合状态空间**：

```
Physical State ────────────→ Resource State
   (轨道机动)               (推进剂减少、电力消耗)
       │
       ▼
Cognitive State ───────────→ Mission State
   (识别到碰撞风险)          (任务优先级调整)
       │                          │
       │                          ▼
       │                    Physical State
       │                  (新轨迹影响轨道演化)
       ▼
Resource State ──(推进剂不足)──→ Mission State
                             (限制可执行行为范围)
```

**整个智能系统的推理过程，就是不断更新这一动态耦合的状态空间。**

### 6.5 USR：USS 的运行时实现

USR 承担以下职责：

| 职能 | 说明 |
|------|------|
| 状态存储 | USS 中所有状态的唯一物理存储位置 |
| 并发控制 | 多个 Runtime 同时读写时的状态一致性保证 |
| 版本管理 | 每次状态更新产生版本号，支持追溯和回滚 |
| 访问接口 | 统一的 Read/Write API，解耦状态组织与 Runtime 实现 |
| 事件通知 | 关键状态变更时（如碰撞预警）主动通知相关 Runtime |

### 6.6 USR 的五项设计原则

**（1）唯一状态源（Single Source of Truth）**
整个系统任意时刻只存在一个统一状态。所有 Runtime 只能访问 USR，不允许维护独立的系统状态副本。

**（2）状态共享而非模块通信**
Runtime 之间不直接调用。例如，Behavior Runtime 不直接调用 Physics Runtime 获取轨道信息，而是从 USR 读取。Behavior Runtime 也不直接调用 Neural Runtime 获取风险评估，同样通过 USR。

**（3）异步运行**
不同 Runtime 具有不同的计算周期。Physics Runtime 的轨道传播可能每秒更新，Neural Runtime 的认知推理可能十秒更新一次，Behavior Runtime 可能仅在行为需要切换时重新评估。USR 不要求统一更新频率。

**（4）状态可追溯**
USR 中每一项状态记录其来源 Runtime、更新时间、可信度分数和变更历史。这为安全审查、异常分析和在线学习提供审计能力。

**（5）统一访问接口**
所有 Runtime 通过一致的 API 访问 USR——无论访问轨道状态、任务状态还是认知状态，接口形式相同。
```
Read(StateType, Query) → StateValue
Write(StateType, Key, Value, Metadata) → Version
```

### 6.7 数据组织

```
USR
├── Physical State
│   ├── Orbit:         {r, v, oe, epoch}
│   ├── Attitude:      {q, ω, mode}
│   ├── Environment:   {gravity, density, srp, third_body}
│   └── Constraint:    {visibility, collision, attitude_forbidden, thrust_limit}
├── Cognitive State
│   ├── SituationAwareness: {targets, debris, em_environment}
│   ├── IntentUnderstanding: {target_behaviors, future_predictions}
│   ├── RiskAssessment:      {collision_prob, resource_risk, mission_risk}
│   └── ExperienceContext:   {similar_cases, strategy_effectiveness}
├── Mission State
│   ├── Goal:            {task_type, target_id, success_criteria}
│   ├── CurrentBehavior: {behavior_node, phase, sub_phase}
│   ├── Progress:        {completed_steps, remaining_steps, time_remaining}
│   └── Priority:        {current_priority, preemption_rules}
└── Resource State
    ├── Propellant:  {remaining_mass, budget_per_phase}
    ├── Power:       {soc, solar_input, load_power, time_to_depletion}
    ├── Thermal:     {temperatures, heater_status}
    └── Communication: {next_pass, pass_duration, data_queue}
```

### 6.8 Runtime 与 USR 的交互循环

每个 Runtime 遵循统一的工作模式：

```
while (system_running):
    state = USR.Read(relevant_state_types)   // 读取当前状态
    updated_state = Compute(state)            // 根据自身职责完成更新
    USR.Write(updated_state, metadata)        // 写回
    Wait(next_cycle)                          // 等待下一周期
```

由于 Runtime 之间不直接交互，整个系统的耦合度从 O(N²) 降至 O(N)——每种 Runtime 只与 USR 耦合。

### 6.9 Behavior Runtime 与 Capability Library

PNIE 将**行为生成（What to do）** 与**能力执行（How to do）** 分离：

- **Behavior Runtime** 负责回答：当前应该执行什么行为？（"自主接近"还是"保持等待"？）
- **Capability Library** 负责回答：如何完成这一行为？（Navigation、Docking、Refueling 等具体实现）

Capability Library 是一组**可复用的能力插件**，不是持续运行的 Runtime。其能力包括：

| Capability | 功能 | 典型实现 |
|-----------|------|---------|
| Navigation | 相对导航、目标跟踪 | 扩展卡尔曼滤波 + 视觉惯性里程计 |
| Orbit Transfer | 轨道机动规划 | 脉冲/连续推力 Lambert 求解器 |
| Docking | 自主对接 | 基于视觉伺服的 MPC |
| Refueling | 推进剂补给 | 机械臂控制 + 流量管理 |
| Collision Avoidance | 碰撞规避 | 概率碰撞检测 + 规避机动规划 |

当 Behavior Runtime 决定当前行为为"自主接近"时，它不关心 Navigation Capability 内部采用的是什么算法（EKF？UKF？粒子滤波？），只调用其统一接口。这种设计保证能力模块的可替换性和可扩展性。

---

## 第七章 Behavior Runtime——行为生成机制

### 7.1 什么是 Behavior？

Behavior Runtime 是 PNIE 中负责自主行为生成的核心运行层。它不直接输出控制指令（那是 GNC 的职责），而是输出**行为（Behavior）**。

行为是一种**高于控制、低于任务**的智能表达：

```
任务（Mission）    : "对目标卫星完成推进剂补给"
  → 行为（Behavior） : "接近目标 → 姿态对准 → 对接 → 传输推进剂 → 撤离"
    → 轨迹（Trajectory）: "从当前轨道经过 3 次机动到达对接点"
      → 控制（Control）  : "打开推力器 1，50% 推力，持续 120 秒"
```

行为本身也是系统状态的一部分——随着环境变化、任务推进和资源变化，Behavior Runtime 不断重新评估当前行为。

### 7.2 行为生成的四阶段流程

每一次行为生成经历四个阶段：

**阶段一：状态感知**

Behavior Runtime 从 USS 读取当前完整的统一状态——物理状态（我在哪里？速度多少？能到哪里？）、认知状态（目标在做什么？环境有风险吗？）、任务状态（我现在处于哪个阶段？还需要完成什么？）、资源状态（我还有多少推进剂？电力够吗？）。

**阶段二：行为推理**

在此基础上回答四个问题：
1. 当前最重要的目标是什么？——依据 Mission State 中的优先级
2. 当前是否存在风险？——依据 Cognitive State 中的风险评估
3. 当前资源是否支持继续执行？——依据 Resource State 中的约束
4. 是否需要切换行为？——依据 Behavior Graph 中的迁移条件

行为推理的结果不是控制命令，而是新的行为状态（例如：由"轨道保持"切换为"主动接近"）。

**阶段三：能力选择**

行为确定后，Behavior Runtime 从 Capability Library 中选择对应能力。例如"自主接近"→ Navigation Capability，"姿态对准"→ Relative Attitude Capability。

**阶段四：行为反馈**

Capability 完成执行后，新的轨道状态、资源状态写回 USR。Behavior Runtime 在下一周期重新读取 USS，评估行为是否需要调整。

**四个阶段持续循环，不存在"决策结束"的时刻。**

### 7.3 行为层级

复杂航天任务需要分层组织行为：

| 层级 | 时间跨度 | 关注问题 | 决策者 |
|------|---------|---------|--------|
| **战略行为（Strategic）** | 数小时至数天 | 完成整个任务（补给、维修、编队维护） | Behavior Runtime |
| **战术行为（Tactical）** | 数分钟至数小时 | 完成当前子任务（接近、对接、传输） | Behavior Runtime |
| **执行行为（Execution）** | 秒至分钟 | 调用具体 Capability 并监测结果 | Capability |

层次化设计使 Behavior Runtime 能够在不同时间尺度上完成决策，而不会将长期规划与短期控制混合。这与人类驾驶员的行为模式一致——先决定变道（战略），再判断当前时机（战术），再打方向盘（执行）。

### 7.4 行为切换机制

行为切换不基于固定阈值或预设时间节点，而是基于 USS 的持续推理结果。以自主补给任务的完整行为演化为例：

```
轨道保持 ──→ 轨道转移 ──→ 自主接近 ──→ 姿态对准 ──→ 自主交会 ──→ 补给执行 ──→ 安全撤离
  │            │            │            │            │            │            │
  │ 任务激活    │ 到达转移    │ 进入相对    │ 姿态匹配    │ 对接成功    │ 补给完成    │ 安全距离
  │            │ 轨道        │ 导航范围    │             │            │            │ 确认
```

每一步切换由统一的 USS 联合条件决定，而非单维度的阈值判断。如果途中 Neural Runtime 检测到目标异常机动（认知状态变化），系统可能从"自主接近"切回"安全撤离"，而不会机械执行完整个预定义流程。

---

## 第八章 Behavior Graph——行为组织与任务演化

### 8.1 定义与设计动机

Behavior Graph 解决的核心问题是：**对于复杂航天任务，自主行为不是孤立的动作，而是由多个行为阶段按照一定逻辑连续演化形成的。**

它不是一个有限状态机（FSM），也不是预定义的任务流程脚本，而是 **USS 上的行为组织模型**——描述系统当前处于何种行为状态，以及不同状态之间如何迁移。

Behavior Graph 是由行为节点（Behavior Node）和行为迁移（Behavior Transition）组成的有向图：

$$G_{behavior} = (V_{node}, E_{transition})$$

其中 $V_{node}$ 表示行为节点集合，$E_{transition}$ 表示行为迁移关系。

与传统 FSM 的核心区别：

| 维度 | 有限状态机（FSM） | Behavior Graph |
|------|------------------|---------------|
| 迁移条件 | 固定规则（如距离 < 100m） | USS 多源联合条件 |
| 状态空间 | 离散状态集合 | USS 连续空间中定义 |
| 可解释性 | 有限 | 高（每种迁移条件可追溯到具体 USS 变化） |
| 动态性 | 静态图 | 可根据任务动态增删节点 |

### 8.2 Behavior Node

每个 Behavior Node 是具有明确语义的行为状态。一个节点包含以下字段：

| 字段 | 含义 | 示例（"自主接近"节点） |
|------|------|---------------------|
| Name | 行为名称 | "Autonomous Approach" |
| Goal | 行为目标 | 缩短与目标航天器的相对距离至 200m 以内 |
| Activation Condition | 激活条件 | 目标已识别 AND 交会窗口存在 AND 推进剂充足 |
| Completion Condition | 完成条件 | 相对距离 < 200m AND 相对速度匹配 |
| Recommended Capability | 推荐能力 | Navigation Capability |
| Safety Constraint | 安全约束 | 最小安全距离 50m，最大接近速度 1m/s |

Behavior Runtime 不关心 Capability 内部实现，只维护节点及其迁移关系。

### 8.3 Behavior Transition

行为迁移由 USS 实时状态联合决定，考虑以下维度：

| 维度 | 检查问题 |
|------|---------|
| 物理状态 | 当前是否满足下一阶段的动力学条件？ |
| 认知状态 | 当前环境是否确认安全？目标是否有异常？ |
| 任务状态 | 当前是否进入新的任务阶段？ |
| 资源状态 | 剩余推进剂、电力是否支持下一阶段？ |
| 历史经验 | 类似场景下该迁移是否曾导致失败？ |

只有当所有维度共同满足时，Behavior Runtime 才触发行为迁移。这不是简单的阈值判断，而是**基于多源状态融合的动态决策**。

### 8.4 Behavior Graph 的运行机制

Behavior Graph 在整个任务期间持续运行，不与任何一次规划绑定：

```
while (mission_active):
    USS = Read_Unified_State_Space()
    CurrentNode = Evaluate_Current_Node(USS)
    if CompletionCondition(CurrentNode, USS) and ActivationCondition(NextNode, USS):
        Transition(NextNode)
    Capability = CurrentNode.RecommendedCapability
    Execute(Capability)
    Wait(next_cycle)
```

整个过程中，Behavior Graph 始终保持连续更新，不存在一次性规划后长期执行的情况——这确保了系统能够根据环境变化动态调整行为路径。

---

## 第九章 Physics–Neural Co-Reasoning 与 Behavior Planner

### 9.1 协同推理的基本思想

前文分别介绍了 Physics Runtime（回答"世界如何演化"）和 Neural Runtime（回答"世界意味着什么"）。本章将它们统一到 **Physics–Neural Co-Reasoning** 框架中，这同时也是 Behavior Planner 的核心工作流程。

在三者协同的架构中，每个角色各司其职：

> **Physics Runtime**：如果什么都不改变，未来物理世界将如何演化？
> **Neural Runtime**：这种演化对于当前任务意味着什么？
> **Behavior Runtime**：基于以上，下一步应该采取什么行为？

协同推理不是多个模型串联调用，而是多个 Runtime 围绕 USS 持续更新认知和行为。

### 9.2 统一的推理与规划循环

Co-Reasoning 和 Behavior Planning 实际上是同一过程的两个视角——**Co-Reasoning 描述的是 Runtime 间的协同机制，Behavior Planning 描述的是决策算法的内部流程。** 合并后统一为五阶段循环：

**阶段一：状态获取**
Behavior Runtime 从 USS 读取统一状态：$S(t) = \{S_p, S_c, S_m, S_r\}$。

**阶段二：物理预测（Physical Reasoning）**
Physics Runtime 根据当前 $S_p$ 完成未来状态预测：
- 轨道演化：未来 N 步的 $\mathbf{r}_{t+k}, \mathbf{v}_{t+k}$
- 相对运动：目标与 chaser 之间的相对几何关系
- 可达域：在当前推进剂约束下可到达的轨道范围
- 资源消耗：完成各种可能行为所需的推进剂和电力

这一阶段回答：**如果什么都不改变，未来物理世界将如何演化？**

**阶段三：认知推理（Cognitive Reasoning）**
Neural Runtime 在物理预测基础上进行认知分析：
- 目标行为预测：目标是否会改变轨道？意图是什么？
- 环境风险评估：是否存在碎片碰撞风险？通信窗口何时结束？
- 任务上下文理解：完成度如何？是否有更高优先级任务抢占？
- 历史经验检索：Memory Runtime 提供类似场景的处理记录
- 不确定性估计：上述预测的置信度如何？

这一阶段回答：**未来演化对于当前任务意味着什么？**

**阶段四：行为推理与规划（Behavior Reasoning & Planning）**
Behavior Runtime 综合前两个阶段的输出，完成：
1. 候选行为生成：基于 USS 生成多个可行行为候选（Behavior Intent）
2. 物理约束筛选：Physics Runtime 剔除不可行候选
3. 行为评估：对剩余候选在任务完成度、安全性、资源消耗、长期收益、鲁棒性五个维度上评分
4. 最优行为确认：选择综合评分最高的行为

**阶段五：执行与状态更新**
选定的行为通过 Capability Library 执行，轨道状态、资源状态和任务进度更新写回 USR。下一周期从阶段一开始。

### 9.3 双预测机制

PNIE 同时维护两种不同性质的预测，二者不可相互替代：

| 特征 | 物理预测 | 认知预测 |
|------|---------|---------|
| 来源 | Physics Runtime（动力学模型） | Neural Runtime（学习模型） |
| 性质 | 确定性 | 概率性 |
| 输出 | 精确的轨道、速度、约束 | 意图概率、风险分数、置信区间 |
| 可解释性 | 完全可解释（源于动力学方程） | 需额外解释机制 |
| 回答 | "未来会发生什么" | "未来意味着什么" |

**Behavior Runtime 在两类预测共同作用下形成最终行为——而不是单独依赖任何一方。**

### 9.4 Physics-First 规划原则

Behavior Planner 的一个核心原则是 **Physics First**。与传统方法将物理模型作为安全检查不同，PNIE 在规划开始阶段就利用物理模型约束整个搜索空间：

- 候选行为需要的 ΔV 超过推进剂能力？→ 直接剔除，不进入后续评估
- 未来轨迹违反最小安全距离约束？→ 直接剔除
- 轨道动力学预测显示无法形成交会窗口？→ 不进入后续规划

大量不合理方案在规划早期即被物理模型排除。神经网络只需关注物理模型无法描述的不确定因素（意图、语义、风险），这大幅降低了神经网络的决策负担。

### 9.5 三重推理一致性约束

Co-Reasoning 不仅要求生成行为，还要求行为满足一致性：

**物理一致性**：所有行为必须满足轨道动力学和工程约束。任何违反动力学规律的候选行为均被拒绝。这是硬约束。

**任务一致性**：行为必须始终服务于当前 Mission State。即使某个行为满足动力学约束（技术上可行），如果偏离当前任务目标（"应该交会却变成避碰"），也不会被采纳。

**认知一致性**：Behavior Runtime 生成的行为必须与当前环境认知保持一致。例如，当 Neural Runtime 判断目标存在异常机动风险时，Behavior Runtime 不应继续执行接近行为，而应主动进入等待或重新规划。

### 9.6 与传统 AI Agent 的本质区别

近年来大语言模型 Agent 通常采用三阶段结构：
```
Observation → Reasoning → Action
```

PNIE 的推理循环与此有本质区别：

```
Observation → Physical Reasoning → Cognitive Reasoning → Behavior Planning → Capability Execution → State Update
```

关键差异：

| 维度 | LLM Agent | PNIE Agent |
|------|-----------|------------|
| 推理基础 | 语言（文本） | 物理（动力学） |
| 物理约束 | 无（靠 prompt 约束） | 内建（Physics Runtime 硬保证） |
| 决策来源 | 单一模型推理 | 物理 + 认知 + 规划 三重协同 |
| 安全保证 | 无 | Physics Runtime 持续验证 |
| 长期记忆 | 上下文窗口 | Memory Runtime 持久化 |

PNIE 是面向物理世界的 Agent，不是面向文本世界的 Agent。

---

## 第十章 Behavior Candidate Modeling——候选行为建模

### 10.1 行为表示的选择

Behavior Planner 的核心任务是在当前状态下寻找最优行为。但这首先要解决一个基础问题：**行为究竟该如何表示？** 行为表示不合理，则无论采用何种 AI 模型都只能在错误的表示空间中学习。

分析三种常见表示方式在 PNIE 中的适用性：

| 表示方式 | 定义 | 不适用的原因 |
|---------|------|-------------|
| **控制量（Control）** | 推力器推力、姿态力矩等执行指令 | 属于 GNC 职责，Behavior Planner 应关注"做什么"而非"如何控制" |
| **轨迹（Trajectory）** | 未来位置序列 $\{x_{t+1}, x_{t+2}, ..., x_{t+T}\}$ | 航天中轨迹由动力学决定，神经网络重复学习效率低且难保证物理一致性 |
| **行为标签（Behavior Label）** | 离散类别如 $\{\text{Approach}, \text{Wait}, \text{Retreat}, \text{Avoid}\}$ | 真实任务行为包含大量连续参数（接近距离、时间窗口、安全约束），离散标签表达能力严重不足 |

### 10.2 Behavior Intent：统一表示

PNIE 采用 **Behavior Intent（行为意图）** 作为 Candidate Behavior 的统一表示。它不是控制指令，不是轨迹，而是对未来行为目标的一种参数化描述：

$$Intent = \{\ Type,\ Goal,\ Constraint,\ TimeWindow,\ RiskPreference,\ Budget \ \}$$

以自主交会任务为例：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| Type | 行为类型 | `Approach` |
| Goal | 行为目标 | `{target_id: "GEO_SAT_1", target_distance: 200m}` |
| Constraint | 约束条件 | `{min_safety_distance: 50m, max_relative_velocity: 1m/s}` |
| TimeWindow | 时间窗口 | `{earliest_start: T0+300s, latest_end: T0+7200s}` |
| RiskPreference | 风险偏好 | `conservative` / `balanced` / `aggressive` |
| Budget | 资源预算 | `{delta_v_budget: 5m/s, propellant_budget: 200g}` |

Behavior Intent 描述的是系统希望达到的行为目标，而非达到目标的具体路径。随后 Physics Runtime 根据 Intent 推导未来物理状态，Behavior Planner 根据预测结果评价行为优劣，Trajectory Planner 再生成最终的可执行轨迹。

**这种设计实现了行为层与控制层的彻底解耦。**

### 10.3 Candidate Behavior Generator 的模型需求

Candidate Behavior Generator 不负责寻找唯一正确答案，而是提出多个具有潜在可行性的行为假设。模型需满足：

1. **一对多生成**：能输出多个候选 Intent（而非单一预测）
2. **连续参数**：能处理距离、时间、预算等连续量（而非仅有离散类别）
3. **上下文感知**：能结合 USS 中的历史经验和当前环境进行推理
4. **Physics Runtime 协同**：能在物理模型提供的约束空间内生成候选，而非独立决策

### 10.4 主流 AI 模型的适用性分析

| 模型 | 核心机制 | 优势 | 局限 | 在 PNIE 中的定位 |
|------|---------|------|------|-----------------|
| **Transformer** | 自注意力序列建模 | 上下文建模强、成熟高效、易于工程部署 | 判别式建模，不适合描述多模态不确定性 | **第一代 Behavior Intent Generator** |
| **Diffusion Model** | 逐步去噪生成 | 擅长学习复杂分布，天然支持一对多生成 | 推理需多步去噪（10-100步），实时性不足 | **离线候选生成 + 在线微调** |
| **Reinforcement Learning** | 策略梯度优化 | 直接学习长期收益最优策略 | 奖励函数设计困难（安全、燃料、时间难以统一量化），缺乏可解释性 | **辅助策略优化（不作为主决策器）** |
| **World Model（Dreamer等）** | 隐空间动力学学习+Imagined Rollout | 可在内部"想象"未来，不需真实环境交互 | 动力学完全由神经网络学习，忽略已有物理规律 | **不直接适用**（见10.5） |

### 10.5 Physics-guided World Model（长期目标）

当前主流 World Model（Dreamer、MuZero）存在一个根本问题：**状态转移 $s_{t+1} = f(s_t, a_t)$ 完全由神经网络学习。** 对航天任务而言，大量动力学规律已有解析模型，重新学习的必要性和可靠性都存疑。

本白皮书提出 **Physics-guided World Model**，对传统 World Model 进行重新诠释：

$$s_{t+1} = f_{physics}(s_t, a_t) \ +\ f_{neural}(s_t, a_t, c_t)$$

其中：

| 分量 | 含义 | 实现 |
|------|------|------|
| $f_{physics}$ | 轨道动力学、姿态动力学及环境模型 | Physics Runtime |
| $f_{neural}$ | 模型残差：未知扰动、执行器误差、未建模动力学 | Neural Runtime |
| $c_t$ | 环境认知上下文（意图、风险、经验） | USS → Cognitive State |

与传统 World Model 的关键区别：

| 维度 | 传统 World Model | Physics-guided World Model |
|------|-----------------|---------------------------|
| 动力学学习 | 纯黑盒神经网络 | 物理模型为主 + 残差学习 |
| 可解释性 | 低 | 高（物理部分完全可解释） |
| 数据需求 | 需要大量训练数据 | 仅需学习残差，数据需求显著降低 |
| 安全性 | 可能产生物理上不可能的预测 | Physics Runtime 硬约束保证 |
| 泛化能力 | 依赖训练数据覆盖 | 物理模型提供外推能力 |

**Physics-guided World Model 是 PNIE 的长期技术目标，但它的实现依赖于神经网络技术与物理模型的成熟融合机制，因此并非第一代系统的建设重点。**

### 10.6 第一代系统的工程建议

基于当前技术成熟度和航天工程对可靠性的要求，第一代 PNIE 建议采用分层实现方案：

```
┌───────────────────────────────────────────────────┐
│              Candidate Behavior Generator          │
│  ┌─────────────────────────────────────────────┐  │
│  │   Transformer Encoder (Behavior Intent)     │  │
│  │   Input: USS (Physical + Cognitive State)   │  │
│  │   Output: {Intent₁, Intent₂, ..., Intentₙ} │  │
│  └─────────────────────────────────────────────┘  │
│                       │                            │
│                       ▼                            │
│  ┌─────────────────────────────────────────────┐  │
│  │   Physics Constraint Filter                 │  │
│  │   剔除不可行候选                               │  │
│  └─────────────────────────────────────────────┘  │
│                       │                            │
│                       ▼                            │
│  ┌─────────────────────────────────────────────┐  │
│  │   Multi-Dimensional Scorer                  │  │
│  │   任务完成度 × 安全性 × 资源 × 长期收益       │  │
│  └─────────────────────────────────────────────┘  │
│                       │                            │
│                       ▼                            │
│               Selected Behavior                    │
│                       │                            │
│                       ▼                            │
│  ┌─────────────────────────────────────────────┐  │
│  │   Trajectory Planner (Capability Library)   │  │
│  │   将 Behavior Intent 转化为具体轨迹           │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

这一方案避免了直接学习复杂动力学，同时充分利用 Transformer 在序列建模和行为生成方面的优势。随着技术演进，Transformer 可逐步替换为 Diffusion 或 Physics-guided World Model——这正是 PNIE 可扩展原则的具体体现。

---

## 第十一章 场景演练：一次自主交会与补给任务

### 11.1 任务设定

为具体说明 PNIE 各组件如何协同工作，本章以一个典型任务为例进行全流程演练：

> **任务**：服务星 A 从 500 km 太阳同步轨道出发，接近目标星 B（同轨道，前方约 50 km），完成推进剂补给后返回原轨道。
> **初始状态**：两星相对距离 50 km，通信窗口 8 分钟后开始（持续 10 分钟）。
> **约束**：剩余推进剂 2 kg，最小安全距离 100 m，最大接近速度 2 m/s。

### 11.2 任务演进全流程

```
T=0s    系统启动，PNIE 初始化
         │
         ▼
        Physics Runtime:  加载轨道参数，开始持续传播
        Neural Runtime:   加载预训练认知模型
        Memory Runtime:   加载目标 B 的历史档案
        Behavior Runtime: USS 初始化完成，确定初始行为
         │
T=1s    USS 状态:
         ├─ Physical:     {r_A, v_A, r_B, v_B, Δr=50km, Δv≈0}
         ├─ Cognitive:    {Intent(B)="station-keeping"(置信度 85%)}
         ├─ Mission:      {Goal="rendezvous_and_refuel", Phase="init"}
         └─ Resource:     {propellant=2kg, power=95%}
         │
         ▼
        Behavior Runtime 推理:
         当前阶段 = "init"
         行为 = "轨道转移" (Behavior Graph 初始节点)
         调用 = Orbit Transfer Capability
         │
         ▼
        Trajectory Planner:
         Physics Runtime 计算可达域: 给定 2kg 推进剂，可达 Δv = 18 m/s
         规划 Hohmann 转移: 两次机动，总 Δv = 12 m/s，剩余裕量充足
         Physics Runtime 验证: ✓ 动力学一致 ✓ 推进剂充足 ✓ 安全距离满足
         │
         ▼
        GNC 执行第一次机动 (Δv = 6 m/s)
         │
T=1800s (30分钟后)  
         │
         机动完成，A 接近 B 约 5 km
         │
         ▼
        USS 更新:
         ├─ Physical:     Δr = 5km
         ├─ Cognitive:    Intent(B) 仍为 station-keeping
         └─ Mission:      Phase = "approach"
         │
         ▼
        Behavior Graph 评估:
         当前节点 "轨道转移" 完成条件 ✓ (已进入 10km 范围)
         下一节点 "自主接近" 激活条件 ✓ (目标稳定、交会窗口可用)
         → Behavior Transition: "轨道转移" → "自主接近"
         │
         ▼
        Neural Runtime 认知更新:
         分析 B 的历史轨道数据 → 确认 B 长期保持稳定，异常概率 < 5%
         Memory Runtime 检索 → 类似轨道上 3 次历史补给任务均成功
         认知一致性 ✓ (没有理由不继续)
         │
         ▼
        调用 Navigation Capability 执行接近
         │
T=3600s (1小时后)
         │
        A 距离 B 约 500 m
         │
         ▼
        Neural Runtime 异常检测:
         B 的轨道开始出现微小漂移 (Δv ≈ 0.05 m/s)
         Intent Estimation 重新评估: 
          "station-keeping" 置信度降至 60%
          "unknown_maneuver" 置信度升至 30%
         │
         ▼
        Risk Assessment: B 的不确定性增大，继续接近风险中等
        认知一致性检查: Neural Runtime 判断存在异常机动风险
         │
         ▼
        Behavior Runtime 重新推理:
         当前行为 "自主接近" 是否仍合适？
         Cognitive State → 中等风险
         Resource State → 推进剂充足 (剩余 1.5 kg)
         Mission State → 未完成补给
         │
         Behavior Planner 生成候选行为:
         ├─ Intent₁: {Type="Hold", Goal=维持当前距离, Risk=conservative}
         ├─ Intent₂: {Type="Approach", Goal=继续接近到 200m, Risk=balanced}
         └─ Intent₃: {Type="Retreat", Goal=返回安全距离 5km, Risk=conservative}
         │
         Physics-First 筛选: 三个均满足物理约束 → 全部保留
         行为评估:
         ├─ Intent₁: 任务完成度 2/10, 安全性 9/10, 资源 7/10 → 总分 6.0
         ├─ Intent₂: 任务完成度 8/10, 安全性 5/10, 资源 6/10 → 总分 6.3
         └─ Intent₃: 任务完成度 0/10, 安全性 10/10, 资源 9/10 → 总分 6.3
         │
         选择 Intent₂ (继续接近但保持警惕)
         │
         ▼
        更新认知状态: {RiskLevel="elevated", MonitorB=true}
        继续执行接近，同时高频监测 B 的轨道变化
         │
T=5400s (1.5小时后)
         │
        B 的轨道恢复正常 (漂移消失)
        Neural Runtime: Intent(B)="station-keeping" 置信度回升至 90%
         │
         ▼
        继续接近 → 姿态对准 → 对接 → 补给执行 → 安全撤离
         │
         ▼
T=10800s (3小时后)
         │
        任务完成
        Behavior Graph 进入终端节点 "任务完成"
        Memory Runtime 记录: 
         - 成功案例 ×1
         - 异常检测经验: B 在 T=3600s 的微小漂移为正常热控推进器脉冲
         - 目标 B 档案更新: 确认存在周期性推力器脉冲行为
```

### 11.3 关键观察

这个场景演练展示了 PNIE 的几个核心特征：

1. **持续推理**：系统在整个 3 小时任务中持续运行，每个决策周期都在重新评估状态
2. **Physics First**：B 轨道漂移后，Physics Runtime 立即完成新的轨道传播，为后续推理提供更新后的物理事实
3. **认知驱动的行为调整**：Neural Runtime 检测到 B 的异常后，Behavior Runtime 生成多个候选行为并进行综合评估，而不是盲目继续接近
4. **松弛耦合**：期间 Neural Runtime 更换意图识别模型、Memory Runtime 更新目标档案，都不影响其他 Runtime 运行
5. **经验积累**：任务结束后，B 的周期性推力器脉冲行为被记录到长期知识库，下次遇到类似特征时可快速识别

---

## 第十二章 总结与展望

### 12.1 核心贡献

本白皮书提出了一种面向未来空间具身智能的系统架构 PSIOS 及其核心引擎 PNIE。其核心贡献可以归纳为以下三个层面：

**范式层面**：提出 Physics-Native Intelligence 概念。与 Physics-Informed Learning 将物理规律作为训练约束不同，Physics-Native 将物理模型作为智能系统持续运行的**原生组件**——物理规律不是在训练阶段约束神经网络，而是在运行阶段成为智能推理的组成部分。

**架构层面**：提出以 Runtime 为基本单元的 PNIE 架构。四种 Runtime（Physics、Neural、Behavior、Memory）围绕 USS 持续协同运行，通过状态驱动（State-Centric）而非模块驱动的设计思想，实现了物理推理、认知推理和自主规划的深度融合。与传统架构最大的区别在于：物理模型不是"被神经网络调用"，而是"与神经网络平等运行"。

**方法层面**：提出 Behavior Intent 作为自主规划的统一建模对象。将行为从控制量和轨迹中解耦出来，通过"候选生成 → 物理筛选 → 综合评估"三阶段机制实现 Physics-First 自主规划。提出 Physics-guided World Model 作为长期技术路线——World Model 的动力学演化不由神经网络独学，而是物理模型负责确定性演化 + 神经网络补偿不确定残差。

### 12.2 与现有工作的关系

PSIOS 的设计融合了多个领域的成熟思想：

| 来源 | 借鉴的概念 | PNIE 的重新诠释 |
|------|-----------|----------------|
| 自动驾驶 PPPC 架构 | 感知→预测→规划→控制的分层思想 | 增加物理推理层，用 Physics Runtime 替代纯数据驱动的预测 |
| ROS/机器人中间件 | 节点间松耦合通信 | 状态驱动替代消息驱动，USS 作为唯一的状态汇集点 |
| World Model（Dreamer/MuZero） | 内部仿真 + Imagined Rollout | 将物理模型从黑盒中拆出，形成 Physics-guided 架构 |
| 操作系统设计 | 进程调度、资源管理、内核/用户态 | Runtime 对应进程，Physics-First 对应安全内核态 |

### 12.3 展望

**近期（1-2 年）**：实现第一代 PNIE 原型系统，采用 Physics Runtime（轨道传播器）+ Transformer Behavior Intent Generator + Behavior Planner 的分层方案，在数字孪生环境中验证自主交会和碰撞规避场景。

**中期（3-5 年）**：建立 Physics-guided World Model 原型。将 Physics Runtime 与 Neural Runtime 的协同从"串行推理"演进为"联合建模"，使系统能在物理约束下进行端到端的行为规划。

**长期（5-10 年）**：PNIE 从单航天器节点扩展为多航天器分布式智能网络。每个航天器运行本地 PNIE 实例，通过 USS 同步机制共享物理环境和任务知识，形成空间智能生态——从一颗卫星的自动驾驶，演进为整个空间基础设施的自主运行。

---

*人工智能的发展正在推动航天技术进入自主智能时代。未来真正的空间具身智能，不应仅依赖越来越大的神经网络，也不能完全依赖传统物理模型，而应建立在二者深度融合的基础之上。*
