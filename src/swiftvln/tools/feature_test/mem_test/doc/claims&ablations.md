

1. **清晰的 Claims（你到底在主张什么）**
2. **对应的 Ablation Setting（怎么做对照，公平、可复现）**
3. **每个 Setting 应该用哪些指标来验证**
4. **你“预期看到什么结果”，以及为什么这样才说明 claim 成立**

---

# 总体实验前提（所有消融共享）

* **模型**：固定同一个训练好的 OverlapVLN checkpoint（uniform memory 训练）
* **prompt 结构**：完全一致（`<history_image>` 位置、长度一致）
* **memory token budget**：固定为 `8 × 88 = 704`
* **评测集**：同一批轨迹 step（含普通 + 回环轨迹）
* **不重新训练模型**

这点你可以在论文里明确写成：

> *We isolate the effect of memory design by keeping the VLN policy frozen.*

---

# Claim 1：

## 「历史信息的 *组织方式* 本身会影响 VLN 决策质量」

### 你在主张什么？

> 即使不改变模型、不重新训练，只要把“历史 token 选得更好、组织得更好”，模型的动作预测和指令对齐就会显著改善。

---

### Ablation Setting（最基础、必须有）

| ID | Memory 方式                 | 时间信息     | 说明                        |
| -- | ------------------------- | -------- | ------------------------- |
| A1 | Uniform frames (baseline) | ✔ hard   | 均匀取 8 帧 × pooling 到 88    |
| A2 | ToMe (global merge)       | ✖ none   | 所有历史 token 全局 merge 到 704 |
| A3 | ToMe (segment merge)      | ✔ coarse | 分 8 段，每段 ToMe → 88        |

---

### 用哪些指标来衡量？

**核心（必须）：**

* **Teacher-forcing next-action NLL / Acc**
  👉 直接反映“模型在同一上下文下是否更容易做出正确动作”

**辅助（解释性）：**

* Memory redundancy（token-level cosine similarity）
* History coverage（memory 是否覆盖历史视觉空间）

---

### 预期结果 & 如何解读？

* A2、A3 的 **next-action NLL < A1**
* A2、A3 的 **memory redundancy 明显更低**
* A3 通常 ≥ A2（因为保留了粗时间）

📌 **如果 A2/A3 在 frozen model 下就优于 A1，你就已经证明：**

> *memory organization is a first-order factor in VLN.*

---

# Claim 2：

## 「Token-level episodic memory 比 frame-based memory 更适合长时、重复场景」

（这是 ToMe 的核心价值）

---

### Ablation Setting

| ID | Memory 方式                   | 描述                      |
| -- | --------------------------- | ----------------------- |
| B1 | Frame-based (Uniform)       | 8 帧 × 88                |
| B2 | Frame-based (best sampling) | 你 Phase 1/2 最强采样        |
| B3 | ToMe (global, no time)      | 纯 episodic token memory |

---

### 指标（这里要换重点）

**核心：**

* **History reconstruction coverage**
  [
  \frac{1}{T}\sum_{u}\max_{m}\cos(e_u, m)
  ]
  👉 memory 是否能“代表整个历史”

* **Redundancy (NN-sim)**
  👉 是否大量浪费 token 在相似走廊/湖边

**决策验证：**

* next-action NLL（同 Claim 1）

---

### 预期结果

* B3 的 **history coverage > B1/B2**
* B3 的 **redundancy 最低**
* B3 的 next-action NLL ≤ B2

📌 **这条消融支撑的 claim 是：**

> frame 是一个过于刚性的 memory 单位，而 token-level memory 更贴近“我来过什么地方”。

---

# Claim 3（非常重要）：

## 「显式时间顺序不是必须的，但 *粗时间锚点* 是有帮助的」

（这是你 ToMe + soft timestamp 的关键论点）

---

### Ablation Setting（这是整篇论文最漂亮的一组）

| ID | Memory 方式                  | 时间处理             |
| -- | -------------------------- | ---------------- |
| C1 | ToMe                       | ✖ 无时间            |
| C2 | ToMe + soft τ (sinusoidal) | ✔ soft           |
| C3 | ToMe + segment τ           | ✔ coarse segment |
| C4 | Frame-based uniform        | ✔ hard time      |

---

### 核心指标（要和 Claim 匹配）

**决策类：**

* next-action NLL / Acc

**对齐类：**

* Instruction coverage（子句级）
* Memory token → instruction attention focus

**时间敏感子集（可选但很强）：**

* “回环 step”上的 next-action NLL

---

### 预期结果（这是你论文的“洞察点”）

* C1 > C4（说明：**硬时间并非必要**）
* C2 ≥ C1（说明：**软时间是有益的**）
* C3 ≥ C2（说明：**分段时间更稳**）

📌 你在论文里可以直接写：

> *Explicit temporal ordering is not required for effective memory usage. However, introducing lightweight temporal grounding further improves decision quality.*

---


# 最终：一张你可以直接用的消融表骨架（示意）

```
Table X: Ablation on Memory Design (Frozen Policy)

Memory            Time     NLL ↓   Acc ↑   Red ↓   HistCov ↑    ↑
---------------------------------------------------------------------------
Uniform (8×88)    Hard     1.23     41.2    0.78     0.62        0.31
ToMe (global)     None     1.15     44.5    0.41     0.79        0.56
ToMe + soft τ     Soft     1.11     46.1    0.42     0.78        0.63
ToMe + segment τ  Coarse   1.09     47.0    0.44     0.77        0.65
ToMe + retrieval  Soft     1.05     49.3    0.46     0.80        0.72
```

（数值是示意，但**趋势就是你要的故事**）

---

# 最后一句非常重要的话

你现在这套消融**不是在“比较模型”**，而是在：

> **系统性地回答：
> “VLN 中，历史信息应该以什么形式存在，模型才能真正用好它？”**

这已经是**一篇完整、成熟的 memory 论文问题意识**了。

如果你愿意，下一步我可以帮你直接把其中一个 Claim（比如 Claim 3 或 Claim 4）**写成论文里的一个完整 subsection（Method + Experiment + Discussion）**，你几乎可以原封不动地用。
