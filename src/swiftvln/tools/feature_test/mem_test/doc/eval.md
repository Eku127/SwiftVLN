好，这里我**重新给你生成一份“评估流程设计文档（Evaluation Protocol Document）”**，专门用于 **OverlapVLN + 冻结模型 + 不同 memory 机制** 的评估。
我会刻意把它写成一种**可以直接交给 Claude / 合作者执行的“规范文档”**，而不是讨论性文字。

> 重点：
>
> * **只用 OverlapVLN**
> * **只做 chunk（K=4）评估**
> * **不讨论训练，只讨论评估怎么做**
> * 每一种评估方法都写清楚：
>   **需要什么数据 / 怎么实现 / 衡量什么问题 / 为什么合理**

---

# OverlapVLN Memory Evaluation Protocol（评估流程文档）

## 0. 总体目标（Scope）

本评估流程的目标是：

> **在固定一个已完成 SFT 的 OverlapVLN 模型的前提下，
> 系统性评估不同历史信息（memory）构造方式
> 对 VLN 决策质量与记忆能力的影响。**

本流程 **不涉及任何再训练**，只在 **评测阶段替换 memory 构造方式**。

---

## 1. 全局固定条件（必须严格遵守）

这些条件在所有评估中 **完全一致**，否则结果不可比。

### 1.1 模型与参数

* 使用 **同一个已训练完成的 OverlapVLN checkpoint**
* 模型参数在评估阶段 **完全冻结**
* 不允许针对不同 memory 策略加载不同模型

### 1.2 Prompt 与接口

* 使用 **完全相同的 prompt 模板**
* `<history_image>` 占位符：

  * 位置固定
  * 接收 **固定数量的 memory tokens（704 = 8×88）**
* 当前观测图像、instruction、overlap_context、window_turns 的拼接方式保持不变

### 1.3 评估单位

* **唯一评估单位：chunk（K = 4）**
* 不做单步（single-step）评估
* 一个 chunk = 模型在一个 turn 中预测的 4 个动作

---

## 2. 评估数据构造流程（所有评估共享）

### 2.1 原始数据格式

每条导航 episode：

```json
{
  "id": 1803,
  "video": "images/scene_xxx_r2r_001803",
  "instructions": ["Go to the kitchen"],
  "actions": [-1, 1, 1, 2, 1, 0]
}
```

### 2.2 图像–动作对齐规则（必须遵守）

* 图像 `{i:03d}.jpg`：执行 `actions[i-1]` 后的观察
* 图像 `{i:03d}.jpg`：用于预测 `actions[i]`
* `actions[0] = -1` 不作为预测目标

### 2.3 Chunk 构造方式

* Chunk 长度：`K = 4`
* Chunk 起点索引：
  [
  i \in {1,\ 1+4,\ 1+8,\ \dots}
  ]
* 对每个起点 `i`：

  * 当前观测：图像 `I_i`
  * 目标动作序列：
    [
    (actions[i], actions[i+1], actions[i+2], actions[i+3])
    ]

### 2.4 STOP 的处理（统一策略）

* 当某个 chunk 中首次出现 STOP（动作 0）：

  * 该 chunk 仍然纳入评估
  * 该 episode 后续 chunks 不再评估
* 所有 memory 策略必须遵守同一规则

---

## 3. 评估执行方式（Teacher Forcing，核心原则）

### 3.1 执行原则

* **不使用 rollout，不与环境交互**
* **不调用 generate() 做采样**
* 使用 **teacher forcing** 方式计算模型对正确 chunk 的概率

### 3.2 Teacher Forcing 的具体含义

对于一个 chunk：

* 输入：

  * instruction
  * history memory（由某种 memory 策略构造）
  * overlap_context / window_turns（若使用完整 OverlapVLN 上下文）
  * 当前图像 `I_i`
* 目标：

  * 4 步动作拼接成的 **目标文本序列**

评估时：

* 强制将 **正确动作前缀**喂给模型
* 计算模型对 **下一个正确 token** 的对数概率
* 累加得到该 chunk 的 **Negative Log-Likelihood（NLL）**

---

# 4. 评估方法一：Chunk Teacher-Forcing NLL（核心评估）

## 评估 1：Chunk-level Teacher Forcing NLL / CE

### 需要的数据

* chunk 起点 `i`
* chunk 对应的目标动作文本（4 步）
* 模型在该 chunk 的输入 prompt（memory 不同，其余相同）

### 实现方法（逻辑）

1. 构造 chunk 输入 prompt（固定模板）
2. 将目标动作文本 tokenized
3. 使用 teacher forcing：

   * 对目标 token 序列逐 token 计算
     [
     -\log p(y_t \mid x, y_{<t})
     ]
4. 对整个 chunk 求和得到 NLL
5. 用 token 数归一化得到 CE（Cross-Entropy）

### 评估目的

* 衡量：
  **在完全相同的决策条件下，模型对“正确 4 步动作序列”的信心有多强**
* 用于回答：

  > 不同 memory 构造方式，是否让模型更容易“理解该怎么走”

### 为什么合理

* 与 OverlapVLN 的 SFT 训练目标完全一致
* 不受错误累积与随机采样影响
* 直接隔离 memory 的影响（唯一变量）

---

# 5. 评估方法二：Chunk-level Action Accuracy（辅助决策评估）

## 评估 2：Chunk 动作准确率指标

### 需要的数据

* 每个 chunk 的 ground-truth 动作序列
* 模型在 teacher forcing 下的 token-level预测

### 实现方法（逻辑）

从 teacher forcing 的输出中解析动作预测，计算：

1. **First-action Accuracy**

   * chunk 第 1 步动作是否预测正确
2. **Acc@4**

   * 4 步动作中，预测正确的比例
3. **Exact Match (EM@4)**

   * 4 步动作是否全部正确

### 评估目的

* First-action Acc：最接近“下一步决策质量”
* Acc@4 / EM@4：反映短期规划一致性

### 为什么合理

* 与 chunk 生成方式严格一致
* 对人类理解友好，可补充 NLL 的概率视角

---

# 6. 评估方法三：Memory 冗余度（Memory Quality）

## 评估 3：Memory Redundancy（冗余度）

### 需要的数据

* 某个 chunk 决策时使用的 memory tokens：
  [
  {m_1,\dots,m_{704}}
  ]

### 实现方法（逻辑）

* 对所有 memory token 计算两两相似度（cosine）
* 常用两种统计：

  1. 最近邻冗余：
     [
     \text{Red}*{nn} = \frac{1}{704}\sum_i \max*{j\neq i}\cos(m_i,m_j)
     ]
  2. 全局平均冗余（可选）

### 评估目的

* 衡量 memory 是否浪费大量 token 在高度相似内容上
* 判断 ToMe / merge 是否真正“压缩了重复信息”

### 为什么合理

* 与任务无关、纯 memory 结构指标
* 能解释“为什么 token-level memory 有效”

---

# 7. 评估方法四：历史覆盖率（Episodic Representation）

## 评估 4：History Coverage

### 需要的数据

* 历史中每一帧的全局视觉特征：
  [
  e_1,\dots,e_T
  ]
* 当前决策使用的 memory tokens：
  [
  m_1,\dots,m_{704}
  ]

### 实现方法（逻辑）

* 对每个历史帧特征 (e_u)，计算：
  [
  \max_k \cos(e_u, m_k)
  ]
* 对所有历史帧取平均：
  [
  \text{Cov}_{hist} = \frac{1}{T}\sum_u \max_k \cos(e_u,m_k)
  ]

### 评估目的

* 判断 memory tokens 是否能“代表”整个历史轨迹
* 特别适用于：

  * ToMe
  * episodic / place-based memory

### 为什么合理

* 不依赖时间顺序
* 能评估“是否记住来过的地方”

---


---

# 9. 评估结果输出（标准化）

### 主表（每种 memory 一行）

* CE ↓
* First-action Acc ↑
* Acc@4 ↑
* EM@4 ↑
* Red_nn ↓
* Cov_hist ↑

### 子集表（可选）

* 长轨迹后段 chunks
* window slide 后的新 chunks
* loop chunks

---

## 最后一句总结（给你也给 Claude）

> **这个评估流程的核心思想是：
> 在完全固定 OverlapVLN 决策模型的前提下，
> 用 teacher forcing + chunk 评估，
> 把“memory 是否更好”拆解成可度量、可解释的多个维度。**
