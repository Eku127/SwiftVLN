# Global Token Clustering：跨帧 Token 合并方案

## 1. 问题背景

### 1.1 场景

在 VLN（Vision-Language Navigation）任务中，智能体在导航过程中积累大量历史观测帧。这些帧需要作为"记忆"输入给模型，但 token 数量会随步数线性增长，造成计算和显存压力。

### 1.2 目标

将 **N 帧历史观测**（每帧约 391 tokens）压缩到**固定 K 个 tokens**（如 704），同时尽可能保留语义信息。

### 1.3 核心思想

不同于传统的"采样+帧内压缩"方案，Global Token Clustering 将所有帧的 tokens 视为一个整体，基于**语义相似度**进行跨帧合并。相似的 token（如重复走廊的背景）会被合并，从而更高效地利用 token 预算。

---

## 2. 算法流程

### 2.1 输入输出

**输入**：
- N 帧历史观测的 ViT 特征
- 每帧 $L$ 个 tokens（Qwen2.5-VL 输出约 391 tokens/帧）
- 总计 $M = N \times L$ 个 tokens

**输出**：
- $K$ 个融合后的 tokens（如 704）

### 2.2 流程概览

```
Step 1: 拼接所有帧的 tokens
        ↓
Step 2: 初始化 K 个质心
        ↓
Step 3: 计算相似度 + Soft Assignment
        ↓
Step 4: 加权聚合得到 K 个输出 tokens
        ↓
(可选) 迭代 1-2 轮优化
```

---

## 3. 数学描述

### 3.1 符号定义

| 符号 | 含义 |
|------|------|
| $N$ | 历史帧数量 |
| $L$ | 每帧 token 数（约 391） |
| $M = N \times L$ | 总 token 数 |
| $K$ | 目标输出 token 数（如 704） |
| $d$ | 特征维度（hidden size） |
| $\mathbf{X} \in \mathbb{R}^{M \times d}$ | 所有 tokens 拼接后的特征矩阵 |
| $\mathbf{C} \in \mathbb{R}^{K \times d}$ | 质心矩阵（输出 tokens） |
| $\tau$ | 温度系数（Temperature） |

### 3.2 Step 1: Token 拼接

将 N 帧的 tokens 按时间顺序拼接：

$$
\mathbf{X} = \text{concat}(\mathbf{Z}_1, \mathbf{Z}_2, \ldots, \mathbf{Z}_N) \in \mathbb{R}^{M \times d}
$$

其中 $\mathbf{Z}_t \in \mathbb{R}^{L \times d}$ 是第 $t$ 帧的 ViT 输出特征。

### 3.3 Step 2: 质心初始化

从 token 池中采样 $K$ 个作为初始质心。

**采样策略**：均匀采样（或基于多样性的 K-Means++ 采样）

$$
\mathbf{C}^{(0)} = \text{Sample}(\mathbf{X}, K) \in \mathbb{R}^{K \times d}
$$

### 3.4 Step 3: 相似度计算与 Soft Assignment

**3.4.1 L2 归一化**

对 tokens 和质心做 L2 归一化，以便计算 cosine 相似度：

$$
\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}, \quad
\hat{\mathbf{c}}_j = \frac{\mathbf{c}_j}{\|\mathbf{c}_j\|_2}
$$

**3.4.2 Cosine 相似度矩阵**

$$
\mathbf{S} = \hat{\mathbf{X}} \cdot \hat{\mathbf{C}}^\top \in \mathbb{R}^{M \times K}
$$

其中 $S_{ij} = \cos(\mathbf{x}_i, \mathbf{c}_j)$ 表示第 $i$ 个 token 与第 $j$ 个质心的相似度。

**3.4.3 Temperature Scaling**

用温度系数 $\tau$ 控制分布的尖锐程度：

$$
\mathbf{S}' = \frac{\mathbf{S}}{\tau}
$$

- $\tau$ 小（如 0.1）：分布尖锐，token 集中分配给最相似的质心
- $\tau$ 大（如 1.0）：分布平滑，token 分散分配给多个质心

**3.4.4 Soft Assignment**

对每个 token，用 softmax 计算其对各质心的分配权重：

$$
\mathbf{A} = \text{softmax}(\mathbf{S}', \text{dim}=1) \in \mathbb{R}^{M \times K}
$$

其中 $A_{ij}$ 表示第 $i$ 个 token 分配给第 $j$ 个质心的权重，满足 $\sum_j A_{ij} = 1$。

### 3.5 Step 4: 加权聚合

每个质心通过加权平均聚合所有 tokens：

$$
\mathbf{c}_j^{\text{new}} = \frac{\sum_{i=1}^{M} A_{ij} \cdot \mathbf{x}_i}{\sum_{i=1}^{M} A_{ij}}
$$

矩阵形式：

$$
\mathbf{C}^{\text{new}} = \text{diag}\left(\mathbf{A}^\top \mathbf{1}\right)^{-1} \mathbf{A}^\top \mathbf{X}
$$

其中 $\mathbf{1} \in \mathbb{R}^{M}$ 是全 1 向量。

### 3.6 可选：迭代优化

重复 Step 3-4 共 1-2 轮，使质心更好地代表其聚类的 tokens。

---

## 4. Token Size 追踪（可选增强）

### 4.1 动机

当连续多轮合并时，某些质心可能代表了更多的原始 tokens。为避免信息稀释，需要追踪每个 token 的"权重"（token size）。

### 4.2 方法

为每个 token 维护一个权重 $w_i$，初始 $w_i = 1$。

**加权聚合时**：

$$
\mathbf{c}_j^{\text{new}} = \frac{\sum_{i=1}^{M} A_{ij} \cdot w_i \cdot \mathbf{x}_i}{\sum_{i=1}^{M} A_{ij} \cdot w_i}
$$

**权重更新**：

$$
w_j^{\text{new}} = \sum_{i=1}^{M} A_{ij} \cdot w_i
$$

---

## 5. 复杂度分析

| 步骤 | 复杂度 |
|------|--------|
| 相似度计算 | $O(M \times K \times d)$ |
| Softmax | $O(M \times K)$ |
| 加权聚合 | $O(M \times K \times d)$ |
| **总计** | $O(M \times K \times d)$ |

对于典型参数（$M=3000$, $K=704$, $d=1536$），单次前向约 3.2G 次乘法，在 GPU 上可高效完成。

---

## 6. 与其他方法对比

| 方法 | 压缩策略 | 跨帧去冗余 | 时间信息 |
|------|----------|-----------|----------|
| 均匀采样 + Pooling | 选 8 帧，每帧池化 | ❌ | ✅ 保留 |
| 均匀采样 + 帧内 ToMe | 选 8 帧，每帧 ToMe | ❌ | ✅ 保留 |
| **Global Token Clustering** | 全部帧，跨帧合并 | ✅ 强 | ❌ 丢失 |
| Segment ToMe | 分段，段内跨帧合并 | ✅ 中 | ✅ 粗粒度保留 |

---

## 7. 适用场景

Global Token Clustering 最适合以下场景：

1. **高冗余历史**：连续相似视角（如走廊、重复场景）
2. **token 预算紧张**：需要大压缩比
3. **不依赖精确时序**：任务更关注"去过哪些地方"而非"什么时候去的"

如果需要保留时间顺序信息，建议使用 **Segment ToMe**（分段方案）。

---

## 8. 总结

Global Token Clustering 的核心是：

> **把所有历史帧的 tokens 视为一个无序集合，基于语义相似度合并到固定数量的质心，实现高效的跨帧去冗余。**

关键公式：

$$
\mathbf{C} = \text{SoftKMeans}(\mathbf{X}, K, \tau) = \text{Normalize}(\mathbf{A}^\top) \cdot \mathbf{X}
$$

其中 $\mathbf{A} = \text{softmax}(\hat{\mathbf{X}} \hat{\mathbf{C}}^\top / \tau)$
