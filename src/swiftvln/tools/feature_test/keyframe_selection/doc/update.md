# 特征显著性优化策略

## 1. 问题背景

在基于视觉惊奇度（Cosine Surprise）的关键帧选择中，我们发现全局平均池化（Mean Pooling）存在**过度平滑（Oversmoothing）**问题：

- VLN 导航场景中，大部分画面是"墙壁、地板、天花板"等背景
- 对一张包含 90% 背景和 10% 新地标的图像做平均池化，那个 10% 的"惊奇"会被淹没在背景中
- 导致不同帧的特征高度相似（cosine similarity > 0.95），区分度极低

**目标**：找到能够放大特征显著性的聚合策略，使得视觉变化能够在特征层面产生更大的差异。

---

## 2. Token 预处理方法

ViT 模型输出的 patch tokens 数量较多（如 DINOv2 为 256 个，Qwen-VL 为 391 个），在进行池化之前可以先进行 Token 压缩。

### 2.1 Direct（直接处理）

不进行任何压缩，直接对所有 tokens 应用池化。

- **特点**：保留全部信息
- **缺点**：背景 tokens 占主导，可能稀释显著特征

### 2.2 ToMe（Token Merging）- 一步软聚类

采用一步 Soft K-Means 的方式将 $N$ 个 tokens 压缩到 $K$ 个：

1. **初始化聚类中心**：使用空间平均池化的结果作为初始锚点
   $$C_{init} = \text{AvgPool2D}(\mathbf{P})$$
   
2. **计算注意力权重**：每个 token 对每个中心的贡献权重
   $$A_{ij} = \text{softmax}\left( \frac{p_i \cdot c_j}{\sqrt{D}} \right)$$

3. **加权聚合**：
   $$c'_j = \sum_{i=1}^{N} A_{ij} \cdot p_i$$

**优势**：
- 利用空间先验保持粗粒度的位置信息
- 通过注意力机制让语义相近的 tokens 聚合
- 计算高效，只需一次矩阵乘法

---

## 3. 特征聚合策略（Pooling）

设预处理后的 tokens 为 $\mathbf{P} = \{p_1, p_2, ..., p_K\}$，其中 $p_i \in \mathbb{R}^D$。

### 3.1 Mean Pooling（均值池化）

$$f_{mean} = \frac{1}{K} \sum_{i=1}^{K} p_i$$

- **特点**：关注整体平均特征
- **问题**：背景占主导时，显著特征被稀释

### 3.2 Max Pooling（最大值池化）

$$f_{max,j} = \max_{i \in [1,K]} (p_{i,j}), \quad j = 1, 2, ..., D$$

- **特点**：在每个特征维度上取最大值
- **优势**：突出最显著的特征响应

### 3.3 Hybrid Pooling（混合池化）

$$f_{hybrid} = [f_{mean}; f_{max}] \in \mathbb{R}^{2D}$$

- **特点**：同时保留整体和局部显著信息
- **代价**：特征维度翻倍

### 3.4 GeM Pooling（广义均值池化）

$$f_{GeM} = \left( \frac{1}{K} \sum_{i=1}^{K} p_i^{\,p} \right)^{1/p}$$

- $p = 1$ 等价于 Mean，$p \to \infty$ 趋近于 Max
- 常用 $p = 3$ 或 $p = 5$

### 3.5 Patch-wise Similarity（逐 patch 相似度）

不进行池化，直接比较对应位置的 patch：

$$Sim(I_t, I_{t-1}) = \frac{1}{K} \sum_{i=1}^{K} \cos(\hat{p}_{t,i}, \hat{p}_{t-1,i})$$

---

## 4. 特征中心化（Centering）

对于池化后的特征序列 $\{f_1, f_2, ..., f_T\}$：

1. **计算全局均值**：$\bar{f} = \frac{1}{T} \sum_{t=1}^{T} f_t$
2. **中心化**：$f'_t = f_t - \bar{f}$
3. **归一化**：$\hat{f}_t = \frac{f'_t}{\|f'_t\|}$

**原理**：ViT 特征在高维空间中分布在狭窄的锥形区域（各向异性），导致任意两个向量的 cosine similarity 都很高（0.8-0.99）。中心化将特征分布拉到原点附近，显著增大区分度。

---

## 5. 组合策略

### 5.1 Max Centered（直接最大值池化 + 中心化）

1. 对每帧应用 Max Pooling：$f_{t,j} = \max_{i} (p_{t,i,j})$
2. 计算全局均值并中心化
3. 归一化后计算惊奇度：$surprise_t = 1 - \hat{f}_t \cdot \hat{f}_{t-1}$

### 5.2 ToMe + Max Centered（ToMe 压缩 + 最大值池化 + 中心化）⭐ 最优

1. **ToMe 压缩**：$N$ tokens → $K$ tokens（如 256 → 64）
2. **Max Pooling**：$K$ tokens → 1 feature
3. **中心化 + 归一化**：消除各向异性
4. **计算惊奇度**

**为什么这个组合最好**：
- ToMe 预处理：语义相近的 tokens 聚合，突出显著区域
- Max Pooling：保留最强的特征响应
- 中心化：拉大特征差异

---

## 6. 评估指标

### 6.1 Surprise Range（惊奇度范围）

$$Range = \max_t(surprise_t) - \min_t(surprise_t)$$

- 范围越大，特征区分度越高

### 6.2 Turn/Forward Ratio（转弯/前进比率）

$$Ratio = \frac{\text{mean}(surprise_{turn})}{\text{mean}(surprise_{forward})}$$

- 比率 > 1 说明能有效区分转弯帧

---

## 7. 实验结果

使用 DINOv2-base 在 Habitat 测试数据（13 个序列）上的完整对比：

### 7.1 Direct Pooling（无预处理）

| 策略 | simple | turns | spin | AVG Range |
|------|--------|-------|------|-----------|
| mean | 0.0826 | 0.1865 | 0.2268 | 0.1653 |
| max | 0.0183 | 0.0283 | 0.0400 | 0.0289 |
| hybrid | 0.0213 | 0.0324 | 0.0552 | 0.0363 |
| gem_p3 | 0.0201 | 0.0306 | 0.0553 | 0.0353 |
| patchwise | 0.1857 | 0.4668 | 0.1968 | 0.2831 |
| mean_centered | 0.4203 | 0.5098 | 0.4549 | 0.4617 |
| **max_centered** | **0.4705** | **0.5225** | **0.5071** | **0.5000** |

### 7.2 ToMe Preprocessing（256 → 64 tokens）

| 策略 | simple | turns | spin | AVG Range |
|------|--------|-------|------|-----------|
| tome_mean | 0.0963 | 0.1894 | 0.2303 | 0.1720 |
| tome_max | 0.0566 | 0.0740 | 0.0729 | 0.0678 |
| tome_hybrid | 0.0554 | 0.0782 | 0.0878 | 0.0738 |
| tome_gem_p3 | 0.0322 | 0.0486 | 0.0735 | 0.0514 |
| tome_patchwise | 0.1742 | 0.4606 | 0.2450 | 0.2933 |
| tome_mean_centered | 0.4364 | 0.5061 | 0.4271 | 0.4565 |
| **tome_max_centered** | **0.6682** | **0.5847** | **0.6038** | **0.6189** |

### 7.3 关键发现

1. **ToMe + Max Centered 综合最优**：
   - 平均 Surprise Range: **0.6189**
   - 相比 Direct Max Centered (0.5000) 提升 **24%**

2. **中心化是核心改进**：
   - 无中心化的策略（mean, max, hybrid, gem）效果都较差
   - 中心化后效果大幅提升

3. **ToMe 预处理的效果**：
   - 对 Max Centered 效果显著（0.5000 → 0.6189）
   - 对简单池化方法效果有限（mean: 0.1653 → 0.1720）
   - 原因：ToMe 聚合后减少了背景干扰，突出了显著区域

4. **在 simple（直行）场景表现尤其突出**：
   - tome_max_centered: 0.6682
   - direct_max_centered: 0.4705
   - 提升 **42%**

---

## 8. 结论与建议

### 推荐策略排序

| 排名 | 策略 | AVG Range | 特点 |
|------|------|-----------|------|
| 1 | **tome_max_centered** | 0.6189 | 最优，需要 ToMe 预处理 |
| 2 | max_centered | 0.5000 | 次优，无需预处理 |
| 3 | mean_centered | 0.4617 | 简单易用 |
| 4 | patchwise | 0.2831 | 保留空间信息 |

### 实施建议

1. **首选：ToMe + Max Pooling + Centering**
   - Token 压缩比例：4x（256 → 64）
   - 能够有效突出视觉变化

2. **在线处理方案**：
   - 使用滑动窗口的局部均值进行近似中心化
   - 窗口大小建议 10-20 帧

3. **模型选择**：
   - 建议使用 DINOv2，其低层特征对视觉变化更敏感
   - Qwen-VL ViT 的语义特征过于平滑，区分度较低
