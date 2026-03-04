# Memory Evaluation Metrics

本文档描述了用于评估不同 Memory 策略的各项指标及其计算逻辑。

## 1. 评估框架概述

### 评估目标
在不重新训练模型的情况下，评估不同历史信息构建方式对 VLN 决策质量和记忆能力的影响。

### 评估方法
- **Teacher Forcing**：使用 ground truth 作为输入，计算模型预测下一步动作的能力
- **Chunk-level 评估**：每次预测 K=4 个动作作为一个评估单元

### 测试集构成
- **common** (100 samples)：历史长度 16-64，代表常见场景
- **long_trajectory** (30 samples)：历史长度 > 64，测试长轨迹记忆能力
- **early_segment** (30 samples)：历史长度 8-15，测试短历史场景

---

## 2. NLL (Negative Log-Likelihood)

### 定义
负对数似然，衡量模型对正确动作序列的预测置信度。

### 计算逻辑
1. 构建输入 prompt（包含 instruction、history memory、current observation）
2. 使用 Teacher Forcing：将 ground truth 动作序列作为目标
3. 计算模型对每个动作 token 的预测概率
4. NLL = -Σ log P(action_i | context)

### 解读
- **越低越好**：表示模型对正确动作更有信心
- 反映了 memory 构建方式对模型决策能力的影响

---

## 3. CE (Cross-Entropy)

### 定义
平均每个 token 的交叉熵，即 NLL 除以 token 数量。

### 计算逻辑
```
CE = NLL / num_action_tokens
```

其中 `num_action_tokens` 是动作序列的 token 数量（K=4 个动作对应的 tokens）。

### 解读
- **越低越好**
- 相比 NLL，CE 消除了序列长度的影响，更便于跨样本比较
- 通常 CE ≈ 0.5 表示模型表现良好

---

## 4. Redundancy (冗余度)

### 4.1 Frame-level Redundancy (帧级冗余)

#### 定义
衡量被选中的历史帧之间的相似程度。高冗余表示选择的帧过于相似，浪费了 memory 容量。

#### 计算逻辑
1. 对每个被选中的帧，将其 88 个 pooled tokens 进行 mean pooling，得到帧级表示 f_i
2. 对所有帧表示进行 L2 归一化
3. 计算帧间余弦相似度矩阵
4. 对每个帧，找到其最近邻（除自身外最相似的帧）
5. **RED_FRAME = 所有帧的最近邻相似度的平均值**

#### 公式
```
RED_FRAME = (1/N) × Σᵢ max_{j≠i} cos(fᵢ, fⱼ)
```

其中：
- N = 被选中的帧数量（通常为 8）
- fᵢ = 第 i 帧的 mean pooled 表示
- cos(·,·) = 余弦相似度

#### 解读
- **越低越好**：表示选择的帧更加多样化
- 预期：long_trajectory < common < early_segment
  - long_trajectory 跨越长历史，帧间差异大
  - early_segment 来自相近时间点，帧间差异小

### 4.2 Token-level Redundancy (Token级冗余)

#### 定义
衡量所有 memory tokens 之间的相似程度。

#### 计算逻辑
1. 将所有 memory tokens (8帧 × 88 tokens = 704 tokens) 进行 L2 归一化
2. 计算 token 间余弦相似度矩阵
3. 对每个 token，找到其最近邻
4. **RED_TOKEN = 所有 tokens 的最近邻相似度的平均值**

#### 公式
```
RED_TOKEN = (1/M) × Σᵢ max_{j≠i} cos(mᵢ, mⱼ)
```

其中：
- M = 总 token 数量（704）
- mᵢ = 第 i 个 memory token

#### 解读
- **越低越好**
- 区分度较 Frame-level 小，因为同一帧内的 tokens 天然相似
- 作为参考指标，观察 pooling 后的 token 冗余情况

### 4.3 两种 Redundancy 的对比

| 指标 | 计算对象 | 区分度 | 用途 |
|------|---------|--------|------|
| RED_FRAME | 8 个帧表示 | **高** | 评估选帧策略 |
| RED_TOKEN | 704 个 tokens | 低 | 参考，观察 pooling 效果 |

---

## 5. Coverage (覆盖率)

### 定义
衡量 memory tokens 对**完整历史轨迹**的表示能力。高覆盖率表示 memory 能够捕获整个历史的信息。

### 计算逻辑
1. 对**所有历史帧**（不仅是被选中的帧）提取 VIT 特征
2. 对每个历史帧进行 mean pooling 得到帧表示 eᵤ
3. 对所有表示进行 L2 归一化
4. 对每个历史帧，找到与其最相似的 memory token
5. **Coverage = 所有历史帧的最大相似度的平均值**

### 公式
```
Coverage = (1/T) × Σᵤ max_k cos(eᵤ, mₖ)
```

其中：
- T = 完整历史的帧数（可能远大于 8）
- eᵤ = 第 u 个历史帧的表示
- mₖ = 第 k 个 memory token

### 关键点
- **使用所有历史帧**，而非仅被选中的帧
- 这样才能真正反映 memory 对完整历史的覆盖程度

### 解读
- **越高越好**：表示 memory 能更好地表示完整历史
- 预期：
  - early_segment 应该较高（历史短，容易覆盖）
  - long_trajectory 可能较低（历史长，难以用 8 帧完全覆盖）

---

## 6. 指标间的关系

```
                    Memory 质量
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    决策能力         信息效率        表示能力
    (NLL/CE)       (Redundancy)    (Coverage)
         │              │              │
    越低越好     Frame: 越低越好    越高越好
                Token: 参考
```

### 理想的 Memory 策略
- **低 NLL/CE**：模型能准确预测动作
- **低 Redundancy**：选择的帧足够多样
- **高 Coverage**：能覆盖完整历史

### 指标间的 Trade-off
- 过低的 Redundancy 可能意味着选择了不相关的帧
- 需要在 Redundancy 和 Coverage 之间找到平衡

---

## 7. 当前基线结果 (Uniform + Pooling)

### 配置
- **Memory 策略**: Uniform 均匀采样
- **压缩方式**: 2D Average Pooling (stride=2)
- **历史帧数**: 8 帧
- **Tokens/帧**: 391 → 88 (压缩后)
- **总 Memory Tokens**: 8 × 88 = 704

### Overall 指标

| 指标 | Mean | Std |
|------|------|-----|
| NLL | 2.3287 | 1.7141 |
| CE | 0.5822 | 0.4285 |
| RED_FRAME | 0.9239 | 0.0342 |
| RED_TOKEN | 0.8943 | 0.0145 |
| Coverage | 0.7909 | 0.0269 |

### 分类别指标

| 类别 | 样本数 | NLL | CE | RED_FRAME | RED_TOKEN | Coverage |
|------|--------|-----|-----|-----------|-----------|----------|
| common | 100 | 2.25 ± 1.65 | 0.56 ± 0.41 | 0.922 ± 0.028 | 0.894 ± 0.013 | 0.791 ± 0.028 |
| long_trajectory | 30 | 2.90 ± 1.87 | 0.72 ± 0.47 | **0.889 ± 0.023** | 0.881 ± 0.009 | 0.779 ± 0.019 |
| early_segment | 30 | 2.04 ± 1.64 | 0.51 ± 0.41 | **0.967 ± 0.008** | 0.909 ± 0.011 | 0.803 ± 0.024 |

### 关键观察

#### 1. RED_FRAME 区分度明显
- **long_trajectory**: 0.889 (最低，帧间差异大)
- **early_segment**: 0.967 (最高，帧间差异小)
- **差异**: ~8%，符合预期

#### 2. RED_TOKEN 区分度较小
- 范围: 0.881 ~ 0.909
- **差异**: ~3%，因为同帧内 tokens 天然相似

#### 3. NLL/CE 反映决策难度
- **long_trajectory** 最高 (CE=0.72)：长历史决策更困难
- **early_segment** 最低 (CE=0.51)：短历史决策较容易

#### 4. Coverage 反映历史覆盖
- **early_segment** 最高 (0.803)：历史短，8帧容易覆盖
- **long_trajectory** 最低 (0.779)：历史长，8帧难以完全覆盖

### 结果可视化

结果文件保存在 `results/uniform/` 目录：
- `results.json` - 详细结果（含每个样本）
- `results.csv` - 表格汇总
- `bar_chart.png` - 分类别柱状图
- `box_plot.png` - 指标分布箱线图
- `scatter.png` - 历史长度 vs 指标散点图

---

## 8. 新增指标（提高区分度）

基于 [require.md](./require.md) 的建议，为解决 max-cosine 指标的"饱和"问题，新增以下指标：

### 8.1 Redundancy 扩展指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **RED_NN** | `(1/N) × Σᵢ max_{j≠i} cos(fᵢ, fⱼ)` | 原始指标，最近邻相似度 |
| **RED@K** | `(1/N) × Σᵢ (1/K) × Σ_{j∈TopK} cos(fᵢ, fⱼ)` | Top-K 平均，对"相似团簇"更敏感 |
| **RED_P90** | 90th percentile of pairwise similarities | 分位数指标，更稳定 |
| **RED_P95** | 95th percentile of pairwise similarities | 排除极端值的影响 |

**预期**：
- RED@K 对 long_trajectory vs early_segment 的区分度应比 RED_NN 更大
- P90/P95 分位数更稳定，适合作为主要对比指标

### 8.2 Coverage 扩展指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **COV_MAX** | `(1/T) × Σᵤ max_k cos(eᵤ, mₖ)` | 原始指标 |
| **COV_SOFT** | `(1/T) × Σᵤ (1/β) × log(Σₖ exp(β × cos))` | Log-Sum-Exp，平滑版 max |
| **COV@τ=0.7** | `(1/T) × Σᵤ 𝟙[max_k cos > 0.7]` | 阈值覆盖率，τ=0.7 |
| **COV@τ=0.8** | `(1/T) × Σᵤ 𝟙[max_k cos > 0.8]` | 阈值覆盖率，τ=0.8 |
| **COV_P10** | 10th percentile of per-frame coverage | 最差覆盖的帧 |
| **COV_P50** | Median of per-frame coverage | 中位数覆盖 |

**预期**：
- COV@τ 对 long_trajectory 应显著下降（更多帧低于阈值）
- COV_P10 反映"最差覆盖"，区分度应更明显

### 8.3 为什么原指标区分度小

```
原因分析：
1. max 操作的饱和效应
   - 只要有一个 memory token 相似，分数就高
   - 差异被压缩到很窄的区间

2. ViT 特征的各向异性
   - 向量都挤在一个锥体里
   - 任意两条向量 cosine 都偏高

3. Mean pooling 的平滑效应
   - 抹掉了局部差异
   - 使帧间更相似
```

---

## 9. 未来扩展

### 可添加的指标
- **Temporal Consistency**：memory 是否保留了时序信息
- **Attention Distribution**：模型对不同 memory tokens 的注意力分布
- **Information Preservation**：压缩前后的信息保留程度
- **Patch-level Coverage**：不做 mean pooling，直接用 patch tokens 匹配

### 可对比的 Memory 策略
- ToMe (Token Merging)
- Soft Token Compression
- Recency-weighted Sampling
- Keyframe Detection
