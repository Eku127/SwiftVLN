# UAV→Satellite Visual Alignment 设计文档（Qwen2.5-VL-3B 版本）

## 1. 目标

### 1.1 项目背景

当前已有一个基于 **satellite image + instruction** 训练的 VLN policy，基础模型使用 **Qwen2.5-VL-3B**。现在目标是将输入从 satellite image 扩展到 **真实 UAV nadir image**，使得模型在没有真实 UAV-VLN 标注数据的前提下，尽可能复用已有 satellite policy 的能力。

当前可用数据：

* `24k` 对真实配对图像：`(uav_image, sat_image)`
* 暂无真实 UAV-VLN trajectory / action 标注

### 1.2 第一阶段目标

第一阶段只做 **视觉域对齐**，不直接训练完整 policy。

目标为：

1. 让 `uav_image` 经过一个 `UAV encoder / adapter` 后，输出的视觉 token / embedding 尽量接近 `sat_image` 经原始 Qwen vision encoder 输出的表示。
2. 让这个表示既具备：

   * **跨域判别性**（contrastive）
   * **teacher 空间兼容性**（distillation）
3. 最终可以将 `uav_image` 的输出替代原本的 `sat_image` 输入，接到后续已有 policy 中。

---

## 2. 核心思路

整体思路：

* 保留原始 `satellite path` 作为 teacher
* 新增 `UAV path`，学习把真实 UAV 图映射到 teacher 使用的视觉表征空间
* 使用两类监督：

  * `contrastive loss`：保证跨域表示空间里，配对图像更近、非配对图像更远
  * `feature distillation loss`：保证 UAV path 输出尽量接近 satellite teacher 输出

一句话概括：

> contrastive 负责“分得开、对得上”；distillation 负责“长得像、接得上”。

---

## 3. 模型结构设计

## 3.1 总体结构

建议采用如下结构：

```text
                  ┌──────────────────────────────┐
                  │  Qwen2.5-VL vision encoder   │  (frozen teacher path)
                  └──────────────┬───────────────┘
                                 │
                        sat_image │
                                 ▼
                         sat_visual_tokens
                                 │
                           sat_global_feat

uav_image
   │
   ▼
┌─────────────────────┐
│ UAV encoder/adapter │  (trainable)
└──────────┬──────────┘
           │
           ▼
     uav_visual_tokens
           │
      uav_global_feat
```

训练时：

* `sat_image -> teacher path`
* `uav_image -> trainable UAV path`
* 两条路径输出用于计算 contrastive + distill

部署时：

* 输入 `uav_image`
* 只走 `UAV encoder / adapter`
* 输出 token 接原先 policy

---

## 3.2 UAV path 的两种推荐实现

### 方案 A：独立 UAV encoder（推荐作为强 baseline）

**结构：**

* 复制一份 Qwen vision encoder 的结构或轻量化版本
* 初始化：

  * 可从原 Qwen vision encoder 权重初始化
  * 然后只训练 UAV encoder
* teacher 的 satellite vision encoder 完全冻结

**优点：**

* 表达能力强
* 容易直接拟合 teacher feature

**缺点：**

* 参数较大
* 训练成本高

适用场景：

* 显存允许
* 想先做一个上限较高的版本

---

### 方案 B：UAV adapter（更推荐工程落地）

**结构：**

* 直接复用原 Qwen vision encoder 作为 backbone
* 在输入侧或中间层插入一个轻量 trainable adapter
* satellite path：原始 Qwen vision encoder，冻结
* uav path：共享 backbone + 独立 adapter，或者复制 backbone 只训 adapter/projector

可以考虑的 adapter 形式：

1. **输入侧 adapter**

   * 在 patch embedding 后接 2~4 层 transformer block
2. **中间层 adapter**

   * 在 vision encoder 的若干 block 后加入 residual adapter
3. **输出侧 adapter/projector**

   * 对最终视觉 token 再做几层 MLP / transformer

推荐最小可行版本：

* 冻结整个 Qwen vision encoder
* 新增一个 `UAVTokenAdapter`
* 输入是 Qwen encoder 输出 token
* 输出是对齐后的 token

```text
uav_image -> frozen Qwen vision encoder -> raw_uav_tokens -> UAVTokenAdapter -> aligned_uav_tokens
sat_image -> frozen Qwen vision encoder -> sat_tokens
```

**优点：**

* 参数少
* 不容易破坏原有视觉表征
* 更适合 24k 数据量

**缺点：**

* 表达能力上限低于独立 encoder

适用场景：

* 第一版实现
* 数据不大、先求稳定

---

## 3.3 推荐的工程选择

优先顺序建议：

### V1（最推荐）

* `sat_encoder`: frozen Qwen2.5-VL vision encoder
* `uav_encoder`: frozen Qwen2.5-VL vision encoder
* `uav_adapter`: trainable transformer/MLP adapter
* 只训练 `uav_adapter + projection heads`

### V2

* `sat_encoder`: frozen
* `uav_encoder`: 从 sat_encoder copy 初始化，部分解冻后几层
* `uav_adapter`: trainable

### V3

* 独立 UAV encoder 全部可训

先从 **V1** 做起。

---

## 4. 输出表示定义

建议保留两级表示：

### 4.1 Token-level feature

记作：

* `T_sat ∈ R^{B×N×D}`
* `T_uav ∈ R^{B×N×D}`

其中：

* `B`: batch size
* `N`: token 数
* `D`: hidden dim

Token-level feature 用于：

* token distillation
* 可选 local alignment
* 后续接 policy

### 4.2 Global feature

记作：

* `g_sat ∈ R^{B×D}`
* `g_uav ∈ R^{B×D}`

生成方式可选：

1. mean pooling over tokens
2. CLS token（如果结构有）
3. attention pooling

推荐：**mean pooling + L2 normalize**

用于：

* contrastive learning

---

## 5. Loss 设计

总损失：

```math
L_total = λ_contrast * L_contrast + λ_global * L_global_distill + λ_token * L_token_distill
```

初始建议：

* `λ_contrast = 1.0`
* `λ_global = 1.0`
* `λ_token = 0.5`

后续再调。

---

## 5.1 Contrastive Loss

### 5.1.1 目的

约束：

* 配对 `(uav_i, sat_i)` 的 global feature 靠近
* 不配对 `(uav_i, sat_j)` 拉远

### 5.1.2 特征准备

```python
g_uav = normalize(pool(T_uav))
g_sat = normalize(pool(T_sat))
```

### 5.1.3 InfoNCE / CLIP 风格 loss

相似度矩阵：

```math
S_{ij} = g_uav_i · g_sat_j / τ
```

其中 `τ` 为 temperature。

双向 loss：

```math
L_u2s = CE(S, labels=[0,1,2,...,B-1])
L_s2u = CE(S^T, labels=[0,1,2,...,B-1])
L_contrast = (L_u2s + L_s2u) / 2
```

### 5.1.4 实现要点

* batch 内其他样本天然作为负样本
* 若支持多卡，建议 all-gather features 扩大负样本池
* temperature 初始可设 `0.07`

### 5.1.5 可选增强

可加入 hard negative：

* 同区域不同位置的 sat 图
* 相邻位置但不完全重合的样本

但 V1 不必先做。

---

## 5.2 Global Feature Distillation Loss

### 5.2.1 目的

约束 UAV global feature 直接逼近 satellite teacher feature。

### 5.2.2 推荐公式

使用 cosine distillation：

```math
L_global_distill = 1 - cosine(g_uav, g_sat)
```

或 L2：

```math
L_global_distill = ||g_uav - g_sat||_2^2
```

推荐优先用 cosine，因为更稳定。

---

## 5.3 Token-level Distillation Loss

### 5.3.1 目的

让 UAV token 表示接近 sat token 表示，以便后续可无缝接 policy。

### 5.3.2 推荐公式

```math
L_token_distill = mean(||T_uav - T_sat||_2^2)
```

也可对 token 先做 normalize 后再做 cosine。

### 5.3.3 注意事项

只有当 UAV 图和 sat 图在视角/裁剪上比较一致时，逐 token 对齐才更可靠。
如果存在较大空间偏移，建议：

* V1 先用 global distill 为主
* token distill 权重设小一点

初始建议：

* `λ_token = 0.25 ~ 0.5`

---

## 5.4 Projection Head

### 5.4.1 作用定位

Projection head 的主要作用不是给最终 policy 直接使用，而是为 **contrastive learning** 提供一个更合适的 embedding 空间。

推荐形式：

```text
T_uav -> pool -> projection_head -> g_uav
T_sat -> pool -> projection_head -> g_sat
```

其中：

* `T_uav`, `T_sat` 是 token-level 表示
* `g_uav`, `g_sat` 是用于 contrastive loss 的 global embedding

### 5.4.2 为什么需要 Projection Head

不直接用 backbone / adapter 输出的 pooled feature 做 contrastive，主要有以下原因：

1. **将“对比学习空间”和“下游任务空间”解耦**

   * token-level 表示最终要接入 VLN policy
   * contrastive 目标更偏向构建判别性 embedding
   * 两者不一定完全一致

2. **减少 contrastive 对主干表示的直接扰动**

   * 若直接在主特征空间做对比学习，容易让主空间过度服务于检索/匹配
   * projection head 可作为训练时的缓冲层

3. **提升对比学习效果**

   * 实践上，CLIP/SimCLR 一类方法通常会在投影空间中进行 contrastive optimization
   * 主干表示则更多保留下游任务可用信息

### 5.4.3 与 UAV Adapter 的区别

#### UAV Adapter

* 处理对象：`token-level` 表示
* 目标：将 UAV 域特征变换到更接近 satellite teacher 的表征空间
* 直接影响：后续是否能无缝接入原有 VLN policy
* 部署时：**需要保留**

#### Projection Head

* 处理对象：`global pooled` 表示
* 目标：构造更适合 contrastive loss 的 embedding 空间
* 直接影响：对比学习的判别性与训练稳定性
* 部署时：**通常不需要保留**

一句话总结：

> UAV adapter 是给 policy 用的；projection head 是给 contrastive loss 用的。

### 5.4.4 Projection Head 如何回传优化到 Adapter

是的，projection head 虽然是为了构造 contrastive loss 而存在，但这个 loss 在反向传播时，不只是更新 projection head 自身，也会继续回传到其上游模块。

在推荐实现中：

```text
uav_image -> frozen Qwen vision encoder -> raw_uav_tokens -> UAV adapter -> pool -> projection head -> contrastive loss
```

因此：

* `contrastive loss` 会更新 `projection head`
* 也会通过梯度继续更新 `UAV adapter`
* 若 `uav_backbone` 被解冻，也会继续更新 backbone 对应层
* frozen 的 `sat_encoder` 不更新

换句话说：

> projection head 负责提供“对比学习的优化接口”，而真正被这个 loss 推着变好的核心模块，是上游的 UAV adapter（以及任何你选择解冻的 student 路径模块）。

### 5.4.5 实践建议

推荐理解为：

* `projection head` 是训练辅助头
* `UAV adapter` 是核心迁移模块

因此：

* 训练时保留 projection head，用它来构造 contrastive objective
* 部署时通常移除 projection head，只保留能输出 `aligned_uav_tokens` 的主路径

推荐结构：

* 2-layer MLP
* hidden dim = `D`
* output dim = `256` 或 `512`

---

## 6. 数据处理

## 6.1 输入格式

每条样本：

```python
{
    "uav_image": ...,
    "sat_image": ...,
    "pair_id": ...
}
```

---

## 6.2 图像增强

对 UAV 和 satellite 图都做增强，但要注意不要破坏配对关系。

推荐增强：

* resize 到统一分辨率
* random horizontal/vertical flip（若任务允许方向不敏感）
* small random rotation
* color jitter
* gaussian blur
* jpeg compression noise
* random resized crop（幅度不要太大）

建议策略：

* `sat_image` 增强较轻
* `uav_image` 增强稍强

目的：

* 提高鲁棒性
* 防止过拟合成像风格

---

## 6.3 Batch 组织

推荐 batch size：

* 单卡小显存：8~16
* 多卡尽量做到全局 batch 64+

contrastive 很吃 batch size，若 batch 小：

* 优先上多卡 all-gather
* 或使用 memory queue（V2 再考虑）

---

## 7. 训练阶段建议

## 7.1 阶段 1：只训 adapter + projection

### 参数设置

* freeze `sat_encoder`
* freeze `uav_backbone`（如果复用 Qwen encoder）
* train `uav_adapter`
* train `proj_head`

### loss

* `L_contrast`
* `L_global_distill`
* `L_token_distill`

### 目的

先验证：

* 是否只用轻量模块就能把 UAV 特征对齐到 satellite 空间

---

## 7.2 阶段 2：解冻 UAV encoder 后几层（可选）

若阶段 1 不足，再做：

* 解冻 `uav_encoder` 后 2~4 层
* 继续训练

目的：

* 提升 domain adaptation 能力

注意：

* 不建议一开始全量解冻
* 学习率要更小

---

## 8. 评估指标

## 8.1 对齐评估

### Retrieval 指标

以 `uav -> sat` 检索：

* Recall@1
* Recall@5
* Recall@10

### Distill 指标

* mean cosine similarity of paired features
* mean token L2 distance

---

## 8.2 对 policy 的可用性评估

如果后续已有 sat-policy：

* 将 `sat_image` 输入得到 `policy_hidden_sat`
* 将 `uav_image` 经 adapter 后输入得到 `policy_hidden_uav`
* 比较：

  * hidden cosine
  * action KL（若已有 policy head）

这是最关键的真实目标指标。

---

## 9. 推荐训练配置

### 9.1 数据配置

* 数据：`24k` 对 `(uav_image, sat_image)`
* 划分建议：

  * train: `80%`
  * val: `10%`
  * test: `10%`
* 划分原则：

  * 尽量按地理区域或场景划分，避免同一区域同时出现在 train 和 test
  * 不建议随机逐样本切分，否则会高估泛化能力

### 9.2 输入与增强

* 输入分辨率：保持与当前 Qwen2.5-VL vision encoder 兼容
* 推荐增强：

  * `uav_image`: 较强增强

    * rotation
    * brightness / contrast jitter
    * blur
    * compression noise
    * small crop / translation
  * `sat_image`: 较轻增强

    * resize
    * mild color jitter
    * small crop
* 原则：

  * 增强应模拟真实 UAV 成像扰动
  * 不要做破坏配对语义的一致性增强

### 9.3 训练参数建议

* optimizer: `AdamW`
* learning rate:

  * `uav_adapter`: `1e-4`
  * `projection_head`: `1e-4`
  * 若后续解冻 `uav_backbone` 后几层：`1e-5`
* weight decay: `0.01`
* scheduler: cosine decay
* warmup ratio: `0.05`
* epochs: `20~50`
* mixed precision: 开启
* gradient clipping: `1.0`

### 9.4 Batch 与负样本

* contrastive 对 batch size 比较敏感
* 推荐：

  * 单卡 batch size: `8~16`
  * 多卡全局 batch size: 尽量 `64+`
* 若多卡可用，建议做 feature all-gather 扩大负样本池

### 9.5 Loss 权重初始建议

```text
L_total = λ_contrast * L_contrast + λ_global * L_global_distill + λ_token * L_token_distill
```

初始建议：

* `λ_contrast = 1.0`
* `λ_global = 1.0`
* `λ_token = 0.25 ~ 0.5`

推荐起始值：

* `λ_contrast = 1.0`
* `λ_global = 1.0`
* `λ_token = 0.5`

若发现 token-level 对齐导致训练不稳，可先降到 `0.25`。

---

## 10. 推荐实验顺序

### 实验 1：Contrastive Only

```text
L = L_contrast
```

目的：

* 验证纯跨域判别性是否能建立有效共享空间

观察指标：

* UAV→SAT Recall@1/5/10
* paired cosine similarity

---

### 实验 2：Contrastive + Global Distill

```text
L = L_contrast + L_global_distill
```

目的：

* 验证 teacher feature 对齐是否提升稳定性与兼容性

这是推荐的强 baseline。

---

### 实验 3：Contrastive + Global Distill + Token Distill

```text
L = L_contrast + L_global_distill + λ_token * L_token_distill
```

目的：

* 验证 token-level 对齐是否进一步帮助 policy 接口兼容

注意：

* 若 UAV / SAT 在视角和裁剪上并不严格一致，token distill 不一定总是正收益

---

### 实验 4：只训 Adapter vs 解冻 UAV Backbone 后几层

目的：

* 判断轻量适配是否足够
* 若只训 adapter 已经足够，则优先保留该方案

推荐顺序：

1. 先只训 adapter
2. 若收益不足，再解冻 UAV backbone 最后 2~4 层

---

## 11. 评估方案

## 11.1 表征层评估

### 检索指标

* UAV→SAT Recall@1
* UAV→SAT Recall@5
* UAV→SAT Recall@10
* SAT→UAV Recall@1/5/10（可选）

### 对齐指标

* paired global cosine similarity
* token MSE / token cosine distance

---

## 11.2 面向 policy 的评估

如果后续已有 satellite-policy，可进一步评估：

* `sat_image` 输入 policy 得到 hidden states / action logits
* `uav_image` 经 adapter 后输入同一 policy 得到 hidden states / action logits

比较：

* hidden-state cosine similarity
* action KL divergence
* top-1 action 一致率

这比 retrieval 更接近最终目标。

---

## 12. 风险与注意事项

### 12.1 Token Distill 可能过强

如果配对数据存在：

* 空间偏移
* 高度变化
* FOV 不一致

逐 token MSE 容易过约束，导致优化困难。

建议：

* 先以 `global distill` 为主
* token distill 作为辅助项，小权重加入

### 12.2 Contrastive 受 batch size 影响较大

若 batch 过小，负样本不足，判别空间不够稳定。

建议：

* 优先增大全局 batch
* 或使用多卡 feature all-gather

### 12.3 不要只看检索指标

高 Recall 不等于对 policy 一定有效。

最终仍应关注：

* policy hidden consistency
* action consistency
* 接入第二阶段后的真实导航表现

---

## 13. 与第二阶段的接口

第一阶段训练完成后，应导出一个模块：

```text
uav_image -> aligned_uav_tokens
```

其输出应尽量兼容原本：

```text
sat_image -> sat_visual_tokens
```

第二阶段可直接将：

* `aligned_uav_tokens`

接到原有 VLN policy 的跨模态模块中，再用 Google Earth 生成的 UAV-like VLN 数据进行 policy-aware finetune。

---

## 14. 最终推荐方案（建议直接采用）

### 推荐版本

* `sat_encoder`: frozen Qwen2.5-VL vision encoder
* `uav_backbone`: frozen Qwen2.5-VL vision encoder
* `uav_adapter`: trainable 2~4 层 Transformer adapter
* `projection_head`: trainable 2-layer MLP

### 训练目标

* `contrastive loss`
* `global cosine distillation`
* `small-weight token distillation`

### 推荐总损失

```text
L_total = 1.0 * L_contrast + 1.0 * L_global_distill + 0.5 * L_token_distill
```

### 推荐实施顺序

1. 先做 `contrastive + global distill`
2. 再加入 `token distill`
3. 若效果仍不足，再尝试解冻 UAV backbone 最后几层

### 推荐判断标准

若以下指标成立，则说明第一阶段成功：

* UAV→SAT Recall 明显提升
* paired cosine similarity 高
* 接入原 policy 后 hidden / action consistency 改善

这说明：

* UAV 图已经被编码到一个既具跨域判别性、又兼容 satellite policy 的视觉空间中
* 可以进入第二阶段 synthetic VLN policy finetune
