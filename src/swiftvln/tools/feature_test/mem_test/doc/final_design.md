下面我给你一份**更新后的“方案计划（可作为论文路线图）”**，它基于你现有的 Global ToMe 文档，但按你提出的两点做了更改：

1. **名字不再叫 ToMe**（避免 reviewer 说“不是真 ToMe”）
2. **方向一（方位/heading）不依赖绝对 heading**：利用你动作序列（前进/左转/右转）在评测/推理过程中可计算的**相对姿态**（相对第一帧 or 相对上一帧）来做“方位锚定”。

我也会先简评你现在的方案合理性，然后给出在你“俯视+更长序列+回环”背景下更像论文贡献的后续创新点（都保持：**输出 704 tokens 注入 `<history_image>`，结构不变**）。

---

## 0) 你当前方案的定位与命名建议

你现在写的 “Global ToMe” 实际是一个 **跨帧 token 的 soft clustering / resampling**：把所有历史 token 拼成大池 (X)，对 (K) 个质心做 soft assignment，再加权聚合得到 (K) 个输出 token，并可迭代 1–2 次。

### 建议改名（避免 ToMe 争议）

选一个“准确且论文友好”的名字：

* **Global Token Clustering (GTC)**
* **Global Soft Token Clustering (GSTC)**
* **Global Token Resampling (GTR)**

你文档里的核心过程更像 “soft k-means / resampling”，而不是 ToMe 原论文的 bipartite matching merge，所以改名是对的。

---

## 1) 对你现有 GSTC/GTC 方案的合理性评估（结论：合理，但要补两块才能变“论文贡献”）

### 合理的地方（值得保留）

* **跨帧去冗余**：把重复背景/相似区域折叠，释放预算给少见信息；对长序列尤其重要。
* **接口完全兼容**：输出固定 (K=704) tokens 注入 `<history_image>`，结构不变。
* **可加 token size**：你文档中提到的 token size 追踪能缓解“越聚越糊”。

### 主要风险（尤其对“俯视 + 旋转”）

* **方位信息被平均抹掉**：俯视里转向前后信息量类似，但“左/右/朝向”是决策关键；纯语义相似聚类容易把不同方位混成一个质心。
* **长序列计算会爆**：你当前复杂度 (O(MKd)) 对超长历史不稳。
* **assignment 塌缩**：softmax 容易让少数质心吃掉大多数 token（背景质心化）。

> 所以：**GSTC 是很好的第一版/强 baseline**。
> 但要成为论文创新点，需要把它“任务特化”：把你任务的核心（方位/回环/长序列）融进去。

---

## 2) 更新后的总体计划（Progressive Plan）

下面每个阶段都满足：**模型不重训也可先评测（frozen policy），输出仍为 704 tokens**。你后面想做“重训版提升”也可以作为 upper bound，但不影响主线。

---

# Phase 1：GSTC/GTC 基线（你已有）

### 方法

* 按你文档：拼接历史 token 池 (X)，soft assignment 得到 (A)，聚合成 (C\in \mathbb{R}^{704\times d})。
* 建议额外**做两点小修补**（不改变核心）：

  1. **输出 token 的排序**：给每个 centroid 一个 “软时间” (\tau_k)（由其吸收 token 的时间加权平均得到），按 (\tau_k) 排序后注入（避免 token 顺序随机）。
  2. **top-r 稀疏分配**：每个 token 只分配给相似度 top-r 的质心再 softmax（减少平均化、加速、避免塌缩）。

### 你能写的 claim

* “跨帧 token 聚类在固定预算下提高历史表示效率，优于帧采样+帧内 pooling。”

---

# Phase 2：Relative-Pose Grounded Clustering（核心更新：不需要绝对 heading）

你说的非常关键：虽然没有绝对 heading，但你的 action 是离散的、可积累的，因此你**可以得到每一帧相对第一帧/上一帧的姿态**。这正是“俯视方位问题”的抓手。

## 2.1 你能得到哪些相对量（不需要地图真值）

从动作序列可以累计得到：

* **相对 yaw（转角）**：
  [
  \Delta\theta_t = \sum_{k=1}^{t} \delta\theta(a_k),\quad \delta\theta(\text{TURN_LEFT})=+15^\circ,\ \delta\theta(\text{TURN_RIGHT})=-15^\circ
  ]
* **相对位移（粗）**：前进步数×25cm 叠加到局部坐标里（只要你知道每步前进距离）。
  即使没有全局坐标，你也能得到相对 ((\Delta x_t,\Delta y_t))（以第一帧为原点、第一帧朝向为 x 轴）。

> 这对你任务很重要：
> “forward 看到更多”本质是视野覆盖变化；“turn”本质是坐标系旋转，信息量相近但方位关系变了。你必须把方位线索显式交给 memory。

## 2.2 具体怎么用到 memory（两种强方案，推荐你都做成消融）

### 方案 2A：Pose-conditioned Similarity（相似度里加入相对姿态）

把每个 token 附上一个 pose embedding（来自相对 yaw / 相对位移）：

* 对帧 (t) 的所有 token (x_{t,i})，构造：
  [
  x'*{t,i} = x*{t,i} + \lambda_\theta E_\theta(\Delta\theta_t) + \lambda_p E_p(\Delta x_t,\Delta y_t)
  ]
  然后 GSTC 的 cosine 相似度用 (x') 计算，而聚合仍用原始 (x) 或 (x')（两种都可以做消融）。

**效果直觉**：
“看起来很像，但来自不同相对朝向/不同位置”的 token 不会被轻易合并，方位信息不再被平均抹掉。

### 方案 2B：Canonicalization（把历史帧“旋回”到统一坐标再聚类）

俯视图非常适合做一个更强的点：**把每帧特征旋转到统一参考系**（以第一帧朝向为参考），再做聚类。

* 对每一帧 (t)，根据 (\Delta\theta_t) 对其 patch token 的空间布局做“等效旋转”（实现上可以在 token 的 2D patch 网格上做旋转重排/插值，或用“旋转位置编码”方式等效实现）。
* 再把对齐后的 tokens 拼池聚类。

**为什么这对你的任务比室内 VLN 更合理**：
室内第一视角旋转会改变可见内容分布（遮挡多）；但俯视图旋转更多是坐标系变化，内容本身更可对齐。你这是“俯视任务专属优势”。

> 论文级 claim：
> “俯视 VLN 的 memory 应该在统一方位坐标系下压缩，否则聚类会把 ‘方位’ 当噪声抹掉。”

---

# Phase 3：Sector-preserving Compression（方位扇区保结构）

如果 Phase 2 是“软注入姿态”，Phase 3 是“硬保结构”，通常区分度更强、写作更像贡献点。

### 方法

把 token 按其在俯视图中的相对方向划分到 S 个扇区（例如 8 个方向扇区，或按极坐标 bins）：

* 每个扇区内部单独做 GSTC（或 smaller-K 的聚类）
* 每个扇区输出固定配额 tokens，拼成 704

**好处**

* 防止把 “左侧地标 token” 和 “右侧地标 token” 合并成一个 centroid
* 特别适合指令里大量 “left/right/upper-left”等方位描述（你前面提到方位 QA 的动机也在这里落地）

---

# Phase 4：Loop-aware Episodic Recall（回环触发记忆）

回环任务是你最好的亮点场景：它能让 reviewer 直观看到“memory 机制是否真的在做事”。

### 方法（仍不改结构）

* 维护轻量 place keys（每隔若干步存一帧全局 embedding (e_t)）
* 当前帧 (e_{cur}) 与历史 keys 做相似检索：

  * 若出现“高相似且来自很早”的匹配（回环信号），则：

    1. 强制把起点/匹配段对应的 token（或其 centroid）塞进 memory 预算（最高优先级）
    2. 剩余预算再用 Phase 2/3 的聚类补齐

**为什么更适合你的俯视任务**

* 绕湖一圈回到起点：视觉会高度相似
* 纯聚类可能“平均掉”起点证据
* 回环触发能保证起点证据被召回 → 你能在 loop recall、loop CE 上打出明显增益

---

# Phase 5：Long-sequence Ready（流式/增量 + 防塌缩）

这是让你真正“长序列可用”的工程与论文点（对俯视长序列尤其必要）。

### 5A Streaming/Incremental Update

不要每次用全部历史重算 (O(MKd))。
改成：

* 维护当前 centroids (C) 和权重 (w)
* 每次窗口推进，只对新增 tokens 做 assignment 更新 centroids
* 加“遗忘衰减”（越久权重越小）以适配超长

### 5B Balanced Assignment（防塌缩）

对 assignment 加“容量/均衡约束”（例如每个 centroid 期望吸收的 token 数接近），避免少数质心吞掉大部分 token。
这通常会显著改善你现在看到的 redundancy/coverage“挤在 0.8–1”的问题，因为塌缩会让 token 分布更各向异性。

---

## 3) 你问“怎样才有更好的创新点？”——我建议你的“主贡献组合”

结合你的任务特点（俯视、方位重要、转向信息量类似、序列更长、回环），最像“主论文贡献”的组合是：

### 主线（最推荐）

1. **GSTC（Global Soft Token Clustering）**：作为跨帧压缩基线（你现有方案）
2. **Relative-Pose Grounded Compression（Phase 2A/2B）**：利用离散动作积分得到相对姿态，实现“方位对齐的记忆压缩”（这是你任务独特性）
3. **Loop-aware Recall（Phase 4）**：在回环任务上形成亮点结果（loop recall + loop chunk CE 显著提升）

这三条叠加，你的创新点会比“纯 token 聚类”强很多，因为它明确回答了：

> 俯视 VLN 的 memory 不只是压缩，还必须保留/对齐方位；并且在回环时能触发 episodic recall。

---

## 4) 你接下来最应该优先做的 3 个“可写成消融”的具体版本

为了让论文实验线很清晰，我建议你按下面顺序落地（每一个都保持 704 tokens）：

1. **GSTC (baseline)**：你文档 +（排序/稀疏分配）小修补
2. **GSTC + pose-conditioned similarity（2A）**：加 (E_\theta(\Delta\theta))（只用相对 yaw 就够先打出效果）
3. **GSTC + loop-trigger recall（4）**：回环子集上做 loop recall + loop CE 的表格

如果你有精力再做第 4 个：
4) **Sector-preserving（3）**：把方向结构硬保住，通常对 “turn/left/right” 指令最敏感

---

如果你希望我把这份计划进一步“产品化”成你论文里能直接用的结构（章节标题、方法段落、消融表、对应 claims 与指标），我可以按你现在评估框架（chunk teacher forcing CE/NLL + redundancy/coverage + loop recall）把每个 phase 应该呈现的结果写成一份“Paper Outline + Ablation Matrix”。
