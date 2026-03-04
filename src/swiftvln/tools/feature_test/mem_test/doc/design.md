基于你这份 OverlapVLN/SFT 框架（滑窗 + overlap_context + 全局 history_cache；窗口滑动后从历史中“均匀采样 K 帧”，每帧做 2D 平均池化压缩后，把 `<history_image>` token 位置替换成对应的视觉 embedding，再拼进 QwenVL system prompt）来看，你的“可动刀点”非常清晰：**只改 compute_history_cache / 历史帧→固定预算 embedding 的生成策略**，其余窗口机制、对话模板、loss masking、prompt 拼接逻辑都不动。

下面我按“能逐步做、每步都有独立测试”的方式，给你一个 **progressive 的改进路线**，同时回答你问的“第一步该做什么”。

---

## 你真正的第一步：先把“memory 好不好”变成可度量的问题

如果你直接从“均匀采样”跳到某个 fancy 模块，最后只能用 SPL/SR 说话，会很难证明你的创新点到底解决了什么、也很难做消融。

你文档里已经有非常关键的可解释性抓手：**History–Instruction Alignment**（用 causal attention 看历史图像 token 对指令 token 的对齐，并给出 diagonal_strength / progress_tracking / focus_score 等指标）。这套东西我强烈建议升级成你论文的“memory 专用评测头”，作为每一步 memory 改动的主指标之一。

所以第一步不是“立刻换采样”，而是：

1. **建立 Memory Eval Harness（离线、可重复）**

   * 固定一批轨迹片段（多个场景、长短、成功/失败都要）。
   * 固定当前帧位置 step_id、固定 instruction。
   * 只替换 history_cache 的生成策略，其它完全一致。
   * 输出：

     * Alignment 指标（你已有）
     * “下一步动作预测”指标（见下文）
     * 冗余/覆盖（diversity & coverage）
     * token/耗时成本

2. **定义 2–3 个不依赖最终导航成功率的“单元测试”**（我后面给你可直接落地的测试流程）

这样你后面每一步改动都能回答：

> 我到底让 memory 更“相关”了？更“覆盖”了？更“能追踪进度”了？还是只是碰巧提高了 SR？

---

## 均匀采样该不该改？结论：要改，但先用“强而简单”的替代基线打地基

**均匀采样**在视频/多模态里几乎永远是被挑战的 baseline：很多视频 LMM 论文都指出 uniform sampling 会漏掉关键片段，尤其长序列下会丢“稀疏但决定性”的事件。([arXiv][1])

但是在你这里，最值得先做的是 **“不用训练、只改选择策略”的强基线**，原因：

* 改动最小：只动 `compute_history_cache` 的 frame index 选择
* 结果最干净：性能变化几乎完全来自“选了哪些历史”
* 很适合作为论文里 progressive ablation 的第 1～2 个台阶

---

## Progressive Memory 改进路线（从最小改动到可发表创新）

我把每一阶段都写成：**做什么 / 怎么接入你现有框架 / 推荐的独立评测**。

### Phase 0：把 baseline 做到“可控可测”

**目标**：你后面所有改进都能“只换一行策略函数”。

* 把历史帧选择抽象成接口：

```python
def select_history_indices(
    all_frame_feats,   # [T, D]，每步一个全局向量（比如 mean pooled）
    instruction_feat,  # [D] 或 [L, D]
    current_feat,      # [D]
    K: int,
    meta=None
) -> List[int]:
    ...
```

* 输出 index 后，再走你现有的：取对应帧的 patch tokens → compress_stride 池化 → 写入 history_cache → system prompt 注入。

**独立评测**：

* 你已有 alignment heatmap + 三个指标（对每个策略固定样本集对比）
* 统计选中帧的时间分布（是否总在前半段/后半段）

---

### Phase 1（最小替换）：从“均匀采样”到“覆盖 + 近因”的分层采样

**动机**：VLN 决策往往需要：

* **近因信息**：刚刚经过什么路口/门牌/楼梯
* **全局进度**：是否已经完成“穿过厨房/到达走廊尽头”等

**策略 1：Two-Scale Stratified Sampling（两段式）**

* K 帧拆成 K1 + K2

  * K1：从最近 `R` 步里均匀（或全取最近）
  * K2：从更早历史里均匀覆盖
* 这是最便宜但通常就能明显好于纯均匀。

**策略 2：Turn/Decision-Point biased（如果你能拿到动作或候选分支数）**

* 在动作序列中找“转向/视角变化大/候选路径多”的 step，把它们优先放进 memory（更贴近“关键决策点”）。

**接入方式**：完全不需要新模型，只返回不同的 index 集合。

**独立评测建议**：

1. **Coverage 指标**：选中帧的 step_id 覆盖率（例如分成 8 个时间 bin，看每个 bin 是否有帧）
2. **动作预测单元测试 A（Teacher-forcing next-action）**：

   * 从数据集中抽 (instruction, history, current) → 让模型预测下一动作（或下一 4 步 action chunk）
   * **环境不 step，只算 action token 的准确率 / logprob**
   * 这样能比 SR 更敏感地衡量 memory 对决策的帮助（HAMT 这类 VLN work 也会用单步动作预测作为训练/评估信号之一）。([arXiv][2])

> Phase 1 的意义：你得到一个“强且干净”的非学习 baseline，后面所有 fancy 方法必须打败它才算真进步。

---

### Phase 2（无训练、但更有信息论味道）：视觉“新颖性/多样性”关键帧选择

**动机**：均匀采样的问题之一是：走廊 20 步都很像，你会浪费大量记忆预算。
视频理解里经常做 **keyframe selection / redundancy reduction**。

**策略 2.1：Farthest Point Sampling（k-center / 最远点采样）**

* 用每帧全局特征向量 `e_t`（比如 patch mean）
* 贪心选 K 帧，使得选中的集合在 embedding 空间尽量“覆盖”历史（最大化最小距离）

伪代码（O(TK)）：

1. 先选最近帧或随机帧作为 seed
2. 每次选 `argmax_t min_{s in S} dist(e_t, e_s)`
3. 可加时间约束（避免全选在一段）

**策略 2.2：Change-point / novelty top-K**

* 计算 `delta_t = 1 - cos(e_t, e_{t-1})`
* 取 delta 最大的 K 个（再做非极大抑制，避免连续帧全中）

**接入方式**：只改 index 选择。

**独立评测建议**：

1. **Redundancy 指标**：选中帧两两 cosine 相似度均值（越低越不冗余）
2. **Memory→Progress Tracking**：用你 alignment 的 progress_tracking_score 看是否更单调、更符合“进度条”直觉
3. **动作预测单元测试 B：分叉点子集评测**

   * 只在 “候选动作多/转弯多”的 step 上评测 next-action accuracy
   * 你会更容易看到 memory selection 的收益集中在哪类 step

---

### Phase 3（真正开始有“论文味”的方向）：Instruction-conditioned Memory Retrieval

到这里，你已经证明“选帧比均匀采样强”。但创新点还不够“语义化”。下一步就是：**记忆应该为“指令”服务**。

这和长视频问答里“根据问题从 memory bank 选相关记忆”的思路非常一致：把 instruction 当作 question，做 query-conditioned memory selection。类似工作在长视频 LMM 里常见：先把历史编码进 memory bank，再做与问题相关的 memory selection 以避免 context 爆炸。([NeurIPS Proceedings][3])

**策略 3.1：Text–Frame Similarity Retrieval（训练-free 版）**

* 得到 instruction 向量 `q`（可以用：

  * QwenVL 自己的文本 encoder 输出
  * 或者用一个外部 CLIP 文本 encoder；但这样会引入额外模型依赖）
* 对每帧算 `rel_t = cos(q, e_t)`
* 选 top-K，但要加 diversity（否则容易都选“同一类相似帧”）

常用做法：**MMR（Maximal Marginal Relevance）**
`score = λ * rel_t - (1-λ) * max_{s in S} sim(e_t, e_s)`

**策略 3.2：对齐驱动的检索（用你已有 attention 对齐作为打分）**
你文档里的 insight 很关键：历史图像 token 在 causal attention 下会回头看指令 token。
所以你可以在离线阶段做一个更“模型一致”的打分：

* 对每个候选帧（或候选小集合）跑一次前向，拿 cross-modal attention（history queries → instruction keys）
* 把“关注 content words 的强度”作为 `rel_t`
* 再做 MMR 选 K

这更像“模型自解释驱动的记忆选择”，写论文会很漂亮：

> 我不靠外部 CLIP，我直接用当前 VLN policy 的注意力结构来评估哪些历史帧与指令更相关。

**独立评测建议**（非常适合写成 paper table）：

1. **Instruction-word coverage**：

   * 把 instruction 分词，去掉 stopwords
   * 统计选中帧对 content token 的 attention mass 覆盖（或你 focus_score）
2. **Clause-level grounding test（可自动构造）**：

   * 用规则把 instruction 按逗号/and/then 切成子句
   * 让模型（或一个小分类器）预测当前 step 属于第几个子句（progress classification）
   * memory 好的话，这个会更准
3. 继续用 next-action accuracy（teacher forcing）做主指标

---

### Phase 4（更强的“记忆”概念）：State-conditioned Episodic Memory（回访唤醒）

VLN 很容易出现：走错路 → 回头 → 重新经过某处。
均匀采样/纯 instruction retrieval 往往不擅长“我来过这里吗？”这类问题。

VLN 领域有“episodic scene memory”思路：当进入某个场景时唤醒过去访问的相关记忆。([arXiv][4])
长视频 LMM 也会做“从历史 memory bank 选择与当前问题/状态相关的记忆”。([arXiv][5])

**策略 4.1：Current-to-Past Retrieval（重访检测）**

* 用 current_feat `c` 对历史帧算 `sim_t = cos(c, e_t)`
* 选出 top-M 个“最像当前”的过去帧（候选为“我以前到过的类似地点”）
* 剩余预算再用 Phase 3 的 instruction-conditioned 选

**策略 4.2：Episodic slots（只存“事件”，不存所有帧）**

* 维护一个 memory bank：只有当新帧与现有记忆都不相似（novelty > τ）才写入
* 这样记忆里天然是“关键场景/节点”
* 窗口滑动时从 bank 中选 K 个

**独立评测建议**：

1. **Loop / revisit 单元测试**（强烈推荐！）

   * 在轨迹中自动找“视觉相似度高但时间间隔大”的帧对（可能是重访）
   * 测试你的 memory 是否能在当前时刻检索到那段过去
   * 指标：Recall@K（当前时刻 history 中是否包含匹配帧）
2. **错误恢复子集评测**：

   * 只在“失败轨迹/偏航后”的片段评估 next-action accuracy
   * episodic memory 往往在这里更有增益

---

### Phase 5（从“选帧”进化到“压缩记忆”：固定预算的 Memory Tokens）

如果你想把创新点做到更“模型化”、更像论文贡献，最终很可能要走到这一步：

> **不再把 memory = K 张图片，而是把历史压缩成固定数量的“记忆 token”。**

这在视频/多模态大模型里是非常主流的思路：把大量视觉特征通过一个模块压成少量 token，再交给 LLM。比如 Flamingo 用 **Perceiver Resampler** 把视觉特征变成固定数量的 visual tokens。([arXiv][6])

你这里可以做到“结构不变”的关键点是：
你本来就用 `<history_image>` token 占位，然后在 embedding 级别替换。
所以 `<history_image>` 不一定非得对应“真实某一帧”，它可以对应“记忆 token”。

给你两个可落地路线：一个训练-free，一个可学习。

#### Phase 5A：训练-free token 聚类/合并（ToMe 思路）

Token Merging（ToMe）核心就是“合并相似 token，减少 token 数”。([arXiv][7])
你可以在 history patch tokens 上做一个简化版：

* 把过去 T 步的 patch tokens（或先选出若干候选帧）堆起来：`X ∈ R^{N_tokens × D}`
* 用 k-means / 层次聚类 / greedy matching，把它们压到固定 `N_mem` 个 centroid tokens
* centroid tokens 直接作为 history_cache 写入 `<history_image>` 位置

优点：无需训练、非常适合做 Phase 5 的第一版。

**评测**：

* compression distortion：原 tokens 与 centroid tokens 的 kNN 覆盖率 / 重构误差（MSE）
* 下游：action prediction + alignment 指标

#### Phase 5B：可学习的 Memory Tokenizer（TokenLearner / Resampler 思路）

TokenLearner 的核心：用少量 learned tokens 自适应“提取重要视觉 token”，适用于图像和视频。([arXiv][8])
你可以加一个很轻的模块（比如 8～32 个 query）：

* 输入：历史 patch tokens（来自若干帧，或来自 episodic bank）
* 输出：固定 N_mem 个 memory tokens
* 然后照旧注入 system prompt

优点：这更像一个完整贡献：**Memory compression module + retrieval/selection policy + 新的评测**。

---

## 给你一套“每阶段都能跑”的独立测试流程（不靠最终 SR/SPL）

下面这些你可以直接做成 `memory_eval.py`，每次换策略跑一遍，输出表格和图。

### Test 1：History–Instruction Alignment（你已有，建议标准化成主指标）

* 输出：diagonal_strength / progress_tracking_score / focus_score + heatmap
* 用途：证明 memory 是否真的在“跟指令走”，而不是随机塞图。

### Test 2：Teacher-forcing Next-Action Accuracy（强烈推荐作为主 quantitative）

* 固定 (instruction, history_cache, current_frame)
* 让模型预测下一步（或下一段 num_future_steps=4）动作 token
* 统计：

  * top-1 action accuracy
  * action sequence edit distance（针对 4-step chunk）
  * 或 action token NLL

优点：

* 不受环境滚动误差影响
* 对 memory 改动敏感
* 很适合做消融表

（HAMT 等 VLN work 也强调用单步 action prediction 等 proxy task 训练/验证 history 编码能力。([arXiv][2])）

### Test 3：Revisit Retrieval Recall@K（专测 episodic/state memory）

* 自动构造：在轨迹里找与当前帧视觉相似、但时间间隔 > Δ 的 past frames
* 看这些 past frames 是否被选进 history_cache（或被压缩 token 表征到）
* 指标：Recall@K / mAP

这能非常清晰地证明 Phase 4 的价值。

### Test 4：Redundancy–Relevance 曲线

对每个策略输出：

* 冗余：avg pairwise sim（越低越多样）
* 相关：avg sim(text,instruction) 或 attention-based relevance
  画一个 scatter：看策略是否落在“高相关、低冗余”的 Pareto 前沿。

---

## 你论文“创新点”最可能长成什么样

如果你想要一个更像论文贡献的标题级创新，我建议你把路线收敛到：

1. **Query-conditioned memory selection**（instruction + current state）
2. **Episodic memory bank**（只存关键事件/场景）
3. **Fixed-budget memory token compression**（把检索到的历史压成 N 个 tokens 注入 prompt）

这套组合在长视频 LMM 里有强共识：在线处理 + memory bank + 相关选择，解决上下文长度限制。([arXiv][5])
同时在 VLN 里也能和 episodic scene memory 的思想呼应。([arXiv][4])

而且最重要的是：**它完全兼容你“不改主体框架”的约束**——因为最终仍然是“产出一段 embedding，注入 system prompt”。

---

## 我给你的最务实建议：按这个顺序推进（每一步都能产出可写的 ablation）

1. **Phase 0：评测 harness + 复现 uniform**
2. **Phase 1：two-scale stratified（近因+覆盖）**
3. **Phase 2：diversity/novelty keyframe（k-center / change-point）**
4. **Phase 3：instruction-conditioned MMR retrieval（最好再加 attention-based 版本）**
5. **Phase 4：current-conditioned episodic retrieval + revisit 单元测试**
6. **Phase 5：memory token compression（先训练-free 聚类/ToMe-style，再考虑 TokenLearner/Resampler 学习版）** ([arXiv][7])

---

如果你愿意，我可以基于你现有的 `compute_history_cache` / `history_cache` 数据结构（你文档里已经写了 evaluator 的类与 slide_window 流程），把 **Phase 1～Phase 3 的选择策略**写成一套“可直接粘进你代码”的伪代码/接口设计（包含复杂度、需要缓存哪些中间量、以及如何保证 token budget 恒定）。此外也可以把 **Test 2/3** 的评测脚本结构一起给出来，确保你每一步都能独立出图出表。

[1]: https://arxiv.org/abs/2502.19680?utm_source=chatgpt.com "M-LLM Based Video Frame Selection for Efficient Video Understanding"
[2]: https://arxiv.org/abs/2110.13309?utm_source=chatgpt.com "History Aware Multimodal Transformer for Vision-and-Language Navigation"
[3]: https://proceedings.neurips.cc/paper_files/paper/2024/file/d7ce06e9293c3d8e6cb3f80b4157f875-Paper-Conference.pdf?utm_source=chatgpt.com "Streaming Long Video Understanding with Large Language Models"
[4]: https://arxiv.org/pdf/2303.01032?utm_source=chatgpt.com "ESceme: Vision-and-Language Navigation with Episodic Scene Memory"
[5]: https://arxiv.org/abs/2404.05726?utm_source=chatgpt.com "MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video ..."
[6]: https://arxiv.org/abs/2204.14198?utm_source=chatgpt.com "Flamingo: a Visual Language Model for Few-Shot Learning"
[7]: https://arxiv.org/abs/2210.09461?utm_source=chatgpt.com "[2210.09461] Token Merging: Your ViT But Faster - arXiv.org"
[8]: https://arxiv.org/abs/2106.11297?utm_source=chatgpt.com "TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?"
