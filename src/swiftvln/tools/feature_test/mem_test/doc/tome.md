对，你现在的需求可以非常具体地表述为：

* 你已经 **sample** 出了 (N) 帧（这一步不讨论）。
* 每帧你已经有 ViT 输出的 token 特征：(\mathbf{Z}_t \in \mathbb{R}^{391\times D})（t=1..N）。
* 你希望最终注入到 `<history_image>` 的 token 数固定为 **(8\times 88=704)**（也就是固定预算 memory tokens）。
* baseline 是：先选 8 帧，再对每帧压成 88 token。
* 你现在想用 **ToMe 的思想**：**把很多 token 合并成更少 token**，但不训练。

下面我就按你给的数字，**一步一步讲清楚 ToMe 在做什么，以及你该怎么把它“搬到你的 memory 压缩里”**。

---

## 1) ToMe 到底在解决什么问题？

ToMe 的核心观点很简单：

> 在 ViT 里很多 token（patch）是高度相似/冗余的；
> 与其让它们都参与后续计算，不如把“相似 token 合并”，用更少 token 代表同样的信息。
> 而且这个过程可以 **不训练**，直接插到现成 ViT 里用。 ([arXiv][1])

在原论文里，ToMe 是为了 **加速 ViT 前向**，所以它是在 transformer 的各层之间“逐步合并 token”。([arXiv][1])
但对你来说，你不一定要把它插进 ViT 内部，你可以把它当作一个 **“训练-free 的 token 压缩算子”**：把历史帧的 tokens 直接压到固定预算后再注入 `<history_image>`。

---

## 2) 你的 baseline（8 帧 × 88 token）在做什么？它的隐含假设

baseline 的隐含假设是：

* “帧”是你 memory 的基本单位；
* 每帧内部做空间 pooling，把 token 数减少到 88；
* 然后取 8 帧拼起来。

这样你保留的是：**时间顺序很清晰**（第 1 帧…第 8 帧），但缺点是：如果 8 帧里很多内容相似（走廊、湖边重复视角），你会浪费 token 预算。

---

## 3) ToMe 思路替代方案：不再把“帧”当作基本单位，而把“token”当基本单位

你现在 sample 得到 (N) 帧，每帧 391 token，总 token 数：

[
M = 391N
]

你的目标 token 数：

[
B = 704
]

你希望做的是：把 (M) 个 token 压到 (B) 个 token。

ToMe 的思想不是 “平均池化”，而是：

> **先计算 token 和 token 的相似度，再把最相似的 token 合并**，尽量把“重复内容”折叠掉。([arXiv][1])

---

## 4) ToMe 真正在做的匹配/合并步骤（Bipartite Soft Matching）

ToMe 的核心算法通常被描述为 **Bipartite Soft Matching (BSM)**：
它不是做全局 (O(M^2)) 的两两聚类，而是用一个非常快的近似策略找“该合并哪些 token”。([arXiv][1])

### Step 4.1 选择用于“相似度比较”的 token 表示

在 ToMe 里，相似度比较不是用原始像素，而是用 transformer 里的 token embedding（论文里讨论的是在网络中逐层做）。([arXiv][1])
在你这里，你可以直接用 ViT 输出的 token 向量（391×D），做归一化后计算 cosine 相似度即可（这就是 ToMe/相关实现常用的相似度度量）。([DeepWiki][2])

### Step 4.2 把 token 分成两组：dst（目的）和 src（来源）

ToMe 会把 token 集合拆成两半：

* dst：保留的“目的 token”（被合并的承载者）
* src：会被合并进 dst 的“来源 token”

拆分方式可以很简单（例如按 token index 奇偶分）。这么做的目的，是让匹配变成“src → dst”的二分匹配，复杂度更低。([arXiv][1])

### Step 4.3 对每个 src，找它最相似的 dst

对每个 src token (s)，找到一个 dst token (d(s))：

[
d(s) = \arg\max_{d\in \text{dst}} \cos(\hat{\mathbf{x}}_s,\hat{\mathbf{x}}_d)
]

这一步的含义是：“每个 src 都想投奔一个最像自己的 dst”。（ToMe 的实现思路就是这种“软匹配/近似匹配”）。([arXiv][1])

### Step 4.4 只选“最值得合并”的那一部分 src（由 ratio 控制）

ToMe 有一个关键超参 **ratio (r)**：控制要合并掉多少 token。ratio 越大，token 数降得越多，但信息损失风险也越大。([Hugging Face][3])

你这里要从 (M) 压到 (B)，那你需要合并掉的 token 数是 (M-B)。
在一次性合并的视角下，相当于选择 (M-B) 个 src token 去合并（dst 和没被选中的 src 继续保留）。

实践上，你可以把每个 src 的“最大相似度分数”当作合并优先级：
[
\text{score}(s)=\max_{d\in \text{dst}} \cos(\hat{\mathbf{x}}_s,\hat{\mathbf{x}}_d)
]
然后选择 score 最大的一批 src 来合并。

这就是 ToMe 的核心直觉：**越像的越先合并**。([arXiv][1])

### Step 4.5 合并的具体形式：加权平均 + token size（可选但很重要）

合并不是简单丢弃，而是把 src 的信息“融进” dst。

最基本的合并：
[
\mathbf{x}'*{d} \leftarrow \frac{\mathbf{x}*{d} + \mathbf{x}_{s}}{2}
]

ToMe 还有一个很关键的细节：**跟踪 token 的“面积/权重”（token size）**，避免连续合并后某些 token 被过度稀释。直觉是：一个 token 可能代表多个 patch，合并时应该按代表的“份额”加权。([arXiv][1])

写成公式就是：
[
\mathbf{x}'*{d} \leftarrow \frac{w_d\mathbf{x}*{d} + w_s\mathbf{x}_{s}}{w_d+w_s},\quad
w'_d \leftarrow w_d+w_s
]

（初始时每个 token (w=1)，合并后权重累加。）

这一步非常适合你，因为你做 memory 压缩时也会担心“合并太多次导致信息偏置”。

---

## 5) 把 ToMe 思路落到你这个“N 帧 → 704 token”的具体操作（两种等价路线）

你现在有 (N) 帧，每帧 391 token。你想输出 704 token。这里有两种做法：

### 路线 A：全局 ToMe（最“像 ToMe”，更偏 episodic）

1. 把所有帧的 tokens 拼成一个集合：总计 (M=391N) 个 token。
2. 对这 (M) 个 token 做上面 Step 4 的 BSM 匹配与合并。
3. 合并直到 token 数 = 704。
4. 把得到的 704 个 token 直接注入 `<history_image>`。

特点：

* **会强烈去冗余**（跨帧相似 token 会合并）
* **时间顺序信息弱**（token 不再天然对应哪一帧）
* 对回环/重访类任务往往更友好（更像“地点记忆”）

这跟 ToMe 论文里“视频 token 跨帧也会合并成物体部件/轨迹”的观察是一致的。([arXiv][1])

### 路线 B：分段 ToMe（保留一点时间结构，更适合 VLN 的“进度感”）

你想要最终是 **8×88** 这种结构，那可以把时间分成 8 段：

1. 把 (N) 帧按时间切成 8 段（每段约 (N/8) 帧）。
2. 每段内把 tokens 拼起来，对该段做 ToMe 合并，输出 **88 个 token**。
3. 8 段各输出 88，拼起来就是 704 token，注入 `<history_image>`。

特点：

* 仍然有“8 段”的粗时间顺序（第 1 段到第 8 段）
* 段内做 ToMe 去冗余（不会浪费在重复走廊）
* 你原来的对齐/进度类指标也更容易沿用

> 你可以把它理解成：“用 ToMe 替代每段的 pooling 压缩”。
> baseline 的“每帧 pooling→88”变成“每段 token merging→88”。

---

## 6) 你问的关键：那我为什么还说“8×88”？到底对应什么？

你现在说的 “目标变成 8×88 个 token”，这其实是一个 **工程约束**：你希望 `<history_image>` 那一段的 token 数固定、最好结构也固定。

* 在路线 A（全局 ToMe）里，“8×88”更多只是个总预算 704；8 这个分组不一定有语义。
* 在路线 B（分段 ToMe）里，“8×88”就是有语义的：8 个时间段，每段 88 个 memory token。

如果你担心 “完全没时间信息”，我会更建议你先做路线 B（更稳、更像你现在系统的渐进替换）。

---

## 7) 最后帮你把“这和 ToMe 原本做法的差异”讲清楚（避免概念混乱）

* **ToMe 原始论文**：在 ViT 的多个 transformer layer 之间逐步合并 token，以减少后续层的计算量；ratio 控制每层合并多少；可不训练。([arXiv][1])
* **你这里的用法（memory 压缩）**：把 ToMe 的“相似 token 合并”当作一个后处理压缩器，用来在注入 LLM 前把历史视觉 tokens 压到固定预算。

思想一致：**相似 token 合并**。
位置不同：一个在 ViT 内部为加速，一个在 memory 管线里为压缩。

---

如果你愿意，我可以接着用你给的 391 这个数字，帮你把两条路线各自的“token 组成”讲得更细（比如：391 里是否包含 CLS、是否要保护 CLS、不参与合并；跨帧合并时是否要避免把“当前关键地标 token”合进背景 token），并给你一个你能直接写进论文的“算法描述（无代码）+ 超参表（ratio/segment数/是否保时间）”。

[1]: https://arxiv.org/abs/2210.09461?utm_source=chatgpt.com "Token Merging: Your ViT But Faster"
[2]: https://deepwiki.com/dbolya/tomesd/3.1-token-merging-algorithm?utm_source=chatgpt.com "Token Merging Algorithm | dbolya/tomesd | DeepWiki"
[3]: https://huggingface.co/docs/diffusers/main/en/optimization/tome?utm_source=chatgpt.com "Token merging - Hugging Face"
