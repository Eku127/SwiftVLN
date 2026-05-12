下面我按“**你现有 SwiftVLN 方案 → 相关最新工作怎么做 history/memory → 针对 BEV SatNav 的可落地方向（不改 Qwen2.5‑VL‑3B 本体）→ 可以写成论文的创新点**”来给建议。

---

## 1) 我理解的你当前方案与痛点在哪里

你现在的 SwiftVLN 核心是：

* **滑动窗口 + 重叠上下文（overlap_context）**来解决长序列上下文断裂（num_frames=32, overlap=16, future_steps=4）。
* 额外再给一个**全局历史记忆块 `<history_memory>`**，目前有三种 history 处理：

  * per-frame：从过去直接采样 `num_history` 帧（可均匀/对数），每帧 256→64 压缩后拼接。
  * GTC：把历史 token 全局聚类成固定 K（512）。
  * Segment-GTC：分段聚类，保留粗粒度时序。

你不满意的点（我也认为是“瓶颈本体”）主要有三类：

1. **固定采样（哪怕对数）不智能**：
   对 Boundary（回环）来说，“关键”不是均匀覆盖时间，而是要保住：起点锚点、关键拐点、与当前相似的 revisited place；
   对 Landmark（精准朝向）来说，“关键”是：哪些历史帧里出现过目标地标/结构，哪些帧提供了可靠的方位参照。
   固定 num_history 很容易漏掉“信息密度最高的那几帧”。

2. **历史 token 的“时序/方位信息”表达不足**：
   你文档里明确写到历史块是“无 ROPE”，而当前帧“有 ROPE”。
   这意味着历史视觉 token 在模型内部很可能缺少显式时空位置编码，模型在“读历史”时更像在看一团无序视觉证据，而不是“按时间展开的轨迹”。

3. **任务本身对方位/回环极敏感**：
   你也强调了 BEV 的旋转会改变参考系、Landmark 需要 15° 精旋、Boundary 要回到起点附近停（10m）。
   所以“history 怎么组织”本质上应该服务于：**方位对齐**与**回环识别**，而不是“给模型多看点过去”。

下面的改进建议会紧扣这三点。

---

## 2) 最新文献里，大家怎么解决“history / memory”问题（你可以借鉴的范式）

### 2.1 不是“存更多帧”，而是“抑噪/选择性使用历史”

* **Memory-Adaptive VLN (MAM, 2024 Pattern Recognition)**：指出 history 里有“有用+有害”混在一起，仅用标量权重不足，会引入噪声；用视觉/文本自适应模块去抑制“有害成分”。([ScienceDirect][1])
  👉 可迁移点：你可以做“token/frame 级别的可靠性/相关性筛选”，替代直接采样。

### 2.2 用“地图/结构化记忆”替代“堆历史帧”

* **GridMM (ICCV 2023)**：把历史观测投影到统一的 top‑down 网格记忆图，再做指令相关聚合。([arXiv][2])
* **MapGPT (ACL 2024)**：在线构建语言形式的 map，给 GPT 一个“global view”再做规划。([ACL Anthology][3])
* **MC‑GPT (2024)**：维护拓扑记忆图（viewpoints/objects/relations），再用 reasoning chains 提升策略多样性和可解释性。([arXiv][4])
* **MapNav (arXiv/ACL 2025)**：用带文字标注的语义地图（ASM）**替代历史帧**作为 VLM 输入。([ar5iv][5])

👉 对你最重要的启示：**“历史”不一定要以“历史帧 token 序列”的形式存在**。对 BEV SatNav，地图化/轨迹化其实更自然（因为你本来就在俯视图）。

### 2.3 走向“流式/慢快记忆 + token 裁剪/压缩”

* **StreamVLN (2025)**：SlowFast context——快流用滑窗 KV cache 保持响应，慢流用 token pruning 压缩历史视觉状态。([arXiv][6])
* **JanusVLN (2025)**：把语义与几何拆开成双隐式记忆，并缓存初始+滑窗的历史 KV，避免重复计算。([arXiv][7])
* 通用 token compression：如 ToMe（token merging）([arXiv][8])、以及更“慢快”的 Vision‑centric Token Compression (ViST, 2025)([arXiv][9])。

👉 你已经有 overlap_context（快流雏形），但“慢流历史”仍偏“静态采样/聚类”。可以向 StreamVLN/JanusVLN 学习“**持续更新的慢记忆**”与“**按需压缩**”。

### 2.4 针对动作-视觉因果：用“逆动力学/动作块”提升稳定性

* **NaVIDA (2026)**：用 chunk‑based inverse dynamics augmentation 学动作导致的视觉变化，并用层级 action chunking + entropy-guided 执行减小累积误差。([arXiv][10])

👉 对你 Landmark 的“精旋”特别 relevant：旋转是强因果操作；让模型更懂“转多少会看到什么”，往往比让它多记几帧更有效。

---

## 3) 面向你这个 BEV SatNav 的“可落地、不改 Qwen2.5‑VL‑3B 本体”的改进方向

下面我给 5 个方向，按“**论文创新性**”和“**你当前代码结构可插拔性**”来设计（基本都能变成新的 `history_processor_type` 或 prompt/输入构造策略）。

---

### 方向 A：把历史当“视频”喂给 Qwen —— 用 M‑RoPE 恢复时序与空间（我认为这是最匹配你现状的强改进）

你现在 history_memory “无 ROPE”。但 Qwen2.5‑VL 本身强调了多模态 3D RoPE（时间T/高度H/宽度W）的编码能力。([arXiv][11])

**核心想法**
把你压缩后的历史 token（比如 8 帧 × 8×8=64 tokens/帧）**当作一个短视频 clip**：

* `T = num_history`
* `H=W=8`（或你压缩后的网格大小）
* 总 token = `T*H*W`（刚好就是你现在拼接出来的 512）

这样，历史不再是“无序拼接 token”，而是“带时间维度的 3D 位置编码序列”。

**为什么这对 BEV 特别有利**

* BEV 的关键难点是“旋转改变参考系 + 长轨迹回环”。
* 把 history 变成 video‑like（显式 time id），模型更容易学到“轨迹演化”，更像在看“走过的路线回放”。

**实现上怎么做（不改模型权重/结构）**
你已经在 evaluator 里拿到了每帧 ViT 特征和 `grid_thw`（当前帧用）。
你要做的是：

1. history per-frame 压缩后，不再简单 cat 成 `[512, d]` 然后把 position ids 全部设成 0；
2. 给这一块 history token 生成对应的 **(T,H,W) pos M‑RoPE 路径（或者在你自定义 embedding 注入时模拟其 position ids）。

**可以写成论文点的角度**
“**History-as-Video Prompting**”：把 VLN 历史序列对齐到 VLM 的视频建模接口（特别是 Qwen 的 M‑R处理历史，而不是把历史当静态记忆包。

---

### 方向 B：方位增强的两条路线（你提的“单帧方位增强”“pose encode 到 feature”我都给可证据化落法）

#### B1) 视觉层：在 BEV 图上画“指南针/航向箭头/刻度”

你不想改模型本体，那“把方位画进像素”是最干净的方式：

* 在每帧 BEV patch 中间画一个箭头：表示 agent heading（相对北/相对图像上方）。
* 画一个小 compass rose（N/E/S/W）或刻度环。
* 甚至在图像四角写上 “N” 标记。

**为什么有理有据**

* 在很多导航设置里，明确提供 compass/heading 是标准做法（PointNav 常用 GPS+Compass 传感器）。([AllenAct][12])
* Qwen2.5‑VL 强项之一就是理解图中图标、图表、布局等视觉符号（technical report + 训练指南都强调这类能力）。([arXiv][11])

**你可以做的 ablation**

* 只加 heading arrow vs 加 compass rose vs 加刻度环；
* 对 Landmark：看“最终 stop 时 yaw 误差分布”；
* 对 Boundary：看“stop 相对起点的距离分布”。

#### B2) embedding 层：Pose‑conditioned token shift（外接小模块，不动 Qwen）

你问“pose 是否能 encode 到每次 vit feature 上”。可以，而且很像很多 map‑based / projection‑based memory 的思想：

* GridMM 的核心就是用几何把历史放到统一的地图上，本质是“让 feature 带上空间坐标”。([arXiv][2])
* MapNav 也用 pose（配合深度）更新 top‑down map。([ar5iv][5])

**一个极简可落地做法**
对每帧（尤其 history 帧）的 token，加一个 pose embedding：

* 输入 pose：`[sin(yaw), cos(yaw), x_norm, y_norm]`（或者只用相对起点的 Δx,Δy + yaw）
* 用一个小 MLP/Linear 投影到 `d=3584`
* 加到这一帧的所有视觉 token：`v_{t,i} ← v_{t,i} + E_pose(p_t)`
* Qwen 参数不变；你只训练这个小 MLP（或者连它都不训，用固定 Fourier feature 映射也行）。

**这对你的两类任务的意义**

* Boundary：模型更容易把“离起点多远/朝向是否回到起点方向”作为停止依据。
* Landmark：模型更容易把“当前 yaw”作为旋转控制变量，减少“看到了但朝向不准”的失败。

---

### 方向 C：从固定采样升级为“检索/事件驱动 keyframe”，把 history 做聪明（直接对应你最不满意的点）

你现在 per-frame 是“从过去取 8 帧”。
最新工作里一个越来越清晰的结论是：**长程记忆最怕“无关信息堆积”和“关键帧缺失”**（MAM 就是专门解决 history noise）。([ScienceDirect][1])
而检索增强记忆在 embodied 里也被系统化提出过（ReMEmbR）。([NVIDIA Developer][13])

我建议你把 history_processor 做成“**Keyframe Bank + Retrieval**”，而不是“按时间采样”。

#### C1) 你可以维护 3 类 keyframe（都很适合 BEV）

1. **转弯触发 keyframe**：只要出现 LEFT/RIGHT，就存（因为 BEV 中旋转会重定义参考系，信息增量大）。
2. **新颖性触发 keyframe**：
   做全局池化得到 descriptor `g_t`

   * 若 `min_{k in bank} cosine(g_t, g_k) < τ`（不相似）→ 存为新 keyframe
3. **回环候选 keyframe**：保留 start 附近的一段关键帧（比如前 N 步的转弯/新颖 keyframes），专门服务 Boundary stop。

#### C2) 每次决策时从 bank 检索，而不是固定取最近/均匀

你可以组合三个检索器（取 top‑k，凑满 512 tokens 预算）：

* **place‑retrieval**：当前 descriptor vs bank descriptor（服务 loop closure、重访位置）
* **instruction‑retrieval**：用指令 embedding/文本关键词去选（服务 landmark 相关地标）
* **pose‑retrieval**：按空间距离选（服务局部路径连续性）

这就是“ReMEmbR 的 embodied retrieval 思路”在你任务上的实现落点。([NVIDIA Developer][13]) 到语义‑几何联合检索的历史构造”，并且可以结合 MAM 的论点强调“history noise”。([ScienceDirect][1])

---

### 方向 D：用“地图式记忆”替代大量历史帧（对 BEV 来说可能是最自然的范式转换）

你是卫星俯视图，这个条件比室内 egocentric VLN 更适合“map memory”。

你可以直接借鉴 MapNav/MapGPT/MC‑GPT/ GridMM 这一系：

* MapNav：用 ASM（带文字标注的语义地图）替代历史帧。([ar5iv][5])
* MapGPT：构建在线语言地图给 GPT 全局视角。([ACL Anthology][3])
* MC‑GPT：拓扑记忆图 + reasoning chains。([arXiv][4])
* GridMM：网格记忆图承载历史。([arXiv][2])

#### D1) 你的“最低成本版本”：Trajectory Sketch Map（强烈建议先做这个）

每一步用 pose 把“走过的轨迹”画到一张小图上（可以纯线稿，甚至不需要叠在卫星底图上）：

* 标记：start 点、current 点、heading arrow、已走路径 polyline
* 可选：把“窗口重叠段”用不同线型画出来，让模型区分近记忆/远记忆
* 把这张图作为 system prompt 的一张额外 image（或替代 history_memory）

**为什么这可能直接大幅改善 Boundary**
Boundary 的 success 本质上是“回到起点附近停下” —— 轨迹草图就是最直接的“回环证据”。

#### D2) 更进阶：Language‑enhanced BEV map（借鉴 Talk2BEV / ConceptFusion）

* Talk2BEV 把 BEV 对象加上语言描述和几何 cues，供 LLM 查询推理。([LLM Bev][14])
* ConceptFusion/VL‑Maps 类工作强调“foundation model 特征融进地图，让地图可被语言检索/查询”。([ConceptFusion][15])

对你来说，可以做一个“轻量语义层”：

* 在卫星图上做道路/水体/建筑粗分割（甚至规则方法也行），形成几个可解释 layer；
* 在 map 上用文字标注“road / river / dense buildings / open field”等（不需要很细）；
* 然后让 Qwen 读这张带标注的 map 来决策。

这条路线特别适合当论文贡献，因为它非常契合“俯视图导航”的独特性。

---

### 方向 E：Boundary vs Landmark 分开优化（你现在 Landmark 明显更难，建议把 memory 设计做“任务自适应”）

你结果里 Landmark SR 明显低于 Boundary，且 initial 策略对 Landmark 还可能有副作用。
这很像“同一套记忆对不同任务有不同最优形态”。

#### E1) Boundary（回环）专项：加入“回环证据”而不是更多历史帧

你可以加一个外部 loop closure signal（不改模型）：

* 用当前帧 descriptor 与 start descriptor 的相似度 `sim(t,start)`
* 或者用 VPR/Loop Closure 检测思想（视觉回环检测是成熟方向）。([arXiv][16])
  把这个信“Loop likelihood: 0.82 (high)”
* “Estimated distance-to-start: close / medium / far”（可离散化）

这会把 Boundary 的 stop 决策从“靠模型自己猜”变成“有可解释证据”。

#### E2) Landmark（精旋）专项：用逆动力学数据增强，让模型更懂“旋转→视野变化”

NaVIDA 的核心是 inverse dynamics augmentation：用视觉变化监督动作因果。([arXiv][10])

你完全可以**不改模型结构**地复用这个思想：

* 构造额外训练样本：输入 `(frame_t, frame_{t+1})`（或小段）+ instruction（可选），输出 “本步动作/动作块”。
* 对旋转尤其有效，因为旋转的视觉差异模式非常稳定。

这条线能显著提升 Landmark 的“朝向控制”，而且是有最新论文背书的。

---

## 4) 我建议你优先做的 2 个“最像论文创新点”的组合（都能对着现有 SwiftVLN 改）

我给你两个“组合拳”，每个都可以写成一个完整方法：

### 组合拳 1：Pose‑Aware Video Memory（强贴合 Qwen2.5‑VL）

1. history token 以 **video‑like (T,H,W) M‑RoPE** 方式输入（方向A）([arXiv][11])
2. 每帧加 **png prompt**（方向B2/B1）
3. history 选择改成 **retrieval keyframes**（方向C），避免 history noise（可引用 MAM）。([ScienceDirect][1])

**论文贡献表达**

* “将 VLN 历史对齐到 VLM 的视频建模接口 + pose 条件化，使 BEV 轨迹推理更自然”。
* 消融：无 M‑RoPE / 无 pose / 无 retrieval。

### 组合拳 2：Trajectory‑Map Memory for BEV‑VLN（最符合“俯视图任务特点”）

1. 用 pose 更新轨迹草图/局部地图（方向D1）
2. Boundary：加 loop closure signal（方向E1）([arXiv][16])
3. Landmark：加 inverse dynamics augmentation（方向E2）([arXiv][10])

**论文贡献表达**

* “把历史从 frame‑tokens 变成可解释的地图式记忆（MapNav/GridMM/MapGPT 系范式迁移到卫星 BEV）”。([ar5iv][5])

---

## 5) 最后给一个“落地优先级”建议（不需要你一次性改很大）

1. **先做 A：history-as-video + M‑RoPE**
   这是“对你现有结构侵入性最小，但潜在收益最大”的改动（因为你现在 history 无 ROPE）。
2. 然后做 C：**keyframe bank + retrieval** 替代固定采样
3. 再做 D1：**trajectory sketch map**（Boundary 大概率立竿见影）
4. 最后做 E2：**inverse dynamics augmentation** 专门救 Landmark（精旋）

---

如果你愿意把“你现在 history token 的 position_ids / grid_thw 具体怎么喂进 Qwen2.5‑VL”那段实现贴一小段（哪怕伪代码），我可以更精确地告诉你：

* “history‑as‑video”在你代码里应该改哪几个张量、怎么生成 (T,H,W) 的 position ids；
* 以及如何在不改变 generate 流程的情况下，把 trajectory sketch 作为额外 image 插进去（尽量不破坏你现有 template）。



[1]: https://www.sciencedirect.com/science/article/pii/S0031320324002620 "https://www.sciencedirect.com/science/article/pii/S0031320324002620"
[2]: https://arxiv.org/abs/2307.12907 "https://arxiv.org/abs/2307.12907"
[3]: https://aclanthology.org/2024.acl-long.529/ "https://aclanthology.org/2024.acl-long.529/"
[4]: https://arxiv.org/abs/2405.10620 "https://arxiv.org/abs/2405.10620"
[5]: https://ar5iv.org/pdf/2502.13451 "https://ar5iv.org/pdf/2502.13451"
[6]: https://arxiv.org/pdf/2507.05240 "https://arxiv.org/pdf/2507.05240"
[7]: https://arxiv.org/abs/2509.22548 "https://arxiv.org/abs/2509.22548"
[8]: https://arxiv.org/abs/2210.09461 "https://arxiv.org/abs/2210.09461"
[9]: https://arxiv.org/abs/2502.00791 "https://arxiv.org/abs/2502.00791"
[10]: https://arxiv.org/html/2601.18188 "https://arxiv.org/html/2601.18188"
[11]: https://arxiv.org/abs/2502.13923 "https://arxiv.org/abs/2502.13923"
[12]: https://allenact.org/projects/pointnav_baselines/ "https://allenact.org/projects/pointnav_baselines/"
[13]: https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr/ "https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr/"
[14]: https://llmbev.github.io/talk2bev/ "https://llmbev.github.io/talk2bev/"
[15]: https://concept-fusion.github.io/ "https://concept-fusion.github.io/"
[16]: https://arxiv.org/abs/2505.21754 "https://arxiv.org/abs/2505.21754"
