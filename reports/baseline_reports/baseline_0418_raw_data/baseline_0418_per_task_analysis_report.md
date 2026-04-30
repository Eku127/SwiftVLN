# Baseline 0418 Per-Task Analysis Report

## 数据来源

- `baseline_0418_val_seen_breakdown.csv`
- `baseline_0418_val_unseen_breakdown.csv`

分析口径：

- `Boundary / Landmark / Road` 三类任务分别看 `SR / OS / NE / Steps`
- 重点结合 SatNav 任务特性解释：
  - `Boundary` 更依赖 stop 决策，容易出现早停、回到目标区后停不下来、绕圈
  - `Landmark` 是强定位任务，一旦 landmark 对齐失败，轨迹很容易整体走飞
  - `Road` 主要考察 road following 与 counting，错误通常是跟丢路段、岔路选择错误、计数偏移

## 总览结论

- 当前 strongest tier 很清晰：`StreamVLN / StreamVLN* / SwiftVLN` 明显领先，`UniNaVid*` 处于第二梯队，传统 baseline 与 NaVILA 系列在 `Landmark` 上基本失效。
- `Boundary` 是 top models 最稳定的任务。`seen/unseen` 上 `StreamVLN` 与 `SwiftVLN` 都能保持 `61%~69%` SR，说明“回家”本身不难，真正的瓶颈在 stop 时机。
- `Landmark` 是最能拉开模型代差的任务。`CMA / Seq2Seq / OpenFly` 在这项上几乎没有竞争力，而 `StreamVLN* / SwiftVLN / StreamVLN / UniNaVid*` 才真正具备 landmark grounding 能力。
- `Road` 在 `seen` 上不算最难，但在 `unseen` 上退化明显，说明它对 route pattern 泛化、路口计数和长链路跟随更敏感。

## 1. Boundary 任务分析

### 结果现象

- `seen` 最强三者：
  - `StreamVLN` `68.86%`
  - `StreamVLN*` `68.53%`
  - `SwiftVLN` `66.47%`
- `unseen` 最强三者：
  - `StreamVLN` `63.19%`
  - `StreamVLN*` `62.87%`
  - `SwiftVLN` `61.37%`

这说明 top tier 模型在 boundary 上已经形成稳定能力，跨 split 的跌幅也不算大：

- `StreamVLN`: `-5.67pt`
- `StreamVLN*`: `-5.66pt`
- `SwiftVLN`: `-5.10pt`

### 关键诊断：OS-SR gap 很重要

对 boundary，`OS` 可以近似理解成“已经成功回到目标区域”，而 `SR` 还要求“在合适时机停住”。因此 `OS - SR` 很适合看 stop failure。

- top models 的 gap 仍然不小：
  - `StreamVLN`: `14.21pt` seen, `15.89pt` unseen
  - `StreamVLN*`: `11.22pt` seen, `14.81pt` unseen
  - `SwiftVLN`: `13.68pt` seen, `13.73pt` unseen
- 弱模型 gap 极大：
  - `CMA`: `79.99pt` seen, `78.24pt` unseen
  - `Seq2Seq`: `76.56pt` seen, `70.86pt` unseen

这对应的任务解释非常直接：

- top models 已经能“回到家附近”，但仍有一部分 episode 没有在正确时机执行 `STOP`
- 弱模型不是完全回不来，而是大量出现“回到目标区但没停住”或者“停在错误时机”

### Steps 与行为模式

- `SwiftVLN` 在 boundary 上的步数比 `StreamVLN` 更短：
  - seen: `56.55` vs `63.77`
  - unseen: `61.80` vs `67.56`
- 同时两者 SR 很接近，说明 `SwiftVLN` 的 boundary 行为更果断，但 stop 决策仍未完全收敛
- `CMA` 和 `UniNaVid` 的步数明显偏长，尤其 seen：
  - `CMA`: `80.05`
  - `UniNaVid`: `102.00`

这和 boundary 任务里典型的“绕圈停不下来”一致。模型可能已经接近 home area，但 stop policy 不稳，导致继续走、错过停止窗口、再绕一圈。

### Boundary 结论

- boundary 不是“导航回家”做不到，而是“回家后如何停对”仍是主问题
- 对 top models，下一步提升空间主要不在路径搜索，而在 stop calibration
- 对弱模型，stop failure 是主要错误类型，不能只看 SR 低就归因为整体导航差

## 2. Landmark 任务分析

### 结果现象

- `seen` 最强：
  - `StreamVLN*` `71.17%`
  - `SwiftVLN` `66.07%`
  - `StreamVLN` `60.04%`
  - `UniNaVid*` `58.85%`
- `unseen` 最强：
  - `StreamVLN*` `59.02%`
  - `SwiftVLN` `46.76%`
  - `UniNaVid*` `45.47%`
  - `StreamVLN` `43.66%`

而传统 / NaVILA baselines 在 landmark 上基本失效：

- `Seq2Seq`: `0.00% / 0.22%`
- `CMA`: `0.00% / 0.00%`
- `OpenFly`: `2.65% / 5.12%`
- `NaVILA`: `8.08% / 3.45%`
- `NaVILA*`: `4.51% / 4.91%`

这说明 landmark 任务对 grounded localization 的要求非常高，没有足够强的视觉定位与 instruction-grounding 能力，很容易直接崩掉。

### 关键诊断：Landmark 是最典型的“定位失败后整体走飞”任务

Landmark 上 `NE` 非常说明问题：

- `SwiftVLN`: `60.57 -> 102.73`
- `StreamVLN`: `98.10 -> 145.66`
- `StreamVLN*`: `92.24 -> 152.97`
- `UniNaVid*`: `186.74 -> 296.14`
- `NaVILA`: `128.96 -> 149.17`
- `NaVILA*`: `146.61 -> 151.93`
- `CMA`: `789.61 -> 853.06`

这和 SatNav 的 landmark 特性高度一致：

- 一旦当前位置估错，后续动作不是局部小偏差，而是整条轨迹被带飞
- 因为 landmark instruction 通常依赖特定地标、局部视觉模式和相对关系，一旦第一个定位锚点对不上，后面纠错会越来越难

### Generalization drop

landmark 在 `unseen` 上对强模型依然有明显打击：

- `SwiftVLN`: `-19.31pt`
- `StreamVLN`: `-16.38pt`
- `StreamVLN*`: `-12.15pt`
- `UniNaVid*`: `-13.38pt`

这说明即便 strongest tier 能做 landmark grounding，它们依然依赖 seen split 中学到的场景统计规律；一旦换到 unseen city，定位鲁棒性立刻下降。

### OS-SR gap 的解释

在 landmark 上，top models 的 `OS-SR gap` 相对 boundary 小很多，通常只在 `4~6pt` 左右：

- `StreamVLN`: `4.64pt` seen, `4.43pt` unseen
- `StreamVLN*`: `6.17pt` seen, `5.99pt` unseen
- `SwiftVLN`: `6.03pt` seen, `5.09pt` unseen

这意味着 landmark 的主矛盾不是 “到了但没停”，而是 “根本没定位准，所以没到”。也就是：

- boundary 更像 stop 问题
- landmark 更像 localization 问题

### Landmark 结论

- landmark 是当前最能体现模型上限的 challenge task
- `SwiftVLN` 比 `StreamVLN` 在 landmark 上更强：
  - seen `+6.03pt SR`
  - unseen `+3.10pt SR`
  - 且 `NE` 更低、步数更少
- 但 `StreamVLN*` 仍是 landmark 的 strongest model
- 如果后续要继续拉开主线模型差距，landmark 是最值得重点优化的一项

## 3. Road 任务分析

### 结果现象

- `seen` 最强：
  - `StreamVLN*` `71.33%`
  - `StreamVLN` `64.02%`
  - `UniNaVid*` `53.82%`
  - `SwiftVLN` `53.75%`
- `unseen` 最强：
  - `StreamVLN*` `53.69%`
  - `StreamVLN` `50.05%`
  - `SwiftVLN` `40.15%`
  - `UniNaVid*` `37.07%`

road 在 seen 上并不一定最难，但在 unseen 上退化很明显：

- `StreamVLN*`: `-17.64pt`
- `StreamVLN`: `-13.97pt`
- `SwiftVLN`: `-13.60pt`
- `UniNaVid*`: `-16.75pt`

### 关键诊断：road 的主要问题是长链路跟随与 counting 漂移

road 更依赖：

- 连续 road following
- 多路口 sequence 记忆
- turning / intersection counting

从 `NE` 的变化看，这类误差在 unseen 上明显累积：

- `StreamVLN*`: `35.60 -> 87.02`
- `StreamVLN`: `46.38 -> 90.43`
- `SwiftVLN`: `55.12 -> 112.53`
- `UniNaVid*`: `47.85 -> 115.72`

这说明 road 上的失败很多不是一开始就完全跑飞，而是：

- 前半段能沿路走
- 但在关键路口、计数节点、或者分支判断处偏了一次
- 偏一次后，后面就会逐步累积成较大的终点误差

### OS-SR gap 相比 Boundary 更小

对 strongest tier，road 的 `OS-SR gap` 没有 boundary 那么夸张：

- `StreamVLN`: `3.52pt` seen, `6.06pt` unseen
- `StreamVLN*`: `4.42pt` seen, `8.94pt` unseen
- `SwiftVLN`: `6.61pt` seen, `8.58pt` unseen

这意味着 road 的首要问题不是 stop，而是“有没有一路跟对、有没有数对”。一旦路段跟随和计数失败，SR 和 OS 会一起掉。

### SwiftVLN 在 road 上的相对短板

与 `StreamVLN` 对比，`SwiftVLN` 在 road 上明显更弱：

- seen: `53.75% vs 64.02%`，差 `-10.27pt`
- unseen: `40.15% vs 50.05%`，差 `-9.90pt`
- `NE` 也更高：
  - seen: `55.12 vs 46.38`
  - unseen: `112.53 vs 90.43`

这说明 `SwiftVLN` 现在的优势更偏向 boundary 与 landmark，但在 road counting / long-horizon following 上还没有追平 StreamVLN 系列。

### Road 结论

- road 是最典型的“过程跟随错误累积”任务
- unseen 下的退化幅度说明它比 boundary 更依赖时序记忆和 route pattern 泛化
- 当前如果要补 `SwiftVLN` 的短板，road 是比 boundary 更值得优先补的方向

## 4. 模型侧结论

### StreamVLN / StreamVLN*

- 仍是最均衡的强 baseline
- `StreamVLN*` 在 `Landmark` 和 `Road` 上提升最明显
- 但在 boundary 上并没有明显超出 `StreamVLN`，说明继续训练更主要提升的是复杂定位与长链路跟随，不是 stop 决策

### SwiftVLN

- strongest characteristics：
  - boundary 强
  - landmark 强
  - 步数普遍更短，行为更干净
- 当前短板：
  - road 弱于 `StreamVLN`
  - boundary 上仍存在约 `13~14pt` 的 OS-SR gap，stop 仍可继续优化

### UniNaVid / UniNaVid*

- `*` 版本提升非常大，尤其 landmark 提升最明显
- 但整体仍不如 `StreamVLN / SwiftVLN`
- 在 unseen 上 NE 仍偏大，说明泛化稳定性不足

### NaVILA / OpenFly / CMA / Seq2Seq

- 在 boundary 与 road 上有部分能力，但 landmark 基本不成立
- 这类模型更容易出现：
  - boundary：回到附近但停不住
  - landmark：定位不到，直接走飞
  - road：能走一段，但计数和分支判断不稳定

## 5. 面向任务特性的改进建议

### Boundary

- 加强 stop-specific supervision，而不是只优化整体路径
- 重点看 “进入 goal area 后若继续移动” 的负样本
- 可以加入 boundary 专门的 stop calibration 或 loop termination 策略

### Landmark

- 优先加强 landmark grounding 与 localization consistency
- 可以重点看：
  - landmark phrase 与视觉区域的对齐
  - 历史记忆是否能稳定支持重定位
  - 当 landmark matching 置信度下降时，是否存在 recovery policy

### Road

- 补强 counting 和 sequential route tracking
- 重点排查：
  - 第几个路口左/右转的计数是否稳定
  - road segment 切换时，history 是否足够保留分叉前上下文
  - unseen 城市中是否存在不同道路形态导致的 counting drift

## 最终判断

- 如果从任务属性看：
  - `Boundary` 主要是 stop 问题
  - `Landmark` 主要是 localization 问题
  - `Road` 主要是 route-following + counting 问题
- 如果从当前结果看：
  - `Boundary` 已经不是主线模型最大的短板
  - `Landmark` 最能体现模型能力上限
  - `Road` 则是 `SwiftVLN` 当前最值得优先补的任务

一句话总结：

- `StreamVLN*` 是当前最均衡的 strongest baseline
- `SwiftVLN` 在 `Boundary + Landmark` 上已经很强，但 `Road` 还有明显补强空间
- 后续如果要继续提升主线模型，建议优先做 `Road counting/following` 与 `Boundary stop calibration`，同时把 `Landmark` 作为最核心的 challenge benchmark 持续跟踪
