# StreamVLN, UniNaVid And OpenFly Analysis

## 核心判断

1. `StreamVLN` 强，首先是因为 `scratch` 起点就很高：底座本身就是强视频 VLM，不需要从零补视频理解。
2. `UniNaVid` 的 `continue` 大幅强于 `scratch`，核心是初始化差距：前者继承了已经做过大规模导航训练的 checkpoint，后者从纯文本 `Vicuna-7B v1.5` 出发。
3. `OpenFly` 相对弱一些，不是单一原因，更像是几种不利因素叠加：
   - 官方模型本来是为 `UAV aerial VLN` 设计的，和 SatNav 不是同一任务分布。
   - 它自己的历史视觉上下文本来就比较短，长程导航上天然吃亏。
   - 它继承的 `OpenVLA/OpenFly` 动作先验与 SatNav 当前四动作接口并不完全对齐。

## StreamVLN

`StreamVLN` 的优势，第一还是底座。`scratch` 直接从 `LLaVA-Video-7B-Qwen2` 这类强视频模型起步，所以它不是“先学会看视频，再学导航”，而是在已有视频理解能力上做导航专项适配。

第二是它的方法设计本身很适合在线导航。它不是简单地把更多历史帧拼进去，而是用 `sliding-window KV cache + slow memory pruning` 做流式上下文管理：当前窗口响应快，长期历史又不会完全丢掉。这对 SatNav 这种持续决策任务很重要。

如果看 `continue` 版本，它还叠加了官方 `StreamVLN` checkpoint 本身的收益。官方仓库明确写了这条线除了 VLN 轨迹训练外，还有第二阶段 co-training，混入了 `LLaVA-Video-178K`、`ScanQA` 和 `MMC4`。所以 `StreamVLN` 强，不只是因为基模好，也因为它的后续 recipe 本身就比较完整。

但即便如此，`continue` 相对 `scratch` 的涨幅通常还是不会像 `UniNaVid` 那么夸张。更稳妥的理解是：它的 `scratch` 起点已经足够高，`continue` 主要是在已有强视频能力上继续做 VLN 适配和数据对齐。

## UniNaVid

`UniNaVid` 的强项在于，它本身就是面向导航的视频 VLA，而不是普通图像问答模型。它的输入输出形式、导航 token、历史帧建模和多步动作预测，都比传统 imitation baseline 更贴近 SatNav。官方项目页也明确写了它的训练数据组成是 `Navigation 3.6M + VQA 2.3M`，这说明它的 `continue` checkpoint 本身就已经具备比较强的导航和视频理解先验。

`continue` 和 `scratch` 的差距则主要来自起点不同，而不是“多训了一轮”。`scratch` 是从纯文本 `Vicuna-7B v1.5` 出发，再外挂视觉编码器和 projector 去学多模态导航；`continue` 则直接继承已经完成大规模导航/VQA 训练的 `Uni-Navid` checkpoint。两者的初始化质量不在一个量级上，所以差距会非常大。

这里要区分两个概念：`scratch` 的基础 checkpoint 确实是纯文本模型，但最终训练出来的系统不是纯文本系统，因为训练时仍然会接上视觉模块和导航相关组件。真正的问题在于，它的一开始没有现成的视频导航先验。

## OpenFly

`OpenFly` 整体不如前两者，最关键的问题仍然是你提到的两点：历史信息处理能力有限，以及动作空间不够匹配。但把论文、官方代码和当前 SatNav 适配放在一起看，细节需要更严谨。

### 为什么它在 SatNav 上容易吃亏

第一，官方 `OpenFly-Agent` 的原始任务是 `aerial VLN`。论文和官方模型卡都明确写了，它是 `UAV` 导航模型，基于 `OpenVLA`，输入语言和图像，输出无人机动作。这里我之前写成“地面导航”是不对的。更准确地说，SatNav 在当前仓库里虽然也是 `4` 个离散动作，但它是卫星/航拍视角下的导航环境，不是地面机器人导航；因此问题不是“地面 vs 空中”这么简单，而是 **OpenFly 的原始 UAV 行为分布、动作表达和 SatNav 当前任务接口并不完全一致**。

第二，`OpenFly` 的历史机制并不是你这边后改弱的。官方 released code 训练默认就是 `history_frames=2`，输入也是 `current + 2 history` 三张图；HF 模型卡的最小推理示例同样是三图输入。所以“只看三张图”本身就是 OpenFly 这条线的原生设定，不是你这里额外削弱出来的。更准确的结论应该是：**OpenFly 自身的历史视觉上下文就偏短**，这在长程 SatNav 上容易吃亏。

这里还有一个容易误判的点。官方论文强调了 `keyframe selection`、`landmark grounding`、`visual token merging` 和一个小 memory bank；但 released code 里真正公开出来的训练/推理接口更像是“三图输入 + history_frames=2”的实现，而不是一个很强的长时记忆系统。也就是说，OpenFly 的“history 不强”这个判断是对的，但更像是方法本身的上限，而不是你这边把它改坏了。

第三，动作空间适配本身就是明显负担，而且这里还有一个“论文和代码不完全一致”的问题。论文版本把 OpenFly-Agent 概括成 `6` 个 UAV 动作；但你 workspace 下的官方 released code 和评测脚本里，动作仍然是基于 `8` 维连续 action vector 来做 tokenization 和反归一化，代码里甚至枚举了 `10` 个模板动作，包括不同前进距离、上下移动和横移。SatNav 当前接口则是 `STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT` 四动作。换句话说，**OpenFly 继承的原始动作先验，与 SatNav 当前四动作接口之间存在明显的表示落差**。

你本地 baseline 后来把默认输出切到 `compact` 四动作文本格式，这其实正说明了原来的 `original` 路径在 SatNav 上不够顺。更准确地说，当前训练时最终输出接口已经被你适配成四动作了，但模型继承的底层动作建模先验并没有天然对齐，这也是为什么 `compact` 路线会更稳。

### 为什么 `continue` 还是会比 `scratch` 好

`continue` 的底座虽然不完全匹配 SatNav，但它至少已经是一个训练过的 `OpenFly-Agent`，已经具备语言到导航动作的映射、以及 `current + 2 history` 这种 OpenFly 原生视觉接口上的导航先验。`scratch` 则是从 `OpenVLA` 这个通用机器人操作 VLA 出发，本来更偏通用 action prediction，不是专门为 UAV VLN 或 SatNav 准备的。

所以 `OpenFly continue > scratch` 是合理的，但它不会像 `UniNaVid continue` 那样把差距一下子拉得特别开。因为 `OpenFly continue` 自己也带着任务迁移误差，它不是“完全对口”的起点，只是比 `scratch` 更接近导航。

### 其他值得注意的因素

除了历史和动作空间，还有三个因素也在拖它：

1. `stop` 偏置。OpenFly 这条线之前明显受过 premature stop 影响，后来把 `stop_window` 调成 `0` 才缓过来。这说明它对“什么时候该停”很敏感，而 SatNav 的成功判定恰好非常依赖这个点。
2. 目标不完全对齐。OpenFly 的训练更像 action-token prediction，但 SatNav 真正看的是长时序闭环导航结果。前者 loss 下降，并不一定自动转化成更高 SR。
3. 论文和 released code 有落差。论文里 OpenFly 的叙述比 released code 更“干净”，例如动作类型和 keyframe 选择机制的描述更理想化；但你 workspace 里的开源实现实际更接近 `OpenVLA-style 8D action token + 2 history frames`。这会让它迁到 SatNav 时更容易出现接口和归纳偏置不匹配。
4. 需要补充一点：你本地 SatNav 适配其实还额外加了 `past actions` 文本历史，而官方 released code 训练 prompt 并没有真正把 dataset 里的 `history` 用进 prompt。因此如果只谈“文字历史”，你这边不是削弱，反而是加了额外线索。真正的瓶颈还是视觉历史窗口太短，以及动作接口不对齐。

## 对比总结

- `StreamVLN` 更像是：`强视频底座 + 合适的流式长历史机制`，所以整体最稳。
- `UniNaVid` 更像是：`continue checkpoint 很强，scratch 起点太弱`，所以 continue/scratch 差距最大。
- `OpenFly` 更像是：`原始 UAV VLA 任务不完全对口 + 方法本身历史窗口短 + 动作接口迁移成本高`，所以即使 recipe 修正后能起量，整体上限还是更容易被压住。
