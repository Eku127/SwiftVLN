下面我把你问的三块拆开讲清楚：**(1) 没有东南西北时，B1 的“航向/指南针”到底标什么**；**(2) 怎么增强“左上/右下”这类图像内部相对方位信息**；**(3) pose embedding 的一个可直接落地的具体方案（表示、归一化、和 2048 维 ViT 特征怎么融合、MLP 怎么训）**。我会尽量给到你可以直接写进代码/论文的形式。

---

## 1) 没有东南西北概念时，航向到底怎么标？

你现在的问题本质是：BEV 里“旋转操作会改变图像方向参考系”，而且任务要求精旋（15°）和回环 stop。
即使没有“北/东”语义，只要你能**定义一个一致的参考方向**，航向标注就成立。可选的参考系有两种（我建议你两种都做 ablation）：

### 方案 B1‑R0：用“Episode 固定参考方向”当你的“北”

**定义：**把第 0 帧（或起点）图像的“图像上方”定义为参考方向 **Ref‑Up**，记为 0°。后续每一步只标 “当前航向相对 Ref‑Up 的角度”。

* 你不需要知道 Ref‑Up 是不是地理北，它只要在一个 episode 内始终一致即可。
* 这对 **Boundary 回环**非常自然：回到起点附近 + 航向接近起始航向，经常是停的强信号。

**图像上怎么画：**

* 在图像中心画一个小箭头表示 agent forward（或者写 `FWD`）。
* 在图像边缘（建议左上角）画一个固定小标记 `Ref`，并用一个小扇形/刻度条显示当前 `Δyaw`（例如 `+45°`）。

> 关键：你标的是“相对起点的航向角”，不是“东南西北”。

---

### 方案 B1‑R1：用“Map 坐标系 +X/+Y”替代东南西北（如果你环境确实有世界坐标）

很多模拟器/数据集都能给 agent 的位置与朝向（哪怕不叫东南西北，也一定有一个世界坐标轴）。Habitat 等基准里甚至会提供 GPS+Compass（位置+朝向相对起点），它被证明对导航成功率影响巨大。([GitHub][1])

**定义：**

* 定义世界坐标轴：`+X` 向右，`+Y` 向上（或者你环境里已有的 x/y 轴）。
* 航向就是 `yaw_world`（相对 +Y 或相对 +X 都行，但要固定一种）。

**图像上怎么画：**

* 左下角画两个小箭头：`+X →`、`+Y ↑`（这不会引入“北”的概念，但给了绝对参考）。
* 中心画 agent heading 箭头，并在旁边写角度（比如 `Yaw=120°`）。

**什么时候更合适？**

* 如果你的 BEV patch 有可能“跟着转”或“不会跟着转”，R1 都能统一，因为它明确给出世界坐标系。
* 对 **Landmark 精旋**也更稳定：模型能把“转 15°”理解成“yaw 改变 15°”。

---

## 2) 如何增强图像内部“左上/右下”等位置信息？

你文档里强调俯视图导航需要理解“左上方/右下角”等相对位置。
但 VLM 在几何/空间定位上经常不稳。增强这件事我建议做两条线：**token 级坐标注入（仍不改模型，只改你喂进去的特征）**。

---

### 2.1 像素级增强（最简单、最“不会碰模型”的）

建议你在每一帧 BEV 图上叠加以下元素（越“规则化”，VLM 越容易学）：

1. **九宫格/十字坐标轴**

* 画一个中心十字（横竖两条细线），把图分成四象限。
* 可选画 3×3 网格（九宫格），每个格子用 1–9 编号（或 TL/TR/BL/BR + 中心 C）。

2. **角标文字**

* 四个角分别写：`左上 / 右上 / 左下 / 右下`（或者 `TL/TR/BL/BR`）。
* 这对 VLM 很友好，因为它“能读字”。（你也可以在 system prompt 里解释这些角标含义。）

3. **图像坐标轴 legend**

* 在角落画：`X→`、`Y↑`，并在 prompt 中说明：
  “图像坐标系：向右为 +X，向上为 +Y。‘左上’= (−X, +Y)。”

这类做法虽然“朴素”，但非常有效，因为它把抽象几何关系变成 VLM 擅长的“读图符号”。

---

### 2.2 Token 级增强：给每个 patch token 加 (x,y) 坐标 embedding（推荐你一定做）

如果你担心“画线/画字会干扰原图语义”，那就走 token 级：**不改 Qwen 本体，只在 ViT 输出 token 上做一个可训练/不可训练的坐标注入**。

这在视觉里是有经典依据的：CoordConv 证明了显式加入坐标通道能显著缓解模型对空间定位的失败，并且常用归一化到 [-1, 1]。([arXiv][2])
而 Fourier features（sin/cos positional mapping）是让低维坐标更容易被 MLP 使用的经典方式。([arXiv][3])

**做法：**

* 你的 ViT 输出 tokens 是 16×16=256（未压缩），或压缩后 8×8=64。
* 每个 token 有一个网格坐标 `(i,j)`：

  * 归一化：
    [
    x = 2\cdot \frac{j}{W-1}-1,\quad y = 2\cdot \frac{i}{H-1}-1
    ]
* 把 `(x,y)` 变成一个 `D`en 上：
  [
  v_{t,i} \leftarrow v_{t,i} + E_{xy}(x_i,y_i)
  ]
  这会让“左上/右下”的信息以**连续坐标**的形式进入模型，不依赖模型自己从 patch 的位置编码里“猜”。

> 你可以把它当作“CoordConv 的 token 版本”。([arXiv][2])

---

## 3) Pose embedding：非常具体的可落地方案（表示 / norm / 融合 / 训练）

你问得很关键：**pose 怎么表示？怎么 norm？2048 维 ViT 特征怎么融合？MLP 单独训还是一起训？**
下面给你一个我认为在你 OverlapVLN 框架里最顺滑的版本（外接模块，不改 Qwen 结构）。

---

### 3.1 Pose 表示：建议同时包含“全局相对起点”和“局部相对上一步”

原因：

* **Boundary 回环**：更依赖“相对起点的位置/朝向”。
* **Landmark 精旋**：更依赖“当前 yaw + 最近动作造成的 yaw 变化”（局部更敏感）。

假设你能从环境拿到连续位姿 `(x_t, y_t, θ_t)`（θ 是 yaw，弧度制），定义：

**(A) 全局相对起点：**

* [
  \Delta x_t = x_t - x_0,\quad \Delta y_
  \Delta \theta_t = \mathrm{wrap}(\theta_t - \theta_0)\in[cos 编码角度（避免 179° 与 −179° 不连续）：
  [
  s_t=\sin(\Delta\theta_t),\quad c_t=\cos(\Delta\theta_t)
  ]

**(B) 局部相对上一步（可选但我建议加）：**

* [
  \delta x_t = x_t - x_{t-1},\quad \delta y_t = y_t - y_{t-1}
  ]
* [
  \delta\theta_t = \mathrm{wrap}(\theta_t - \theta_{t-1})
  ]
* 同样用 sin/cos：(\sin(\delta\theta_t),\cos(\delta\theta_t))

最终 pose 向量可以是：
[
p_t = [\Delta x_t,\Delta y_t,\sin\Delta\theta_t,\cos\Delta\theta_t,\ \delta x_t,\delta y_t,\sin\delta\theta_t,\cos\delta\theta_t]
]
维度 8，很干净。

---

### 3.2 Pose 归一化（norm）：给你 3 个可选策略（按推荐顺序）

你问“怎么 norm”，这里非常重要——norm 不好会直接训练崩。

#### Norm‑1（推荐）：按“观测裁剪半径/视野尺度”归一化

BEV patch 一般对应一个固定的物理范围（比如半径 R 米）。如果你知道这个 R：

* [
  \hat{\Delta x}=\mathrm{clip}(\Delta x/R,\ -1,1),\quad
  \hat{\Delta y}=\mathrm{clip}(\Delta y/R,\ -1,1)
  ]
* 局部步长同理：(\hat{\delta x}=\delta x / R)（或除以 step_size）

优点：尺度稳定；跨 episode 不漂。

#### Norm‑2：用数据集统计量做标准化（z-score）

离线扫一遍训练集，统计所有 (\Delta x, \Delta y) 的均值方差：

* [
  \hat{\Delta x}=(\Delta x - \mu_x)/\sigma_x
  ]
  然后再 `clip` 到例如 [-3,3]。

优点：泛化好；缺点：要做一次统计。

#### Norm‑3：只用“相对量 + tanh 压缩”

如果你不想考虑尺度：

* [
  \hat{\Delta x}=\tanh(\Delta x / s)
  ]
  s 是手动常数（例如 30m/50m）。这很工程，但能跑。

角度部分用 sin/cos 本来就是 [-1,1]，不需要额外 norm。

---

### 3.3 怎么把 pose 融合到 2048 维 ViT token 上？

你说 ViT 特征是 **2048 维**，那我们就以 `D=2048` 来写（如果你后面其实还有投影层，把 D 替换成投影后的 hidden size 即可）。

假设单帧 token：

* 未压缩：`V_t ∈ R^{256×2048}`
* 压缩后：`Ṽ_t ∈ R^{64×2048}`（你现有 per-frame 压缩就是这个概念）

#### 融合方式 1（最稳、最常用）：Additive Bias（推荐先做它）

用一个小 MLP 把 pose 映射到同维度，然后广播相加：

* [
  e_t = \mathrm{MLP}(p_t) \in R^{2048}
  ]
* [
  Ṽ'_t = Ṽ_t + \mathbf{1}\cdot e_t^\top
  ]

优点：实现最简单，不改变 token 数、不融合方式 2（更强一点）：FiLM / Feature-wise affine（建议作为增强版）
FiLM 是成熟的“条件调制”方法：产生每个通道的缩放和偏置。([arXiv][4])

* [
  [\gamma_t,\beta_t] = \mathrm{MLP}(p_t),\quad \gamma_t,\beta_t\in R^{2048}
  ]
* [
  Ṽ'_t = Ṽ_t \odot (1+\gamma_t) + \beta_t
  ]

**训练稳定小技巧：**把 MLP 的最后一层初始化为 0（权重/偏置全 0），这样一开始 (\gamma=\beta=0)，网络等价于原模型，训练更稳。

---

### 3.4 这个小 MLP 怎么设计？

非常简单的两层就够了：

* `Linear(p_dim=8 → h=256)`
* `GELU`
* `Linear(256 → out_dim=2048)` （Additive）
  或 `Linear(256 → 4096)`（FiLM 输出 γ+β）

参数量很小，几乎不增加负担。

---

### 3.5 这个 MLP 是单独训练还是混在一起训练？

这里分两种“你想不想动 Qwen 权重”的哲学（都不改结构，只是训不训权重）：

#### 训练策略 S1：**冻结 Qwen，只训练 pose‑MLP（最符合“不改模型本身”）**

* Qwen 全部参数 `requires_grad=False`
* 只训练 pose‑MLP（以及你自己的 history processor 里可训练部分，如果有）
* 优点：完全不动基础模型；工程风险低；论文里也容易强调“plug‑and‑play”
* 缺点：上限可能受限（因为模型没机会适配 pose 注入带来的分布变化）

**我建议的 lr：**`1e-3 ~ 3e-4`（看 batch size），weight decay 很小或不用。

#### 训练策略 S2：**和你现有 OverlapVLN 一起端到端训练（推荐你最终要做）**

你本来就在训练 OverlapVLN（调 prompt / 记忆处理 / 可能也在微调模型），那 pose‑MLP 最自然就是放进同一个 loss 里一起训。

**关键建议：分参数组学习率**

* Qwen：`lr = 2e-5`（你现在类似这个量级）
* pose‑MLP：`lr = 2e-4`（通常比 backbone 大 5~10 倍）

这样 MLP 学得动、Qwen 不会被冲坏。

> 小结：如果你非常强调“完全不动 Qwen 权重”，选 S1；如果你追求性能与论文贡献，选 S2，并用参数组 lr 控制风险。

---

## 4) 你可以怎么在 OverlapVLN 代码流里放进去（对应t 会把 `<history_memory>` 替换成历史特征缓存，而且历史块目前是“无 ROPE”的统一块。

所以 pose‑embedding 最自然的插入点是：

1. `encode_frame()` 得到 `V_t`
2. `compress_frame(V_t)` 得到 `Ṽ_t`（64×2048）
3. 从 env 拿 `p_t`，算 norm 后进 MLP 得 `e_t`
4. `Ṽ'_t = Ṽ_t + e_t`（或 FiLM）
5. `history_cache.append(Ṽ'_t)`，后面照旧拼进 systrrent frame** 也可以做同样处理（因为 Landmark 精旋更依赖当前 yaw），即：当前帧的 256 tokens 也加同一个 pose embedding。

---

## 5) 建议你做的最小 ablation（能快速判断是不是方向对了）

1. 只做 **角标+九宫格**（像素增强） vs 不做
2. 只做 **pose additive**（S1 冻结 Qwen） vs 不做
3. pose additive vs pose FiLM
4. pose 向量只用全局（Δx,Δy,sinΔθ,cosΔθ） vs 全局+局部（再加 δx,δy,sinδθ,cosδθ）

你会很快看到：

* Boundary 是否更敢 stop（dist-to-start 更“像样”）
* Landmark 是否 yaw 误差分布显著变窄（精旋更稳）

---

如果你愿意，我可以按你现在的张量形状再把伪代码写得更贴近你项目（比如你是 `[B,256,2048]` 还是 `[256,2048]`，以及 pose 是按 step 还是按 turn 对齐），并且给你一个“不会破坏原有训练稳定性”的初始化与 optimizer 参数组写法。

[1]: https://github.com/facebookresearch/habitat-challenge?utm_source=chatgpt.com "GitHub - facebookresearch/habitat-challenge: Code for the habitat challenge"
[2]: https://arxiv.org/pdf/1807.03247?utm_source=chatgpt.com "and the CoordConv solution - arXiv.org"
[3]: https://arxiv.org/abs/2006.10739?utm_source=chatgpt.com "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
[4]: https://arxiv.org/abs/1709.07871?utm_source=chatgpt.com "FiLM: Visual Reasoning with a General Conditioning Layer"
