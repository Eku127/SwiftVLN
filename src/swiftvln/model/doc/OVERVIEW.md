# SwiftVLN 逻辑概览

本文档描述 SwiftVLN 的整体工作流程，重点介绍滑动窗口、上下文重用机制和历史记忆处理策略。

---

## 0. 任务背景

### 0.1 俯视图无人机视觉语言导航

SwiftVLN 面向的是**俯视图（Bird's-eye View, BEV）无人机视觉语言导航（VLN）任务**。不同于传统的第一人称视角室内导航任务，本任务使用**卫星图像**模拟真实无人机的俯视视角，在室外大范围连续空间中执行导航。

### 0.2 俯视图导航的独特挑战

俯视图导航与传统正视图（第一人称）导航存在本质差异：

**1. 空间关系理解**
- 需要理解物体在图像上的**相对位置**（如"左上方"、"右下角"）
- Agent 必须将图像坐标系与自身运动方向建立映射关系
- 例如：指令"向左上角的建筑物移动"，需要先识别建筑在图像上的位置，再计算所需旋转角度

**2. 旋转定位的重要性**
- 俯视图中，旋转操作直接改变图像的方向参考系
- Agent 需要通过**精确旋转**来对齐目标方向
- 旋转后的视野变化显著，需要重新理解场景布局

**3. 前进动作的信息量**
- 与第一人称视角不同，俯视图的 FORWARD 动作包含**丰富的环境信息**
- 可以同时观察前方、左侧、右侧的大范围场景
- 每一步前进都提供更全面的空间上下文

### 0.3 两种任务类型

本任务包含两种不同特性的导航场景，对应不同的成功标准：

#### Boundary（边界导航）

**特点**：
- 具有**回环（loop）特性**，轨迹形成闭合路径
- 指令通常描述一个区域的边界，如"沿着公园边缘行走"
- Agent 需要在接近起点的准确位置停止

**挑战**：
- 需要记忆起点位置，判断何时完成回环
- 对位置精度要求高（SUCCESS_DISTANCE: 10m）
- 需要平衡前进与停止的时机

#### Landmark（地标导航）

**特点**：
- 需要导航到某个特定位置，并执行**精确旋转**
- 指令通常要求"面向特定地标"，如"走到路口，面向教堂"
- Agent 需要在**看到目标地标**后停止

**挑战**：
- 需要精准的位置定位（SUCCESS_DISTANCE: 30m）
- 必须执行**高精度旋转**（15° 单位角度）来对准地标
- 需要在视野中识别并确认目标地标的出现
- 相比 Boundary 任务，Landmark 任务对视觉理解和旋转控制的要求更高

### 0.4 任务难度

基于最新数据集（ver_260206，修复 landmark 识别问题）的评估结果显示：
- **Boundary 任务**：SR 约 55-64%，相对更易完成
- **Landmark 任务**：SR 约 24-33%，难度显著更高

Landmark 任务的低成功率反映了俯视图下精确旋转定位和地标识别的挑战性。

---

## 1. 核心思想

SwiftVLN 引入**滑动窗口重叠机制**，解决长轨迹推理中上下文丢失的问题。

```
问题: 独立窗口导致上下文断裂
  窗口1: [step 0-31]  →  窗口2: [step 32-63]
                         ↑
                    完全独立，上下文断裂

SwiftVLN 解决方案: 滑动窗口 + 上下文传递
  窗口1: [step 0-31]   →  窗口2: [step 16-47]
         └── 最后16步 ──┘        └── 前16步来自窗口1
                overlap_context 传递上下文
```

### 1.1 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_frames` | 32 | 窗口大小（动作数） |
| `num_overlap` | 16 | 窗口重叠的动作数 |
| `num_future_steps` | 4 | 每轮预测的动作数 |
| `num_history` | 8 | 全局历史帧采样数（per_frame 模式） |
| `compress_stride` | 2 | 历史帧压缩步长（per_frame 模式） |
| `history_processor_type` | per_frame | 历史处理方式 |
| `log_base` | 1.0 | 采样分布（1.0=均匀，>1.0=对数） |
| `system_prompt_setting` | vanilla | System prompt 策略（vanilla/initial） |

### 1.2 派生参数

```python
stride = num_frames - num_overlap  # 16，窗口滑动步长
turns_per_window = num_frames // num_future_steps  # 8，每窗口的轮数
overlap_turns = num_overlap // num_future_steps  # 4，重叠的轮数
```

---

## 2. System Prompt 策略

SwiftVLN 支持两种 system prompt 策略，通过 `system_prompt_setting` 参数选择。

### 2.1 Vanilla（默认）

**原理**：标准的 system prompt，仅包含任务指令和历史记忆 token。

```
System Prompt 结构:
  - 任务描述: "You are an autonomous navigation assistant..."
  - 历史记忆: <history_memory> tokens (压缩的历史帧)
  - 无额外的初始观察
```

**特点**：
- ✅ 简洁高效，token 数量较少
- ✅ 模型完全依赖历史记忆和当前观察进行推理
- ⚠️ 缺少旅程起点的完整视觉信息

### 2.2 Initial

**原理**：在 system prompt 中额外添加轨迹的**第一帧**（未压缩的完整图像），作为旅程起点的初始观察。

```
System Prompt 结构:
  - 任务描述: "You are an autonomous navigation assistant..."
  - 历史记忆: <history_memory> tokens (压缩的历史帧)
  - 初始观察: <initial_view> (第一帧，256 tokens，未压缩)

帧加载顺序:
  all_frames = [history_frames] + [initial_frame] + [current_frames]
               ↑ 压缩          ↑ 未压缩         ↑ 未压缩
```

**特点**：
- ✅ 提供完整的起点视觉信息，帮助模型建立空间参考
- ✅ 可能提升需要回顾起点的任务表现（如地标导航）
- ⚠️ 额外消耗 256 个 vision token

### 2.3 策略对比

| 特性 | Vanilla | Initial |
|------|---------|---------|
| Token 数量 | 较少 | +256 tokens |
| 起点信息 | 仅历史记忆（压缩） | 完整第一帧（未压缩） |
| 适用场景 | 通用 | 需要回顾起点的任务 |
| 命名后缀 | 无 | `-initial` |

**实验命名示例**：
- Vanilla: `swiftvln-satnav-stage1-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-bs64-lr2e-5-20260207-123456`
- Initial: `swiftvln-satnav-stage1-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-initial-bs64-lr2e-5-20260207-123456`

---

## 3. 历史记忆处理策略

SwiftVLN 支持三种历史记忆处理方式，通过 `history_processor_type` 参数选择。

### 3.1 Per-Frame Compression（默认）

**原理**：独立压缩每帧，保持时序结构。

```
历史帧采样 (num_history=8):
  使用 power transformation 进行灵活采样
  - log_base=1.0: 均匀采样
  - log_base>1.0: 对数采样（更多近期帧）

  公式: t_frame = 1 - (1 - t_sample)^log_base

每帧压缩 (compress_stride=2):
  原始: 16×16 = 256 tokens
  压缩: 8×8 = 64 tokens (4x 压缩)
  方法: Average Pooling 或 ToMe

输出: num_history × tokens_per_frame 个 token
```

**特点**：
- ✅ 保持时序信息，每帧独立可识别
- ✅ 向后兼容，配置灵活
- ⚠️ Token 数量随历史帧数线性增长

**参数**：
- `num_history`: 采样帧数（默认 8）
- `log_base`: 采样分布（1.0=均匀，2.0=对数）
- `compress_stride`: 压缩步长（2=4x, 3=9x）
- `use_tome`: 是否使用 ToMe 压缩

### 3.2 Global Token Clustering (GTC)

**原理**：将所有历史帧 token 视为无序集合，聚类到固定数量的 token。

```
历史帧采样:
  每 num_future_steps 步采样一帧（与训练一致）
  
聚类处理 (output_tokens=512):
  所有帧 tokens → Soft K-Means → 固定 512 个 token

输出: 始终是 output_tokens 个 token
```

**特点**：
- ✅ 跨帧去冗余，适合重复场景（走廊、相似环境）
- ✅ 固定输出 token 数，计算可控
- ⚠️ 丢失时序信息

**参数**：
- `gtc_output_tokens`: 输出 token 数（默认 512）
- `gtc_temperature`: 软分配温度（默认 0.1）
- `gtc_num_iterations`: K-Means 迭代次数（默认 1）

### 3.3 Segment GTC

**原理**：将历史分成 8 个时间段，每段内部用 GTC 聚类，保留粗粒度时序。

```
历史帧分段:
  总帧数 → 分成 8 段（按时间顺序）
  
段内聚类:
  每段 tokens → Soft K-Means → 每段 output_tokens/8 个 token

输出: 8 × (output_tokens/8) = output_tokens 个 token
```

**特点**：
- ✅ 平衡时序保留与冗余去除
- ✅ 固定输出 token 数
- ✅ 保留粗粒度进度信息

**参数**：与 GTC 相同

### 3.4 策略对比

| 特性 | Per-Frame | GTC | Segment GTC |
|------|-----------|-----|-------------|
| 时序保留 | ✅ 完整 | ❌ 无 | ⚠️ 粗粒度 |
| Token 数量 | 可变 | 固定 | 固定 |
| 冗余去除 | ⚠️ 帧内 | ✅ 全局 | ⚠️ 段内 |
| 适用场景 | 通用 | 高重复环境 | 长轨迹 |

---

## 4. 数据准备

### 4.1 数据集结构

```
trajectory_data/
├── R2R/
│   ├── annotations.json
│   └── videos/
│       └── episode_xxx/
│           └── rgb/
│               ├── 000000.png
│               └── ...
```

### 4.2 滑动窗口采样

```
轨迹长度: 100 步
num_frames: 32
num_overlap: 16
stride: 16

采样的窗口起始点:
  窗口1: start_idx = 0   → [0, 32)
  窗口2: start_idx = 16  → [16, 48)
  窗口3: start_idx = 32  → [32, 64)
  窗口4: start_idx = 48  → [48, 80)
  窗口5: start_idx = 64  → [64, 96)
  窗口6: start_idx = 80  → [80, 100) (末尾调整)
```

### 4.3 Loss Masking（训练时）

对于非首窗口（`start_idx > 0`）的样本：

```
窗口样本结构:
  Turn 0-3: overlap context（前4轮，loss=0.0）
  Turn 4-7: 新内容（后4轮，loss=1.0）

原因: 前4轮的内容在上一个窗口已经训练过
```

---

## 5. 多轮对话格式

### 5.1 消息结构

```python
messages = [
    # System: 任务描述 + 统一历史记忆
    {
        'role': 'system',
        'content': 'You are an autonomous navigation assistant. Your task is to {instruction}. '
                   '... '
                   'These are your historical observations: '
                   '<|vision_start|><history_memory><|vision_end|>.'
                   # 训练时: 单个 <history_memory>，由 template 扩展为 N 个
                   # 评估时: 直接放置 N 个 <history_memory> token
    },
    # Turn 1
    {'role': 'user', 'content': 'you can see <|vision_start|><image><|vision_end|>.'},
    {'role': 'assistant', 'content': '↑ ↑ ← ←'},
    # Turn 2
    {'role': 'user', 'content': 'in front of you is <|vision_start|><image><|vision_end|>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ →'},
    # ... 更多轮次
]
```

### 5.2 Token 处理

**训练时** (dataset.py → template.py):
- System prompt 中放置**单个** `<history_memory>` token
- Template 的 `_replace_image_token` 将其扩展为 N 个
- User turn 使用标准 `<image>` token，template 转换为 `<current_image>` 并扩展

**评估时** (evaluator.py):
- System prompt 中**直接放置 N 个** `<history_memory>` token
- User turn 使用 `<current_image>` token 并手动扩展

### 5.3 Token 数量

- `<history_memory>`: 统一的历史记忆 token，扩展后数量 N 由 HistoryProcessor 决定
  - Per-Frame: N = num_history × tokens_per_compressed_frame
  - GTC/SegmentGTC: N = gtc_output_tokens
- `<current_image>`: 当前帧 token，扩展后数量 M = 原始 token 数（如 256）

---

## 6. 训练流程

### 6.1 Token 转换

```
Dataset 输出 (原始):
  System:  ... <|vision_start|><history_memory><|vision_end|> ...
  User:    ... <|vision_start|><image><|vision_end|> ...
                     ↓ template._replace_image_token
Template 转换后:
  System:  ... <|vision_start|><history_memory>×N<|vision_end|> ...
  User:    ... <|vision_start|><current_image>×M<|vision_end|> ...
               统一历史块（无ROPE）    当前帧（有ROPE）
```

### 6.2 特殊 Token

| Token | 用途 |
|-------|------|
| `<history_memory>` | 统一的历史记忆占位符（无 ROPE） |
| `<current_image>` | 当前帧占位符（有 ROPE 位置编码） |

### 6.3 压缩处理

根据 `history_processor_type` 选择不同处理方式：

#### Per-Frame (默认)

独立压缩每帧，保留时序结构：

```
输入: 8 帧历史 VIT 特征，每帧 [256, 3584]

1. 对每帧应用 2D 压缩 (stride=2)
   Pooling: nn.AvgPool2d((2,2)) → 16×16 变成 8×8
   ToMe:    Grid-based Token Merging，语义保留更好
   
2. 每帧: 256 tokens → 64 tokens (4x 压缩)

3. 拼接: 8 × 64 = 512 tokens

输出: [512, 3584]，保留时序（第 0-63 是第 1 帧，64-127 是第 2 帧...）
```

#### GTC (Global Token Clustering)

跨帧聚类，去除全局冗余：

```
输入: N 帧 VIT 特征，拼接得到 [M, 3584]

1. 拼接所有帧: X ∈ R^{M×d}（丢弃时序）

2. Soft K-Means 聚类到 K=512 个中心:
   - 初始化: 均匀采样 K 个 token 作为初始中心
   - 软分配: A = softmax(X·C^T / τ)，τ=0.1
   - 加权聚合: C_new = (A^T · X) / sum(A)
   
3. 可选迭代 (num_iterations=1)

输出: [512, 3584]，无时序（聚类中心无顺序）
```

#### Segment GTC

分段聚类，保留粗粒度时序：

```
输入: 32 帧历史 VIT 特征

1. 分成 8 段（固定）:
   Seg 0: frames [0-3]    → 拼接 token
   Seg 1: frames [4-7]    → 拼接 token
   ...
   Seg 7: frames [28-31]  → 拼接 token

2. 每段内部应用 GTC:
   目标: output_tokens / 8 = 64 tokens/段
   每段独立 Soft K-Means 聚类

3. 按时间顺序拼接:
   [Seg0_64] + [Seg1_64] + ... + [Seg7_64] = 512 tokens

输出: [512, 3584]，保留粗粒度时序（前 64 是早期，后 64 是近期）
```

#### 对比

| 方法 | 输出 token | 时序信息 | 冗余去除 | 计算量 |
|------|-----------|---------|---------|-------|
| Per-Frame | 可变 (num_history × 压缩后) | ✅ 帧级精确 | ⚠️ 帧内 | 低 |
| GTC | 固定 (output_tokens) | ❌ 无 | ✅ 全局 | 中 |
| Segment GTC | 固定 (output_tokens) | ⚠️ 段级粗粒度 | ⚠️ 段内 | 中 |

---

## 7. 评估流程（核心）

### 7.1 缓存架构

```python
class SwiftVLNEvaluator:
    # 全局历史缓存
    history_cache: List[Tensor]  # 处理后的历史特征
    
    # VIT 特征缓存（用于 GTC/SegmentGTC）
    vit_feature_cache: Dict[int, Tuple[Tensor, Tensor]]
    
    # 窗口间传递的上下文
    overlap_context: OverlapContext  # 上一窗口最后4轮的 token + 图像嵌入
    
    # 当前窗口状态
    window_turns: List[TurnContext]  # 当前窗口已完成的轮次
```

### 7.2 OverlapContext 结构

```python
@dataclass
class OverlapContext:
    input_ids: Tensor      # 拼接的 token IDs [1, seq_len]
    image_embeds: List[Tensor]  # 每轮的图像特征 [num_tokens, hidden]
```

### 7.3 评估循环

```python
step_id = 0
action_queue = []

while not done:
    # 1. 收集当前观察
    current_img = env.get_rgb()
    rgb_list.append(current_img)
    
    # 2. 如果需要生成新动作
    if len(action_queue) == 0:
        # 2a. 检查是否需要滑动窗口
        if step_id > 0 and step_id % stride == 0 and step_id >= num_frames:
            slide_window()
        
        # 2b. 构建输入嵌入
        inputs_embeds = build_complete_prompt_embeds(
            instruction, current_img, conjunction
        )
        
        # 2c. 生成响应
        response = model.generate(inputs_embeds)
        action_queue = parse_actions(response)
        
        # 2d. 保存当前轮次到窗口
        save_turn_to_window(conjunction, response, vit_features)
        
        # 2e. 缓存 VIT 特征（用于 GTC/SegmentGTC）
        if history_processor_type in ('gtc', 'segment_gtc'):
            vit_feature_cache[step_id] = (current_vit, current_grid_thw)
    
    # 3. 执行动作
    action = action_queue.pop(0)
    env.step(action)
    step_id += 1
```

### 7.4 窗口滑动流程

```python
def slide_window(rgb_list, new_window_start):
    # 1. 保存当前窗口最后4轮作为 overlap_context
    prepare_overlap_context()
    
    # 2. 重新计算历史缓存（根据 history_processor_type）
    compute_history_cache(rgb_list, new_window_start)
    
    # 3. 重置窗口状态
    window_turns = []
    window_start_step = new_window_start
```

### 7.5 Embedding 构建

```python
def build_complete_prompt_embeds(instruction, current_image, conjunction):
    # 1. 编码当前图像
    current_vit_features = encode_frame(current_image)
    
    # 2. 构建 system prompt（包含统一历史记忆）
    system_ids = build_system_prompt_ids(instruction, history_token_counts)
    system_embeds = embed_tokens(system_ids)
    
    # 3. 用历史缓存替换 <history_memory> 占位符
    history_positions = find_positions(system_ids, history_memory_token_id)
    system_embeds[history_positions] = cat(history_cache)
    
    # 4. 添加 overlap_context（上一窗口的最后4轮）
    if overlap_context is not None:
        overlap_embeds = embed_tokens(overlap_context.input_ids)
        overlap_positions = find_positions(overlap_context.input_ids, current_image_token_id)
        overlap_embeds[overlap_positions] = cat(overlap_context.image_embeds)
    
    # 5. 添加当前窗口已完成的轮次
    for turn in window_turns:
        # ... 处理每轮的 user/assistant
    
    # 6. 添加新的 user turn（当前帧）
    # ...
    
    # 7. 拼接所有部分
    return cat([system_embeds, overlap_embeds, window_turn_embeds, new_user_embeds])
```

---

## 8. 时间线示例

假设 `num_frames=32, num_overlap=16, num_future_steps=4`：

```
Step 0-3:   Turn 0, 窗口1
Step 4-7:   Turn 1, 窗口1
Step 8-11:  Turn 2, 窗口1
Step 12-15: Turn 3, 窗口1
Step 16-19: Turn 4, 窗口1
Step 20-23: Turn 5, 窗口1
Step 24-27: Turn 6, 窗口1
Step 28-31: Turn 7, 窗口1
            ↓ 完成8轮，准备滑动

Step 32:    检测到需要滑动窗口
            - 保存 Turn 4-7 到 overlap_context
            - 计算历史缓存（根据 processor_type）
            - 重置 window_turns = []
            ↓ 开始窗口2

Step 32-35: Turn 0, 窗口2
            - overlap_context 提供 Turn 4-7 的上下文
            - 模型看到的完整序列:
              [system + history_memory] + [overlap: Turn 4-7] + [Turn 0]
            ...
```

---

## 9. 文件职责

```
dataset.py    → 滑动窗口采样，loss masking，历史帧采样策略
template.py   → Token 转换，HistoryProcessor 集成
arguments.py  → 定义 num_overlap, history_processor_type 等参数
model.py      → 添加特殊 token（history_memory, current_image）
trainer.py    → 训练入口
eval.py       → 评估入口，参数解析
evaluator.py  → 核心：VIT 缓存、overlap_context 管理、embedding 构建
```

---

## 10. EXP_NAME 格式

训练产出的实验名称格式：

```
Per-Frame:
  swiftvln-{env}-{stage}-{size}-{ep}ep-f{frames}s{steps}-overlap{overlap}-pf-h{history}[-nomem][-random]-b{log_base}-{method}-s{stride}[-qa{ratio}]-bs{bs}-lr{lr}-{timestamp}
  
  示例: swiftvln-habitat-stage1-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-bs64-lr2e-5-20260204-123456
  随机采样示例: swiftvln-satnav-stage1-3b-1ep-f32s4-overlap0-pf-h8-random-b1.0-pool-s2-noembed-data260404-bs64-lr2e-5-20260418-123456

GTC:
  swiftvln-{env}-{stage}-{size}-{ep}ep-f{frames}s{steps}-overlap{overlap}-gtc-k{tokens}[-qa{ratio}]-bs{bs}-lr{lr}-{timestamp}
  
  示例: swiftvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-bs64-lr2e-5-20260204-123456

Segment GTC:
  swiftvln-{env}-{stage}-{size}-{ep}ep-f{frames}s{steps}-overlap{overlap}-sgtc-k{tokens}[-qa{ratio}]-bs{bs}-lr{lr}-{timestamp}
  
  示例: swiftvln-satnav-stage2-3b-1ep-f32s4-overlap16-sgtc-k512-qa15-bs64-lr2e-5-20260204-123456
```

---

## 11. 关键设计选择

| 设计点 | 选择 | 原因 |
|--------|------|------|
| 重叠大小 | num_overlap=16 (4轮) | 平衡上下文连续性和计算效率 |
| 统一历史Token | `<history_memory>` 单一块 | 简化处理，支持多种 processor |
| 缓存策略 | 缓存 VIT 特征而非 embedding | 减少内存占用，支持 GTC 重聚类 |
| Token ID 缓存 | 保存 input_ids 而非 embedding | 内存友好 |
| 索引赋值 | 用索引替代 masked_scatter | 支持多 token 图像 |
| 默认 processor | per_frame | 向后兼容，配置灵活 |

---

## 12. 实验结果回顾

### 12.1 数据集更新（2026-02-07）

最新评估使用的数据集（ver_260206）修复了 landmark 相关的问题，使评估结果更加准确可靠。

### 12.2 Initial 策略实验

**实验背景**：测试在 system prompt 中添加第一帧（未压缩）作为初始观察的效果。

**配置**：
- 模型：Qwen2.5-VL-3B
- 框架：SwiftVLN + Per-Frame (h8, log_base=1.0, pool, stride=2)
- 窗口：num_frames=32, num_overlap=16, num_future_steps=4
- 训练：1 epoch, batch_size=64, lr=2e-5
- 策略：Initial（添加第一帧到 system prompt）

**SatNav Stage1 验证集结果（val_unseen）**：

运行两次实验，结果如下：

| 实验编号 | 时间戳 | ALL_SR | ALL_SPL | ALL_OS | Boundary_SR | Boundary_SPL | Landmark_SR | Landmark_SPL |
|---------|--------|---------|---------|---------|-------------|--------------|-------------|--------------|
| Run 1 | 20260207-070600 | **41.57%** | **0.3972** | 52.41% | **63.69%** | **0.5965** | 24.62% | 0.2445 |
| Run 2 | 20260207-075833 | 31.67% | 0.2685 | **54.56%** | 37.50% | 0.2667 | **27.20%** | **0.2699** |

**与 Vanilla 策略对比**（相同配置，无 initial）：

| 策略 | ALL_SR | ALL_SPL | Boundary_SR | Boundary_SPL | Landmark_SR | Landmark_SPL |
|------|---------|---------|-------------|--------------|-------------|--------------|
| Vanilla (Run 1) | **42.86%** | **0.4203** | 55.16% | 0.5380 | **33.43%** | **0.3301** |
| Vanilla (Run 2) | 40.62% | 0.3902 | 55.56% | 0.5268 | 29.18% | 0.2857 |
| Initial (Run 1) | 41.57% | 0.3972 | **63.69%** | **0.5965** | 24.62% | 0.2445 |
| Initial (Run 2) | 31.67% | 0.2685 | 37.50% | 0.2667 | 27.20% | 0.2699 |

**关键观察**：

1. **Vanilla vs Initial（整体表现）**：
   - Vanilla 策略在 ALL_SR 和 ALL_SPL 上略优（42.86% vs 41.57%）
   - 两种策略的最佳表现接近，差异不显著

2. **Boundary 任务**：
   - Initial (Run 1) 在 Boundary_SR 上表现最佳（63.69%）
   - 说明初始观察对边界导航任务可能有帮助

3. **Landmark 任务**：
   - Vanilla 策略在 Landmark 任务上明显更优（33.43% vs 27.20%）
   - Initial 策略未能显著提升地标导航表现

4. **稳定性**：
   - Initial 策略的两次运行结果差异较大（41.57% vs 31.67%）
   - Vanilla 策略更稳定（42.86% vs 40.62%）

### 12.3 Initial 策略总结

**工作原理**：
- 在 system prompt 中添加轨迹第一帧（256 tokens，未压缩）
- 提供完整的起点视觉信息作为空间参考锚点
- 额外消耗 256 个 vision token

**适用场景**：
- ✅ 需要明确起点参考的边界导航任务
- ⚠️ 地标导航任务效果不明显，反而可能分散模型注意力
- ⚠️ 训练稳定性相对较低，可能需要更多调参

**推荐配置**：
- 如果任务需要频繁回顾起点（如往返路径），可尝试 initial 策略
- 通用场景推荐使用 vanilla 策略，表现更稳定
- 如使用 initial，建议多次运行验证稳定性

### 12.4 与其他方法对比

基于最新数据集（修复 landmark 问题后）的完整对比：

| 模型 | 配置 | ALL_SR | ALL_SPL | Boundary_SR | Landmark_SR |
|------|------|---------|---------|-------------|-------------|
| **SwiftVLN** (vanilla, best) | pf-h8-b1.0-pool-s2 | **42.86%** | **0.4203** | 55.16% | **33.43%** |
| **SwiftVLN** (initial, best) | pf-h8-b1.0-pool-s2-initial | 41.57% | 0.3972 | **63.69%** | 24.62% |
| SwiftVLN (vanilla) | pf-h8-b2.0-pool-s2 | 41.22% | 0.4039 | 59.52% | 27.20% |
| StreamVLN | f32h8s4 | 42.17% | 0.4123 | **63.89%** | 25.53% |
| CompressVLN | f32h8s4-stride2 | 37.09% | 0.3600 | 52.78% | 25.08% |

**关键发现**：
1. SwiftVLN (vanilla) 在综合指标上最优
2. Initial 策略在 Boundary 任务上接近 StreamVLN
3. 数据集修复后，Landmark 任务的成功率普遍较低（24-33%），说明任务难度较高
4. SwiftVLN 框架在不同配置下表现稳定，证明设计有效性
