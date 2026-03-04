# OverlapVLN 样本格式说明

本文档描述 OverlapVLN 在训练和推理时如何构建样本，重点说明滑动窗口和 loss masking 机制。

---

## 1. 滑动窗口采样

### 1.1 窗口参数

```
num_frames: 32       # 窗口大小（动作数）
num_overlap: 16      # 窗口重叠（动作数）
stride: 16           # 滑动步长 = num_frames - num_overlap
num_future_steps: 4  # 每轮预测的动作数
```

### 1.2 采样示例

```
轨迹长度: 80 步
actions: [a0, a1, a2, ..., a79]

滑动窗口采样:
  窗口1: start_idx=0,  范围=[0, 32),  训练 Turn 0-7
  窗口2: start_idx=16, 范围=[16, 48), 训练 Turn 0-7 (但 Turn 0-3 masked)
  窗口3: start_idx=32, 范围=[32, 64), 训练 Turn 0-7 (但 Turn 0-3 masked)
  窗口4: start_idx=48, 范围=[48, 80), 训练 Turn 0-7 (但 Turn 0-3 masked)
```

### 1.3 重叠区域

```
窗口1: [████████████████████████████████]
       step 0                       step 31

窗口2:                 [████████████████████████████████]
                       step 16                      step 47
                       └──── 重叠区域 ────┘
                       step 16-31 (Turn 4-7 of 窗口1)
```

---

## 2. Loss Masking 机制

### 2.1 为什么需要 Masking

重叠区域在连续窗口中出现两次：
- 窗口 N 中作为新内容训练
- 窗口 N+1 中作为上下文（不应重复训练）

### 2.2 Masking 规则

```python
if start_idx > 0:  # 非首窗口
    # 前 overlap_turns 轮的 assistant response 设置 loss=0.0
    overlap_turns = num_overlap // num_future_steps  # 4
    for i in range(overlap_turns):
        messages[assistant_turn_index].update({'loss': 0.0})
```

### 2.3 示例

```python
# 窗口2 (start_idx=16) 的消息结构
messages = [
    {'role': 'system', 'content': '... <history_image>×8 ...'},
    
    # Turn 0-3: 来自窗口1的重叠内容，loss=0.0
    {'role': 'user', 'content': 'you can see <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ← ←', 'loss': 0.0},  # ← masked
    
    {'role': 'user', 'content': 'in front of you is <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ →', 'loss': 0.0},  # ← masked
    
    {'role': 'user', 'content': 'there is <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ ↑', 'loss': 0.0},  # ← masked
    
    {'role': 'user', 'content': 'ahead of you is <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ ↑', 'loss': 0.0},  # ← masked
    
    # Turn 4-7: 新内容，正常训练
    {'role': 'user', 'content': 'you can spot <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ← ←'},  # ← 正常训练
    
    {'role': 'user', 'content': 'you are toward the <current_image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ →'},  # ← 正常训练
    
    # ... Turn 6-7
]
```

---

## 3. 训练样本格式

### 3.1 Dataset 输出

```python
{
    'messages': [
        {'role': 'system', 'content': '... <image>×8 ...'},
        {'role': 'user', 'content': 'you can see <image>.'},
        {'role': 'assistant', 'content': '↑ ↑ ← ←', 'loss': 0.0},  # masked
        # ... 更多轮次
    ],
    'images': [PIL.Image, ...],  # 历史帧在前，当前帧在后
    'num_history_images': 8,     # 告诉 Template 前 8 张是历史帧
}
```

### 3.2 Template 处理后

```
Token 序列:
[system_tokens] [history_image×64]×8 [user_tokens] [current_image×256] ...

其中:
- <history_image>×64: 每张历史帧压缩后的 token 数
- <current_image>×256: 当前帧原始 token 数
```

---

## 4. 推理时的 Embedding 构建

### 4.1 完整 Prompt 结构

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. System Prompt                                                 │
│    - 任务描述                                                    │
│    - 历史帧: <history_image>×64 × num_history                    │
├─────────────────────────────────────────────────────────────────┤
│ 2. Overlap Context (如果有)                                      │
│    - 上一窗口最后 4 轮的完整对话                                  │
│    - 每轮: user + <current_image>×256 + assistant response       │
├─────────────────────────────────────────────────────────────────┤
│ 3. Window Turns (当前窗口已完成的轮次)                            │
│    - 当前窗口中已生成的轮次                                       │
│    - 每轮: user + <current_image>×256 + assistant response       │
├─────────────────────────────────────────────────────────────────┤
│ 4. New User Turn (当前帧)                                        │
│    - user + <current_image>×256 + generation_prompt              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Token 数量估算

```
假设:
- num_history = 8, 每张压缩后 64 tokens
- overlap_turns = 4, 每张当前帧 256 tokens
- window_turns = 3 (已完成3轮)
- 每轮文本约 50 tokens

Token 数估算:
  System:         ~100 + 8×64 = 612 tokens
  Overlap:        4 × (50 + 256 + 50) = 1424 tokens
  Window Turns:   3 × (50 + 256 + 50) = 1068 tokens
  New User:       50 + 256 + 10 = 316 tokens
  ─────────────────────────────────────────
  总计:           ~3420 tokens
```

---

## 5. 缓存数据结构

### 5.1 OverlapContext

```python
@dataclass
class OverlapContext:
    # 上一窗口最后 overlap_turns 轮的拼接 token IDs
    input_ids: Tensor  # [1, seq_len], 包含 user + assistant 的所有 token
    
    # 对应的图像嵌入（已通过 VIT 编码）
    image_embeds: List[Tensor]  # 每个元素 [num_tokens, hidden_size]
```

### 5.2 TurnContext

```python
@dataclass
class TurnContext:
    # user turn 的 token IDs (不含 generation prompt)
    user_input_ids: Tensor  # [1, seq_len]
    
    # assistant 的响应文本
    assistant_response: str  # 如 "↑ ↑ ← ←"
    
    # 该轮图像的 VIT 特征
    image_embed: Tensor  # [num_tokens, hidden_size]
```

---

## 6. 图像 Token 替换

### 6.1 索引赋值方式

```python
# 找到所有 <current_image> token 的位置
positions = (input_ids[0] == current_image_token_id).nonzero(as_tuple=True)[0]

# 将 VIT 特征按顺序填入
# image_embed: [256, 2048]
# positions: [pos0, pos1, ..., pos255]
inputs_embeds[0, positions[:256]] = image_embed
```

### 6.2 多图像替换

当有多个图像时（如 overlap_context 中有 4 张）：

```python
# 拼接所有图像特征
all_embeds = torch.cat([img1, img2, img3, img4], dim=0)  # [1024, 2048]

# 找到所有 token 位置
positions = find_all_positions(input_ids, current_image_token_id)  # 1024 个位置

# 按顺序填入
inputs_embeds[0, positions[:1024]] = all_embeds
```

---

## 7. 动作格式

与 CompressVLN 相同：

| 动作 | 符号 | 含义 |
|------|------|------|
| FORWARD | ↑ | 前进 0.25 米 |
| LEFT | ← | 左转 15 度 |
| RIGHT | → | 右转 15 度 |
| STOP | STOP | 停止 |

---

## 8. 训练 vs 推理对比

| 维度 | 训练 | 推理 |
|------|------|------|
| 窗口采样 | 预先滑动窗口采样 | 实时滑动 |
| Overlap Context | 隐式（数据重叠） | 显式缓存传递 |
| Loss Masking | 重叠轮次 loss=0.0 | 不适用 |
| VIT 特征 | 每次编码 / 预计算 | 缓存复用 |
| 图像来源 | 预先加载 | 实时观察 |

---

## 9. 与 CompressVLN 格式的区别

| 维度 | CompressVLN | OverlapVLN |
|------|-------------|------------|
| 窗口重叠 | 无 | num_overlap=16 |
| Loss Masking | 无 | 前 overlap_turns 轮 |
| 推理缓存 | 仅 history | history + overlap_context + window_turns |
| 上下文连续性 | 窗口间断裂 | 窗口间平滑过渡 |
