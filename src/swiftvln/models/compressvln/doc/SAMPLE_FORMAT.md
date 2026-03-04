# CompressVLN 样本格式说明

本文档描述 CompressVLN 在训练和推理时如何构建样本，以及多轮对话的具体格式。

---

## 1. 训练样本构建流程

### 1.1 采样索引

每个训练样本由三元组 `(episode_id, instruction_id, start_idx)` 确定：
- `episode_id`: 哪个导航轨迹
- `instruction_id`: 使用哪条指令（同一轨迹可能有多条指令）
- `start_idx`: 从轨迹的第几步开始

### 1.2 帧采样

```
轨迹总帧数: 100 帧
start_idx: 32
num_frames: 32  (窗口大小)
num_history: 8  (历史帧数)
num_future_steps: 4  (每轮预测的动作数)

历史帧范围: [0, 32) → 均匀采样 8 张
当前帧范围: [32, 64) → 每隔 4 步采样 → [32, 36, 40, 44, 48, 52, 56, 60]
```

### 1.3 图片顺序

```python
images = [历史帧0, 历史帧1, ..., 历史帧7, 当前帧0, 当前帧1, ..., 当前帧7]
#         └─────── 8 张 ───────┘  └─────────── 8 张 ──────────┘
```

---

## 2. 多轮对话格式

### 2.1 训练时的对话结构

```python
messages = [
    # System: 任务描述 + 历史帧
    {
        'role': 'system',
        'content': 'You are an autonomous navigation assistant. Your task is to walk to the kitchen. '
                   'Based on your observations, output a sequence of actions using: '
                   '↑ (forward), ← (turn left), → (turn right), or STOP (when goal is reached). '
                   'Output actions directly without explanation. '
                   'These are your historical observations: <image> <image> <image> <image> <image> <image> <image> <image>.'
    },
    # Turn 1: 当前帧0 → 动作序列
    {'role': 'user', 'content': 'you can see <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ← ←'},
    # Turn 2: 当前帧1 → 动作序列
    {'role': 'user', 'content': 'in front of you is <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ →'},
    # Turn 3: 当前帧2 → 动作序列
    {'role': 'user', 'content': 'there is <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ ↑'},
    # ... 更多轮次
    # 最后一轮
    {'role': 'user', 'content': 'ahead of you is <image>.'},
    {'role': 'assistant', 'content': '↑ STOP'},
]
```

### 2.2 关键元数据

```python
{
    'messages': messages,
    'images': [PIL.Image, ...],  # 历史帧在前，当前帧在后
    'num_history_images': 8,     # 告诉 Template 前 8 张是历史帧
}
```

Template 根据 `num_history_images` 区分：
- 前 8 张 `<image>` → 转换为 `<history_image>` → 压缩处理
- 后续 `<image>` → 转换为 `<current_image>` → 标准处理

---

## 3. 推理时的对话构建

### 3.1 窗口管理

推理时同样使用窗口机制，每 `num_frames` 步重置窗口：

```
步骤 0-31:  窗口1，无历史
步骤 32-63: 窗口2，历史来自 [0, 32)
步骤 64-95: 窗口3，历史来自 [0, 64)
```

### 3.2 推理时的对话构建

```python
# 假设当前在步骤 45，窗口起始于步骤 32

# 1. 采样历史帧 (窗口开始前的观察)
history_images = sample_from([0, 32), num_history=8)  # 8 张

# 2. 收集当前窗口的观察
#    步骤 32, 36, 40, 44 的观察（每 num_future_steps=4 一轮）
window_images = [rgb_32, rgb_36, rgb_40, rgb_44]  # 4 张

# 3. 构建消息
messages = [
    {'role': 'system', 'content': '... <image>×8 ...'},  # 8 个历史帧 token
    
    # 历史轮次（已执行）
    {'role': 'user', 'content': 'you can see <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ← ←'},  # 步骤 32-35 的动作
    
    {'role': 'user', 'content': 'in front of you is <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ →'},  # 步骤 36-39 的动作
    
    {'role': 'user', 'content': 'there is <image>.'},
    {'role': 'assistant', 'content': '↑ ↑ ↑ ↑'},  # 步骤 40-43 的动作
    
    # 当前轮次（待生成）
    {'role': 'user', 'content': 'ahead of you is <image>.'},
    # 模型生成: '↑ ↑ ← STOP'
]

# 4. 合并图片
images = history_images + window_images  # 8 + 4 = 12 张
num_history_images = 8
```

### 3.3 编码时传递历史帧数量

```python
# 关键：传递 num_history_images 触发压缩
encoded = template.encode({
    'messages': messages,
    'images': images,
    'num_history_images': num_history_images,  # 必须传递！
})
```

---

## 4. 动作格式

### 4.1 动作符号

| 动作 | 符号 | 含义 |
|------|------|------|
| FORWARD | ↑ | 前进 0.25 米 |
| LEFT | ← | 左转 15 度 |
| RIGHT | → | 右转 15 度 |
| STOP | STOP | 停止，到达目标 |

### 4.2 动作序列示例

```
↑ ↑ ← ←     # 前进2次，左转2次
↑ ↑ ↑ →     # 前进3次，右转1次
↑ STOP      # 前进1次，停止
```

---

## 5. 数据格式示例

### 5.1 annotations.json

```json
[
    {
        "video": "videos/episode_001",
        "instructions": [
            "Walk to the kitchen and stop near the refrigerator.",
            "Go to the kitchen area."
        ],
        "actions": [1, 1, 2, 1, 1, 3, 1, 0]
    }
]
```

动作编码：`0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT`

### 5.2 视频帧目录

```
videos/episode_001/
└── rgb/
    ├── 000000.png
    ├── 000001.png
    ├── 000002.png
    └── ...
```

### 5.3 预计算特征（可选）

```
features/
└── episode_001.pt
```

每个 `.pt` 文件包含：
```python
{
    'vit_features': [tensor, tensor, ...],  # 每帧的 ViT 特征
    'grid_thw': [tensor, tensor, ...],      # 每帧的网格尺寸
}
```

---

## 6. 训练 vs 推理对比

| 维度 | 训练 | 推理 |
|------|------|------|
| 历史帧来源 | 预先采样 | 实时收集 |
| 当前帧来源 | 预先采样 | 实时观察 |
| 动作来源 | Ground Truth | 模型生成 |
| 多轮对话 | 全部预构建 | 逐步累积 |
| 窗口重置 | 数据集划分 | 步数触发 |

---

## 7. 与 UniNaVid 的区别

| 维度 | CompressVLN | UniNaVid |
|------|-------------|----------|
| 对话格式 | 多轮对话 | 单轮对话 |
| 历史分类 | 历史/当前 两类 | 长期/短期/当前 三类 |
| 压缩方式 | 历史帧 pooling | 短期 pooling + 长期相似度合并 |
| 用户输入 | 每轮一张当前图 | 所有历史+当前一起 |
