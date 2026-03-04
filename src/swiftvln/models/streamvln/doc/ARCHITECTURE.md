# StreamVLN 架构文档

## 1. 项目概述

StreamVLN 是基于 Qwen2.5-VL 的视觉语言导航模型，集成于 ms-swift 框架。模型使用固定窗口切分轨迹，支持多轮对话推理和 KV Cache 流式推理。

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│  历史帧采样(8帧) + 当前窗口帧 ───▶ Qwen2.5-VL Vision Encoder │
│         │                          (ViT + spatial_merge)    │
│         │                                  │                │
│         ▼                                  ▼                │
│  导航指令 ───▶ Tokenizer ───▶ Qwen2.5-VL LLM + KV Cache     │
│                                        │                    │
│                                        ▼                    │
│                   多轮对话: "↑←→↑" × N 轮                   │
└──────────────────────────────────────────────────────────────┘
```

## 3. 核心组件

### 3.1 模型 (`model.py`)

```python
StreamVLNQwen25VLForConditionalGeneration
        ↓ 继承
Qwen2_5_VLForConditionalGeneration (transformers)
        +
KV Cache 管理方法（流式推理）
```

- 直接继承 Qwen2.5-VL
- **关键差异**：实现 KV Cache 管理用于流式推理
  - `reset()`: 初始化多环境缓存
  - `reset_for_env()`: 单环境重置（每 32 帧）
  - `get_cache()` / `update_cache()`: 缓存管理
  - `get_step_count()`: 步数追踪

### 3.2 数据集 (`dataset.py`)

```python
StreamVLNDataset(Dataset)
├── data_path           # 数据目录（支持逗号分隔多路径）
├── num_frames          # 窗口大小（默认 32）
├── num_history         # 历史帧采样数（默认 8）
├── num_future_steps    # 每轮预测动作数（默认 4）
├── use_random          # 历史帧采样方式（True=随机, False=均匀）
└── max_samples         # 限制训练样本数量
```

### 3.3 训练器 (`trainer.py`)

```python
StreamVLNSft(SwiftSft)
├── _get_dataset()           # 检测 annotations.json 创建 StreamVLNDataset
├── _encode_dataset()        # 跳过 HuggingFace 预处理
└── _post_process_datasets() # LazyLLMDataset 包装
```

### 3.4 评估器 (`evaluator.py`)

```python
VLNEvaluator
├── 流式推理              # 使用 KV Cache 高效推理
├── 窗口管理              # 每 num_frames 步重置
├── 对话积累              # 窗口内多轮对话累积
└── Habitat 集成          # R2R/RxR 数据集评估
```

## 4. 数据格式

> **注**: StreamVLN 与 Navid 使用相同的底层数据格式

### 4.1 目录结构

```
dataset_root/
├── annotations.json
└── episode_xxx/
    └── rgb/
        ├── 000000.png    # 执行 actions[-1] 后的观察，用于预测 actions[0]
        ├── 000001.png    # 执行 actions[0] 后的观察，用于预测 actions[1]
        └── ...
```

### 4.2 annotations.json

```json
{
    "id": 1803,
    "video": "episode_xxx",
    "instructions": ["Go to the kitchen"],
    "actions": [-1, 1, 1, 2, 1, 0]
}
```

### 4.3 动作编码

| 索引 | 动作 | 符号 | 说明 |
|------|------|------|------|
| -1 | 初始占位符 | - | 起始状态 |
| 0 | STOP | STOP | 停止/等待 |
| 1 | MOVE_FORWARD | ↑ | 前进 25cm |
| 2 | TURN_LEFT | ← | 左转 15° |
| 3 | TURN_RIGHT | → | 右转 15° |

### 4.4 图像-动作对应关系

**核心规则**: 当前图像用于预测下一个动作

```
图像 {i:03d}.png = 执行 actions[i-1] 后的观察
图像 {i:03d}.png → 预测 actions[i]
```

**代码实现**:
```python
# 将动作序列向后偏移一位
actions = data['actions'][1:] + [0]  # 去掉 -1，末尾补 STOP
```

## 5. 数据切分策略

### 5.1 固定窗口切分

StreamVLN 将长轨迹切分为固定大小的窗口：

```
轨迹: [帧0, 帧1, ..., 帧95]  (96帧)
                ↓
        切分为 32 帧窗口
                ↓
窗口0: [帧0-31]   → 样本0
窗口1: [帧32-63]  → 样本1
窗口2: [帧64-95]  → 样本2
```

### 5.2 窗口内帧采样

每个窗口按 `num_future_steps` 间隔采样图像：

```
32 帧窗口，num_future_steps = 4
        ↓
采样: [帧0, 帧4, 帧8, 帧12, 帧16, 帧20, 帧24, 帧28]
        ↓
8 张图像 → 8 轮对话
```

### 5.3 历史帧采样

非首窗口会采样历史帧作为上下文：

```
窗口1 (帧32-63) 需要历史帧
        ↓
从 [帧0-31] 中采样 8 帧
        ↓
均匀采样: [帧0, 帧4, 帧8, 帧12, 帧16, 帧20, 帧24, 帧28]
随机采样: [帧2, 帧7, 帧11, 帧15, 帧19, 帧22, 帧27, 帧30]
```

## 6. 对话 Prompt 格式

### 6.1 多轮对话结构

```
SYSTEM: You are an autonomous navigation assistant. Your task is to {instruction}. 
        Devise an action sequence to follow the instruction using the four actions: 
        TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, 
        MOVE FORWARD (↑) by 25 centimeters, or STOP.
        These are your historical observations: <image> <image> ... <image>.

USER:   you can see <image>.
ASSISTANT: ↑↑←↑

USER:   in front of you is <image>.
ASSISTANT: ↑→↑↑

USER:   there is <image>.
ASSISTANT: ←↑↑↑

...（共 8 轮）
```

### 6.2 样本构建逻辑

```
__getitem__(i):
  1. 解析索引 → (ep_id, ins_id, start_frame)
  2. 计算窗口范围 [start_frame, start_frame + num_frames)
  3. 采样历史帧（如果 start_frame > 0）
  4. 按 num_future_steps 间隔采样当前窗口帧
  5. 构建多轮对话
     - System: 指令 + 历史帧占位符
     - 每轮: User (图像) + Assistant (4动作)
  6. 返回 {messages, images}
```

## 7. StreamVLN vs Navid 对比

| 维度 | StreamVLN | Navid |
|------|-----------|-------|
| **数据格式** | annotations.json + rgb/ | **相同** |
| **动作偏移** | `actions[1:] + [0]` | **相同** |
| **切分方式** | 固定 32 帧窗口 | 按 step 生成样本 |
| **历史帧** | 采样 8 帧，无 memory token | 全部使用或均匀采样 |
| **对话结构** | 多轮（每轮 4 动作） | 单轮（一次 4 动作） |
| **样本数量** | `轨迹长度 // 32` | `轨迹长度 // sample_interval` |
| **KV Cache** | ✅ 支持流式推理 | ❌ 无 |
| **推理效率** | 高（缓存复用） | 中（每次完整推理） |

## 8. 流式推理机制

### 8.1 KV Cache 管理

```python
# 初始化
model.reset(env_num=8)  # 8 个并行环境

# 每个 episode 开始
model.reset_for_env(env_idx)

# 推理时
cache = model.get_cache(env_idx)
output = model.generate(..., past_key_values=cache)
model.update_cache(env_idx, output.past_key_values)

# 每 32 步重置
if model.get_step_count(env_idx) >= num_frames:
    model.reset_for_env(env_idx)
```

### 8.2 窗口内推理流程

```
窗口开始 → reset_for_env()
    │
    ├─ 第1帧: 完整 prompt + System + User
    │         生成 4 动作，缓存 KV
    │
    ├─ 第2帧: 增量 User turn
    │         复用 KV Cache，生成 4 动作
    │
    ├─ ...（共 8 轮）
    │
窗口结束 → 重置缓存，采样新历史帧
```

## 9. 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_frames` | 32 | 窗口大小（帧数） |
| `num_history` | 8 | 历史帧采样数 |
| `num_future_steps` | 4 | 每轮预测的动作数 |
| `use_random` | False | 历史帧采样方式（True=随机, False=均匀） |
| `max_samples` | None | 限制训练样本总数 |

## 10. 使用示例

### 10.1 训练

```bash
python src/swiftvln/models/streamvln/trainer.py \
    --custom_register_path src/swiftvln/models/streamvln \
    --model_type streamvln_qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset /path/to/vln_data \
    --num_frames 32 \
    --num_history 8 \
    --num_future_steps 4 \
    --use_random false
```

### 10.2 评估

```bash
python src/swiftvln/models/streamvln/eval.py \
    --model /path/to/checkpoint \
    --eval_split val_unseen \
    --num_frames 32 \
    --num_history 8 \
    --save_video true
```

## 11. Token 数量估算

图像 Token 数 = `(H/14) × (W/14) / spatial_merge_size²`

| 图像尺寸 | spatial_merge=2 | spatial_merge=4 |
|----------|-----------------|-----------------|
| 644×476 | 391 tokens | 97 tokens |
| 1344×1344 | 2304 tokens | 576 tokens |

### 单窗口 Token 估算（默认配置）

- 历史帧: 8 张 × 391 tokens = 3,128 tokens
- 当前帧: 8 张 × 391 tokens = 3,128 tokens
- 文本: ~500 tokens
- **总计**: ~6,756 tokens / 窗口

**注**: spatial_merge_size 可通过环境变量 `SPATIAL_MERGE_SIZE` 调整。
