# CompressVLN 技术文档

## 概述

CompressVLN 是基于 StreamVLN 的改进版本，核心改进是对**历史帧的视觉 token 进行压缩**，从而减少计算开销，同时保持当前帧的完整分辨率。

## 与 StreamVLN 的对比

| 特性 | StreamVLN | CompressVLN |
|------|-----------|-------------|
| 基础模型 | Qwen2.5-VL | Qwen2.5-VL |
| 历史帧处理 | 保持原始分辨率 | 2D 平均池化压缩 |
| 当前帧处理 | 保持原始分辨率 | 保持原始分辨率 |
| Dataset 图像 token | 统一 `<image>` | 统一 `<image>`（Template 转换） |
| Template 图像 token | 统一 `<\|image_pad\|>` | 区分 `<history_image>` / `<current_image>` |
| padding_free | 支持 | 不支持（需设为 false） |
| batch_size | 任意 | 任意（已支持按样本处理） |
| 训练速度 | 基准 | 更快（历史帧 token 数减少） |
| 显存占用 | 基准 | 更低 |

## 核心设计

### 1. 差异化图像 Token

CompressVLN 使用两种特殊 token 来区分图像类型：

- **`<history_image>`**：用于历史观察帧，会被压缩
- **`<current_image>`**：用于当前观察帧，保持原始分辨率

**实现方式**：
- Dataset 使用标准 `<image>` token（与 ms-swift 兼容）
- Dataset 返回 `num_history_images` 元数据
- Template 的 `replace_tag` 根据图片索引将 `<image>` 转换为对应的特殊 token

### 2. 压缩策略

采用 **2D 平均池化** 进行压缩：

- 压缩步长 `compress_stride=2` 时，每张历史图像的 token 数减少为原来的 1/4
- 压缩步长 `compress_stride=3` 时，token 数减少为原来的 1/9
- 压缩在空间维度上进行，保留了图像的整体语义信息

**Token 数量计算**：

```
原始图片 (640×480)
    ↓
Qwen2.5-VL 分块: grid_thw = [1, 28, 28]
    ↓
Spatial Merge (merge_size=2): [1, 14, 14] = 196 tokens
    ↓
CompressVLN 池化 (stride=2): [1, 7, 7] = 49 tokens

压缩率: 196 → 49 = 4倍压缩
```

### 3. 压缩位置

压缩发生在 **视觉编码器之后、语言模型之前**：

```
图像 → 视觉编码器 → 图像嵌入 → [历史帧压缩] → 与文本嵌入融合 → 语言模型
```

## 详细数据流程

### 阶段 1: Dataset 构建

**文件**: `dataset.py`

```
CompressVLNDataset.__getitem__(i):
  ├── 加载图片: [历史帧 H1, H2, ...] + [当前帧 C1, C2, ...]
  ├── 构建 messages:
  │     - system: "... <image> <image> <image> ..."  (历史帧)
  │     - user: "... <image> ..."  (当前帧)
  │     - assistant: "↑ ← → ..."
  └── 返回:
        {
          'messages': [...],
          'images': [H1, H2, ..., C1, C2, ...],
          'num_history_images': N  # 关键元数据
        }
```

### 阶段 2: Template.replace_tag()

**文件**: `template.py` 第 107-137 行

将标准 `<image>` token 转换为差异化 token：

```python
def replace_tag(self, media_type, index, inputs):
    num_history = inputs.extra_kwargs.get('num_history_images', 0)
    
    if index < num_history:
        return ['<|vision_start|><history_image><|vision_end|>']
    else:
        return ['<|vision_start|><current_image><|vision_end|>']
```

### 阶段 3: Template._encode()

**文件**: `template.py` 第 139-272 行

计算并扩展占位符 token：

```
输入 input_ids: [..., <history_image>, ..., <current_image>, ...]
                       ↓                      ↓
历史图片 token 数量:    49 (压缩后)            -
当前图片 token 数量:    -                     196 (标准)
                       ↓                      ↓
扩展后 input_ids: [..., <history_image>×49, ..., <current_image>×196, ...]
```

**关键函数**: `_extend_tokens()` - ms-swift 提供的工具函数

### 阶段 4: Template._post_encode()

**文件**: `template.py` 第 311-493 行

执行实际的压缩和 embedding 替换：

```
1. 获取文本 embedding: inputs_embeds = embed_tokens(input_ids)
2. 获取图像 embedding: all_image_embeds = model.visual(pixel_values)
3. 按样本处理（支持 batch_size > 1）:
   for each sample:
     for each history image:
       pooled = avg_pool2d(img_embeds)  # 2D 池化压缩
       history_embeds_list.append(pooled)
     for each current image:
       current_embeds_list.append(img_embeds)  # 不压缩
4. 替换占位符:
   inputs_embeds.masked_scatter(history_mask, history_embeds)
   inputs_embeds.masked_scatter(current_mask, current_embeds)
```

### 阶段 5: 模型前向传播

模型接收融合后的 `inputs_embeds`，进行标准的语言模型训练。

## Batch 处理机制

### 问题背景

当 `batch_size > 1` 时，图片在 `image_grid_thw` 中的顺序是：

```
[样本1所有图片, 样本2所有图片, ...]
```

而不是：

```
[所有历史图片, 所有当前图片]  # 错误假设
```

### 解决方案

**文件**: `template.py` 第 332-370 行

按样本顺序处理，而不是全局处理：

```python
# 获取每个样本的历史/当前图片数量
history_counts = [3, 2, 0]  # 样本1有3张历史, 样本2有2张, 样本3有0张
current_counts = [2, 1, 3]  # 样本1有2张当前, 样本2有1张, 样本3有3张

# 按样本处理
for sample_idx in range(num_samples):
    # 先处理该样本的历史图片（压缩）
    for h_idx in range(history_counts[sample_idx]):
        compress(img_idx)
        img_idx += 1
    # 再处理该样本的当前图片（不压缩）
    for c_idx in range(current_counts[sample_idx]):
        keep(img_idx)
        img_idx += 1
```

### 示例

```
Batch 2 个样本:
- 样本1: 3 历史 + 2 当前
- 样本2: 2 历史 + 1 当前

image_grid_thw 顺序: [S1_H, S1_H, S1_H, S1_C, S1_C, S2_H, S2_H, S2_C]
                      ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
处理方式:            压缩  压缩  压缩  保持  保持  压缩  压缩  保持
```

## 文件结构

```
compressvln/
├── __init__.py       # 模块注册（model + template）
├── dataset.py        # 数据集（返回 num_history_images 元数据）
├── model.py          # 模型定义（添加特殊 token 到 tokenizer）
├── template.py       # 模板（压缩逻辑核心，521 行）
├── compressor.py     # 2D 平均池化实现（未在 template 中直接使用）
├── arguments.py      # 训练参数（含 compress_stride）
├── trainer.py        # 训练入口
├── eval.py           # 评估入口
└── script/
    ├── train/        # 训练脚本
    └── eval/         # 评估脚本
```

### 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| Token 转换 | `template.py` | 107-137 |
| Token 扩展 | `template.py` | 193-231 |
| Batch 处理 | `template.py` | 332-370 |
| 2D 池化 | `template.py` | 274-309 |
| Embedding 替换 | `template.py` | 479-491 |

## 使用限制

### padding_free 必须设为 false

CompressVLN 使用自定义的 `<history_image>` 和 `<current_image>` token，这与 Qwen2.5-VL 的 `get_rope_index` 函数不兼容。因此：

- 训练脚本中需设置 `PADDING_FREE=false`
- 这会略微增加训练开销（约 10-30%），但保证了压缩功能的正常工作

### 替代优化

虽然不能使用 padding_free，但 CompressVLN 通过历史帧压缩节省的计算量可以部分弥补：

```
8 张历史图片 × 196 tokens = 1568 tokens
         ↓ 压缩后 (stride=2)
8 张历史图片 × 49 tokens = 392 tokens

每样本节省: 1176 tokens
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `compress_stride` | 2 | 池化步长，2 表示 4 倍压缩 |
| `num_history` | 8 | 历史帧采样数量 |
| `num_frames` | 32 | 窗口大小 |
| `num_future_steps` | 4 | 每轮预测的动作数 |

## 训练与评估

### 训练

```bash
bash src/swiftvln/models/compressvln/script/train/train_compressvln_qwen2_5_vl.sh
```

### 评估

```bash
MODEL_PATH=/path/to/checkpoint \
ENV_TYPE=habitat \
bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh
```

### Debug 配置

在 `.vscode/launch.json` 中已配置 `CompressVLN: Train (Single GPU Debug)`，可直接 F5 启动调试。

建议断点位置：
1. `dataset.py:180` - 验证 `num_history_images` 返回值
2. `template.py:127` - 验证 token 转换逻辑
3. `template.py:321` - 验证 `_post_encode` 输入
4. `template.py:430` - 验证压缩后的 token 数量

## 设计考量

### 为什么只压缩历史帧？

1. **当前帧最重要**：导航决策主要依赖当前观察
2. **历史帧提供上下文**：压缩后仍能提供轨迹记忆
3. **计算效率**：历史帧数量多，压缩收益大

### 为什么在 Template 层区分 token？

1. **ms-swift 兼容性**：Dataset 使用标准 `<image>` token，与 ms-swift 无缝集成
2. **灵活性**：通过 `num_history_images` 元数据传递压缩信息
3. **复用性**：可复用 StreamVLN 的大部分 Dataset 逻辑

### 为什么推理时也压缩？

1. **保持一致性**：训练和推理行为一致
2. **降低推理成本**：长轨迹导航时历史帧累积，压缩可显著减少计算量
