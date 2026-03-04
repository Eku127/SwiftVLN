# MonoVLN 技术文档

## 概述

MonoVLN 是一个基于**单轮对话**的视觉语言导航系统。与 StreamVLN 的多轮对话不同，MonoVLN 将指令、历史观察和当前观察整合到单个用户消息中，模型直接输出动作序列。同时，MonoVLN 对**历史帧的视觉 token 进行压缩**，减少计算开销。

## 与 StreamVLN 的对比

| 特性 | StreamVLN | MonoVLN |
|------|-----------|---------|
| 基础模型 | Qwen2.5-VL | Qwen2.5-VL |
| 对话格式 | 多轮对话 | 单轮对话 |
| 历史帧位置 | System prompt | User message |
| 当前帧数量 | 窗口内多帧 | 固定 1 帧 |
| 历史帧处理 | 保持原始分辨率 | 2D 平均池化压缩 |
| 当前帧处理 | 保持原始分辨率 | 保持原始分辨率 |
| KV Cache | 需要管理 | 不需要 |
| Dataset 继承 | StreamVLNDataset | torch.utils.data.Dataset |
| Episode 采样 | 滑动窗口 | 均匀采样固定数量 |

## 核心设计

### 1. 单轮对话格式

MonoVLN 每个训练/推理样本是一个简单的 user-assistant 对话：

```
User: "Task: {instruction}. Historical observations: <image> <image> .... Current observation: <image>. Output actions: ↑/←/→/STOP."
Assistant: "↑ ↑ ← → STOP"
```

**优势**：
- 简化推理流程，无需管理多轮对话状态
- 不需要 KV cache，每步独立调用
- 代码更简洁，易于理解和维护

### 2. 差异化图像 Token

MonoVLN 使用两种特殊 token 来区分图像类型：

- **`<history_image>`**：用于历史观察帧，会被压缩
- **`<current_image>`**：用于当前观察帧，保持原始分辨率

**实现方式**：
- Dataset 使用标准 `<image>` token（与 ms-swift 兼容）
- Dataset 返回 `num_history_images` 元数据
- Template 的 `replace_tag` 根据图片索引将 `<image>` 转换为对应的特殊 token

### 3. Episode 均匀采样

每个 episode 均匀采样 `samples_per_episode` 个训练样本：

```python
# samples_per_episode = 8, episode 有 100 个 action, num_future_steps = 4
# 
# 可采样范围: [0, 96]  (96 = 100 - 4)
# 采样点: [0, 13, 27, 41, 54, 68, 82, 96]
#
# 保证:
# - 第 0 个样本：无历史帧
# - 第 96 个样本：预测的动作包含 STOP
```

### 4. 压缩策略

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
MonoVLN 池化 (stride=2): [1, 7, 7] = 49 tokens

压缩率: 196 → 49 = 4倍压缩
```

## 详细数据流程

### 阶段 1: Dataset 构建

**文件**: `dataset.py`

```
MonoVLNDataset.__init__():
  ├── 加载导航数据
  ├── 对每个 episode 均匀采样 samples_per_episode 个时间点
  └── 构建 data_list: [(ep_id, ins_id, current_idx), ...]

MonoVLNDataset.__getitem__(i):
  ├── 获取 (episode_id, instruction_id, current_idx)
  ├── 采样历史帧: 从 [0, current_idx) 均匀采样 num_history 帧
  ├── 获取当前帧: video_frames[current_idx] (1 帧)
  ├── 获取动作序列: actions[current_idx : current_idx + num_future_steps]
  └── 返回:
        {
          'messages': [
              {'role': 'user', 'content': 'Task: ... Historical: <image>... Current: <image>...'},
              {'role': 'assistant', 'content': '↑ ↑ ← STOP'}
          ],
          'images': [H1, H2, ..., C1],
          'num_history_images': N
        }
```

### 阶段 2: Template.replace_tag()

**文件**: `template.py`

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

**文件**: `template.py`

计算并扩展占位符 token：

```
输入 input_ids: [..., <history_image>, ..., <current_image>, ...]
                       ↓                      ↓
历史图片 token 数量:    49 (压缩后)            -
当前图片 token 数量:    -                     196 (标准)
                       ↓                      ↓
扩展后 input_ids: [..., <history_image>×49, ..., <current_image>×196, ...]
```

### 阶段 4: Template._post_encode()

**文件**: `template.py`

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

## 评估流程

### MonoVLNEvaluator

**文件**: `evaluator.py`

MonoVLN 评估器每步独立构建完整的单轮对话消息：

```python
class MonoVLNEvaluator(VLNEvaluator):
    def build_complete_messages(self, instruction, rgb_list, current_step):
        # 1. 采样历史帧
        history_images = []
        if current_step > 0:
            history_indices = self.sample_history_indices(current_step, self.num_history)
            history_images = [rgb_list[i] for i in history_indices]
        
        # 2. 获取当前帧（1帧）
        current_image = rgb_list[current_step]
        
        # 3. 构建单轮对话
        images = history_images + [current_image]
        user_content = f"Task: {instruction}. Historical observations: {history_tokens}. Current observation: <image>. ..."
        messages = [{'role': 'user', 'content': user_content}]
        
        return messages, images, len(history_images)
    
    def eval_episode(self, env_wrapper, episode, env_idx=0):
        rgb_list = []
        while not done:
            rgb_list.append(env.get_rgb())
            
            # 每步独立构建消息
            messages, images, num_history = self.build_complete_messages(
                instruction, rgb_list, step_id
            )
            
            # 编码（传递 num_history_images 触发压缩）
            encoded = template.encode({
                'messages': messages,
                'images': images,
                'num_history_images': num_history
            })
            
            # 生成动作
            action = model.generate(**encoded)
            
            # 执行
            obs, done = env.step(action)
```

## 文件结构

```
monovln/
├── __init__.py       # 模块注册（model + template）
├── dataset.py        # 独立数据集（单轮对话，均匀采样）
├── model.py          # 模型定义（添加特殊 token，reset 为 no-op）
├── template.py       # 模板（压缩逻辑核心）
├── compressor.py     # 2D 平均池化实现
├── arguments.py      # 训练参数（含 samples_per_episode）
├── trainer.py        # 训练入口
├── eval.py           # 评估入口
├── evaluator.py      # 单轮对话评估器
├── doc/
│   ├── OVERVIEW.md   # 逻辑概览
│   └── README_CN.md  # 本文件
└── script/
    ├── train/        # 训练脚本
    └── eval/         # 评估脚本
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `compress_stride` | 2 | 池化步长，2 表示 4 倍压缩 |
| `num_history` | 8 | 历史帧采样数量 |
| `num_future_steps` | 4 | 每次预测的动作数 |
| `samples_per_episode` | 8 | 每个 episode 采样的训练样本数 |

## 训练与评估

### 训练

```bash
bash src/swiftvln/models/monovln/script/train/train_monovln_qwen2_5_vl.sh
```

### 评估

```bash
MODEL_PATH=/path/to/checkpoint \
ENV_TYPE=habitat \
bash src/swiftvln/models/monovln/script/eval/eval_monovln_qwen2_5_vl_distributed.sh
```

## 设计考量

### 为什么使用单轮对话？

1. **简化推理**：不需要管理多轮对话状态和 KV cache
2. **代码简洁**：每步独立调用，逻辑清晰
3. **灵活性**：历史帧数量可以动态变化

### 为什么只压缩历史帧？

1. **当前帧最重要**：导航决策主要依赖当前观察
2. **历史帧提供上下文**：压缩后仍能提供轨迹记忆
3. **计算效率**：历史帧数量多，压缩收益大

### 为什么均匀采样 episode？

1. **覆盖全面**：覆盖轨迹的不同阶段（开始、中间、结束）
2. **边界条件**：保证无历史和 STOP 场景被包含
3. **效率**：避免过多重复相似的训练样本
