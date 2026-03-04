# MonoVLN 逻辑概览

本文档描述 MonoVLN 的整体工作流程，不涉及具体实现细节。

---

## 1. 核心思想

MonoVLN 采用**单轮对话**格式进行视觉语言导航，同时对**历史帧进行 token 压缩**，当前帧保持原始分辨率。

```
单轮对话:
User: "Task: {instruction}. Historical observations: <img1> <img2> .... Current observation: <img>. Output actions."
Assistant: "↑ ↑ ← → STOP"

Token 压缩:
历史帧（8张）→ 压缩 → 每张 49 tokens
当前帧（1张）→ 不压缩 → 每张 196 tokens

总 token 数: 8×49 + 1×196 = 588 (vs StreamVLN 多帧: 8×196 + N×196)
```

---

## 2. 数据准备

### 2.1 数据集结构

```
trajectory_data/
├── R2R/
│   ├── annotations.json          # 导航标注
│   └── videos/
│       └── episode_xxx/
│           └── rgb/
│               ├── 000000.png    # 第0帧
│               ├── 000001.png    # 第1帧
│               └── ...
```

### 2.2 annotations.json 格式

```json
{
  "episode_id": {
    "video": "/path/to/episode_xxx",
    "instructions": ["Walk to the door..."],
    "actions": [1, 1, 2, 0, ...]  // 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
  }
}
```

---

## 3. 获取训练样本 (`__getitem__`)

### 3.1 Episode 采样策略

MonoVLN 对每个 episode 均匀采样固定数量的训练样本：

```
Episode: 100 个 action, samples_per_episode=8, num_future_steps=4

可采样范围: [0, 96]  # 96 = 100 - 4，保证能预测完整的 action 序列
采样点: [0, 13, 27, 41, 54, 68, 82, 96]

- 第0个样本: 当前位置=0, 无历史, 预测 actions[0:4]
- 第96个样本: 当前位置=96, 有96帧历史, 预测 actions[96:100] (包含 STOP)
```

### 3.2 处理流程

```
1. 获取数据索引
   - (episode_id, instruction_id, current_idx) = data_list[i]

2. 采样历史帧（如果 current_idx > 0）
   - 范围: [0, current_idx)
   - 均匀采样 min(num_history, current_idx) 张
   - 如果可用帧数 < num_history，使用全部

3. 获取当前帧
   - 只取 1 帧: video_frames[current_idx]

4. 获取预测动作
   - actions[current_idx : current_idx + num_future_steps]
```

### 3.3 输出

```python
{
    'messages': [
        {
            'role': 'user', 
            'content': 'Task: Walk to the door. Historical observations: <image> <image> <image>. Current observation: <image>. Output actions: ↑/←/→/STOP.'
        },
        {
            'role': 'assistant', 
            'content': '↑ ↑ ← STOP'
        }
    ],
    'images': [H1, H2, H3, C1],  # 3张历史 + 1张当前
    'num_history_images': 3      # 告诉 Template 前3张是历史
}
```

---

## 4. 训练流程

### 4.1 Token 转换 (`replace_tag`)

```
Dataset 输出:  <image> <image> <image>  <image>
               ↓ index < num_history    ↓ index >= num_history
Template 转换: <history_image> ×3       <current_image> ×1
```

### 4.2 Token 扩展 (`_encode`)

```
输入: ... <history_image> <history_image> <history_image> <current_image> ...
          ↓ 计算压缩后数量                                ↓ 计算标准数量
输出: ... <history_image>×49 ×3 ...                       <current_image>×196 ...
```

### 4.3 Embedding 替换 (`_post_encode`)

```
1. 文本 embedding: embed_tokens(input_ids)
2. 图像 embedding: model.visual(pixel_values)
3. 历史帧: 2D 池化压缩
4. 当前帧: 保持原样
5. masked_scatter 替换占位符
6. 返回 inputs_embeds 给模型
```

---

## 5. 推理/评估流程

### 5.1 单轮对话评估

MonoVLN 评估时每步独立构建完整消息，不需要 KV cache：

```python
# MonoVLNEvaluator.build_complete_messages()
def build_complete_messages(instruction, rgb_list, current_step):
    # 1. 采样历史帧
    history_images = sample_history(rgb_list, current_step, num_history)
    
    # 2. 获取当前帧
    current_image = rgb_list[current_step]
    
    # 3. 构建单轮对话
    images = history_images + [current_image]
    user_content = f"Task: {instruction}. Historical observations: {history_tokens}. Current observation: <image>. ..."
    messages = [{'role': 'user', 'content': user_content}]
    
    return messages, images, len(history_images)
```

### 5.2 评估流程

```
1. 收集观察
   - rgb_list.append(current_image)

2. 构建单轮消息（每步独立）
   - messages, images, num_history = build_complete_messages(
       instruction, rgb_list, current_step
   )

3. 编码（传递 num_history_images 触发压缩）
   - encoded = template.encode({
       'messages': messages,
       'images': images,
       'num_history_images': num_history
   })

4. 生成动作
   - outputs = model.generate(**model_inputs)
   - action_seq = parse_actions(output_text)

5. 执行动作
   - env.step(action)
```

### 5.3 环境交互循环

```python
rgb_list = []
while not done:
    # 1. 收集观察
    rgb_list.append(env.get_rgb())
    
    # 2. 构建单轮消息
    messages, images, num_history = build_complete_messages(
        instruction, rgb_list, step_id
    )
    
    # 3. 编码（触发历史帧压缩）
    encoded = template.encode({
        'messages': messages,
        'images': images,
        'num_history_images': num_history,
    })
    
    # 4. 生成
    action = model.generate(**encoded)
    
    # 5. 执行
    obs, done = env.step(action)
```

---

## 6. 关键设计选择

| 设计点 | 选择 | 原因 |
|--------|------|------|
| 对话格式 | 单轮对话 | 简化推理，无需 KV cache |
| 当前帧数量 | 固定 1 帧 | 与历史帧形成清晰区分 |
| 压缩目标 | 只压缩历史帧 | 当前帧对决策最重要 |
| 压缩方法 | 2D 平均池化 | 简单、无参数、保留空间结构 |
| Episode 采样 | 均匀采样 | 覆盖轨迹不同阶段 |
| 首尾保证 | 包含第0帧和最后帧 | 保证无历史和 STOP 场景 |

---

## 7. 与 StreamVLN 的区别

```
StreamVLN:
  对话格式: System(历史) + 多轮 User-Assistant(当前帧 + 动作)
  当前帧: 窗口内多帧
  推理: 需要 KV cache 管理
  所有图片 → 标准处理 → 每张 196 tokens

MonoVLN:
  对话格式: 单轮 User-Assistant
  当前帧: 固定 1 帧
  推理: 每步独立，无 KV cache
  历史图片 → 压缩处理 → 每张 49 tokens
  当前图片 → 标准处理 → 每张 196 tokens
```

---

## 8. 文件职责

```
dataset.py    → 构建训练样本，均匀采样 episode，单轮对话格式
template.py   → 区分 token 类型，执行压缩
model.py      → 添加特殊 token，reset/reset_for_env (no-op)
trainer.py    → 训练入口
eval.py       → 评估入口
evaluator.py  → 单轮对话评估器，每步独立构建 messages
arguments.py  → 参数定义，包含 samples_per_episode
```
