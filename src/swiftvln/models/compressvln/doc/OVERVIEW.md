# CompressVLN 逻辑概览

本文档描述 CompressVLN 的整体工作流程，不涉及具体实现细节。

---

## 1. 核心思想

CompressVLN 对 VLN（视觉语言导航）任务中的**历史帧进行 token 压缩**，当前帧保持原始分辨率。

```
历史帧（8张）→ 压缩 → 每张 49 tokens
当前帧（8张）→ 不压缩 → 每张 196 tokens

总 token 数: 8×49 + 8×196 = 1960 (vs 原本 8×196 + 8×196 = 3136)
```

---

## 2. 数据准备

### 2.1 数据集结构

```
trajectory_data/
├── R2R/
│   ├── annotations.json          # 导航标注
│   ├── features/                  # 预计算 ViT 特征（可选）
│   │   ├── episode_xxx.pt
│   │   └── ...
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

### 3.1 输入

- 样本索引 `i`
- 从 `data_list[i]` 获取 `(episode_id, instruction_id, start_idx)`

### 3.2 处理流程

```
1. 确定当前窗口
   - 时间范围: [start_idx, start_idx + num_frames]
   - 当前帧: 在窗口内每隔 num_future_steps 采样

2. 采样历史帧（如果不是轨迹起点）
   - 范围: [0, start_idx)
   - 均匀采样 num_history 张

3. 加载图片
   - 顺序: [历史帧..., 当前帧...]
   - 返回 PIL.Image 列表

4. 构建对话
   - system: 任务描述 + 历史帧 <image> tokens
   - 多轮对话: 每轮一张当前帧 + 预测动作
```

### 3.3 输出

```python
{
    'messages': [
        {'role': 'system', 'content': '... <image> <image> <image> ...'},  # 历史帧
        {'role': 'user', 'content': '当前观察: <image>.'},                   # 当前帧1
        {'role': 'assistant', 'content': '↑ ↑ ← →'},
        {'role': 'user', 'content': '下一步: <image>.'},                     # 当前帧2
        {'role': 'assistant', 'content': '↑ STOP'},
        ...
    ],
    'images': [PIL.Image, PIL.Image, ...],  # 历史在前，当前在后
    'num_history_images': 8                  # 告诉 Template 前8张是历史
}
```

---

## 4. 预计算特征模式（可选）

当 ViT 冻结时，可以预先计算所有帧的 ViT 特征，训练时直接加载，跳过 ViT 前向传播。

### 4.1 优势

- **显存节省**：ViT 不参与计算，显存占用大幅降低
- **训练加速**：跳过 ViT 编码，加速训练

### 4.2 生成预计算特征

```bash
# 使用共享的预计算脚本
bash src/swiftvln/scripts/vit_feat_precompute/precompute_r2r.sh
```

特征保存到 `{data_path}/features/` 目录，每个 episode 一个 `.pt` 文件：
```python
{
    'vit_features': List[torch.Tensor],  # [num_tokens, 2048], bfloat16
    'grid_thw': List[torch.Tensor],      # [3] per frame
}
```

### 4.3 启用预计算特征训练

在训练脚本中设置：
```bash
USE_PRECOMPUTED_FEATURES=true
# 自动强制 FREEZE_VIT=true
```

### 4.4 数据流变化

```
普通模式:
  images → image_processor → pixel_values → model.visual() → embeddings → 压缩

预计算模式:
  features/*.pt → Dataset 加载 → embeddings（跳过 ViT）→ 压缩
```

---

## 5. 训练流程

### 5.1 Token 转换 (`replace_tag`)

```
Dataset 输出:  <image> <image> ... <image>  <image> ...
                ↓ index < num_history_images  ↓ index >= num_history_images
Template 转换: <history_image> ...           <current_image> ...
```

### 5.2 Token 扩展 (`_encode`)

```
输入: ... <history_image> ... <current_image> ...
           ↓ 计算压缩后数量    ↓ 计算标准数量
输出: ... <history_image>×49 ... <current_image>×196 ...
```

### 5.3 Embedding 替换 (`_post_encode`)

```
1. 文本 embedding: embed_tokens(input_ids)
2. 图像 embedding: model.visual(pixel_values)
3. 历史帧: 2D 池化压缩
4. 当前帧: 保持原样
5. masked_scatter 替换占位符
6. 返回 inputs_embeds 给模型
```

---

## 6. 推理/评估流程

### 6.1 关键：评估时传递 num_history_images

```
评估时必须在 template.encode() 调用中传递 num_history_images：

encoded = template.encode({
    'messages': messages,
    'images': images,
    'num_history_images': len(history_images),  # 关键！
})
```

**为什么重要**：
- Template 的 `replace_tag()` 依赖此元数据区分历史/当前图片
- 没有它，所有图片都会被当作当前图片，不进行压缩

### 6.2 评估流程（CompressVLNEvaluator）

```
1. 收集观察
   - rgb_list.append(current_image)

2. 窗口管理
   - 每 num_frames 步重置窗口
   - 采样历史帧（窗口开始前的图片）

3. 构建消息
   - messages = [...], images = [历史... + 当前...]
   - num_history_images = len(history_images)

4. 编码（传递 num_history_images）
   - encoded = template.encode({..., 'num_history_images': n})
   - Template 自动压缩历史帧

5. 生成动作
   - outputs = model.generate(**model_inputs)
   - action_seq = parse_actions(output_text)

6. 执行动作
   - env.step(action)
```

### 6.3 环境交互循环

```python
rgb_list = []
while not done:
    # 1. 收集观察
    rgb_list.append(env.get_rgb())
    
    # 2. 构建消息（区分历史和当前）
    messages, images, num_history = build_messages(...)
    
    # 3. 编码（关键：传递 num_history_images）
    encoded = template.encode({
        'messages': messages,
        'images': images,
        'num_history_images': num_history,  # 触发压缩
    })
    
    # 4. 生成
    action = model.generate(**encoded)
    
    # 5. 执行
    obs, done = env.step(action)
```

---

## 7. 关键设计选择

| 设计点 | 选择 | 原因 |
|--------|------|------|
| 压缩目标 | 只压缩历史帧 | 当前帧对决策最重要 |
| 压缩方法 | 2D 平均池化 | 简单、无参数、保留空间结构 |
| 压缩时机 | 训练 + 推理 | 保持一致性 |
| Token 区分 | Dataset 用标准 token | 与 ms-swift 兼容 |
| 预计算特征 | 可选，ViT 冻结时 | 节省显存、加速训练 |

---

## 8. 与 StreamVLN 的区别

```
StreamVLN:
  所有图片 → 标准处理 → 每张 196 tokens

CompressVLN:
  历史图片 → 压缩处理 → 每张 49 tokens
  当前图片 → 标准处理 → 每张 196 tokens

CompressVLN (预计算模式):
  历史特征 → 加载 → 压缩处理 → 每张 49 tokens  (跳过 ViT)
  当前特征 → 加载 → 标准处理 → 每张 196 tokens (跳过 ViT)
```

---

## 9. 文件职责

```
dataset.py    → 构建训练样本，标记历史/当前图片数量，支持加载预计算特征
template.py   → 区分 token 类型，执行压缩，支持预计算特征模式
arguments.py  → 定义训练参数，包括预计算特征相关参数
model.py      → 添加特殊 token 到词表
trainer.py    → 训练入口，传递预计算特征参数
eval.py       → 评估入口
evaluator.py  → 评估器，传递 num_history_images 触发压缩

# 共享脚本（与 UniNaVid 共用）
src/swiftvln/scripts/vit_feat_precompute/
├── precompute_features.py  → ViT 特征预计算脚本
└── precompute_r2r.sh       → R2R/RxR 数据集预计算脚本
```
