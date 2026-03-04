# UniNaVid - 基于 ms-swift 的三级记忆架构 VLN 训练

本模块实现了 Uni-NaVid 风格的三级记忆架构，用于视觉语言导航 (VLN) 训练。

## 架构概述

UniNaVid 使用三级记忆架构来处理历史帧：

| 记忆类型 | 帧范围 | 压缩方式 | Token 数量 |
|---------|--------|---------|-----------|
| 长期记忆 | 较早的历史帧 | Pooling + 均值 + 相似度合并 | M (合并后) |
| 短期记忆 | 最近 N 帧 | Pooling | 每帧 pooling 后 tokens |
| 当前观测 | 当前帧 | 无压缩 | 原始 tokens |

### 关键特性

- **相似度合并**：长期记忆中相似帧会被合并（余弦相似度 > 阈值）
- **动态序列裁剪**：`_post_encode` 阶段将 N 个占位符裁剪为 M 个合并 token
- **随机丢帧增强**：训练时随机丢弃部分历史帧（默认 10%）
- **Uni-NaVid 风格动作格式**：`forward/left/right/stop`

## 文件结构

```
uninavid/
├── __init__.py          # 模块注册（模型 + 模板）
├── model.py             # UniNaVidQwen25VLForConditionalGeneration
├── template.py          # UniNaVidQwen25VLTemplate（核心压缩 + 裁剪逻辑）
├── dataset.py           # UniNaVidDataset（数据加载 + 帧分类）
├── arguments.py         # UniNaVidArguments（参数定义）
├── trainer.py           # UniNaVidSft（训练入口）
├── eval.py              # 评估入口
├── evaluator.py         # UniNaVidEvaluator（增量特征缓存）
├── script/
│   ├── train/
│   │   └── train_uninavid_qwen2_5_vl.sh  # 多卡训练脚本
│   └── eval/
│       └── eval_uninavid_qwen2_5_vl_distributed.sh  # 分布式评估脚本
└── README.md            # 本文件
```

## 特殊 Token

| Token | 用途 | 压缩方式 |
|-------|-----|---------|
| `<long_term_image>` | 长期记忆帧 | Pooling + 均值 + 相似度合并 |
| `<short_term_image>` | 短期记忆帧 | Pooling |
| `<current_image>` | 当前观测帧 | 无压缩 |

## 参数说明

### UniNaVid 特有参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `short_term_frames` | 32 | 短期记忆帧数阈值 |
| `similarity_threshold` | 0.985 | 长期记忆合并的余弦相似度阈值 |
| `compress_stride` | 3 | Pooling 步长（2=4x, 3=9x, 4=16x 压缩）|
| `drop_frame_prob` | 0.1 | 训练时随机丢帧概率 |
| `num_future_steps` | 4 | 每步预测的动作数量 |
| `samples_per_episode` | 5 | 每个 episode 采样的时间点数量 |
| `image_resize_stride` | 1.5 | 图像缩放比例（1.5=缩至 2/3，2=缩至 1/2），用于减少 ViT 显存 |
| `history_frame_stride` | 2 | 历史帧时间降采样（2=每隔 1 帧取 1 帧），在随机丢帧前应用 |
| `use_precomputed_features` | false | 是否使用预计算的 ViT 特征 |

## 预计算特征（可选）

当 ViT 冻结时，可以预先计算所有帧的 ViT 特征，训练时直接加载，跳过 ViT 前向传播。

### 优势
- **显存节省**：ViT 不参与计算，显存占用大幅降低
- **训练加速**：跳过 ViT 编码，加速训练

### 生成预计算特征

```bash
# 单 GPU
python src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
    --data_path /path/to/R2R \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct

# 多 GPU（推荐）
torchrun --nproc_per_node=8 src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
    --data_path /path/to/R2R \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct
```

特征会保存到 `{data_path}/features/` 目录，每个 episode 一个 `.pt` 文件。

### 使用预计算特征训练

在训练脚本中设置：
```bash
USE_PRECOMPUTED_FEATURES=true
```

### 预计算模式下无效的参数

| 参数 | 原因 |
|------|------|
| `IMAGE_RESIZE_STRIDE` | 图像在预计算时已处理，使用原始尺寸 |
| `FREEZE_VIT` | ViT 不参与计算，自动强制为 true |

其他参数（`history_frame_stride`、`drop_frame_prob` 等）仍然有效。

## 使用方法

### 训练

```bash
# 使用默认配置
bash src/swiftvln/models/uninavid/script/train/train_uninavid_qwen2_5_vl.sh

# 或直接运行 trainer.py
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    src/swiftvln/models/uninavid/trainer.py \
    --custom_register_path src/swiftvln/models/uninavid \
    --model_type uninavid_qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset /path/to/vln_data \
    --short_term_frames 32 \
    --similarity_threshold 0.985 \
    --compress_stride 3 \
    --drop_frame_prob 0.1 \
    --num_future_steps 4 \
    --samples_per_episode 5 \
    --image_resize_stride 1.5 \
    --history_frame_stride 2 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 2e-5
```

### 评估

```bash
# Habitat 环境评估
ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint \
    bash src/swiftvln/models/uninavid/script/eval/eval_uninavid_qwen2_5_vl_distributed.sh

# SatNav 环境评估
ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint \
    bash src/swiftvln/models/uninavid/script/eval/eval_uninavid_qwen2_5_vl_distributed.sh

# 自定义参数（应与训练参数匹配）
ENV_TYPE=habitat \
MODEL_PATH=/path/to/checkpoint \
SHORT_TERM_FRAMES=32 \
SIMILARITY_THRESHOLD=0.985 \
COMPRESS_STRIDE=3 \
IMAGE_RESIZE_STRIDE=1.5 \
EVAL_SPLIT=val_unseen \
    bash src/swiftvln/models/uninavid/script/eval/eval_uninavid_qwen2_5_vl_distributed.sh

# 调试模式（限制 episode 数量）
MAX_EPISODES=10 DEBUG_TIMING=true VERBOSE=true \
MODEL_PATH=/path/to/checkpoint \
    bash src/swiftvln/models/uninavid/script/eval/eval_uninavid_qwen2_5_vl_distributed.sh
```

#### 评估特性

- **增量特征缓存**：每步只编码当前帧，历史帧特征从缓存读取
- **三级记忆推理**：完全复现 Uni-NaVid 的 online 推理流程
- **高效推理**：避免重复 ViT 计算，显著加速评估

### 数据格式

数据集应包含 `annotations.json` 文件，格式如下：

```json
[
    {
        "video": "episode_001",
        "instructions": ["Navigate to the kitchen"],
        "actions": [-1, 1, 1, 2, 1]
    }
]
```

其中 `video` 目录下应包含 `rgb/` 子目录，存放按序排列的帧图像。

**动作编码：**
| 索引 | 动作 | 说明 |
|-----|------|------|
| -1 | 初始占位符 | 起始状态，无动作 |
| 0 | stop | 停止/到达目标 |
| 1 | forward | 前进 25cm |
| 2 | left | 左转 15° |
| 3 | right | 右转 15° |

**图像-动作对应关系：**
- 图像 `{i:03d}.png` = 执行 `actions[i-1]` 后的观察
- 图像 `{i:03d}.png` → 预测 `actions[i]`

**代码处理：**
```python
# 去掉起始的 -1，末尾补 STOP (0)
actions = data['actions'][1:] + [0]
```

## 技术细节

### 占位符动态裁剪（方案 B）

由于长期记忆合并后的 token 数量 M 在 `_encode` 阶段无法确定，我们采用以下策略：

1. **`_encode` 阶段**：为每个长期帧分配 1 个占位符（共 N 个）
2. **`_post_encode` 阶段**：
   - 视觉编码 + Grid Pooling
   - 长期帧取均值 + 相似度合并 → M 个
   - 裁剪序列：删除 N-M 个多余占位符
   - 填入所有 embedding

### 相似度合并算法

```python
def _merge_long_term_memory(self, embeddings_list, threshold=0.985):
    """按时间顺序合并相似的长期记忆帧"""
    merged = [embeddings_list[0].mean(dim=0, keepdim=True)]
    weights = [1]
    
    for i in range(1, len(embeddings_list)):
        current = embeddings_list[i].mean(dim=0, keepdim=True)
        sim = F.cosine_similarity(merged[-1], current, dim=-1).mean()
        
        if sim > threshold:
            # 合并：加权平均
            new_weight = weights[-1] + 1
            merged[-1] = (merged[-1] * weights[-1] + current) / new_weight
            weights[-1] = new_weight
        else:
            # 不合并：添加新条目
            merged.append(current)
            weights.append(1)
    
    return torch.cat(merged, dim=0)  # [M, hidden_size]
```

### 评估时的增量处理流程

```
Step N: 新观测图像到达
========================================

1. ViT 编码阶段
   新观测图像 ──► ViT 编码 ──► current_features [H×W tokens]
                               （完整特征，不压缩）

2. 特征组装阶段
   long_term_cache  + short_term_cache  + current_features
   [M tokens]         [N × pooled]        [H×W tokens]
        └───────────────┼──────────────────┘
                        ▼
                拼接 + text embeddings
                        ▼
                 complete inputs_embeds

3. LLM 推理
   inputs_embeds ──► LLM.generate() ──► "forward forward left stop"

4. 缓存更新（推理完成后）
   current_features ──► Grid Pooling ──► short_term_cache.append()
   
   如果 short_term 满了：
   oldest ──► Mean Pooling ──► 与 long_term_cache 相似度检查
        ├── 相似度 > threshold: 加权平均合并
        └── 否则: 追加为新 token
```

## 与 MonoVLN 的差异

| 维度 | MonoVLN | UniNaVid |
|-----|---------|----------|
| Token 类型 | 2 种（history/current）| 3 种（long-term/short-term/current）|
| 历史帧数量 | 固定 num_history | 不限制（使用所有历史帧）|
| 长期记忆 | 无 | 相似度合并（余弦相似度 > 阈值时合并）|
| 占位符处理 | 固定数量 | 动态裁剪（N 个占位符 → M 个合并 token）|
| 动作格式 | 符号（↑←→）| 文本（forward/left/right/stop）|
| 数据增强 | 无 | 随机丢帧 + 时间降采样 |
| 图像预处理 | 原始分辨率 | 可配置缩放（image_resize_stride）|
| 评估方式 | 每步重新编码所有帧 | 增量特征缓存（只编码新帧）|
| Prompt 格式 | 直接拼接 | 分层描述（Earlier/Recent Observations）|

## 注意事项

1. **padding_free 必须为 false**：自定义 token 与 Qwen2.5-VL 的 get_rope_index 不兼容
2. **显存优化**：
   - 使用 `image_resize_stride` 缩小图像以减少 ViT 显存占用
   - 使用 `history_frame_stride` 降采样历史帧以减少帧数
   - 建议使用 DeepSpeed ZeRO-2 或 ZeRO-3
3. **长序列**：建议设置 `max_length=32768` 或更高
4. **评估参数一致性**：评估时的参数（如 `short_term_frames`、`compress_stride` 等）应与训练时一致