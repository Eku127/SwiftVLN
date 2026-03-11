# SatNav 数据微调 Uni-NaVid 可行性分析与实施计划

> 本文档记录使用 SatNav 卫星导航数据对 Uni-NaVid 模型进行 finetune 的完整可行性分析和分步实施计划。

---

## 1. 两侧数据/模型对比

| 维度 | SatNav | Uni-NaVid |
|------|--------|-----------|
| **视觉输入** | 448×448 卫星鸟瞰图（BEV） | 第一人称室内 RGB（处理为 224×224） |
| **Vision Encoder** | — | EVA-CLIP（预训练于自然/室内图像） |
| **LLM Backbone** | — | Vicuna-7B |
| **数据存储** | 独立 JPEG 帧 `{episode}/rgb/001.jpg` | 单个 .mp4 视频文件 |
| **标注格式** | `annotations.json`，int 编码动作 | JSON 列表，对话格式（human/gpt） |
| **动作编码** | int: -1(初始)/0(停)/1(前)/2(左)/3(右) | text: "forward"/"left"/"right"/"stop" |
| **动作粒度** | 前进 10m，转弯 15° | 前进 0.25m，转弯 30° |
| **预测方式** | 逐步预测 1 个 action | 每步预测 4 个 actions |
| **数据规模** | 15,689 episodes，平均 60 步 | 示例 500 条（完整数据更大） |

---

## 2. 可行性评估

### 2.1 可行的方面

**数据格式转换技术上可行**

SatNav 的 `annotations.json` 可以转换为 Uni-NaVid 的训练 JSON 格式：

- 独立帧 `{episode}/rgb/*.jpg` → 合成 `.mp4`（ffmpeg/opencv）
- 动作映射：`1→forward`, `2→left`, `3→right`, `0→stop`
- 指令文本直接嵌入 Uni-NaVid 的 prompt template

**动作空间语义兼容**

两者都是离散动作空间（forward/left/right/stop），仅编码形式不同。

**训练框架兼容**

两者都用 DeepSpeed + 多卡全参训练。Uni-NaVid stage_2 脚本已支持从已有 checkpoint finetune。

**序列长度可控**

SatNav 平均 ~60 步 × 4 tokens/帧（grid:2 压缩）= ~240 visual tokens，远小于 `model_max_length=2048`。

### 2.2 主要挑战和风险

#### 风险 1：视觉域差异巨大（高风险）

EVA-CLIP 的 vision encoder 预训练于自然图像/室内场景，SatNav 的卫星鸟瞰图是完全不同的视觉域：

- 默认 `tune_vision_encoder=False` 时，EVA-CLIP 对卫星图可能无法提取有效语义特征
- 即使 `tune_vision_encoder=True`，初始特征质量差可能导致收敛困难
- 448→224 降采样会丢失卫星图中道路/建筑细节

**缓解策略**：先做 EVA-CLIP 特征探查实验（步骤 5），根据特征质量决定训练策略。

#### 风险 2：动作物理含义不同（中等风险）

| | SatNav | Uni-NaVid (Habitat) |
|---|--------|---------------------|
| forward | 10m | 0.25m |
| left/right | 15° | 30° |

模型可能已内化 Habitat 的动作粒度。SatNav 的 15° 转弯与 Uni-NaVid 的 30° 不对等。

**缓解策略**：模型 finetune 后应能学到新的物理含义，但需要充足的训练数据和轮次。

#### 风险 3：4-action 窗口化设计需精心设计

Uni-NaVid 每步预测 4 个 action，需要对 SatNav 逐步 action 序列进行窗口化：

```
Episode (70 steps): [-1, 1, 1, 2, 1, 3, 1, 1, ..., 0]

窗口化后：
Sample 1: video=帧[0:1],   label="forward forward left forward"
Sample 2: video=帧[0:5],   label="right forward forward forward"
Sample 3: video=帧[0:9],   label="forward ... forward stop"
...
```

15,689 episodes × ~17 windows/episode ≈ **~267K 训练样本**。

#### 风险 4：环境依赖冲突

Uni-NaVid 锁定了较老的依赖版本：

```
torch==2.0.1, transformers==4.31.0, deepspeed==0.9.5, flash-attn==2.5.9.post1
```

与现有 `swift-vln-train` 环境不兼容，需要单独创建 conda 环境。

---

## 3. Uni-NaVid 训练数据格式详解

### 3.1 JSON 格式

```json
[
  {
    "id": "NAV_ID_VLN_35150_002",
    "video": "nav_videos/35150_002.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation. Your assigned task is: '{instruction}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
      },
      {
        "from": "gpt",
        "value": "right right forward forward"
      }
    ]
  }
]
```

**关键要求：**

- `id` 必须包含 `"NAV_ID"` 字符串 → 触发导航专用处理路径（帧 drop augmentation + color jitter）
- `video` 为相对于 `--video_folder` 的 mp4 文件路径
- `conversations[0].value` 中必须包含 `NAVIGATION_IDENTIFIER`（即 `"a video of historical observations and an image of the current observation"`） → 触发导航 token 处理（video + current image + navigation token）
- `conversations[1].value` 为空格分隔的 4 个 action 词

### 3.2 Prompt Template

```
Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation. Your assigned task is: '{instruction}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop.
```

### 3.3 视觉处理流水线

1. decord VideoReader 读取 mp4
2. 按 `video_fps`（默认 1）采样帧
3. 导航数据专用增强：随机丢帧（最多 10%）+ color jitter
4. EVA-CLIP image processor 处理为 224×224
5. grid:2 压缩 → 每帧 4 tokens
6. Token merging：短期记忆（最近 64 帧全精度）+ 长期记忆（余弦相似度合并）

---

## 4. SatNav → Uni-NaVid 数据转换方案

### 4.1 动作映射

| SatNav code | SatNav meaning | Uni-NaVid text |
|-------------|----------------|----------------|
| -1 | INITIAL（起始帧标记） | 跳过 |
| 0 | STOP | "stop" |
| 1 | MOVE_FORWARD | "forward" |
| 2 | TURN_LEFT | "left" |
| 3 | TURN_RIGHT | "right" |

### 4.2 窗口化策略

对每个 episode 的 action 序列（跳过 actions[0]=-1），按步长 4 进行滑动窗口：

```python
actions = episode["actions"][1:]  # 去掉 INITIAL
for window_start in range(0, len(actions), 4):
    window_end = min(window_start + 4, len(actions))
    action_window = actions[window_start:window_end]

    # 不足 4 个 action 的最后一个窗口：补 stop 或丢弃
    if len(action_window) < 4:
        action_window += [0] * (4 - len(action_window))  # 补 stop

    # 视频：从第 1 帧到第 window_start+1 帧（含当前观察）
    video_frames = frames[0 : window_start + 1]
    label = " ".join(ACTION_MAP[a] for a in action_window)
```

### 4.3 视频合成

将每个 episode 的 JPEG 帧序列合成为 mp4：

```bash
ffmpeg -framerate 1 -i {episode}/rgb/%03d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
```

注意：Uni-NaVid 训练时按 `video_fps=1` 采样，因此合成视频时 framerate=1 即可保留全部帧。

### 4.4 输出目录结构

```
satnav_uninavid_data/
├── nav_videos/
│   ├── Geneva-1_satnav_000000_w00.mp4   # episode 0, window 0
│   ├── Geneva-1_satnav_000000_w01.mp4   # episode 0, window 1
│   └── ...
└── satnav_uninavid_train.json            # 训练标注
```

---

## 5. 分步实施计划

### 步骤 1：环境搭建（0.5 天）

```bash
conda create -n uninavid python=3.10 -y
conda activate uninavid
cd /mnt/data1/home/jiangjiajun/workspace/Uni-NaVid
pip install --upgrade pip
pip install -e .
pip install flash-attn==2.5.9.post1
```

**验证**：`python -c "import uninavid; print('OK')"`

### 步骤 2：模型下载（~1 小时）

```bash
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
bash baseline/uninavid/scripts/download_uninavid_models.sh --all
```

需要下载：
- EVA-CLIP `eva_vit_g.pth`（~3.9 GB）
- Uni-NaVid pretrained weights（~14 GB）
- Vicuna-7B（~13 GB，可选，stage 1 训练用）

### 步骤 3：数据转换脚本（1-2 天）

编写 `baseline/uninavid/src/convert_satnav_to_uninavid.py`：

**输入**：
- SatNav `annotations.json`：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json`
- SatNav 帧图像：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/images/`

**输出**：
- 窗口化训练 JSON（Uni-NaVid 格式）
- 合成的 mp4 视频文件

**核心逻辑**：
1. 读取 annotations.json
2. 对每个 episode 进行 4-action 窗口化
3. 对每个窗口：截取历史帧 → 合成 mp4
4. 生成对话格式 JSON 条目
5. 输出统计信息

### 步骤 4：小规模 Smoke Test（0.5 天）

用 ~100 个 episode 的转换数据做训练验证：

```bash
# 只转换前 100 个 episode
python baseline/uninavid/src/convert_satnav_to_uninavid.py \
  --max-episodes 100 \
  --output-dir /mnt/data3/jiangjiajun/dataset/satnav_uninavid_data/smoke

# 单卡或少卡训练几步，确认 loss 正常下降
deepspeed uninavid/train/train_mem.py \
  --model_name_or_path baseline/uninavid/model/uninavid-7b-full-224-video-fps-1-grid-2 \
  --data_path /mnt/data3/jiangjiajun/dataset/satnav_uninavid_data/smoke/satnav_uninavid_train.json \
  --video_folder /mnt/data3/jiangjiajun/dataset/satnav_uninavid_data/smoke \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  ...
```

**验证标准**：训练 loss 能正常下降（不是 NaN 或不变）。

### 步骤 5：EVA-CLIP 特征探查（0.5 天）

编写 `baseline/uninavid/src/probe_eva_clip_features.py`：

1. 加载 EVA-CLIP 模型
2. 分别对 SatNav 卫星图和 Habitat 室内图提取特征
3. 对比特征分布（均值、方差、余弦相似度分布）
4. 可视化 t-SNE/PCA 降维结果

**决策依据**：
- 若两域特征有一定区分度但不完全乱序 → 可以 finetune，建议 `tune_vision_encoder=True`
- 若卫星图特征退化为噪声 → 需考虑替换 vision encoder 或放弃此方案

### 步骤 6：全量训练（视步骤 4-5 结果）

基于步骤 4-5 的结论选择训练策略：

**方案 A：从 Uni-NaVid 继续 finetune（推荐首选）**

```bash
# stage_2 style: 基于已训练的 Uni-NaVid
PREV_MODEL="baseline/uninavid/model/uninavid-7b-full-224-video-fps-1-grid-2"
DATA_PATH="/mnt/data3/jiangjiajun/dataset/satnav_uninavid_data/full/satnav_uninavid_train.json"

deepspeed uninavid/train/train_mem.py \
  --model_name_or_path $PREV_MODEL \
  --data_path $DATA_PATH \
  --video_folder /mnt/data3/jiangjiajun/dataset/satnav_uninavid_data/full \
  --tune_vision_encoder True \
  ...
```

**方案 B：从 Vicuna-7B 从头训练（如果域差异太大）**

```bash
# stage_1 style: 从底座开始
PREV_MODEL="baseline/uninavid/model/vicuna-7b-v1.5"
```

### 步骤 7：评测适配（1 天）

修改 `offline_eval_uninavid.py` 以支持 SatNav：

- 替换图像加载为 SatNav 卫星图裁剪
- 动作输出映射回 SatNav 环境语义
- 接入 SatNav 评测环境（episodes/eval）

---

## 6. 数据路径速查

| 数据 | 路径 |
|------|------|
| SatNav 训练标注 | `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json` |
| SatNav 训练帧 | `.../trajectory_data/images/{episode_id}/rgb/` |
| SatNav 评测 episodes | `.../ver_260306/episodes/eval/all_episodes.json` |
| Uni-NaVid 仓库 | `/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid` |
| Uni-NaVid 模型（SwiftVLN 内） | `baseline/uninavid/model/` |
| EVA-CLIP processor | `Uni-NaVid/uninavid/processor/clip-patch14-224` |

---

## 7. 风险评估总结

| 风险 | 等级 | 影响 | 缓解策略 |
|------|------|------|----------|
| EVA-CLIP 对卫星图特征提取失效 | 高 | 训练无法收敛 | 步骤 5 特征探查；tune vision encoder |
| 动作粒度不匹配 | 中 | 预测偏差 | 充分 finetune；可考虑调整窗口策略 |
| 窗口化设计不合理 | 中 | 训练效果差 | smoke test 验证；迭代调整 |
| 环境依赖冲突 | 低 | 搭建耗时 | 独立 conda 环境 |
| 训练数据量不足 | 低 | 过拟合 | ~267K 样本应足够 |

**总体结论**：技术上可行，但视觉域迁移风险显著。建议先用 3 天完成步骤 1-5 的快速验证，再决定是否投入全量训练。
