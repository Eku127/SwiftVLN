# Uni-NaVid 训练数据格式参考

> 基于 [Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) 源码分析的训练数据格式完整说明。

---

## 1. 训练 JSON 文件格式

训练数据为 JSON 数组，每个元素代表一个训练样本。支持三种模态：

### 1.1 导航视频数据（Navigation）

```json
{
  "id": "NAV_ID_VLN_35150_002",
  "video": "nav_videos/35150_002.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation. Your assigned task is: 'Walk towards the door...'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
    },
    {
      "from": "gpt",
      "value": "right right forward forward"
    }
  ]
}
```

**关键标识**：

| 标识 | 位置 | 触发行为 |
|------|------|----------|
| `"NAV_ID"` | `id` 字段中包含 | 启用导航专用数据增强（帧 drop + color jitter） |
| `NAVIGATION_IDENTIFIER` | prompt 中包含 `"a video of historical observations and an image of the current observation"` | 启用导航 token 处理路径（video tokens + current image tokens + `[Navigation]` token） |

### 1.2 普通视频数据（Video）

```json
{
  "id": "VIDEO_some_id",
  "video": "videos/some_video.mp4",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this video."},
    {"from": "gpt", "value": "The video shows..."}
  ]
}
```

### 1.3 图像数据（Image）

```json
{
  "id": "IMAGE_some_id",
  "image": "images/some_image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in this image?"},
    {"from": "gpt", "value": "This image shows..."}
  ]
}
```

---

## 2. 对话格式规范

- 固定为 `[human, gpt]` 交替的 turn-based 对话
- `conversations[0].from = "human"`，`conversations[1].from = "gpt"`
- 对于导航任务：gpt 回复为空格分隔的 **4 个动作词**
- 可用动作词：`"forward"`, `"left"`, `"right"`, `"stop"`

---

## 3. 训练脚本参数

关键参数（来自 `scripts/uninavid_stage_2.sh`）：

```bash
--model_name_or_path $PREV_MODEL      # 基模型路径
--version imgsp_v1                      # 使用 imgsp_v1 预处理（含特殊 token 注入）
--data_path $DATA_PATH                  # 训练 JSON 路径
--image_folder ./data/Nav-Finetune      # 图像根目录
--video_folder ./data/Nav-Finetune      # 视频根目录
--vision_tower ./model_zoo/eva_vit_g.pth  # EVA-CLIP 权重路径
--image_processor ./uninavid/processor/clip-patch14-224  # CLIP processor 配置
--tune_vision_encoder False             # 是否微调 vision encoder
--mm_projector_type mlp2x_gelu          # 视觉投射器类型
--video_fps 1                           # 视频采样帧率
--compress_type "grid:2"                # 视觉 token 压缩方式（grid:2 → 4 tokens/帧）
--model_max_length 2048                 # 最大序列长度
```

---

## 4. 特殊 Token 体系

定义在 `uninavid/constants.py`：

| Token | 值 | 用途 |
|-------|-----|------|
| `<image>` | DEFAULT_IMAGE_TOKEN | 图像占位符（prompt 中使用） |
| `<video_special>` | VIDEO_START_SPECIAL_TOKEN | 视频特征起始 |
| `</video_special>` | VIDEO_END_SPECIAL_TOKEN | 视频特征结束 |
| `<image_special>` | IMAGE_START_TOKEN | 当前帧特征起始 |
| `</image_special>` | IMAGE_END_TOKEN | 当前帧特征结束 |
| `[Navigation]` | NAVIGATION_SPECIAL_TOKEN | 导航模式标记 |
| `<image_sep>` | IAMGE_SEPARATOR | 帧间分隔符 |

导航数据的 token 注入顺序（在 `preprocess_imgsp_v1` 中）：

```
<video_special> <image_sep> [image_features] </video_special> <image_special> </image_special> [Navigation]
```

---

## 5. 视频处理流水线

```
mp4 文件
  ↓ decord VideoReader 加载
  ↓ 按 video_fps (default=1) 采样帧 → frame_idx = [0, sample_fps, 2*sample_fps, ...]
  ↓ [仅导航数据] 随机帧 drop（最多丢弃 10% 非末尾帧）
  ↓ [仅导航数据] 随机帧复制（3% 概率）
  ↓ [仅导航数据] Color Jitter（5% 概率 per frame）
  ↓ CLIP processor → resize 到 224×224
  ↓ EVA-CLIP 特征提取 → (n_frames, 256, 1408)
  ↓ grid:2 avg pooling → (n_frames, 4, hidden_dim)
  ↓ MLP projector → (n_frames, 4, llm_hidden_dim)
  ↓ Token merging:
      - 最近 64 帧：全精度（4 tokens/帧）
      - 更早帧：按余弦相似度 > 0.985 合并为 1 token/group
  ↓ 拼接特殊 token 后送入 LLM
```

---

## 6. 训练模式对照

| 模式 | 基模型 | 脚本 | 适用场景 |
|------|--------|------|----------|
| Stage 1 | Vicuna-7B | `uninavid_stage_1.sh` | 从 LLM 底座开始训练 |
| Stage 2 | Uni-NaVid pretrained | `uninavid_stage_2.sh` | 在已有能力基础上 finetune |

Stage 2 差异：无 `--hostfile` 参数，其余超参完全相同。
