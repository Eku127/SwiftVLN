# SatNav → Uni-NaVid 数据转换规格

> 详细定义从 SatNav annotations 到 Uni-NaVid 训练数据的转换规则。

---

## 1. 输入数据

| 数据 | 路径 |
|------|------|
| SatNav 标注 | `$SATNAV_DATA_ROOT/ver_260306/trajectory_data/annotations.json` |
| SatNav 帧图像 | `.../trajectory_data/images/{episode_video}/rgb/{NNN}.jpg` |

SatNav annotations.json 单条示例：

```json
{
  "id": 0,
  "trajectory_id": "0",
  "steps": 70,
  "video": "images/Geneva-1_satnav_000000",
  "instructions": [
    "Begin at the storage yard by the roadside; orbit counter-clockwise..."
  ],
  "actions": [-1, 1, 1, 1, 2, 1, 3, ..., 0]
}
```

帧对齐关系：`rgb/001.jpg` ↔ `actions[0]`(-1)，`rgb/002.jpg` ↔ `actions[1]`，...

---

## 2. 输出数据

### 2.1 目标目录结构

```
{output_dir}/
├── nav_videos/
│   ├── satnav_{episode_id:06d}_w{window_idx:02d}.mp4
│   └── ...
└── satnav_uninavid_train.json
```

### 2.2 输出 JSON 格式

```json
[
  {
    "id": "NAV_ID_SATNAV_000000_w00",
    "video": "nav_videos/satnav_000000_w00.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation. Your assigned task is: 'Begin at the storage yard by the roadside; orbit counter-clockwise...'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
      },
      {
        "from": "gpt",
        "value": "forward forward forward left"
      }
    ]
  }
]
```

---

## 3. 转换规则

### 3.1 动作映射

```python
ACTION_MAP = {
    1: "forward",
    2: "left",
    3: "right",
    0: "stop",
}
```

`actions[0] = -1`（INITIAL）仅标记起点，不参与动作转换。

### 3.2 窗口化策略

对每个 episode，从 `actions[1:]` 开始，以步长 4 进行非重叠滑动窗口：

```python
real_actions = episode["actions"][1:]  # 去掉 INITIAL marker
window_size = 4

for w_idx, w_start in enumerate(range(0, len(real_actions), window_size)):
    w_end = min(w_start + window_size, len(real_actions))
    action_window = real_actions[w_start:w_end]

    # 不足 4 个 action：用 stop 填充
    while len(action_window) < window_size:
        action_window.append(0)

    # 历史帧范围：第 1 帧到第 w_start+1 帧（1-indexed）
    # 即 rgb/001.jpg ~ rgb/{w_start+1:03d}.jpg
    frame_start = 1
    frame_end = w_start + 1  # 包含当前观察帧

    label = " ".join(ACTION_MAP[a] for a in action_window)
```

**示例**（episode steps=70, actions 长度=71）：

| 窗口 | w_start | 历史帧范围 | action_window | label |
|------|---------|-----------|---------------|-------|
| w00 | 0 | 001.jpg (仅初始帧) | actions[1:5] | "forward forward left forward" |
| w01 | 4 | 001~005.jpg | actions[5:9] | "right forward forward forward" |
| w02 | 8 | 001~009.jpg | actions[9:13] | ... |
| ... | | | | |
| w17 | 68 | 001~069.jpg | actions[69:71]+[stop,stop] | "forward stop stop stop" |

### 3.3 视频合成规格

- 输入帧：`{trajectory_data}/images/{video}/rgb/{frame_start:03d}.jpg ~ {frame_end:03d}.jpg`
- 合成格式：H.264 MP4，`yuv420p` pixel format
- 帧率：1 fps（Uni-NaVid 训练时 `video_fps=1` 会 1:1 采样）
- 分辨率：保持原始 448×448（CLIP processor 会在训练时缩放为 224×224）

```bash
# 单个视频合成命令示例
ffmpeg -framerate 1 \
  -start_number 1 -i "{episode_dir}/rgb/%03d.jpg" \
  -frames:v {frame_count} \
  -c:v libx264 -pix_fmt yuv420p \
  -y output.mp4
```

Python 等效（推荐，更灵活控制帧范围）：

```python
import cv2

def frames_to_mp4(frame_paths, output_path, fps=1):
    img = cv2.imread(frame_paths[0])
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )
    for p in frame_paths:
        writer.write(cv2.imread(p))
    writer.release()
```

### 3.4 ID 命名规范

```
NAV_ID_SATNAV_{episode_id:06d}_w{window_idx:02d}
```

示例：`NAV_ID_SATNAV_000000_w00`, `NAV_ID_SATNAV_000000_w01`, ...

### 3.5 Prompt Template

固定使用以下模板（与 Uni-NaVid 原始训练一致）：

```
Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation. Your assigned task is: '{instruction}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop.
```

`{instruction}` 取自 SatNav `episode["instructions"][0]`。

---

## 4. 数据统计预估

基于 SatNav ver_260306 数据：

| 指标 | 值 |
|------|-----|
| 原始 episode 数 | 15,689 |
| 平均 steps/episode | 59.6 |
| 平均 windows/episode | ~15（= ceil(59.6 / 4)） |
| **预估总训练样本数** | **~235K** |
| 预估视频文件数 | ~235K 个 mp4 |
| 预估单个 mp4 大小 | ~50KB ~ 5MB（取决于帧数） |
| 预估总视频存储 | ~100-200 GB |

---

## 5. 转换脚本接口设计

```
baseline/uninavid/src/convert_satnav_to_uninavid.py

用法:
  python convert_satnav_to_uninavid.py \
    --annotations "$SATNAV_DATA_ROOT/ver_260306/trajectory_data/annotations.json" \
    --image-root "$SATNAV_DATA_ROOT/ver_260306/trajectory_data" \
    --output-dir "$UNINAVID_DATA_ROOT/smoke" \
    [--max-episodes N]        # 限制转换 episode 数（用于 smoke test）
    [--window-size 4]         # 动作窗口大小（默认 4）
    [--video-fps 1]           # 合成视频帧率（默认 1）
    [--workers 8]             # 并行视频合成线程数
```

**输出**：
- `{output_dir}/nav_videos/*.mp4`
- `{output_dir}/satnav_uninavid_train.json`
- `{output_dir}/conversion_stats.json`（转换统计信息）
