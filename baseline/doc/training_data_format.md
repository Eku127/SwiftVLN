# SwiftVLN 训练数据格式与结构文档

> 本文档描述 SwiftVLN 项目中训练数据的完整格式与目录结构，供后续 agent 和开发者参考。

---

## 1. 数据集概览

SwiftVLN 支持两类 VLN（Vision-and-Language Navigation）环境的训练数据：

| 环境类型 | 名称 | 场景类型 | 前进步长 | 转弯角度 | 图像分辨率 |
|---------|------|---------|---------|---------|-----------|
| SatNav  | 卫星导航 | 2D 卫星地图（鸟瞰图） | 10m | 15° | 448×448 |
| Habitat | 室内导航 | 3D 室内场景 (MP3D) | 0.25m | 15° | 640×480 |

两类环境共享相同的 **annotations.json + RGB 帧** 核心数据格式，但在具体路径、图像尺寸和动作语义上有所不同。

---

## 2. 目录结构

### 2.1 SatNav 数据集

数据集根目录：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/`

```
satnav_datasets/
├── scenes/                          # 卫星地图 .tif 文件（评测时使用）
│   ├── Geneva-1.tif
│   ├── Berlin-1.tif
│   └── ...
└── ver_260306/                      # 数据版本目录（当前默认版本）
    ├── trajectory_data/             # ★ VLN 训练核心数据
    │   ├── annotations.json         # 所有 episode 的标注文件
    │   └── images/                  # 轨迹帧图像
    │       ├── Geneva-1_satnav_000000/
    │       │   └── rgb/
    │       │       ├── 001.jpg      # 448×448 卫星鸟瞰裁剪图
    │       │       ├── 002.jpg
    │       │       └── ...          # 帧数 = actions 序列长度
    │       ├── Geneva-1_satnav_000001/
    │       └── ...
    ├── data/                        # 原始数据（按城市组织，评测时使用）
    │   ├── Geneva-1/
    │   │   └── VLN_episodes.json    # 评测用 episode 定义
    │   ├── Berlin-1/
    │   └── ...
    └── episodes/                    # 处理后的 episode 文件（评测时使用）
        ├── train/
        │   ├── all_episodes.json
        │   ├── boundary_episodes.json
        │   ├── landmark_episodes.json
        │   └── road_episodes.json
        └── eval/
            ├── all_episodes.json
            ├── boundary_episodes.json
            ├── landmark_episodes.json
            └── road_episodes.json
```

### 2.2 Habitat (R2R) 数据集

数据集根目录：`/mnt/data3/jiangjiajun/dataset/streamvln_datasets/`

```
streamvln_datasets/
└── trajectory_data/
    ├── R2R/                          # Room-to-Room 数据集
    │   ├── annotations.json          # 所有 episode 的标注文件
    │   └── images/                   # 轨迹帧图像
    │       ├── 17DRP5sb8fy_r2r_001803/
    │       │   └── rgb/
    │       │       ├── 001.jpg       # 640×480 室内第一人称视角
    │       │       ├── 002.jpg
    │       │       └── ...
    │       └── ...
    ├── RxR/                          # RxR 数据集（同构）
    ├── EnvDrop/                      # EnvDrop 数据集（同构）
    └── ScaleVLN/                     # ScaleVLN 数据集（同构）
```

### 2.3 城市划分（SatNav 专属）

SatNav 城市按用途划分，训练与评测城市严格分离：

| 用途 | 城市列表 | 数量 |
|------|---------|------|
| 训练 | Geneva-1, TheBayArea-1, TheBayArea-2, Minneapolis-1, Minneapolis-2, Paris-1, Rome-1, Boston-1, NewYork-1 | 9 个 |
| 评测 | Berlin-1, LosAngeles-1 | 2 个 |

---

## 3. 核心数据格式

### 3.1 annotations.json（VLN 轨迹标注）

这是训练时直接加载的核心文件，格式为 **JSON 数组**，每个元素代表一条 episode：

```json
[
  {
    "id": 0,
    "trajectory_id": 0,
    "steps": 70,
    "video": "images/Geneva-1_satnav_000000",
    "instructions": [
      "Begin at the storage yard by the roadside; orbit counter-clockwise around the adjoining sports-field complex, keeping the turf on your left. Stop at the yard."
    ],
    "actions": [-1, 1, 1, 1, 2, 1, 1, 2, 1, ..., 0]
  },
  ...
]
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | Episode 全局唯一 ID |
| `trajectory_id` | int | 轨迹 ID（同一条轨迹路径可对应多条不同指令的 episode） |
| `steps` | int | 轨迹步数（不含初始帧，即 `len(actions) - 1`） |
| `video` | string | 相对于 annotations.json 所在目录的帧图像文件夹路径 |
| `instructions` | list[string] | 自然语言导航指令列表（SatNav 通常 1 条，R2R 通常 3 条） |
| `actions` | list[int] | 动作序列，长度 = `steps + 1` |

### 3.2 动作编码

| 编码 | 动作 | 符号 | 说明 |
|------|------|------|------|
| `-1` | INITIAL | — | 初始帧标记，**固定出现在 `actions[0]`**，不对应任何实际动作 |
| `0` | STOP | STOP | 停止，到达目标时触发 |
| `1` | MOVE_FORWARD | ↑ | 前进（SatNav: 10m；Habitat: 0.25m） |
| `2` | TURN_LEFT | ← | 左转（统一 15°） |
| `3` | TURN_RIGHT | → | 右转（统一 15°） |

**重要约定：**

- `actions[0]` 固定为 `-1`（INITIAL），仅标记起点，不执行任何动作
- `actions[-1]` 通常为 `0`（STOP），表示导航终止
- **帧与动作对齐关系**：`rgb/001.jpg` 对应 `actions[0]` 时刻的观察，`rgb/002.jpg` 对应 `actions[1]` 时刻的观察，以此类推
- 帧文件总数 = `len(actions)` = `steps + 1`

### 3.3 RGB 帧图像

每个 episode 的帧图像存储在 `{video}/rgb/` 目录下：

```
{video}/rgb/
├── 001.jpg    # 第 1 帧（初始位置观察，对应 actions[0] = -1）
├── 002.jpg    # 第 2 帧（执行 actions[1] 后的新观察）
├── 003.jpg
└── ...
```

- **命名规则**：三位数字编号，从 `001` 开始，格式为 `%03d.jpg`
- **SatNav 图像**：448×448 像素，以当前朝向为正前方裁剪的卫星鸟瞰图
- **Habitat 图像**：640×480 像素，室内第一人称 RGB 视角

### 3.4 VLN_episodes.json（评测用 Episode 定义）

每个城市目录下的 `VLN_episodes.json` 包含基于真实地理坐标的评测 episode 定义：

```json
{
  "episodes": [
    {
      "episode_id": 0,
      "trajectory_id": 0,
      "trajectory_type": "Boundary",
      "scene_id": "Geneva-1",
      "start_position": [6.104124787582547, 46.171727042526335, 50],
      "start_rotation": 54.75357977128277,
      "goals": [{"position": [6.104124787582547, 46.171727042526335, 50]}],
      "waypoints": [[6.104124, 46.171727, 50], ...],
      "reference_path": [[6.104124, 46.171727, 50], ...],
      "instruction": {
        "instruction_text": "Begin at the storage yard...",
        "instruction_type": "..."
      },
      "aux_info": {
        "boundary_id": "3",
        "direction": "ccw",
        "length": 685.92,
        "label": "leisure,pitch"
      }
    }
  ]
}
```

**坐标说明**：`[经度, 纬度, 海拔(固定50)]`，使用 WGS84 地理坐标系。

**轨迹类型与成功判定阈值：**

| trajectory_type | 说明 | 成功距离阈值 |
|----------------|------|-------------|
| `Boundary` | 沿区域边界行走 | 10m |
| `LandmarkSet` | 地标间导航 | 30m |
| `Road` | 沿道路行走 | 10m |

处理后的 episodes 按 split（`train` / `eval`）及类型分别存入 `episodes/` 目录，供评测时使用。

---

## 4. 数据统计（ver_260306）

### 4.1 训练数据（trajectory_data）

| 指标 | 值 |
|------|-----|
| 总 episode 数 | 15,689 |
| 总动作数 | 934,744 |
| 平均 episode 长度 | 59.6 步 |
| 覆盖城市数 | 9（训练城市） |
| 每条 episode 指令数 | 1 |

各城市 episode 分布：

| 城市 | Episode 数 |
|------|-----------|
| Rome-1 | 2,580 |
| TheBayArea-1 | 2,530 |
| Minneapolis-1 | 2,402 |
| TheBayArea-2 | 1,628 |
| Minneapolis-2 | 1,662 |
| Paris-1 | 1,625 |
| NewYork-1 | 1,365 |
| Geneva-1 | 1,108 |
| Boston-1 | 789 |

### 4.2 评测数据（episodes/eval）

| 指标 | 值 |
|------|-----|
| 总评测 episode 数 | 2,728 |
| — Boundary | 504 |
| — LandmarkSet | 658 |
| — Road | 1,566 |
| 评测城市 | Berlin-1（1,675），LosAngeles-1（1,053） |

### 4.3 图像规格对比

| 数据集 | 分辨率 | 视角类型 | 格式 |
|--------|--------|---------|------|
| SatNav 轨迹帧 | 448×448 | 卫星鸟瞰（以朝向为正前方裁剪） | JPEG RGB |
| Habitat (R2R) 轨迹帧 | 640×480 | 室内第一人称 | JPEG RGB |

---

## 5. 数据路径速查

| 数据 | 绝对路径 |
|------|---------|
| SatNav 数据集根 | `/mnt/data3/jiangjiajun/dataset/satnav_datasets` |
| 当前默认版本 | `ver_260306` |
| **VLN 训练标注** | `.../ver_260306/trajectory_data/annotations.json` |
| **VLN 训练图像** | `.../ver_260306/trajectory_data/images/{episode_id}/rgb/` |
| 评测 episodes（全量） | `.../ver_260306/episodes/eval/all_episodes.json` |
| 卫星地图 | `.../scenes/{city}.tif` |
| **Habitat R2R 训练数据** | `/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R/` |
| Habitat VLN-CE 评测数据 | `/mnt/data3/jiangjiajun/dataset/vlnce_datasets/R2R_VLNCE_v1-3_preprocessed/` |
