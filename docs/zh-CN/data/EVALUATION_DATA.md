# 评测数据准备

简体中文 | [English](../../en-US/data/EVALUATION_DATA.md)


## 1. 复用训练数据

完成对应环境的 [SatNav 训练数据](TRAINING_DATA_SATNAV.md)或
[Habitat 训练数据](TRAINING_DATA_HABITAT.md)准备后，以下资源可以直接用于评测：

| 环境 | 评测复用资源 | 评测不需要 |
| --- | --- | --- |
| SatNav | SatNav-v0.1 `val_seen` / `val_unseen` Episode、GeoTIFF 场景 | `trajectory_data/` |
| Habitat | R2R VLN-CE v1-3 `val_seen` / `val_unseen` Episode、MP3D 场景 | R2R / RxR / EnvDrop 离线轨迹 |

此时只需在 `${SWIFTVLN_ROOT}/.local/env.sh` 中补充评测路径：

```bash
# SatNav
export SWIFTVLN_SATNAV_EVAL_ROOT="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/episodes/eval"
export SWIFTVLN_SATNAV_EVAL_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT}/{split}/all_episodes.json"
export SWIFTVLN_SATNAV_SCENES_DIR="${SWIFTVLN_SATNAV_DATA_ROOT}/scenes"

# Habitat
export SWIFTVLN_HABITAT_SCENES_DIR="${SWIFTVLN_HABITAT_DATA_ROOT}/scene_datasets"
export SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/datasets/r2r/{split}/{split}.json.gz"
```

完成上述配置后，无需继续阅读本章，直接进入 [SwiftVLN 评测](../evaluation/README.md)。
以下章节面向仅准备评测数据的研究者。

## 2. SatNav 评测数据

### 2.1 下载 Episode

按照 SatNav 的
[Episode 数据下载](https://github.com/Eku127/SatNav/blob/master/docs/zh-CN/dataset/DATA_DOWNLOAD.md)
下载并校验 SatNav-Episodes-v0.1。评测使用：

| Split | Episodes |
| --- | ---: |
| `val_seen` | 4,574 |
| `val_unseen` | 8,756 |

### 2.2 准备 GeoTIFF 场景

选择以下任一种方式准备 59 个场景：

- **下载现成场景：** 在 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) 填写申请表并同意条款，申请通过系统检查后使用同一账号下载。
- **通过 API 生成：** 注册地图服务并配置自己的凭据，按照 [SatNav 卫星场景下载](https://github.com/Eku127/SatNav/blob/master/docs/zh-CN/applications/MAP_DOWNLOAD.md)运行脚本。

下载现成场景的命令：

```bash
pip install -U huggingface_hub
hf auth login
hf download Eku127/SatNav-Scenes-v0.1 --repo-type dataset \
  --include "scenes/*.tif" --include SHA256SUMS \
  --local-dir /path/to/satnav_datasets
```

评测数据目录如下：

```text
satnav_datasets/
├── SatNav-v0.1/
│   ├── episodes/eval/
│   │   ├── val_seen/all_episodes.json
│   │   └── val_unseen/all_episodes.json
│   ├── scenes_list.yaml
│   └── SHA256SUMS
└── scenes/
    ├── Amsterdam-1.tif
    └── ...
```

### 2.3 配置路径

```bash
export SWIFTVLN_SATNAV_DATA_ROOT="/path/to/satnav_datasets"
export SWIFTVLN_SATNAV_DATASET="SatNav-v0.1"
export SWIFTVLN_SATNAV_EVAL_ROOT="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/episodes/eval"
export SWIFTVLN_SATNAV_EVAL_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT}/{split}/all_episodes.json"
export SWIFTVLN_SATNAV_SCENES_DIR="${SWIFTVLN_SATNAV_DATA_ROOT}/scenes"
```

### 2.4 校验数据

```bash
swiftvln validate-evaluation-data --env-type satnav
```

命令检查 Episode 数量、字段、重复 ID、场景引用和 GeoTIFF 文件。

## 3. Habitat 评测数据

### 3.1 准备 MP3D 场景

按照 [Matterport3D 官方说明](https://niessner.github.io/Matterport/)申请并下载 Habitat 场景，
将 90 个场景保存至：

```text
/path/to/streamvln_datasets/scene_datasets/mp3d/
```

每个场景包含 Habitat 使用的 `.glb` 和 `.navmesh`：

```text
scene_datasets/mp3d/
├── 17DRP5sb8fy/
│   ├── 17DRP5sb8fy.glb
│   └── 17DRP5sb8fy.navmesh
└── ...
```

### 3.2 下载 R2R VLN-CE v1-3

从 [VLN-CE 数据页](https://jacobkrantz.github.io/vlnce/data)下载
`R2R_VLNCE_v1-3.zip`：

```bash
python -m pip install gdown
mkdir -p /path/to/streamvln_datasets/datasets
cd /path/to/streamvln_datasets/datasets

gdown --id 1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
unzip R2R_VLNCE_v1-3.zip
mv R2R_VLNCE_v1-3 r2r
```

评测使用：

| Split | Episodes | 路径 |
| --- | ---: | --- |
| `val_seen` | 778 | `datasets/r2r/val_seen/val_seen.json.gz` |
| `val_unseen` | 1,839 | `datasets/r2r/val_unseen/val_unseen.json.gz` |

完整目录如下：

```text
streamvln_datasets/
├── datasets/r2r/
│   ├── val_seen/val_seen.json.gz
│   └── val_unseen/val_unseen.json.gz
└── scene_datasets/mp3d/
    └── <scene-id>/
        ├── <scene-id>.glb
        └── <scene-id>.navmesh
```

### 3.3 配置路径

```bash
export SWIFTVLN_HABITAT_DATA_ROOT="/path/to/streamvln_datasets"
export SWIFTVLN_HABITAT_SCENES_DIR="${SWIFTVLN_HABITAT_DATA_ROOT}/scene_datasets"
export SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/datasets/r2r/{split}/{split}.json.gz"
```

### 3.4 校验数据

```bash
swiftvln validate-evaluation-data --env-type habitat
```

命令检查 Episode 数量、字段、重复 ID，以及 MP3D `.glb` 与 `.navmesh` 场景文件。
