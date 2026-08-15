# SatNav 训练数据准备

SwiftVLN 使用 SatNav 生成的离线 expert trajectory 进行训练。数据准备流程为：

```text
Episode + GeoTIFF
        │
        ▼
SatNav Trajectory Generation
        │
        ├── annotations.json
        ├── summary.json
        └── images/<episode>/rgb/*.jpg
                    │
                    ▼
             SwiftVLN Training
```

## 1. 准备 SatNav

完成[环境安装](../getting-started/INSTALLATION.md)后，初始化 SatNav 子模块并安装数据生成组件：

```bash
cd "${SWIFTVLN_ROOT}"
git submodule update --init third_party/SatNav

conda activate swiftvln-train
cd third_party/SatNav
python -m pip install -e '.[applications]'
```

## 2. 下载 Episode

按照 SatNav 的
[Episode 数据下载](https://github.com/Eku127/SatNav/blob/4bd6652c875af00236a09730d76c4b391046e6a4/docs/zh-CN/dataset/DATA_DOWNLOAD.md)
下载 SatNav-Episodes-v0.1，并将解压目录保存为：

```text
/path/to/satnav_datasets/SatNav-v0.1
```

校验下载文件：

```bash
cd /path/to/satnav_datasets/SatNav-v0.1
sha256sum -c SHA256SUMS
```

各数据划分包含：

| Split | Episodes |
| --- | ---: |
| train | 105,164 |
| val_seen | 4,574 |
| val_unseen | 8,756 |

## 3. 准备 GeoTIFF 场景

SatNav-v0.1 包含场景范围，不包含卫星影像。按照 SatNav 的
[卫星场景下载](https://github.com/Eku127/SatNav/blob/4bd6652c875af00236a09730d76c4b391046e6a4/docs/zh-CN/applications/MAP_DOWNLOAD.md)
配置地图服务凭据，并使用允许下载、存储和模型训练的影像来源。

以下命令使用 Google Map Tiles API 批量生成 59 个场景。先运行 dry run：

```bash
cd "${SWIFTVLN_ROOT}/third_party/SatNav"
mkdir -p /path/to/satnav_datasets/scenes
export GOOGLE_MAPS_API_KEY=your-api-key

python -m applications.map_downloader google \
  --scene-config /path/to/satnav_datasets/SatNav-v0.1/scenes_list.yaml \
  --output-dir /path/to/satnav_datasets/scenes \
  --dry-run
```

确认场景范围和输出路径后开始下载：

```bash
python -m applications.map_downloader google \
  --scene-config /path/to/satnav_datasets/SatNav-v0.1/scenes_list.yaml \
  --output-dir /path/to/satnav_datasets/scenes
```

验证 Episode 和 GeoTIFF：

```bash
SATNAV_DATA_ROOT=/path/to/satnav_datasets/SatNav-v0.1 \
SATNAV_SCENES_DIR=/path/to/satnav_datasets/scenes \
bash scripts/validation/data_validation.sh
```

验证通过后输出：

```text
GeoTIFF scenes: 59/59
SatNav data configuration is complete.
```

## 4. 生成离线轨迹

先使用 SatNav 内置的两个示例 Episode 测试生成流程：

```bash
cd "${SWIFTVLN_ROOT}/third_party/SatNav"

python -m applications.trajectory_generation.generate \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/trajectory_generation_test
```

完整 train split 使用 SatNav 的生产配置和并行入口：

```bash
SATNAV_TRAIN_EPISODES_PATH=/path/to/satnav_datasets/SatNav-v0.1/episodes/train/all_episodes.json \
SATNAV_SCENES_DIR=/path/to/satnav_datasets/scenes \
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir /path/to/satnav_datasets/SatNav-v0.1/trajectory_data
```

完整数据约占 233 GB。生成器默认根据场景数量和 CPU 核数选择 worker；也可以通过
`--num_workers N` 指定并行数。任务中断后，使用相同输出目录重新执行命令即可继续。

详细参数见 SatNav 的
[轨迹数据生成](https://github.com/Eku127/SatNav/blob/4bd6652c875af00236a09730d76c4b391046e6a4/docs/zh-CN/applications/TRAJECTORY_GENERATION.md)。

## 5. 轨迹格式

生成目录包含：

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <scene_id>_satnav_<episode-index>/
        ├── .done
        ├── .annotation.json
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

`annotations.json` 是 SwiftVLN 的训练入口。单条记录如下：

```json
{
  "id": 0,
  "trajectory_id": "0",
  "steps": 3,
  "video": "images/Amsterdam-1_satnav_000000",
  "instructions": ["Continue along the road and stop at the junction."],
  "actions": [-1, 1, 1, 0]
}
```

| 字段 | 说明 |
| --- | --- |
| `id` | Episode 在源 JSON 列表中的索引 |
| `trajectory_id` | 源 Episode 的路线标识 |
| `steps` | 可执行动作数量，等于 `len(actions) - 1` |
| `video` | RGB 帧目录相对于 `trajectory_data/` 的路径 |
| `instructions` | 该轨迹对应的导航指令列表 |
| `actions` | 与 RGB observation 对齐的离散动作序列 |

动作编码如下：

| ID | Action |
| ---: | --- |
| -1 | `INIT` |
| 0 | `STOP` |
| 1 | `MOVE_FORWARD` |
| 2 | `TURN_LEFT` |
| 3 | `TURN_RIGHT` |

每条轨迹满足：

```text
JPEG 数量 = len(actions) = steps + 1
```

完整字段定义见 SatNav 的
[数据格式](https://github.com/Eku127/SatNav/blob/4bd6652c875af00236a09730d76c4b391046e6a4/docs/zh-CN/dataset/DATASET_FORMAT.md#6-离线-trajectory)。

## 6. 校验完整数据

使用 SatNav 校验器检查 annotation、Episode、生成配置、场景和全部 JPEG：

```bash
cd "${SWIFTVLN_ROOT}/third_party/SatNav"
mkdir -p /path/to/satnav_datasets/validation

python scripts/validation/validate_trajectory_output.py \
  --annotations /path/to/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json \
  --output-root /path/to/satnav_datasets/SatNav-v0.1/trajectory_data \
  --source-episodes /path/to/satnav_datasets/SatNav-v0.1/episodes/train/all_episodes.json \
  --generation-config applications/episode_processing/configs/trajectory_generation.yaml \
  --scenes-dir /path/to/satnav_datasets/scenes \
  --expected-count 105164 \
  --decode-images \
  --report /path/to/satnav_datasets/validation/trajectory_data.json
```

完整 train split 的生成统计应为：

```text
Success (incl. cached): 105164
Discarded (max steps): 0
Failed: 0
Generated annotations: 105164 / 105164 episodes
```

## 7. 整理目录并配置 SwiftVLN

完成 Episode、GeoTIFF 和离线轨迹准备后，确认目录结构如下：

```text
satnav_datasets/
├── SatNav-v0.1/
│   ├── episodes/
│   │   ├── train/all_episodes.json
│   │   └── eval/
│   │       ├── val_seen/all_episodes.json
│   │       └── val_unseen/all_episodes.json
│   ├── scenes_list.yaml
│   ├── SHA256SUMS
│   └── trajectory_data/
│       ├── annotations.json
│       ├── summary.json
│       └── images/
└── scenes/
    ├── Amsterdam-1.tif
    └── ...
```

在 `${SWIFTVLN_ROOT}/.local/env.sh` 中设置数据路径：

```bash
export SWIFTVLN_SATNAV_REPO="${SWIFTVLN_ROOT}/third_party/SatNav"
export SWIFTVLN_SATNAV_DATA_ROOT="/path/to/satnav_datasets"
export SWIFTVLN_SATNAV_DATASET="SatNav-v0.1"
export SWIFTVLN_SATNAV_TRAIN_DATA_PATH="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/trajectory_data"
export SWIFTVLN_SATNAV_TRAIN_EPISODES_PATH="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/episodes/train/all_episodes.json"
export SWIFTVLN_SATNAV_SCENES_DIR="${SWIFTVLN_SATNAV_DATA_ROOT}/scenes"
```

不同训练模式使用的数据如下：

| 训练模式 | 必需数据 |
| --- | --- |
| `MEMORY_METHOD=history` | `annotations.json`、`images/` |
| `MEMORY_METHOD=map` | `annotations.json`、`images/`、`summary.json`、train Episode、`scenes/` |

`MEMORY_METHOD=map` 按上述完整目录结构解析 Episode 和 GeoTIFF。数据配置完成后，进入
[SwiftVLN 训练](../training/README.md)。
