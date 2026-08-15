# Habitat 训练数据准备

本文的数据生产方法参考
[StreamVLN](https://github.com/InternRobotics/StreamVLN)：在 Matterport3D（MP3D）场景中
运行 Habitat，使用 `ShortestPathFollower` 沿 VLN-CE Episode 的 `reference_path` 生成离散
动作与 RGB observation。实际生产使用 SwiftVLN 自带的 trajectory generation CLI，不需要
安装 StreamVLN。

> R2R 固定使用 `R2R_VLNCE_v1-3`。不要使用 `R2R_VLNCE_v1`，也不要混用旧版 R2R
> trajectory annotation。

本文生产以下三组训练数据：

| 数据集 | VLN-CE Episode | 训练轨迹数 |
| --- | --- | ---: |
| R2R | `R2R_VLNCE_v1-3/train` | 10,819 |
| RxR | `RxR_VLNCE_v0/train_guide` 中的英语 Episode | 19,990 |
| EnvDrop | `R2R_VLNCE_v1-3_preprocessed/envdrop` | 146,304 |

## 1. 准备生成环境

完成[环境安装](../getting-started/INSTALLATION.md)后，进入 Habitat 评测环境：

```bash
cd "${SWIFTVLN_ROOT}"
conda activate swiftvln-eval

swiftvln generate-habitat-trajectories --help
swiftvln validate-habitat-trajectories --help
```

该 CLI 使用 SwiftVLN 的 Habitat 配置，生成参数为：

| 参数 | 值 |
| --- | --- |
| RGB 分辨率 | `640 × 480` |
| 前进步长 | `0.25 m` |
| 转向角 | `15°` |
| Expert | Habitat `ShortestPathFollower` |
| 最长轨迹 | 498 个 observation-action pair |

`swiftvln generate-habitat-trajectories` 默认使用单进程，适合单 Episode smoke。完整 R2R、
RxR 和 EnvDrop 应使用 `torchrun` 并行生成；尤其是 EnvDrop 包含 146,304 个 Episode，并会
产生约千万张 RGB，单进程生产会持续很久。

## 2. 准备 MP3D 场景

按照 [Matterport3D 官方说明](https://niessner.github.io/Matterport/)申请并下载 Habitat
场景文件，将 90 个场景保存至：

```text
/path/to/streamvln_datasets/scene_datasets/mp3d/
```

场景目录格式如下：

```text
scene_datasets/mp3d/
├── 17DRP5sb8fy/
│   ├── 17DRP5sb8fy.glb
│   └── 17DRP5sb8fy.navmesh
├── 1LXtFkjw3qL/
└── ...
```

## 3. 准备 VLN-CE Episode

安装下载工具并创建数据目录：

```bash
conda activate swiftvln-eval
python -m pip install gdown

mkdir -p /path/to/streamvln_datasets/datasets
cd /path/to/streamvln_datasets/datasets
```

### 3.1 R2R v1-3

从 [VLN-CE 数据页](https://jacobkrantz.github.io/vlnce/data)下载
`R2R_VLNCE_v1-3.zip`：

```bash
gdown --id 1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
unzip R2R_VLNCE_v1-3.zip
mv R2R_VLNCE_v1-3 r2r
```

训练 Episode 路径为：

```text
datasets/r2r/train/train.json.gz
```

### 3.2 RxR v0

下载 `RxR_VLNCE_v0.zip`：

```bash
gdown --id 145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t
unzip RxR_VLNCE_v0.zip
mv RxR_VLNCE_v0 rxr
```

下载包提供 multilingual guide train split：

```text
datasets/rxr/train/train_guide.json.gz
```

生成时使用 `--instruction-language en` 选取其中 19,996 个英语 Episode。生成器按照
StreamVLN 的数据规则移除动作数超过 498 的 6 条轨迹，最终保留 19,990 条训练轨迹。

### 3.3 EnvDrop

下载 `R2R_VLNCE_v1-3_preprocessed.zip`：

```bash
gdown --id 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr
unzip R2R_VLNCE_v1-3_preprocessed.zip
mv R2R_VLNCE_v1-3_preprocessed/envdrop envdrop
```

训练 Episode 路径为：

```text
datasets/envdrop/envdrop.json.gz
```

## 4. 生成 R2R 轨迹

先生成一个 Episode，检查 Habitat、MP3D 和 VLN-CE Episode 是否能够正常加载：

```bash
swiftvln generate-habitat-trajectories \
  --dataset r2r \
  --data-path /path/to/streamvln_datasets/datasets/r2r/train/train.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/smoke/R2R \
  --episode-id 1
```

完整生产使用一张 GPU 对应一个进程。以下命令使用 8 张 GPU；使用其他 GPU 数量时同步修改
`--nproc_per_node`：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset r2r \
  --data-path /path/to/streamvln_datasets/datasets/r2r/train/train.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/R2R
```

每个 rank 创建独立的 Habitat Env，并按照 scene 分组后使用
`episodes[rank::world_size]` 分配 Episode。各 rank 完成后，rank 0 自动合并
`annotations_<rank>.json`，生成训练入口 `annotations.json`。

## 5. 生成 RxR 轨迹

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset rxr \
  --data-path /path/to/streamvln_datasets/datasets/rxr/train/train_guide.json.gz \
  --instruction-language en \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/RxR
```

## 6. 生成 EnvDrop 轨迹

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset envdrop \
  --data-path /path/to/streamvln_datasets/datasets/envdrop/envdrop.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/EnvDrop
```

任务中断后，使用相同的进程数和输出目录，并增加 `--resume`。生成器会读取每个 rank 的
annotation shard 或 journal，跳过已经完成的 Episode：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset envdrop \
  --data-path /path/to/streamvln_datasets/datasets/envdrop/envdrop.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/EnvDrop \
  --resume
```

## 7. 轨迹格式

每个数据集的输出目录包含：

```text
trajectory_data/R2R/
├── annotations.json
├── annotations_0.json
├── annotations_1.json
├── ...
├── generation_summary.json
├── generation_summary_0.json
├── generation_summary_1.json
└── images/
    └── 1LXtFkjw3qL_r2r_000087/
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

单条 annotation 格式如下：

```json
{
  "id": 87,
  "video": "images/1LXtFkjw3qL_r2r_000087",
  "instructions": ["Walk past the table and stop near the doorway."],
  "actions": [-1, 1, 1, 2, 1]
}
```

| 字段 | 说明 |
| --- | --- |
| `id` | VLN-CE Episode ID |
| `video` | Episode 图像目录相对于当前 trajectory 根目录的路径 |
| `instructions` | 导航指令列表 |
| `actions` | 与 RGB 帧一一对应的 Habitat 离散动作 |

动作编码如下：

| ID | Action |
| ---: | --- |
| -1 | `INIT` |
| 0 | `STOP`，原始 annotation 中省略 |
| 1 | `MOVE_FORWARD`，前进 0.25 m |
| 2 | `TURN_LEFT`，左转 15° |
| 3 | `TURN_RIGHT`，右转 15° |

每条轨迹满足：

```text
JPEG 数量 = len(actions)
```

SwiftVLN 训练时移除开头的 `INIT`，并在序列末尾补充 `STOP`。

## 8. 校验轨迹

使用 SwiftVLN 校验器检查 annotation 字段、Episode ID、动作编码、图像目录、孤立目录以及
逐轨迹的 RGB/action 数量：

```bash
swiftvln validate-habitat-trajectories \
  --data-dir /path/to/streamvln_datasets/trajectory_data/R2R \
  --expected-count 10819

swiftvln validate-habitat-trajectories \
  --data-dir /path/to/streamvln_datasets/trajectory_data/RxR \
  --expected-count 19990

swiftvln validate-habitat-trajectories \
  --data-dir /path/to/streamvln_datasets/trajectory_data/EnvDrop \
  --expected-count 146304
```

增加 `--decode-images` 可以逐张解码 JPEG；完整数据解码会显著增加校验时间。

## 9. 整理目录并配置 SwiftVLN

完成生成后，目录结构如下：

```text
streamvln_datasets/
├── datasets/
│   ├── r2r/train/train.json.gz
│   ├── rxr/train/train_guide.json.gz
│   └── envdrop/envdrop.json.gz
├── scene_datasets/
│   └── mp3d/<scene-id>/<scene-id>.glb
└── trajectory_data/
    ├── R2R/
    │   ├── annotations.json
    │   └── images/<episode>/rgb/*.jpg
    ├── RxR/
    │   ├── annotations.json
    │   └── images/<episode>/rgb/*.jpg
    └── EnvDrop/
        ├── annotations.json
        └── images/<episode>/rgb/*.jpg
```

在 `${SWIFTVLN_ROOT}/.local/env.sh` 中设置训练路径：

```bash
export SWIFTVLN_HABITAT_DATA_ROOT="/path/to/streamvln_datasets"
export SWIFTVLN_HABITAT_R2R_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/R2R"
export SWIFTVLN_HABITAT_RXR_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/RxR"
export SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/EnvDrop"
```

训练时通过 `VLN_DATA_PATH` 选择数据组合：

```bash
export VLN_ENV_TYPE=habitat
export VLN_DATA_PATH="${SWIFTVLN_HABITAT_R2R_TRAIN_PATH},${SWIFTVLN_HABITAT_RXR_TRAIN_PATH},${SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH}"
```

数据配置完成后，进入 [SwiftVLN 训练](../training/README.md)。
