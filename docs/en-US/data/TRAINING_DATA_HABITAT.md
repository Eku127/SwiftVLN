# Habitat Training Data

[简体中文](../../zh-CN/data/TRAINING_DATA_HABITAT.md) | English

This data-generation workflow follows [StreamVLN](https://github.com/InternRobotics/StreamVLN). It runs Habitat in Matterport3D (MP3D) scenes and uses `ShortestPathFollower` to generate discrete actions and RGB observations along each VLN-CE episode's `reference_path`. Generation is handled by SwiftVLN's own trajectory CLI; the StreamVLN repository is not required.

> R2R uses `R2R_VLNCE_v1-3`. Do not use `R2R_VLNCE_v1` or mix trajectories generated from older R2R annotations.

This article produces the following three sets of training data:

| Dataset | VLN-CE Episode | Number of training trajectories |
| --- | --- | ---: |
| R2R | `R2R_VLNCE_v1-3/train` | 10,819 |
| RxR | English Episode in `RxR_VLNCE_v0/train_guide`| 19,990 |
| EnvDrop | `R2R_VLNCE_v1-3_preprocessed/envdrop` | 146,304 |

## 1. Prepare the generation environment

After completing [environment installation](../getting-started/INSTALLATION.md), enter the Habitat evaluation environment:

```bash
cd "${SWIFTVLN_ROOT}"
conda activate swiftvln-eval

swiftvln generate-habitat-trajectories --help
swiftvln validate-habitat-trajectories --help
```

The CLI uses SwiftVLN’s Habitat configuration, and the generated parameters are:

| Parameter | Value |
| --- | --- |
| RGB Resolution |`640 × 480`|
| Forward step |`0.25 m`|
| Turn angle |`15°`|
| Expert | Habitat `ShortestPathFollower` |
| Longest trajectory | 498 observation-action pairs |

`swiftvln generate-habitat-trajectories` uses one process by default and is suitable for a one-episode smoke test. Generate complete R2R, RxR, and EnvDrop datasets in parallel with `torchrun`. EnvDrop alone contains 146,304 episodes and produces roughly ten million RGB images, so single-process generation takes a long time.

## 2. Prepare MP3D scenes

Follow [Matterport3D official instructions](https://niessner.github.io/Matterport/)to apply and download Habitat Scene files, save 90 scenes to:

```text
/path/to/streamvln_datasets/scene_datasets/mp3d/
```

The scene directory format is as follows:

```text
scene_datasets/mp3d/
├── 17DRP5sb8fy/
│   ├── 17DRP5sb8fy.glb
│   └── 17DRP5sb8fy.navmesh
├── 1LXtFkjw3qL/
└── ...
```

## 3. Prepare VLN-CE episodes

Install the download tool and create the data directory:

```bash
conda activate swiftvln-eval
python -m pip install gdown

mkdir -p /path/to/streamvln_datasets/datasets
cd /path/to/streamvln_datasets/datasets
```

### 3.1 R2R v1-3

Download from [VLN-CE data page](https://jacobkrantz.github.io/vlnce/data) `R2R_VLNCE_v1-3.zip`:

```bash
gdown --id 1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
unzip R2R_VLNCE_v1-3.zip
mv R2R_VLNCE_v1-3 r2r
```

The training Episode path is:

```text
datasets/r2r/train/train.json.gz
```

### 3.2 RxR v0

Download `RxR_VLNCE_v0.zip`:

```bash
gdown --id 145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t
unzip RxR_VLNCE_v0.zip
mv RxR_VLNCE_v0 rxr
```

The archive provides the multilingual guide training split:

```text
datasets/rxr/train/train_guide.json.gz
```

Use `--instruction-language en` to select the 19,996 English episodes. Following the StreamVLN data rule, the generator removes six trajectories with more than 498 actions and retains 19,990 training trajectories.

### 3.3 EnvDrop

Download `R2R_VLNCE_v1-3_preprocessed.zip`:

```bash
gdown --id 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr
unzip R2R_VLNCE_v1-3_preprocessed.zip
mv R2R_VLNCE_v1-3_preprocessed/envdrop envdrop
```

The training Episode path is:

```text
datasets/envdrop/envdrop.json.gz
```

## 4. Generate R2R trajectories

First generate an episode and check whether Habitat, MP3D and VLN-CE episodes can be loaded normally:

```bash
swiftvln generate-habitat-trajectories \
  --dataset r2r \
  --data-path /path/to/streamvln_datasets/datasets/r2r/train/train.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/smoke/R2R \
  --episode-id 1
```

Full generation uses one GPU per process. The following command uses eight GPUs; set `--nproc_per_node` to the number of GPUs you want to use:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset r2r \
  --data-path /path/to/streamvln_datasets/datasets/r2r/train/train.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/R2R
```

Each rank creates an independent Habitat environment. Episodes are grouped by scene and assigned with `episodes[rank::world_size]`. After all ranks finish, rank 0 merges `annotations_<rank>.json` into the training entry point, `annotations.json`.

## 5. Generate RxR trajectories

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset rxr \
  --data-path /path/to/streamvln_datasets/datasets/rxr/train/train_guide.json.gz \
  --instruction-language en \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/RxR
```

## 6. Generate EnvDrop trajectories

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset envdrop \
  --data-path /path/to/streamvln_datasets/datasets/envdrop/envdrop.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/EnvDrop
```

After the task is interrupted, use the same number of processes and output directory, and increase `--resume`. The generator reads each rank's annotation shard or journal, skipping completed Episodes:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m swiftvln.data.habitat_trajectory \
  --dataset envdrop \
  --data-path /path/to/streamvln_datasets/datasets/envdrop/envdrop.json.gz \
  --scenes-dir /path/to/streamvln_datasets/scene_datasets \
  --output-dir /path/to/streamvln_datasets/trajectory_data/EnvDrop \
  --resume
```

## 7. Trajectory format

The output directory for each dataset contains:

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

The format of a single annotation is as follows:

```json
{
  "id": 87,
  "video": "images/1LXtFkjw3qL_r2r_000087",
  "instructions": ["Walk past the table and stop near the doorway."],
  "actions": [-1, 1, 1, 2, 1]
}
```

| Field | Description |
| --- | --- |
| `id` | VLN-CE Episode ID |
|`video`| The path of the Episode image directory relative to the current trajectory root directory |
|`instructions`| Navigation command list |
|`actions`| Habitat discrete actions corresponding one-to-one to RGB frames |

The action is coded as follows:

| ID | Action |
| ---: | --- |
| -1 | `INIT` |
| 0 |`STOP`, omitted from the original annotation |
| 1 |`MOVE_FORWARD`, forward 0.25 m |
| 2 |`TURN_LEFT`, turn left 15° |
| 3 |`TURN_RIGHT`, turn right 15° |

Each trajectory satisfies:

```text
JPEG count = len(actions)
```

During SwiftVLN training, `INIT` at the beginning is removed and `STOP` is added at the end of the sequence.

## 8. Validate trajectories

Use SwiftVLN validators to check annotation fields, Episode IDs, action encodings, image directories, orphan directories, and Number of RGB/actions per trajectory:

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

Add `--decode-images` to decode JPEGs frame by frame; full data decoding will significantly increase the checksum time.

## 9. Organize directories and configure SwiftVLN

After the generation is completed, the directory structure is as follows:

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

Set the training path in `${SWIFTVLN_ROOT}/.local/env.sh`:

```bash
export SWIFTVLN_HABITAT_DATA_ROOT="/path/to/streamvln_datasets"
export SWIFTVLN_HABITAT_R2R_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/R2R"
export SWIFTVLN_HABITAT_RXR_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/RxR"
export SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/trajectory_data/EnvDrop"
```

Use `VLN_DATA_PATH` to select the data combination during training:

```bash
export VLN_ENV_TYPE=habitat
export VLN_DATA_PATH="${SWIFTVLN_HABITAT_R2R_TRAIN_PATH},${SWIFTVLN_HABITAT_RXR_TRAIN_PATH},${SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH}"
```

After the data configuration is completed, enter [SwiftVLN training](../training/README.md).
