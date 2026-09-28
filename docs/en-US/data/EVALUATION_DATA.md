# Evaluation Data

[简体中文](../../zh-CN/data/EVALUATION_DATA.md) | English


## 1. Reuse prepared training data

Complete the [SatNav training data](TRAINING_DATA_SATNAV.md)or [After the Habitat training data](TRAINING_DATA_HABITAT.md)is prepared, the following resources can be used directly for evaluation:

| Environment | Evaluation reuse resources | Not required for evaluation |
| --- | --- | --- |
| SatNav | SatNav-v0.1 `val_seen`/`val_unseen` Episode, GeoTIFF scene |`trajectory_data/`|
| Habitat | R2R VLN-CE v1-3 `val_seen`/`val_unseen` Episode, MP3D scene | R2R / RxR / EnvDrop offline trajectory |

At this time, you only need to add the evaluation path in `${SWIFTVLN_ROOT}/.local/env.sh`:

```bash
# SatNav
export SWIFTVLN_SATNAV_EVAL_ROOT="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/episodes/eval"
export SWIFTVLN_SATNAV_EVAL_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT}/{split}/all_episodes.json"
export SWIFTVLN_SATNAV_SCENES_DIR="${SWIFTVLN_SATNAV_DATA_ROOT}/scenes"

# Habitat
export SWIFTVLN_HABITAT_SCENES_DIR="${SWIFTVLN_HABITAT_DATA_ROOT}/scene_datasets"
export SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/datasets/r2r/{split}/{split}.json.gz"
```

If these resources are already configured, proceed directly to [SwiftVLN evaluation](../evaluation/README.md). The remaining sections are for users preparing evaluation data without the training datasets.

## 2. SatNav evaluation data

### 2.1 Download episodes

Follow SatNav’s [Episode data download](https://github.com/Eku127/SatNav/blob/master/docs/en-US/dataset/DATA_DOWNLOAD.md) Download and verify SatNav-Episodes-v0.1. Evaluation uses:

| Split | Episodes |
| --- | ---: |
| `val_seen` | 4,574 |
| `val_unseen` | 8,756 |

### 2.2 Prepare GeoTIFF scenes

Choose either option to prepare the 59 scenes:

- **Download prepared scenes:** complete the [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) access form, accept the terms, and download with the same account after your request passes the system checks.
- **Generate through an API:** register for an imagery service and configure your credentials using the [SatNav scene download guide](https://github.com/Eku127/SatNav/blob/master/docs/en-US/applications/MAP_DOWNLOAD.md).

To download prepared scenes:

```bash
pip install -U huggingface_hub
hf auth login
hf download Eku127/SatNav-Scenes-v0.1 --repo-type dataset \
  --include "scenes/*.tif" --include SHA256SUMS \
  --local-dir /path/to/satnav_datasets
```

The evaluation data directory is as follows:

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

### 2.3 Configure paths

```bash
export SWIFTVLN_SATNAV_DATA_ROOT="/path/to/satnav_datasets"
export SWIFTVLN_SATNAV_DATASET="SatNav-v0.1"
export SWIFTVLN_SATNAV_EVAL_ROOT="${SWIFTVLN_SATNAV_DATA_ROOT}/${SWIFTVLN_SATNAV_DATASET}/episodes/eval"
export SWIFTVLN_SATNAV_EVAL_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT}/{split}/all_episodes.json"
export SWIFTVLN_SATNAV_SCENES_DIR="${SWIFTVLN_SATNAV_DATA_ROOT}/scenes"
```

### 2.4 Validate the data

```bash
swiftvln validate-evaluation-data --env-type satnav
```

The command checks episode numbers, fields, duplicate IDs, scene references, and GeoTIFF files.

## 3. Habitat evaluation data

### 3.1 Prepare MP3D scenes

Follow [Matterport3D official instructions](https://niessner.github.io/Matterport/)to apply and download the Habitat scene, Save 90 scenes to:

```text
/path/to/streamvln_datasets/scene_datasets/mp3d/
```

Each scene contains `.glb` and `.navmesh` used by Habitat:

```text
scene_datasets/mp3d/
├── 17DRP5sb8fy/
│   ├── 17DRP5sb8fy.glb
│   └── 17DRP5sb8fy.navmesh
└── ...
```

### 3.2 Download R2R VLN-CE v1-3

Download from [VLN-CE data page](https://jacobkrantz.github.io/vlnce/data) `R2R_VLNCE_v1-3.zip`:

```bash
python -m pip install gdown
mkdir -p /path/to/streamvln_datasets/datasets
cd /path/to/streamvln_datasets/datasets

gdown --id 1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
unzip R2R_VLNCE_v1-3.zip
mv R2R_VLNCE_v1-3 r2r
```

Evaluation uses:

| Split | Episodes | Path |
| --- | ---: | --- |
| `val_seen` | 778 | `datasets/r2r/val_seen/val_seen.json.gz` |
| `val_unseen` | 1,839 | `datasets/r2r/val_unseen/val_unseen.json.gz` |

The complete catalog is as follows:

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

### 3.3 Configure paths

```bash
export SWIFTVLN_HABITAT_DATA_ROOT="/path/to/streamvln_datasets"
export SWIFTVLN_HABITAT_SCENES_DIR="${SWIFTVLN_HABITAT_DATA_ROOT}/scene_datasets"
export SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH="${SWIFTVLN_HABITAT_DATA_ROOT}/datasets/r2r/{split}/{split}.json.gz"
```

### 3.4 Validate the data

```bash
swiftvln validate-evaluation-data --env-type habitat
```

The command checks Episode numbers, fields, duplicate IDs, and MP3D `.glb` and `.navmesh` scene files.
