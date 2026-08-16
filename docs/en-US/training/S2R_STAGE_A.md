# Satellite-to-UAV Stage-A Training

[简体中文](../../zh-CN/training/S2R_STAGE_A.md) | English

Satellite-to-UAV Stage-A trains an adapter on paired UAV–satellite images. The Qwen2.5-VL vision tower remains frozen, while the adapter transforms UAV visual tokens to align them with the corresponding satellite features.

| Components | Role in Stage-A |
| --- | --- |
| Qwen2.5-VL vision tower | Encodes UAV and satellite images; frozen during training |
| Satellite-to-UAV adapter | Transforms UAV visual tokens; trainable |
| Projection head | Computes contrastive-learning and retrieval features; trainable |

The objective combines bidirectional UAV–satellite contrastive loss with global cosine-alignment loss. Retrieval evaluation uses the shared projection head to compute pairwise image similarity.

## 1. Prepare the environment and teacher

Follow [Installation](../getting-started/INSTALLATION.md) to create the `swiftvln-train` environment, then download the default SatNav model described in [Models and checkpoints](../getting-started/CHECKPOINTS.md).

```bash
cd /path/to/SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
source .local/env.sh
source "${SWIFTVLN_CONDA_SH}"
conda activate swiftvln-train

python -m pip install -e ".[s2r-data]"
```

The default experiment uses the trained SwiftVLN SatNav model as the frozen teacher:

```bash
export TEACHER_MODEL_PATH="${SWIFTVLN_SATNAV_MODEL_PATH}"
```

## 2. Prepare SatDronePair

Stage-A generates SatDronePair using the following four data sources:

- DenseUAV
- GTA-UAV
- SUES-200
- UAV-VisLoc

For upstream downloads, conversion commands, and image-quality checks, see [SatDronePair data generation](../data/SATDRONEPAIR.md). The converted directory is:

```text
SatDronePair/
├── denseuav/{drone,satellite,pairs.csv,dataset_info.json}
├── gta/{drone,satellite,pairs.csv,dataset_info.json}
├── sues/{drone,satellite,pairs.csv,dataset_info.json}
└── uavvisloc/{drone,satellite,pairs.csv,dataset_info.json}
```

## 3. Build the manifest

Set data and manifest path:

```bash
export PAIR_ROOT=/path/to/SatDronePair
export MANIFEST_PATH="${SWIFTVLN_ROOT}/runtime/s2r/manifests/manifest_v1.jsonl"
```

Build training and validation splits:

```bash
python -m tools.s2r.scripts.build_manifest \
  --data_root "${PAIR_ROOT}" \
  --output_path "${MANIFEST_PATH}" \
  --val_ratio 0.1 \
  --seed 42 \
  --skip_missing false
```

The manifest is split into `train` and `val` by location. Paired images from one location never span both splits:

| Data source | Split grouping |
| --- | --- |
| DenseUAV | Base location ID |
| GTA-UAV | Satellite tile |
| SUES-200 | Scene ID |
| UAV-VisLoc | Sequence ID |

The command reports the total number of manifest records and a per-source, per-split summary.

## 4. Train the Stage-A adapter

### 4.1 Default training configuration

| Parameters | Default value | Meaning |
| --- | --- | --- |
|`--batch_size`|`8`| Number of paired images per GPU |
|`--epochs`|`10`| Number of training epochs |
|`--learning_rate`|`1e-4`| Learning rate of Adapter and projection head |
| `--weight_decay` | `0.01` | AdamW weight decay |
|`--warmup_ratio`|`0.05`| Linear warmup as a proportion of the total number of steps |
|`--grad_accum_steps`|`1`| Number of gradient accumulation steps |
|`--temperature`|`0.07`| Bidirectional contrastive-loss temperature |
|`--adapter_layers`|`2`| Transformer adapter layers |
| `--adapter_heads` | `8` | Adapter attention heads |
| `--adapter_mlp_ratio` | `4.0` | Adapter MLP expansion ratio |
|`--projection_dim`|`512`| Retrieval projection dimension |
|`--teacher_dtype`|`auto`| BF16 on CUDA, FP32 on CPU |


### 4.2 Single-GPU training

`train_s2r_stagea.sh` uses a GPU and automatically loads `.local/env.sh` with `swiftvln-train` environment:

```bash
export OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/s2r-stagea-swiftvln-3b-10ep-bs8-lr1e-4"

bash scripts/train/train_s2r_stagea.sh
```

### 4.3 Multi-GPU training

Use `torchrun` to start multi-GPU training. The following configuration uses 8 GPUs, with an effective batch size of 64:

```bash
export OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/s2r-stagea-swiftvln-3b-10ep-bs64-lr1e-4"

torchrun --standalone --nproc_per_node=8 -m tools.s2r.trainer \
  --manifest_path "${MANIFEST_PATH}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}"
```

Stage-A's bidirectional contrastive loss aggregates image features across all GPUs and preserves gradients across ranks.

## 5. Outputs and checkpoints

Training output is saved in `OUTPUT_DIR`:

```text
<OUTPUT_DIR>/
├── best.pt
├── latest.pt
├── train_args.json
├── progress.json
├── metrics.jsonl
└── checkpoints/
    └── step_XXXXXXX.pt
```

| File | Content |
| --- | --- |
|`best.pt`|`U2S R@1` highest checkpoint on `val` split |
|`latest.pt`| The most recently saved checkpoint |
|`train_args.json`| Training arguments |
|`progress.json`| Current step count, training status and final metrics |
|`metrics.jsonl`| Metrics from each retrieval evaluation |

By default, retrieval evaluation on `val` runs at the end of training. Set `--eval_every_steps <N>` to evaluate and update `best.pt` every N steps; use `--save_every_steps <N>` to save additional periodic checkpoints.

Load the existing Stage-A weights and continue training:

```bash
OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/<new-run-name>" \
MANIFEST_PATH="${MANIFEST_PATH}" \
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH}" \
bash scripts/train/train_s2r_stagea.sh \
  --resume_checkpoint /path/to/stage-a/latest.pt
```

`resume_checkpoint` loads the adapter, projection head, global step, and best metric. The optimizer, scheduler, and random-number state are reinitialized.

## 6. Retrieval evaluation

Run retrieval evaluation on the `val` split with `best.pt`:

```bash
python -m tools.s2r.eval \
  --manifest_path "${MANIFEST_PATH}" \
  --checkpoint_path "${OUTPUT_DIR}/best.pt" \
  --split val \
  --batch_size 8 \
  --num_workers 4 \
  | tee "${OUTPUT_DIR}/eval_val.json"
```

Teacher path is saved in Stage-A checkpoint. When you need to switch the local model directory, add `--teacher_model_path /path/to/teacher`.

| Metric | Meaning |
| --- | --- |
|`u2s_r@1/5/10`| Use UAV image to retrieve Recall@K corresponding to Satellite image |
|`s2u_r@1/5/10`| Use Satellite image to retrieve Recall@K corresponding to UAV image |
|`paired_cosine_mean`| Average cosine similarity of paired image projection features |
|`by_source`| Sub-data source metrics of DenseUAV, GTA-UAV, SUES-200 and UAV-VisLoc |
