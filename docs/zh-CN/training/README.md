# SwiftVLN 训练

SwiftVLN 使用离线 expert trajectory 进行监督微调。SatNav 与 Habitat 共用训练入口；
每次训练通过 `VLN_ENV_TYPE` 选择一种环境。

```text
trajectory annotations + RGB frames
                  │
                  ▼
          SwiftVLNDataset
                  │
                  ▼
       trajectory window + history
                  │
                  ▼
          ms-swift full SFT
                  │
                  ▼
             checkpoint
```

## 1. 准备环境与数据

按照[安装](../getting-started/INSTALLATION.md)创建 `swiftvln-train` 环境，并完成所需数据准备：

- [SatNav 训练数据](../data/TRAINING_DATA_SATNAV.md)
- [Habitat 训练数据](../data/TRAINING_DATA_HABITAT.md)

进入仓库并加载本机配置：

```bash
cd /path/to/SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
source .local/env.sh
source "${SWIFTVLN_CONDA_SH}"
conda activate swiftvln-train
```

本页默认使用 Qwen2.5-VL 3B：

```bash
export MODEL_FAMILY=qwen2_5_vl
export MODEL_PATH="${SWIFTVLN_QWEN25_MODEL_PATH}"
```

使用 Qwen3-VL 时设置：

```bash
export MODEL_FAMILY=qwen3_vl
export MODEL_PATH="${SWIFTVLN_QWEN3_MODEL_PATH}"
export USE_LIGER_KERNEL=false
```

## 2. 选择训练任务

一次训练只运行以下一种配置。

### 2.1 SatNav

```bash
export VLN_ENV_TYPE=satnav
export VLN_DATA_PATH="${SWIFTVLN_SATNAV_TRAIN_DATA_PATH}"

test -f "${VLN_DATA_PATH}/annotations.json"
```

SatNav 的 `history` 训练读取 `annotations.json` 和 `images/`。`map` 训练还会读取 train
Episode、GeoTIFF 场景与 `summary.json`。

### 2.2 Habitat

完整 Habitat 训练组合 R2R、RxR 和 EnvDrop：

```bash
export VLN_ENV_TYPE=habitat
export VLN_DATA_PATH="${SWIFTVLN_HABITAT_R2R_TRAIN_PATH},${SWIFTVLN_HABITAT_RXR_TRAIN_PATH},${SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH}"

test -f "${SWIFTVLN_HABITAT_R2R_TRAIN_PATH}/annotations.json"
test -f "${SWIFTVLN_HABITAT_RXR_TRAIN_PATH}/annotations.json"
test -f "${SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH}/annotations.json"
```

仅使用部分数据集时，按逗号连接对应的 trajectory 目录：

```bash
export VLN_DATA_PATH="${SWIFTVLN_HABITAT_R2R_TRAIN_PATH},${SWIFTVLN_HABITAT_RXR_TRAIN_PATH}"
```

## 3. 默认训练配置

训练入口为：

```bash
bash scripts/train/train_swiftvln_qwen_vl.sh
```

主线配置如下：

| 参数 | 默认值 |
| --- | --- |
| 模型 | Qwen2.5-VL 3B |
| 训练方式 | Full SFT |
| Epoch | `1` |
| 轨迹窗口 | `NUM_FRAMES=32` |
| 每轮预测动作数 | `NUM_FUTURE_STEPS=4` |
| 相邻窗口重叠 | `NUM_OVERLAP=0` |
| Memory | `MEMORY_METHOD=history` |
| History processor | `HISTORY_PROCESSOR_TYPE=per_frame` |
| 历史帧数 | `NUM_HISTORY=8` |
| 历史帧压缩 | `COMPRESS_STRIDE=2`，average pooling |
| 每卡 batch size | `BATCH_SIZE=8` |
| 梯度累积 | `GRAD_ACCUM_STEPS=1` |
| Learning rate | `2e-5` |
| Precision | `bfloat16` |
| Attention | FlashAttention 2 |
| 分布式优化 | DeepSpeed ZeRO-2 |

有效 batch size 为：

```text
BATCH_SIZE × GRAD_ACCUM_STEPS × GPU 数量
```

使用前 `N` 张可见 GPU：

```bash
TRAIN_NUM_GPUS=8 bash scripts/train/train_swiftvln_qwen_vl.sh
```

指定 GPU：

```bash
TRAIN_CUDA_DEVICES=0,2,4,6 bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 4. Dry run

Dry run 解析 GPU、模型、数据路径、训练参数和实验名称，不加载模型或启动 `torchrun`：

```bash
TRAIN_NUM_GPUS=1 \
TRAIN_DRY_RUN=true \
USE_SWANLAB=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

输出中应包含当前选择的环境和数据：

```text
SwiftVLN Training
Model Family: qwen2_5_vl
Environment: satnav|habitat
Data: <trajectory path>
Memory Method: history
[INFO] TRAIN_DRY_RUN=true, skip torchrun launch after config validation.
```

## 5. Smoke 训练

使用 16 个训练 sample 运行 2 个 optimizer step：

```bash
export SMOKE_OUTPUT="output/swiftvln/smoke-${VLN_ENV_TYPE}"

TRAIN_NUM_GPUS=2 \
MAX_SAMPLES=16 \
MAX_STEPS=2 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
SAVE_STEPS=1 \
SAVE_TOTAL_LIMIT=1 \
LOGGING_STEPS=1 \
DATALOADER_NUM_WORKERS=2 \
DATALOADER_PREFETCH_FACTOR=2 \
DATASET_NUM_PROC=1 \
OUTPUT_DIR_OVERRIDE="${SMOKE_OUTPUT}" \
USE_SWANLAB=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

检查 optimizer step 和 checkpoint：

```bash
python - "${SMOKE_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
states = sorted(root.glob("v*/checkpoint-*/trainer_state.json"))
assert states, f"checkpoint not found: {root}"
state = json.loads(states[-1].read_text(encoding="utf-8"))
assert state["global_step"] > 0, state["global_step"]
print("checkpoint", states[-1].parent)
print("global_step", state["global_step"])
PY
```

## 6. 完整训练

以下命令使用 8 张 GPU、1 个 epoch 和默认 history 配置。

### 6.1 SatNav

```bash
VLN_ENV_TYPE=satnav \
VLN_DATA_PATH="${SWIFTVLN_SATNAV_TRAIN_DATA_PATH}" \
MODEL_FAMILY=qwen2_5_vl \
MODEL_PATH="${SWIFTVLN_QWEN25_MODEL_PATH}" \
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
NUM_EPOCHS=1 \
BATCH_SIZE=8 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=2e-5 \
TRAIN_NUM_GPUS=8 \
USE_SWANLAB=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

### 6.2 Habitat

```bash
VLN_ENV_TYPE=habitat \
VLN_DATA_PATH="${SWIFTVLN_HABITAT_R2R_TRAIN_PATH},${SWIFTVLN_HABITAT_RXR_TRAIN_PATH},${SWIFTVLN_HABITAT_ENVDROP_TRAIN_PATH}" \
MODEL_FAMILY=qwen2_5_vl \
MODEL_PATH="${SWIFTVLN_QWEN25_MODEL_PATH}" \
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
NUM_EPOCHS=1 \
BATCH_SIZE=8 \
GRAD_ACCUM_STEPS=1 \
LEARNING_RATE=2e-5 \
TRAIN_NUM_GPUS=8 \
USE_SWANLAB=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 7. Memory 配置

以下变量可以加入完整训练命令。

### 7.1 Per-frame history

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
LOG_BASE=1.0 \
COMPRESS_STRIDE=2 \
USE_TOME=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

`NUM_HISTORY=0` 对应 no-memory：

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=0 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

### 7.2 GTC

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=gtc \
GTC_OUTPUT_TOKENS=512 \
GTC_TEMPERATURE=0.1 \
GTC_NUM_ITERATIONS=1 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

分段 GTC 使用：

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=segment_gtc \
GTC_OUTPUT_TOKENS=512 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

### 7.3 SatNav map memory

Map memory 仅用于 SatNav，并与 per-frame pooling 配合使用：

```bash
VLN_ENV_TYPE=satnav \
VLN_DATA_PATH="${SWIFTVLN_SATNAV_TRAIN_DATA_PATH}" \
MEMORY_METHOD=map \
HISTORY_PROCESSOR_TYPE=per_frame \
USE_RANDOM=false \
USE_TOME=false \
EMBEDDING_MODE=none \
MAP_GLOBAL_SIDE_M=1000 \
MAP_LOCAL_SIDE_M=400 \
MAP_RENDER_PX=448 \
MAP_MASK_METHOD=dilate20 \
MAP_CACHE_DIR=auto \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

默认 cache 位于 SatNav 数据集目录下的 `map_cache/`。指定其他目录：

```bash
MAP_CACHE_DIR=/path/to/map_cache bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 8. Embedding enhancement

`EMBEDDING_MODE` 每次选择一种模式：

| 模式 | 配置 |
| --- | --- |
| 无 enhancement | `EMBEDDING_MODE=none` |
| Additive pose embedding | `EMBEDDING_MODE=pose` |
| FiLM pose embedding | `EMBEDDING_MODE=posefilm` |
| S2R Stage-A adapter | `EMBEDDING_MODE=uav` |

Pose embedding：

```bash
EMBEDDING_MODE=pose \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

FiLM pose embedding：

```bash
EMBEDDING_MODE=posefilm \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

加载 [S2R Stage-A](S2R_STAGE_A.md) adapter：

```bash
EMBEDDING_MODE=uav \
UAV_ADAPTER_PATH=/path/to/stage-a-checkpoint.pt \
UAV_ADAPTER_TYPE=transformer_v1 \
UAV_ADAPTER_APPLY_SCOPE=all_images \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 9. 训练输出

默认输出目录由实验配置自动命名：

```text
output/swiftvln/
└── swiftvln-<env>-<model>-<training-config>-<timestamp>/
    └── v0-<date>-<time>/
        ├── args.json
        ├── logging.jsonl
        ├── runs/
        ├── images/
        └── checkpoint-<step>/
            ├── config.json
            ├── model*.safetensors
            ├── trainer_state.json
            ├── scheduler.pt
            ├── rng_state_*.pth
            └── global_step<step>/
```

列出一次训练生成的 checkpoint：

```bash
find output/swiftvln -type d -name 'checkpoint-*' -print
```

设置固定输出根目录：

```bash
OUTPUT_DIR_OVERRIDE=/path/to/output/run-name \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

启用 SwanLab：

```bash
USE_SWANLAB=true \
SWANLAB_PROJECT=SwiftVLN \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 10. 恢复训练

完整恢复会加载模型、optimizer、scheduler、随机数状态和 global step。重新使用原训练任务、
模型、数据、GPU 数量与训练参数：

```bash
export RUN_ROOT=/path/to/output/run-name
export CHECKPOINT="${RUN_ROOT}/v0-YYYYMMDD-HHMMSS/checkpoint-1000"

OUTPUT_DIR_OVERRIDE="${RUN_ROOT}" \
RESUME_FROM_CHECKPOINT="${CHECKPOINT}" \
RESUME_ONLY_MODEL=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

同一 `RUN_ROOT` 下会创建新的 `vN-<date>-<time>/` 目录保存后续 checkpoint。

只加载模型权重并重新创建 optimizer 与 scheduler：

```bash
OUTPUT_DIR_OVERRIDE=/path/to/output/new-run \
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-1000 \
RESUME_ONLY_MODEL=true \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

## 11. 下一步

- [SwiftVLN 评测](../evaluation/README.md)
- [配置参考](../reference/CONFIGURATION.md)
- [实验命名](../reference/EXPERIMENT_NAMING.md)
- [输出格式](../reference/OUTPUTS.md)
