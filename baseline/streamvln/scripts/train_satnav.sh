#!/bin/bash
# ==============================================================================
# Train StreamVLN Baseline on SatNav trajectory data (8-GPU full-param).
#
# Usage:
#   bash scripts/train_satnav.sh [continue|scratch]
#
# Modes:
#   continue  (default) — fine-tune from official StreamVLN checkpoint
#   scratch             — start from LLaVA-Video-7B-Qwen2 base model
#
# Environment variables (all optional):
#   SATNAV_DATASET   — Data dir name under satnav_datasets (default: SatNav-v0.1)
#   SATNAV_VERSION   — Deprecated alias for SATNAV_DATASET, kept for old launchers
#   SATNAV_TRAIN_DATA_DIR — Explicit trajectory_data dir override
#   SATNAV_MAX_EPISODES / SATNAV_MAX_SAMPLES — Optional dataset caps for smoke tests
#   NUM_EPOCHS       — Training epochs (default: 1)
#   LEARNING_RATE    — Learning rate (default: 2e-5)
#   BATCH_SIZE       — Per-device batch size (default: 3)
#   GRAD_ACCUM       — Gradient accumulation steps (default: 2)
#   GPUS_PER_NODE    — Number of GPUs (default: 8)
#   USE_SWANLAB      — Enable SwanLab reporting (default: false)
#   SAVE_STRATEGY    — "epoch" or "steps" (default: epoch)
#   SAVE_STEPS       — Save every N steps when SAVE_STRATEGY=steps (default: 1000)
#   SMOKE_TEST       — true/false, when true save under output/streamvln-baseline/smoketest (default: false)
#   STREAMVLN_OFFLINE — true/false, keep HF/Transformers offline for local model dirs (default: true)
#
# Output:
#   output/streamvln-baseline/<EXP_NAME>/                (normal)
#   output/streamvln-baseline/smoketest/<EXP_NAME>/      (smoke test)
#
# EXP_NAME format:
#   streamvln-baseline-{mode}-{epochs}ep-f{frames}h{history}s{future}-data{dataset}-bs{eff_bs}-lr{lr}-{timestamp}
#
# Environment: conda env streamvln-baseline
# ==============================================================================

set -euo pipefail

# ---- Parse mode argument ----
MODE="${1:-continue}"
if [ "$MODE" != "continue" ] && [ "$MODE" != "scratch" ]; then
    echo "[ERROR] Unknown mode: $MODE. Must be 'continue' or 'scratch'."
    exit 1
fi

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"

TRAIN_SCRIPT="${BASELINE_DIR}/src/train_satnav.py"
DEEPSPEED_CFG="${BASELINE_DIR}/configs/zero2.json"

# ---- SatNav Data ----
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
SATNAV_DATASET="${SATNAV_DATASET:-${SATNAV_VERSION:-SatNav-v0.1}}"
echo "[INFO] Using SatNav dataset: ${SATNAV_DATASET}"

SATNAV_DATA_DIR="${SATNAV_TRAIN_DATA_DIR:-${SATNAV_DATA_ROOT}/${SATNAV_DATASET}/trajectory_data}"
if [ ! -d "$SATNAV_DATA_DIR" ]; then
    echo "[ERROR] SatNav trajectory data not found: ${SATNAV_DATA_DIR}"
    echo "[ERROR] StreamVLN training expects <dataset>/trajectory_data/annotations.json and image folders."
    echo "[ERROR] Set SATNAV_DATASET or SATNAV_TRAIN_DATA_DIR if the training trajectory export lives elsewhere."
    exit 1
fi

if [ ! -f "${SATNAV_DATA_DIR}/annotations.json" ]; then
    echo "[ERROR] SatNav trajectory annotations not found: ${SATNAV_DATA_DIR}/annotations.json"
    exit 1
fi

VERSION_TAG="$(echo "$SATNAV_DATASET" | sed -E 's/^ver_//; s/[^A-Za-z0-9._-]+/-/g')"

# ---- Vision model (local copy to avoid network download) ----
VISION_MODEL_VERSION="${BASELINE_DIR}/model/siglip-so400m-patch14-384"

# ---- Training hyperparameters ----
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SMOKE_TEST="${SMOKE_TEST:-false}"
STREAMVLN_OFFLINE="${STREAMVLN_OFFLINE:-true}"

if [ "${STREAMVLN_OFFLINE}" = "true" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- SwanLab configuration ----
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_MODE="cloud"

# ---- VLN parameters ----
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4

# ---- Checkpoint selection ----
if [ "$MODE" = "scratch" ]; then
    MODEL_NAME_OR_PATH="${BASELINE_DIR}/model/LLaVA-Video-7B-Qwen2"
    if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
        echo "[ERROR] Base model not found: ${MODEL_NAME_OR_PATH}"
        exit 1
    fi
    echo "[INFO] Mode: from scratch — base: ${MODEL_NAME_OR_PATH}"
else
    MODEL_NAME_OR_PATH="${BASELINE_DIR}/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
    if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
        echo "[ERROR] Official StreamVLN checkpoint not found: ${MODEL_NAME_OR_PATH}"
        exit 1
    fi
    echo "[INFO] Mode: continue — checkpoint: ${MODEL_NAME_OR_PATH}"
fi

# ---- Build EXP_NAME ----
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * GPUS_PER_NODE))
EXP_NAME="streamvln-baseline-${MODE}-${NUM_EPOCHS}ep-f${NUM_FRAMES}h${NUM_HISTORY}s${NUM_FUTURE_STEPS}-data${VERSION_TAG}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
MODEL_OUTPUT_ROOT="${REPO_ROOT}/output/streamvln-baseline"
if [ "${SMOKE_TEST}" = "true" ]; then
    OUTPUT_DIR="${MODEL_OUTPUT_ROOT}/smoketest/${EXP_NAME}"
else
    OUTPUT_DIR="${MODEL_OUTPUT_ROOT}/${EXP_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

# ---- SwanLab args ----
SWANLAB_ARGS=()
if [ "$USE_SWANLAB" = "true" ]; then
    # SwanLab HuggingFace Trainer integration uses environment variables
    export SWANLAB_PROJECT="${SWANLAB_PROJECT}"
    export SWANLAB_NAME="${EXP_NAME}"
    export SWANLAB_MODE="${SWANLAB_MODE}"
    SWANLAB_ARGS=(--report_to swanlab)
else
    SWANLAB_ARGS=(--report_to none)
fi

# ---- Save strategy args ----
SAVE_ARGS=(--save_strategy "${SAVE_STRATEGY}" --save_total_limit 1)
if [ "$SAVE_STRATEGY" = "steps" ]; then
    SAVE_ARGS+=(--save_steps "${SAVE_STEPS}")
fi

# ---- Max steps (optional, for smoke test) ----
MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

echo "=========================================="
echo "StreamVLN Baseline Training"
echo "=========================================="
echo "  Mode        : ${MODE}"
echo "  Model       : ${MODEL_NAME_OR_PATH}"
echo "  Dataset     : ${SATNAV_DATASET}"
echo "  Data dir    : ${SATNAV_DATA_DIR}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  EXP_NAME    : ${EXP_NAME}"
echo "  GPUs        : ${GPUS_PER_NODE}"
echo "  Batch       : ${BATCH_SIZE} x ${GRAD_ACCUM} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "  LR          : ${LEARNING_RATE}"
echo "  Epochs      : ${NUM_EPOCHS}"
echo "  Save        : ${SAVE_STRATEGY}"
echo "  SwanLab     : ${USE_SWANLAB}"
echo "  Smoke Test  : ${SMOKE_TEST}"
echo "=========================================="

# ---- PYTHONPATH ----
export PYTHONPATH="/mnt/data1/home/jiangjiajun/workspace/StreamVLN:\
/mnt/data1/home/jiangjiajun/workspace/StreamVLN/streamvln:\
${BASELINE_DIR}:${PYTHONPATH:-}"

# ---- Python / distributed launcher ----
DEFAULT_STREAMVLN_PYTHON="/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-baseline/bin/python"
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$DEFAULT_STREAMVLN_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_STREAMVLN_PYTHON"
    else
        PYTHON_BIN="$(command -v python3 || command -v python || true)"
    fi
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python interpreter not found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
if [ -d "$PYTHON_BIN_DIR" ]; then
    export PATH="${PYTHON_BIN_DIR}:${PATH}"
fi

if command -v torchrun >/dev/null 2>&1; then
    DIST_LAUNCH=(torchrun)
else
    DIST_LAUNCH=("${PYTHON_BIN}" -m torch.distributed.run)
fi

# Prefer CUDA 13 nvcc on H100 (CUDA 11.5 nvcc cannot compile sm_90 ops).
if [ -x "/usr/local/cuda-13.0/bin/nvcc" ]; then
    export PATH="/usr/local/cuda-13.0/bin:${PATH}"
    export CUDA_HOME="/usr/local/cuda-13.0"
fi

# ---- Launch ----
"${DIST_LAUNCH[@]}" \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=$((RANDOM % 10000 + 20000)) \
    "${TRAIN_SCRIPT}" \
    --deepspeed "${DEEPSPEED_CFG}" \
    \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version qwen_1_5 \
    \
    --video_folder "${SATNAV_DATA_DIR}" \
    --group_by_task False \
    \
    --num_history ${NUM_HISTORY} \
    --num_future_steps ${NUM_FUTURE_STEPS} \
    --num_frames ${NUM_FRAMES} \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    \
    --bf16 True \
    --run_name "${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
    "${MAX_STEPS_ARG[@]}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    "${SAVE_ARGS[@]}" \
    --learning_rate ${LEARNING_RATE} \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.075 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1.85e-05}' \
    --logging_steps ${LOGGING_STEPS} \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    "${SWANLAB_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "  EXP_NAME : ${EXP_NAME}"
echo "  Output   : ${OUTPUT_DIR}"
echo "=========================================="
