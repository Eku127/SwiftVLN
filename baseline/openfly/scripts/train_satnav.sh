#!/usr/bin/env bash
set -euo pipefail

SWIFTVLN_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASELINE_DIR="${SWIFTVLN_ROOT}/baseline/openfly"

OPENFLY_BACKEND="${OPENFLY_BACKEND:-continue}"
if [ "${OPENFLY_BACKEND}" = "scratch" ]; then
    DEFAULT_MODEL_PATH="${BASELINE_DIR}/model/openvlaopenvla-7b-prismatic"
    DEFAULT_PROCESSOR_PATH="${BASELINE_DIR}/model/openfly-agent-7b"
else
    DEFAULT_MODEL_PATH="${BASELINE_DIR}/model/openfly-agent-7b"
    DEFAULT_PROCESSOR_PATH=""
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${DEFAULT_PROCESSOR_PATH}}"
DATA_PATH="${DATA_PATH:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data/annotations.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data}"
NUM_GPUS="${NUM_GPUS:-1}"
TRAIN_BSZ="${TRAIN_BSZ:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
GRID_SIZE="${GRID_SIZE:-16}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEEPSPEED_MODE="${DEEPSPEED_MODE:-zero2}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
OPENFLY_ACTION_FORMAT="${OPENFLY_ACTION_FORMAT:-}"
OPENFLY_UNNORM_KEY="${OPENFLY_UNNORM_KEY:-satnav_original}"
MAX_EPISODES="${SATNAV_MAX_EPISODES:-}"
MAX_SAMPLES="${SATNAV_MAX_SAMPLES:-}"
USE_FLASH_ATTENTION_2="${USE_FLASH_ATTENTION_2:-true}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
REPORT_TO="${REPORT_TO:-none}"
SATNAV_HEAD_KEEP="${SATNAV_HEAD_KEEP-7}"
SATNAV_SAMPLE_STRIDE="${SATNAV_SAMPLE_STRIDE-3}"
SATNAV_STOP_REPEAT="${SATNAV_STOP_REPEAT-2}"
SATNAV_STOP_HISTORY_AUG="${SATNAV_STOP_HISTORY_AUG-1}"
SATNAV_STOP_WINDOW="${SATNAV_STOP_WINDOW-0}"
SATNAV_TAIL_KEEP="${SATNAV_TAIL_KEEP-5}"
OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS="${OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS:-0.4,1.2,1.2,1.2}"
OPENFLY_ACTION_HISTORY_LIMIT="${OPENFLY_ACTION_HISTORY_LIMIT:-16}"

if [ -z "${OPENFLY_ACTION_FORMAT}" ]; then
    OPENFLY_ACTION_FORMAT="compact"
fi

VERSION_NUM="$(echo "${DATA_PATH}" | grep -oP 'ver_\K\d+' | head -1 || true)"
[ -z "${VERSION_NUM}" ] && VERSION_NUM="unknown"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
EFFECTIVE_BSZ=$((TRAIN_BSZ * GRAD_ACCUM * NUM_GPUS))
BACKEND_SUFFIX="-bk${OPENFLY_BACKEND}"
FORMAT_SUFFIX="-act${OPENFLY_ACTION_FORMAT}"
HISTORY_SUFFIX="-hist${OPENFLY_ACTION_HISTORY_LIMIT}"
SAMPLE_TAG="-sample-hk${SATNAV_HEAD_KEEP:-off}-fs${SATNAV_SAMPLE_STRIDE:-off}-stopx${SATNAV_STOP_REPEAT:-1}-stopw${SATNAV_STOP_WINDOW:-0}-tail${SATNAV_TAIL_KEEP:-0}-stoph${SATNAV_STOP_HISTORY_AUG:-1}${HISTORY_SUFFIX}"
EXP_NAME="${EXP_NAME:-openfly-baseline-1ep-data${VERSION_NUM}${BACKEND_SUFFIX}${FORMAT_SUFFIX}${SAMPLE_TAG}-bs${EFFECTIVE_BSZ}-lr${LEARNING_RATE}-${TIMESTAMP}}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"
if [ -n "${OUTPUT_DIR_OVERRIDE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
else
    OUTPUT_DIR="${SWIFTVLN_ROOT}/output/openfly-baseline/${EXP_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate openfly-baseline

export PYTHONPATH="${BASELINE_DIR}/src:${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export SATNAV_HEAD_KEEP SATNAV_SAMPLE_STRIDE SATNAV_STOP_REPEAT SATNAV_STOP_HISTORY_AUG SATNAV_STOP_WINDOW SATNAV_TAIL_KEEP
export OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS
export OPENFLY_ACTION_HISTORY_LIMIT
unset WANDB_PROJECT WANDB_API_KEY WANDB_ENTITY

if [ -z "${DEEPSPEED_CONFIG}" ]; then
    case "${DEEPSPEED_MODE}" in
        off|none|ddp)
            DEEPSPEED_CONFIG=""
            ;;
        zero1)
            DEEPSPEED_CONFIG="${BASELINE_DIR}/configs/zero1.json"
            ;;
        zero2)
            DEEPSPEED_CONFIG="${BASELINE_DIR}/configs/zero2.json"
            ;;
        *)
            echo "Unsupported DEEPSPEED_MODE: ${DEEPSPEED_MODE}" >&2
            exit 1
            ;;
    esac
fi

ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --backend "${OPENFLY_BACKEND}"
    --data_path "${DATA_PATH}"
    --image_folder "${IMAGE_FOLDER}"
    --action_format "${OPENFLY_ACTION_FORMAT}"
    --unnorm_key "${OPENFLY_UNNORM_KEY}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --per_device_train_batch_size "${TRAIN_BSZ}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --save_strategy steps
    --evaluation_strategy no
    --gradient_checkpointing
    --grid_size "${GRID_SIZE}"
    --torch_dtype "${TORCH_DTYPE}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --report_to "${REPORT_TO}"
)

if [ -n "${PROCESSOR_PATH}" ]; then
    ARGS+=(--processor_name_or_path "${PROCESSOR_PATH}")
fi

if [ -n "${DEEPSPEED_CONFIG}" ]; then
    ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [ "${USE_FLASH_ATTENTION_2}" = "true" ]; then
    ARGS+=(--use_flash_attention_2)
fi

if [[ "${TORCH_DTYPE}" == "bfloat16" || "${TORCH_DTYPE}" == "auto" ]]; then
    ARGS+=(--bf16)
else
    ARGS+=(--fp16)
fi

if [ -n "${MAX_STEPS}" ]; then
    ARGS+=(--max_steps "${MAX_STEPS}")
fi

if [ -n "${MAX_EPISODES}" ]; then
    ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    _MASTER_ADDR="${MASTER_ADDR}"
    _MASTER_PORT="${MASTER_PORT}"
    unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT
    export MASTER_ADDR="${_MASTER_ADDR}"
    export MASTER_PORT="${_MASTER_PORT}"
    torchrun --nnodes 1 --nproc-per-node "${NUM_GPUS}" --master_addr "${_MASTER_ADDR}" --master_port "${_MASTER_PORT}" \
        "${BASELINE_DIR}/src/train_satnav.py" "${ARGS[@]}"
else
    python "${BASELINE_DIR}/src/train_satnav.py" "${ARGS[@]}"
fi
