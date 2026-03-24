#!/bin/bash
# Uni-NaVid SatNav fine-tuning launcher.
# Usage:
#   bash baseline/uninavid/scripts/train_satnav.sh [EXP_NAME]
#
# Optional env overrides:
#   DATA_PATH=...
#   VIDEO_FOLDER=...
#   NUM_GPUS=8
#   TRAIN_BSZ=24
#   EVAL_BSZ=1
#   GRAD_ACCUM=1
#   NUM_EPOCHS=1
#   LEARNING_RATE=1e-5
#   MAX_STEPS=8
#   SAVE_STRATEGY=no
#   SAVE_STEPS=2000
#   SAVE_TOTAL_LIMIT=1
#   USE_SWANLAB=false
#   REPORT_TO=none
#   DATALOADER_WORKERS=2
#   MODEL_MAX_LENGTH=1536
#   UNINAVID_USE_FLASH_ATTN=1
#   MASTER_PORT=29500
#   GROUP_BY_MODALITY_LENGTH=False
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────
SWIFTVLN_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
UNINAVID_REPO="/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
BASELINE_DIR="${SWIFTVLN_ROOT}/baseline/uninavid"

PREV_MODEL="${BASELINE_DIR}/model/Uni-Navid"
VISION_TOWER="${BASELINE_DIR}/model/eva_vit_g.pth"
IMAGE_PROCESSOR="${UNINAVID_REPO}/uninavid/processor/clip-patch14-224"
DS_CONFIG="${DS_CONFIG:-${BASELINE_DIR}/configs/zero1.json}"

DATA_PATH="${DATA_PATH:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data}"

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BSZ="${TRAIN_BSZ:-24}"
EVAL_BSZ="${EVAL_BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1536}"
MASTER_PORT="${MASTER_PORT:-29500}"
GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-False}"
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SEED_ARG=()
MAX_STEPS_ARG=()
CUSTOM_EXP_NAME="${1:-}"

VERSION_NUM="$(echo "${DATA_PATH}" | grep -oP 'ver_\K\d+' | head -1 || true)"
if [[ -z "${VERSION_NUM}" ]]; then
    VERSION_NUM="unknown"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
EFFECTIVE_BATCH_SIZE=$((TRAIN_BSZ * GRAD_ACCUM * NUM_GPUS))

if [[ -n "${CUSTOM_EXP_NAME}" ]]; then
    EXP_NAME="${CUSTOM_EXP_NAME}"
else
    EXP_NAME="uninavid-baseline-${NUM_EPOCHS}ep-data${VERSION_NUM}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
fi

OUTPUT_DIR="${SWIFTVLN_ROOT}/output/uninavid-baseline/${EXP_NAME}"

REPORT_TO_ARG=()
if [[ -n "${REPORT_TO}" ]]; then
    REPORT_TO_ARG=(--report_to "${REPORT_TO}")
elif [[ "${USE_SWANLAB}" == "true" ]]; then
    export SWANLAB_PROJECT
    export SWANLAB_NAME="${EXP_NAME}"
    export SWANLAB_MODE
    REPORT_TO_ARG=(--report_to swanlab)
else
    REPORT_TO_ARG=(--report_to none)
fi

if [[ -n "${SEED:-}" ]]; then
    SEED_ARG=(--seed "${SEED}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

# ── Conda ──────────────────────────────────────────────────
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline

echo "=========================================="
echo "Uni-NaVid Baseline Training"
echo "=========================================="
echo "  Data path   : ${DATA_PATH}"
echo "  Video dir   : ${VIDEO_FOLDER}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  EXP_NAME    : ${EXP_NAME}"
echo "  GPUs        : ${NUM_GPUS}"
echo "  Batch       : ${TRAIN_BSZ} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${EFFECTIVE_BATCH_SIZE}"
echo "  LR          : ${LEARNING_RATE}"
echo "  Epochs      : ${NUM_EPOCHS}"
echo "  Save        : ${SAVE_STRATEGY}"
echo "  SwanLab     : ${USE_SWANLAB}"
echo "  Report To   : ${REPORT_TO_ARG[*]}"
echo "=========================================="

# ── Launch ─────────────────────────────────────────────────
deepspeed --master_port "${MASTER_PORT}" --num_gpus "${NUM_GPUS}" "${BASELINE_DIR}/src/train_satnav.py" \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "${PREV_MODEL}" \
    --version imgsp_v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${VIDEO_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --image_processor "${IMAGE_PROCESSOR}" \
    --tune_vision_encoder False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}" \
    --video_fps 1 \
    --compress_type "grid:2" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    "${MAX_STEPS_ARG[@]}" \
    --per_device_train_batch_size "${TRAIN_BSZ}" \
    --per_device_eval_batch_size "${EVAL_BSZ}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --lazy_preprocess True \
    "${REPORT_TO_ARG[@]}" \
    "${SEED_ARG[@]}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "  EXP_NAME : ${EXP_NAME}"
echo "  Output   : ${OUTPUT_DIR}"
echo "=========================================="
