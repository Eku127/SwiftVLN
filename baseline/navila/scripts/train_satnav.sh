#!/bin/bash
# Train NaVILA baseline on SatNav trajectory data.
# Usage:
#   bash baseline/navila/scripts/train_satnav.sh [EXP_NAME]
#
# Optional env overrides:
#   DATA_PATH=...
#   IMAGE_FOLDER=...
#   NUM_GPUS=8
#   TRAIN_BSZ=4
#   GRAD_ACCUM=1
#   NUM_EPOCHS=1
#   LEARNING_RATE=3e-5
#   MAX_STEPS=120
#   SAVE_STEPS=100
#   SAVE_TOTAL_LIMIT=2
#   MODEL_MAX_LENGTH=4096
#   DATALOADER_WORKERS=16
#   MASTER_PORT=29500
#   USE_SWANLAB=false
#   SWANLAB_PROJECT=baseline
#   SWANLAB_MODE=cloud
#   REPORT_TO=...
set -euo pipefail

SWIFTVLN_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASELINE_DIR="${SWIFTVLN_ROOT}/baseline/navila"

MODEL_PATH="${MODEL_PATH:-${BASELINE_DIR}/model/navila-siglip-llama3-8b-v1.5-pretrain}"
VISION_TOWER="${VISION_TOWER:-${BASELINE_DIR}/model/navila-siglip-llama3-8b-v1.5-pretrain/vision_tower}"
DS_CONFIG="${DS_CONFIG:-${BASELINE_DIR}/configs/zero2.json}"

DATA_PATH="${DATA_PATH:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data}"

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BSZ="${TRAIN_BSZ:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_STEPS="${MAX_STEPS:-120}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
REPORT_TO="${REPORT_TO:-}"
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
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
    EXP_NAME="navila-baseline-${NUM_EPOCHS}ep-8f-data${VERSION_NUM}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
fi

OUTPUT_DIR="${SWIFTVLN_ROOT}/output/navila-baseline/${EXP_NAME}"
MAX_STEPS_ARG=()
REPORT_TO_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

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

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline

echo "=========================================="
echo "NaVILA Baseline Training"
echo "=========================================="
echo "  Model      : ${MODEL_PATH}"
echo "  Data path  : ${DATA_PATH}"
echo "  Image root : ${IMAGE_FOLDER}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  GPUs       : ${NUM_GPUS}"
echo "  Batch      : ${TRAIN_BSZ} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${EFFECTIVE_BATCH_SIZE}"
echo "  LR         : ${LEARNING_RATE}"
echo "  SwanLab    : ${USE_SWANLAB}"
echo "  Report To  : ${REPORT_TO_ARG[*]}"
echo "=========================================="

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${BASELINE_DIR}/src/train_satnav.py" \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version llama_3 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --data_mixture satnav \
    --longvila_sampler True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    "${MAX_STEPS_ARG[@]}" \
    --per_device_train_batch_size "${TRAIN_BSZ}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --fps 0.0 \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --lazy_preprocess True \
    "${REPORT_TO_ARG[@]}"

echo "=========================================="
echo "Training completed!"
echo "  EXP_NAME : ${EXP_NAME}"
echo "  Output   : ${OUTPUT_DIR}"
echo "=========================================="
