#!/bin/bash
# Uni-NaVid SatNav fine-tuning launcher.
# Usage:
#   bash baseline/uninavid/scripts/train_satnav.sh [EXP_NAME]
#
# Setting: 8-GPU full-param finetune，视觉编码器开放训练，grad_accum=2
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────
SWIFTVLN_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
UNINAVID_REPO="/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
BASELINE_DIR="${SWIFTVLN_ROOT}/baseline/uninavid"

PREV_MODEL="${BASELINE_DIR}/model/Uni-Navid"
VISION_TOWER="${BASELINE_DIR}/model/eva_vit_g.pth"
IMAGE_PROCESSOR="${UNINAVID_REPO}/uninavid/processor/clip-patch14-224"
DS_CONFIG="${UNINAVID_REPO}/scripts/zero2.json"

DATA_PATH="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json"
VIDEO_FOLDER="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data"

EXP_NAME="${1:-satnav_ft_$(date +%y%m%d_%H%M%S)}"
OUTPUT_DIR="${SWIFTVLN_ROOT}/output/uninavid-baseline/${EXP_NAME}"

# ── Conda ──────────────────────────────────────────────────
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline

# ── Launch ─────────────────────────────────────────────────
deepspeed "${BASELINE_DIR}/src/train_satnav.py" \
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
    --group_by_modality_length True \
    --video_fps 1 \
    --compress_type "grid:2" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
