#!/bin/bash
# OverlapVLN Initial Strategy Eval Debug Script
# 
# Usage:
#   bash src/swiftvln/models/overlapvln/script/eval/eval_debug_initial.sh

set -e

# Conda
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval

# GPU Config - Use 2 GPUs for quick test
CUDA_DEVICES="0,1"
NUM_GPUS=2
MASTER_PORT=29602

# Model - use archived checkpoint
MODEL_PATH="/mnt/data4/jiangjiajun/archive/ms-swift/output/overlapvln/overlapvln-habitat-stage1-3b-1ep-f32h8s4-overlap16-pf-s2-bs64-lr2e-5-20260203-012501/v0-20260203-012535/checkpoint-2239"

# Environment
ENV_TYPE="habitat"
CONFIG_PATH="configs/vln_r2r.yaml"
EVAL_SPLIT="val_unseen"

# VLN Parameters (must match training)
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
NUM_OVERLAP=16

# History Processor - per_frame (match training)
HISTORY_PROCESSOR_TYPE="per_frame"
COMPRESS_STRIDE=2
USE_TOME=false
LOG_BASE=1.0

# *** Test initial strategy ***
SYSTEM_PROMPT_SETTING="initial"

# Debug Config
MAX_EPISODES=2  # Only test 2 episodes

# Output
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results/eval_debug/initial_test_${TIMESTAMP}"

# Save video for verification
SAVE_VIDEO=false

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=7200
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# *** Enable debug logging ***
export OVERLAPVLN_DEBUG=1

# Silence Habitat logs
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
VLN_DIR="${SWIFTVLN_ROOT}/src/swiftvln"

echo "=========================================="
echo "OverlapVLN Initial Strategy Eval Test"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Env: ${ENV_TYPE}, Split: ${EVAL_SPLIT}"
echo "System Prompt: ${SYSTEM_PROMPT_SETTING}"
echo "History: ${HISTORY_PROCESSOR_TYPE}, h=${NUM_HISTORY}, stride=${COMPRESS_STRIDE}"
echo "Max Episodes: ${MAX_EPISODES} (debug test)"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "OVERLAPVLN_DEBUG: ${OVERLAPVLN_DEBUG}"
echo "=========================================="

cd "$SWIFTVLN_ROOT"

# Build history args
HISTORY_PROCESSOR_ARGS="--history_processor_type ${HISTORY_PROCESSOR_TYPE}"
HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --compress_stride ${COMPRESS_STRIDE} --log_base ${LOG_BASE}"

# Build video args
VIDEO_ARGS=""
[ "$SAVE_VIDEO" = "true" ] && VIDEO_ARGS="--save_video"

echo ""
echo "Starting evaluation with OVERLAPVLN_DEBUG=1..."
echo ""

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m swiftvln.models.overlapvln.eval \
    --model_path "${MODEL_PATH}" \
    --env-type "${ENV_TYPE}" \
    --habitat_config_path "${VLN_DIR}/${CONFIG_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames "${NUM_FRAMES}" \
    --num_history "${NUM_HISTORY}" \
    --num_future_steps "${NUM_FUTURE_STEPS}" \
    --num_overlap "${NUM_OVERLAP}" \
    --system_prompt_setting "${SYSTEM_PROMPT_SETTING}" \
    ${HISTORY_PROCESSOR_ARGS} \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    --max_episodes "${MAX_EPISODES}" \
    ${VIDEO_ARGS}

echo ""
echo "=========================================="
echo "Eval test completed!"
echo "Results: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "Check log for [INITIAL DEBUG] lines to verify initial strategy"
echo ""
