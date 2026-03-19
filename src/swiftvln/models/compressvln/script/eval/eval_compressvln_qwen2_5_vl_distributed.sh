#!/bin/bash
# CompressVLN Distributed Evaluation Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   # Habitat evaluation (default)
#   ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh
#
#   # SatNav evaluation
#   ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh
#
# This script runs distributed CompressVLN evaluation with history frame compression.

set -e  # Exit on error

# ============================================================================
# CRITICAL: Force NVIDIA EGL BEFORE anything else (must be set early!)
# This prevents Mesa software rendering fallback on machines with both
# NVIDIA and Mesa EGL vendors installed
# ============================================================================
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval

# ============================================================================
# GPU Configuration
# ============================================================================
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29600}"

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_PATH="${MODEL_PATH:-/path/to/your/trained/checkpoint}"

# ============================================================================
# Environment Type Configuration
# ============================================================================
ENV_TYPE="${ENV_TYPE:-habitat}"  # habitat or satnav

# ============================================================================
# Environment-specific Configuration
# ============================================================================
if [ "$ENV_TYPE" == "habitat" ]; then
    CONFIG_PATH="configs/vln_r2r.yaml"
elif [ "$ENV_TYPE" == "satnav" ]; then
    CONFIG_PATH="configs/satnav_task.yaml"
else
    echo "[ERROR] Unknown ENV_TYPE: $ENV_TYPE. Must be 'habitat' or 'satnav'."
    exit 1
fi

# SatNav 默认 val_seen；Habitat 或未指定 ENV_TYPE 时默认 val_unseen
if [ -z "$EVAL_SPLIT" ]; then
    if [ "$ENV_TYPE" == "satnav" ]; then
        EVAL_SPLIT="val_seen"
    else
        EVAL_SPLIT="val_unseen"
    fi
fi

# ============================================================================
# VLN Parameters (should match training)
# Can be overridden via environment variables
# ============================================================================
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-4}"

# ============================================================================
# CompressVLN-Specific Parameters
# ============================================================================
COMPRESS_STRIDE="${COMPRESS_STRIDE:-2}"  # Should match training

# ============================================================================
# Debug Options
# ============================================================================
MAX_EPISODES="${MAX_EPISODES:-}"

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Extract model name from MODEL_PATH
MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/output/compressvln/\([^/]*\)/.*|\1|p')
MODEL_NAME="${MODEL_NAME:-unknown_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/eval/compressvln/${MODEL_NAME}/${EVAL_SPLIT}/${TIMESTAMP}}"

# ============================================================================
# Video Options
# ============================================================================
SAVE_VIDEO="${SAVE_VIDEO:-false}"
VIDEO_COMPRESSION="${VIDEO_COMPRESSION:-false}"

# ============================================================================
# Environment Setup
# ============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=4
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
VLN_DIR="${SWIFTVLN_ROOT}/src/swiftvln"

# ============================================================================
# Print Configuration
# ============================================================================
echo "=============================================="
echo "CompressVLN Distributed Evaluation"
echo "=============================================="
echo "Environment:     ${ENV_TYPE}"
echo "Config Path:     ${CONFIG_PATH}"
echo "Model Path:      ${MODEL_PATH}"
echo "Eval Split:      ${EVAL_SPLIT}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "CUDA Devices:    ${CUDA_DEVICES}"
echo "Compress Stride: ${COMPRESS_STRIDE}"
echo "Save Video:      ${SAVE_VIDEO}"
echo "=============================================="

# ============================================================================
# Pre-check
# ============================================================================
if [ "$MODEL_PATH" == "/path/to/your/trained/checkpoint" ]; then
    echo "[ERROR] Please set MODEL_PATH!"
    echo "Usage: MODEL_PATH=/path/to/checkpoint EVAL_SPLIT=val_seen bash $0"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path does not exist: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Run Distributed Evaluation
# ============================================================================
cd "$SWIFTVLN_ROOT"

# Silence Habitat-sim C++ logs
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2

echo "[INFO] Starting CompressVLN distributed evaluation on ${NUM_GPUS} GPUs..."

# Build video arguments
VIDEO_ARGS=""
[ "$SAVE_VIDEO" = "true" ] && VIDEO_ARGS="--save_video"
[ "$VIDEO_COMPRESSION" = "true" ] && VIDEO_ARGS="${VIDEO_ARGS} --video_compression"

# Build max_episodes argument if set
MAX_EPISODES_ARG=""
if [ -n "$MAX_EPISODES" ]; then
    MAX_EPISODES_ARG="--max_episodes ${MAX_EPISODES}"
    echo "[INFO] Debug mode: limiting to ${MAX_EPISODES} episodes"
fi

# Build debug_timing argument if set
DEBUG_TIMING_ARG=""
if [ "${DEBUG_TIMING:-false}" = "true" ]; then
    DEBUG_TIMING_ARG="--debug_timing"
    echo "[INFO] Debug timing enabled: will print detailed timing statistics"
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m swiftvln.models.compressvln.eval \
    --model_path "${MODEL_PATH}" \
    --env-type "${ENV_TYPE}" \
    --habitat_config_path "${VLN_DIR}/${CONFIG_PATH}" \
    --satnav-config "${VLN_DIR}/${CONFIG_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames "${NUM_FRAMES}" \
    --num_history "${NUM_HISTORY}" \
    --num_future_steps "${NUM_FUTURE_STEPS}" \
    --compress_stride "${COMPRESS_STRIDE}" \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    ${VIDEO_ARGS} \
    ${MAX_EPISODES_ARG} \
    ${DEBUG_TIMING_ARG}

echo "=============================================="
echo "CompressVLN Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
