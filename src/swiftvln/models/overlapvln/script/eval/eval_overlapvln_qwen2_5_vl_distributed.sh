#!/bin/bash
# OverlapVLN Distributed Evaluation Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   # Habitat evaluation (default)
#   ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/overlapvln/script/eval/eval_overlapvln_qwen2_5_vl_distributed.sh
#
#   # SatNav evaluation
#   ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/overlapvln/script/eval/eval_overlapvln_qwen2_5_vl_distributed.sh
#
# This script runs distributed OverlapVLN evaluation with history frame compression.

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

EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"

# ============================================================================
# VLN Parameters (should match training)
# Can be overridden via environment variables
# ============================================================================
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-4}"

# ============================================================================
# OverlapVLN-Specific Parameters
# ============================================================================
NUM_OVERLAP="${NUM_OVERLAP:-16}"  # Number of overlapping actions between windows

# ============================================================================
# History Processor Configuration
# ============================================================================
HISTORY_PROCESSOR_TYPE="${HISTORY_PROCESSOR_TYPE:-per_frame}"  # "per_frame", "gtc", or "segment_gtc"

# ---------- Per-frame parameters (used when HISTORY_PROCESSOR_TYPE="per_frame") ----------
COMPRESS_STRIDE="${COMPRESS_STRIDE:-2}"  # Should match training
USE_TOME="${USE_TOME:-false}"  # Use GridToMe compression (must match training)
LOG_BASE="${LOG_BASE:-1.0}"  # Sampling distribution: 1.0 = uniform, >1.0 = logarithmic

# ---------- GTC/SegmentGTC parameters (used when HISTORY_PROCESSOR_TYPE="gtc" or "segment_gtc") ----------
GTC_OUTPUT_TOKENS="${GTC_OUTPUT_TOKENS:-512}"
GTC_TEMPERATURE="${GTC_TEMPERATURE:-0.1}"
GTC_NUM_ITERATIONS="${GTC_NUM_ITERATIONS:-1}"

# ============================================================================
# System Prompt Setting
# ============================================================================
# System prompt strategy: "vanilla" (default) or "initial"
# - vanilla: Standard prompt without initial view image (must match training)
# - initial: Add first frame (uncompressed) as initial observation
SYSTEM_PROMPT_SETTING="${SYSTEM_PROMPT_SETTING:-vanilla}"

# Embedding enhancement (must match training checkpoint setup)
USE_PIXEL_EMBED="${USE_PIXEL_EMBED:-false}"
USE_POSE_EMBED="${USE_POSE_EMBED:-false}"
POSE_FUSION_METHOD="${POSE_FUSION_METHOD:-additive}"
POSE_NORM_SCALE="${POSE_NORM_SCALE:-100.0}"

# ============================================================================
# Debug Options
# ============================================================================
MAX_EPISODES="${MAX_EPISODES:-}"
# Debug logging for specific rank (-1 to disable)
SATNAV_DEBUG_RANK="${SATNAV_DEBUG_RANK:--1}"

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Extract model name from MODEL_PATH
MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/output/overlapvln/\([^/]*\)/.*|\1|p')
MODEL_NAME="${MODEL_NAME:-unknown_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/eval/overlapvln/${MODEL_NAME}/${ENV_TYPE}_${EVAL_SPLIT}_${TIMESTAMP}}"

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

# Debug logging for SatNav (rank-specific logging)
export SATNAV_DEBUG_RANK="${SATNAV_DEBUG_RANK}"
if [ "${SATNAV_DEBUG_RANK}" != "-1" ]; then
    export SATNAV_DEBUG_LOG="${OUTPUT_DIR}/debug_rank${SATNAV_DEBUG_RANK}.log"
    echo "Debug logging enabled for rank ${SATNAV_DEBUG_RANK}"
    echo "Debug log will be saved to: ${SATNAV_DEBUG_LOG}"
fi

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
VLN_DIR="${SWIFTVLN_ROOT}/src/swiftvln"

# ============================================================================
# Print Configuration
# ============================================================================
echo "=============================================="
echo "OverlapVLN Distributed Evaluation"
echo "=============================================="
echo "Environment:     ${ENV_TYPE}"
echo "Config Path:     ${CONFIG_PATH}"
echo "Model Path:      ${MODEL_PATH}"
echo "Eval Split:      ${EVAL_SPLIT}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "CUDA Devices:    ${CUDA_DEVICES}"
echo "Num Overlap:     ${NUM_OVERLAP}"
echo "History Processor: ${HISTORY_PROCESSOR_TYPE}"
if [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = "true" ] && COMPRESS_METHOD="tome"
    SAMPLING_TYPE="uniform"
    [ "$LOG_BASE" != "1.0" ] && [ "$LOG_BASE" != "1" ] && SAMPLING_TYPE="logarithmic (b=$LOG_BASE)"
    echo "  Sampling: $SAMPLING_TYPE, ${NUM_HISTORY} frames"
    echo "  Compression: stride=$COMPRESS_STRIDE, method=$COMPRESS_METHOD"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    echo "  GTC Output Tokens: ${GTC_OUTPUT_TOKENS}"
    echo "  GTC Temperature:   ${GTC_TEMPERATURE}"
    echo "  GTC Iterations:    ${GTC_NUM_ITERATIONS}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    echo "  Segment GTC Output Tokens: ${GTC_OUTPUT_TOKENS}"
    echo "  Segment GTC Temperature:   ${GTC_TEMPERATURE}"
    echo "  Segment GTC Iterations:    ${GTC_NUM_ITERATIONS}"
    echo "  Num Segments: 8 (fixed)"
fi
echo "System Prompt:   ${SYSTEM_PROMPT_SETTING}"
echo "Pixel Embed:     ${USE_PIXEL_EMBED}"
echo "Pose Embed:      ${USE_POSE_EMBED} (fusion=${POSE_FUSION_METHOD}, norm_scale=${POSE_NORM_SCALE})"
echo "Save Video:      ${SAVE_VIDEO}"
echo "=============================================="

# ============================================================================
# Pre-check
# ============================================================================
if [ "$MODEL_PATH" == "/path/to/your/trained/checkpoint" ]; then
    echo "[ERROR] Please set MODEL_PATH!"
    echo "Usage: MODEL_PATH=/path/to/checkpoint EVAL_SPLIT=val_unseen bash $0"
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

echo "[INFO] Starting OverlapVLN distributed evaluation on ${NUM_GPUS} GPUs..."

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

# Build history processor arguments
HISTORY_PROCESSOR_ARGS="--history_processor_type ${HISTORY_PROCESSOR_TYPE}"
if [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --compress_stride ${COMPRESS_STRIDE} --log_base ${LOG_BASE}"
    [ "$USE_TOME" = "true" ] && HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --use_tome"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ] || [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_output_tokens ${GTC_OUTPUT_TOKENS}"
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_temperature ${GTC_TEMPERATURE}"
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_num_iterations ${GTC_NUM_ITERATIONS}"
fi

# Build embedding enhancement arguments
EMBED_ENHANCE_ARGS=""
[ "$USE_PIXEL_EMBED" = "true" ] && EMBED_ENHANCE_ARGS="--use_pixel_embed"
if [ "$USE_POSE_EMBED" = "true" ]; then
    EMBED_ENHANCE_ARGS="${EMBED_ENHANCE_ARGS} --use_pose_embed --pose_fusion_method ${POSE_FUSION_METHOD} --pose_norm_scale ${POSE_NORM_SCALE}"
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m swiftvln.models.overlapvln.eval \
    --model_path "${MODEL_PATH}" \
    --env-type "${ENV_TYPE}" \
    --habitat_config_path "${VLN_DIR}/${CONFIG_PATH}" \
    --satnav-config "${VLN_DIR}/${CONFIG_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames "${NUM_FRAMES}" \
    --num_history "${NUM_HISTORY}" \
    --num_future_steps "${NUM_FUTURE_STEPS}" \
    --num_overlap "${NUM_OVERLAP}" \
    --system_prompt_setting "${SYSTEM_PROMPT_SETTING}" \
    ${EMBED_ENHANCE_ARGS} \
    ${HISTORY_PROCESSOR_ARGS} \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    ${VIDEO_ARGS} \
    ${MAX_EPISODES_ARG} \
    ${DEBUG_TIMING_ARG}

echo "=============================================="
echo "OverlapVLN Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
