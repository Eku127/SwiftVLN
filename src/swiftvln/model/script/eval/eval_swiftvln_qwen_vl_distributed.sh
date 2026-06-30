#!/bin/bash
# SwiftVLN Distributed Evaluation Script - Qwen VL families (ms-swift)
# 
# Usage:
#   # Habitat evaluation (default)
#   ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh
#
#   # SatNav evaluation
#   ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint bash src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh
#
# This script runs distributed SwiftVLN evaluation with history frame compression.

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
SWIFTVLN_EVAL_CONDA_ENV="${SWIFTVLN_EVAL_CONDA_ENV:-swift-vln-eval-update}"
conda activate "$SWIFTVLN_EVAL_CONDA_ENV"

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
MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5_vl}"  # qwen2_5_vl | qwen3_vl
case "$MODEL_FAMILY" in
    qwen2_5_vl|qwen25|qwen2.5)
        MODEL_FAMILY="qwen2_5_vl"
        DEFAULT_MODEL_TYPE="swiftvln_qwen2_5_vl"
        DEFAULT_TEMPLATE_TYPE="swiftvln_qwen2_5_vl"
        ;;
    qwen3_vl|qwen3)
        MODEL_FAMILY="qwen3_vl"
        DEFAULT_MODEL_TYPE="swiftvln_qwen3_vl"
        DEFAULT_TEMPLATE_TYPE="swiftvln_qwen3_vl"
        ;;
    *)
        echo "[ERROR] Unknown MODEL_FAMILY: $MODEL_FAMILY. Available: qwen2_5_vl, qwen3_vl."
        exit 1
        ;;
esac
MODEL_TYPE="${MODEL_TYPE:-$DEFAULT_MODEL_TYPE}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-$DEFAULT_TEMPLATE_TYPE}"

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
# SwiftVLN-Specific Parameters
# ============================================================================
NUM_OVERLAP="${NUM_OVERLAP:-0}"  # Number of overlapping actions between windows

# ============================================================================
# History Processor Configuration
# ============================================================================
HISTORY_PROCESSOR_TYPE="${HISTORY_PROCESSOR_TYPE:-per_frame}"  # "per_frame", "gtc", or "segment_gtc"

# ---------- Per-frame parameters (used when HISTORY_PROCESSOR_TYPE="per_frame") ----------
COMPRESS_STRIDE="${COMPRESS_STRIDE:-2}"  # Should match training
USE_TOME="${USE_TOME:-false}"  # Use GridToMe compression (must match training)
LOG_BASE="${LOG_BASE:-1.0}"  # Sampling distribution: 1.0 = uniform, >1.0 = logarithmic
USE_RANDOM="${USE_RANDOM:-false}"  # Use random history sampling (must match training)

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
MEMORY_METHOD="${MEMORY_METHOD:-history}"
MAP_GLOBAL_SIDE_M="${MAP_GLOBAL_SIDE_M:-1000}"
MAP_LOCAL_SIDE_M="${MAP_LOCAL_SIDE_M:-400}"
MAP_RENDER_PX="${MAP_RENDER_PX:-448}"
MAP_MASK_METHOD="${MAP_MASK_METHOD:-dilate20}"

# Map-memory render cache.
# "auto" (default): let the Python layer derive {dataset_root}/map_cache from
# DATA_PATH (e.g. SatNav-v0.1/map_cache or a custom abcd/map_cache), so eval
# warms / reuses the same cache as training. Any absolute path overrides; set to one of
# {off,false,none,0,disable,disabled,no} to disable caching.
MAP_CACHE_DIR="${MAP_CACHE_DIR:-auto}"

# Embedding enhancement (must match training checkpoint setup)
USE_POSE_EMBED="${USE_POSE_EMBED:-false}"
USE_UAV_ADAPTER="${USE_UAV_ADAPTER:-false}"
UAV_ADAPTER_PATH="${UAV_ADAPTER_PATH:-}"
UAV_ADAPTER_TYPE="${UAV_ADAPTER_TYPE:-transformer_v1}"
UAV_ADAPTER_APPLY_SCOPE="${UAV_ADAPTER_APPLY_SCOPE:-all_images}"
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
MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/output/swiftvln/\([^/]*\)/.*|\1|p')
MODEL_NAME="${MODEL_NAME:-unknown_model}"
AUTO_RESUME_EVAL="${AUTO_RESUME_EVAL:-true}"
OUTPUT_DIR_WAS_SET=false
if [ -n "${OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR_WAS_SET=true
fi
DEFAULT_OUTPUT_PARENT="./results/eval/swiftvln/${MODEL_NAME}/${EVAL_SPLIT}"

is_truthy() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

find_latest_incomplete_output_dir() {
    local output_parent="$1"
    local candidate

    [ -d "$output_parent" ] || return 1

    while IFS= read -r candidate; do
        [ -d "$candidate" ] || continue

        # evaluation_summary.json is written only after rank0 completes the
        # offline merge, so treat such directories as completed runs.
        [ -f "${candidate}/evaluation_summary.json" ] && continue

        # Only reuse directories with resume evidence. Empty setup-only
        # directories should not steal a fresh run.
        if [ -s "${candidate}/result.jsonl" ] || [ -d "${candidate}/.dist_sync" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(
        find "$output_parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | \
            sort -rn | cut -d' ' -f2-
    )

    return 1
}

AUTO_RESUME_EVAL_USED=false
if [ "$OUTPUT_DIR_WAS_SET" = false ] && is_truthy "$AUTO_RESUME_EVAL"; then
    RESUME_OUTPUT_DIR="$(find_latest_incomplete_output_dir "$DEFAULT_OUTPUT_PARENT" || true)"
    if [ -n "$RESUME_OUTPUT_DIR" ]; then
        OUTPUT_DIR="$RESUME_OUTPUT_DIR"
        AUTO_RESUME_EVAL_USED=true
        echo "[INFO] AUTO_RESUME_EVAL=true: reusing incomplete output dir: ${OUTPUT_DIR}"
    else
        OUTPUT_DIR="${DEFAULT_OUTPUT_PARENT}/${TIMESTAMP}"
    fi
else
    OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_PARENT}/${TIMESTAMP}}"
fi

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

# Map-memory render cache: forward MAP_CACHE_DIR to the Python layer via the
# SWIFTVLN_MAP_CACHE_DIR env var. "auto" keeps the code default (derive
# {dataset_root}/map_cache from DATA_PATH); explicit paths or "off"-
# family sentinels are passed through verbatim.
if [ "$MEMORY_METHOD" = "map" ]; then
    if [ -n "$MAP_CACHE_DIR" ] && [ "$MAP_CACHE_DIR" != "auto" ]; then
        export SWIFTVLN_MAP_CACHE_DIR="$MAP_CACHE_DIR"
    fi
fi

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
echo "SwiftVLN Distributed Evaluation"
echo "=============================================="
echo "Environment:     ${ENV_TYPE}"
echo "Model Family:    ${MODEL_FAMILY}"
echo "Model Type:      ${MODEL_TYPE}"
echo "Template Type:   ${TEMPLATE_TYPE}"
echo "Config Path:     ${CONFIG_PATH}"
echo "Model Path:      ${MODEL_PATH}"
echo "Eval Split:      ${EVAL_SPLIT}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Auto Resume:     ${AUTO_RESUME_EVAL} (used=${AUTO_RESUME_EVAL_USED}, explicit_output_dir=${OUTPUT_DIR_WAS_SET})"
echo "Num GPUs:        ${NUM_GPUS}"
echo "CUDA Devices:    ${CUDA_DEVICES}"
echo "Num Overlap:     ${NUM_OVERLAP}"
echo "Memory Method:   ${MEMORY_METHOD}"
if [ "$MEMORY_METHOD" = "map" ]; then
    echo "  Map: global=${MAP_GLOBAL_SIDE_M}m, local=${MAP_LOCAL_SIDE_M}m, render=${MAP_RENDER_PX}px, mask=${MAP_MASK_METHOD}"
    echo "  Render cache: MAP_CACHE_DIR=${MAP_CACHE_DIR} (env SWIFTVLN_MAP_CACHE_DIR=${SWIFTVLN_MAP_CACHE_DIR:-<unset, will derive from DATA_PATH>})"
    echo "  Compression: stride=${COMPRESS_STRIDE}, method=pool"
else
    echo "History Processor: ${HISTORY_PROCESSOR_TYPE}"
fi
if [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = "true" ] && COMPRESS_METHOD="tome"
    if [ "$USE_RANDOM" = "true" ]; then
        SAMPLING_TYPE="random (log_base ignored)"
    else
        SAMPLING_TYPE="uniform"
        [ "$LOG_BASE" != "1.0" ] && [ "$LOG_BASE" != "1" ] && SAMPLING_TYPE="logarithmic (b=$LOG_BASE)"
    fi
    echo "  Sampling: $SAMPLING_TYPE, ${NUM_HISTORY} frames"
    echo "  Use Random: $USE_RANDOM"
    echo "  Compression: stride=$COMPRESS_STRIDE, method=$COMPRESS_METHOD"
elif [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    echo "  GTC Output Tokens: ${GTC_OUTPUT_TOKENS}"
    echo "  GTC Temperature:   ${GTC_TEMPERATURE}"
    echo "  GTC Iterations:    ${GTC_NUM_ITERATIONS}"
elif [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    echo "  Segment GTC Output Tokens: ${GTC_OUTPUT_TOKENS}"
    echo "  Segment GTC Temperature:   ${GTC_TEMPERATURE}"
    echo "  Segment GTC Iterations:    ${GTC_NUM_ITERATIONS}"
    echo "  Num Segments: 8 (fixed)"
fi
echo "System Prompt:   ${SYSTEM_PROMPT_SETTING}"
echo "Pose Embed:      ${USE_POSE_EMBED} (fusion=${POSE_FUSION_METHOD}, norm_scale=${POSE_NORM_SCALE})"
echo "UAV Adapter:     ${USE_UAV_ADAPTER} (path=${UAV_ADAPTER_PATH:-<none>}, type=${UAV_ADAPTER_TYPE}, scope=${UAV_ADAPTER_APPLY_SCOPE})"
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

if [ "$MEMORY_METHOD" = "map" ]; then
    if [ "$ENV_TYPE" != "satnav" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently supports only ENV_TYPE=satnav."
        exit 1
    fi
    if [ "$HISTORY_PROCESSOR_TYPE" != "per_frame" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently requires HISTORY_PROCESSOR_TYPE=per_frame."
        exit 1
    fi
    if [ "$USE_TOME" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently requires USE_TOME=false."
        exit 1
    fi
    # Map images are synthesized top-down views, so RGB-frame embed
    # enhancements (pose / uav_adapter) are not meaningful and must
    # match the training-time constraint of staying disabled.
    if [ "$USE_POSE_EMBED" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map requires USE_POSE_EMBED=false."
        exit 1
    fi
    if [ "$USE_UAV_ADAPTER" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map requires USE_UAV_ADAPTER=false."
        exit 1
    fi
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

echo "[INFO] Starting SwiftVLN distributed evaluation on ${NUM_GPUS} GPUs..."

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
    [ "$USE_RANDOM" = "true" ] && HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --use_random"
    [ "$USE_TOME" = "true" ] && HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --use_tome"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ] || [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_output_tokens ${GTC_OUTPUT_TOKENS}"
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_temperature ${GTC_TEMPERATURE}"
    HISTORY_PROCESSOR_ARGS="${HISTORY_PROCESSOR_ARGS} --gtc_num_iterations ${GTC_NUM_ITERATIONS}"
fi

# Build the eval command with arrays so optional args cannot break shell parsing.
EVAL_CMD=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    -m swiftvln.model.eval
    --model_path "${MODEL_PATH}"
    --model_type "${MODEL_TYPE}"
    --template_type "${TEMPLATE_TYPE}"
    --env-type "${ENV_TYPE}"
    --habitat_config_path "${VLN_DIR}/${CONFIG_PATH}"
    --satnav-config "${VLN_DIR}/${CONFIG_PATH}"
    --eval_split "${EVAL_SPLIT}"
    --num_frames "${NUM_FRAMES}"
    --num_history "${NUM_HISTORY}"
    --num_future_steps "${NUM_FUTURE_STEPS}"
    --num_overlap "${NUM_OVERLAP}"
    --system_prompt_setting "${SYSTEM_PROMPT_SETTING}"
    --memory_method "${MEMORY_METHOD}"
    --map_global_side_m "${MAP_GLOBAL_SIDE_M}"
    --map_local_side_m "${MAP_LOCAL_SIDE_M}"
    --map_render_px "${MAP_RENDER_PX}"
    --map_mask_method "${MAP_MASK_METHOD}"
)

if [ "$USE_POSE_EMBED" = "true" ]; then
    EVAL_CMD+=(--use_pose_embed --pose_fusion_method "${POSE_FUSION_METHOD}" --pose_norm_scale "${POSE_NORM_SCALE}")
fi
if [ "$USE_UAV_ADAPTER" = "true" ]; then
    EVAL_CMD+=(--use_uav_adapter --uav_adapter_type "${UAV_ADAPTER_TYPE}" --uav_adapter_apply_scope "${UAV_ADAPTER_APPLY_SCOPE}")
    if [ -n "$UAV_ADAPTER_PATH" ]; then
        EVAL_CMD+=(--uav_adapter_path "${UAV_ADAPTER_PATH}")
    fi
fi

if [ -n "${HISTORY_PROCESSOR_ARGS}" ]; then
    read -r -a _HISTORY_ARGS <<< "${HISTORY_PROCESSOR_ARGS}"
    EVAL_CMD+=("${_HISTORY_ARGS[@]}")
fi

EVAL_CMD+=(--output_dir "${OUTPUT_DIR}" --distributed)

if [ -n "${VIDEO_ARGS}" ]; then
    read -r -a _VIDEO_ARGS <<< "${VIDEO_ARGS}"
    EVAL_CMD+=("${_VIDEO_ARGS[@]}")
fi
if [ -n "${MAX_EPISODES_ARG}" ]; then
    read -r -a _MAX_EPISODES_ARGS <<< "${MAX_EPISODES_ARG}"
    EVAL_CMD+=("${_MAX_EPISODES_ARGS[@]}")
fi
if [ -n "${DEBUG_TIMING_ARG}" ]; then
    read -r -a _DEBUG_TIMING_ARGS <<< "${DEBUG_TIMING_ARG}"
    EVAL_CMD+=("${_DEBUG_TIMING_ARGS[@]}")
fi

"${EVAL_CMD[@]}"

echo "=============================================="
echo "SwiftVLN Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
