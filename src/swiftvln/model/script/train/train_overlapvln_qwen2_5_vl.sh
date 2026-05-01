#!/bin/bash
# OverlapVLN Training Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   bash src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh
#
# This script trains OverlapVLN with history frame compression.
# Key difference from StreamVLN: adds compress_stride parameter

set -e  # Exit on error

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
SWIFTVLN_TRAIN_CONDA_ENV="${SWIFTVLN_TRAIN_CONDA_ENV:-swift-vln-train-update}"
conda activate "$SWIFTVLN_TRAIN_CONDA_ENV"

# ============================================================================
# GPU Configuration
# ============================================================================
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-}"   # Explicit GPU list, e.g. "0,1,2"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-}"           # Number of GPUs to take from current visible set
TRAIN_DRY_RUN="${TRAIN_DRY_RUN:-false}"        # true = print config and exit before torchrun
CUDA_DEVICES="${CUDA_DEVICES:-auto}"           # "auto" = all currently visible GPUs
MASTER_PORT="${MASTER_PORT:-29500}"            # Master port for distributed training
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"  # Optional full training resume checkpoint
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"       # true = load weights only from resume checkpoint
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"        # Optional output root override before versioning

normalize_cuda_device_list() {
    local raw="$1"
    echo "$raw" | tr -d '[:space:]' | sed 's/^,*//; s/,*$//; s/,,*/,/g'
}

get_current_visible_cuda_devices() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        normalize_cuda_device_list "${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    python - <<'PY'
import sys
import torch

count = torch.cuda.device_count()
if count <= 0:
    sys.exit(1)
print(",".join(str(i) for i in range(count)))
PY
}

take_first_n_cuda_devices() {
    local device_list="$1"
    local request_count="$2"
    local -a devices=()
    local -a selected=()
    local idx=0

    IFS=',' read -r -a devices <<< "$device_list"
    if (( request_count < 1 )); then
        echo "[ERROR] TRAIN_NUM_GPUS must be >= 1, got: ${request_count}" >&2
        return 1
    fi
    if (( request_count > ${#devices[@]} )); then
        echo "[ERROR] TRAIN_NUM_GPUS=${request_count} exceeds available visible GPUs (${#devices[@]}): ${device_list}" >&2
        return 1
    fi

    while (( idx < request_count )); do
        selected+=("${devices[$idx]}")
        ((idx++))
    done

    local joined=""
    local item=""
    for item in "${selected[@]}"; do
        if [[ -n "$joined" ]]; then
            joined+=","
        fi
        joined+="$item"
    done
    echo "$joined"
}

resolve_cuda_devices() {
    local visible_devices=""

    if [[ -n "$TRAIN_CUDA_DEVICES" ]]; then
        normalize_cuda_device_list "$TRAIN_CUDA_DEVICES"
        return 0
    fi

    if [[ -n "$CUDA_DEVICES" && "$CUDA_DEVICES" != "auto" ]]; then
        normalize_cuda_device_list "$CUDA_DEVICES"
        return 0
    fi

    visible_devices="$(get_current_visible_cuda_devices)" || {
        echo "[ERROR] No visible CUDA devices detected in current environment." >&2
        return 1
    }
    visible_devices="$(normalize_cuda_device_list "$visible_devices")"

    if [[ -n "$TRAIN_NUM_GPUS" ]]; then
        take_first_n_cuda_devices "$visible_devices" "$TRAIN_NUM_GPUS"
        return 0
    fi

    echo "$visible_devices"
}

CUDA_DEVICES="$(resolve_cuda_devices)"
if [[ -z "$CUDA_DEVICES" ]]; then
    echo "[ERROR] Failed to resolve CUDA devices." >&2
    exit 1
fi

# Auto-detect GPU count from resolved CUDA_DEVICES
GPUS_PER_NODE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [[ "$GPUS_PER_NODE" -lt 1 ]]; then
    echo "[ERROR] Resolved GPU count is invalid: ${GPUS_PER_NODE} (CUDA_DEVICES=${CUDA_DEVICES})" >&2
    exit 1
fi

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_TYPE="overlapvln_qwen2_5_vl"

# Training stage: "stage1" (from base Qwen) or "stage2" (from trained VLN model)
TRAIN_STAGE="${TRAIN_STAGE:-stage1}"

# Model paths for each stage
# Stage1 defaults to the local offline cache path to avoid ModelScope hub resolution.
# Default remains the local 3B cache path. For 7B, override STAGE1_MODEL_PATH
# via env, e.g.:
#   STAGE1_MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct}"
STAGE2_MODEL_PATH="${STAGE2_MODEL_PATH:-/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-3b-1ep-f32h8s4-overlap16-stride2-bs64-lr2e-5-20260124-214153/v0-20260124-214234/checkpoint-2239}"

# Select model path based on stage
if [ "$TRAIN_STAGE" == "stage1" ]; then
    MODEL_PATH="$STAGE1_MODEL_PATH"
elif [ "$TRAIN_STAGE" == "stage2" ]; then
    MODEL_PATH="$STAGE2_MODEL_PATH"
else
    echo "[ERROR] Unknown TRAIN_STAGE: $TRAIN_STAGE. Must be 'stage1' or 'stage2'."
    exit 1
fi

# Extract model size for experiment naming
MODEL_SIZE=$(echo "$MODEL_PATH" | grep -oE '[0-9]+B' | tr '[:upper:]' '[:lower:]')
MODEL_SIZE=${MODEL_SIZE:-"3b"}

# ============================================================================
# VLN Data Configuration
# ============================================================================
# Environment type: "habitat" (forward=0.25m) or "satnav" (forward=10m)
VLN_ENV_TYPE="${VLN_ENV_TYPE:-satnav}"

# Define data paths for each environment
HABITAT_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
)
SATNAV_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/trajectory_data"
)

# Select data paths based on VLN_ENV_TYPE (using nameref)
declare -n VLN_DATA_PATHS="${VLN_ENV_TYPE^^}_DATA_PATHS"
if [ ${#VLN_DATA_PATHS[@]} -eq 0 ]; then
    echo "[ERROR] Unknown VLN_ENV_TYPE: $VLN_ENV_TYPE. Available: habitat, satnav"
    exit 1
fi

VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")

# VLN-Specific Parameters
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-4}"
USE_RANDOM="${USE_RANDOM:-false}"
# Default to baseline full-data training.
# 0 means "use all available samples".
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ============================================================================
# Mixed Training: QA Dataset Configuration (Optional)
# ============================================================================
# Set USE_QA_MIXED_TRAINING=true to enable mixed training with VLN + QA data
USE_QA_MIXED_TRAINING="${USE_QA_MIXED_TRAINING:-false}"
QA_DATASET="${QA_DATASET:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/data/qa_swift.jsonl}"
QA_RATIO="${QA_RATIO:-0.15}"             # Ratio of QA samples (0.15 = 15% QA, 85% VLN)
QA_MAX_SAMPLES="${QA_MAX_SAMPLES:-0}"    # Max QA samples (0 = use all available)

# ============================================================================
# OverlapVLN-Specific Parameters
# ============================================================================
# History Processor Type: "per_frame", "gtc", or "segment_gtc"
# - per_frame: Per-frame compression (default), each frame compressed independently
#   - log_base=1.0: Uniform sampling
#   - log_base>1.0: Logarithmic sampling (more recent frames preserved)
# - gtc: Global Token Clustering, cross-frame clustering to fixed tokens
# - segment_gtc: Segment-wise GTC, splits history into 8 segments
HISTORY_PROCESSOR_TYPE="${HISTORY_PROCESSOR_TYPE:-per_frame}"

# ---------- Per-frame parameters (used when HISTORY_PROCESSOR_TYPE="per_frame") ----------
# Compression stride (2 = 4x compression, 3 = 9x, 4 = 16x)
COMPRESS_STRIDE="${COMPRESS_STRIDE:-2}"

# Compression method: false = average pooling, true = GridToMe
# GridToMe provides better semantic preservation for small objects
USE_TOME="${USE_TOME:-false}"

# Sampling distribution: 1.0 = uniform, >1.0 = logarithmic (more recent frames)
# 2.0 = moderate, 3.0+ = aggressive concentration on recent frames
LOG_BASE="${LOG_BASE:-1.0}"

# ---------- GTC/SegmentGTC parameters (used when HISTORY_PROCESSOR_TYPE="gtc" or "segment_gtc") ----------
# Fixed number of output tokens for Global Token Clustering / Segment GTC
GTC_OUTPUT_TOKENS="${GTC_OUTPUT_TOKENS:-512}"

# Temperature for soft assignment in Soft K-Means (lower = sharper)
GTC_TEMPERATURE="${GTC_TEMPERATURE:-0.1}"

# Number of Soft K-Means iterations (1-2 usually sufficient)
GTC_NUM_ITERATIONS="${GTC_NUM_ITERATIONS:-1}"

# ---------- Sliding window overlap configuration ----------
# Baseline default (2026-04-17): overlap=0.
# Historical baseline used overlap=16 (50% overlap with num_frames=32, stride=16).
# num_overlap: Number of overlapping actions between consecutive windows
# When num_overlap > 0, stride = num_frames - num_overlap
# First (num_overlap / num_future_steps) turns in non-first samples have loss masked
NUM_OVERLAP="${NUM_OVERLAP:-0}"
# Legacy tail window adjustment for overlap training.
# false: keep strict stride-aligned overlap windows (current default for overlap > 0)
# true: move short tail windows backward to cover end-of-episode/STOP data
OVERLAP_TAIL_WINDOW_ADJUST="${OVERLAP_TAIL_WINDOW_ADJUST:-false}"

# ---------- System prompt setting ----------
# System prompt strategy: "vanilla" (default, no initial view) or "initial"
# - vanilla: Standard prompt without initial view image
# - initial: Add the first frame of the episode (uncompressed) to the system prompt
#   as the initial observation at the starting point of the journey
SYSTEM_PROMPT_SETTING="${SYSTEM_PROMPT_SETTING:-vanilla}"

# ---------- Memory method ----------
# history: original historical RGB frames
# map: SatNav explored-map memory (global + local), replaces history frames
MEMORY_METHOD="${MEMORY_METHOD:-history}"
MAP_GLOBAL_SIDE_M="${MAP_GLOBAL_SIDE_M:-1000}"
MAP_LOCAL_SIDE_M="${MAP_LOCAL_SIDE_M:-400}"
MAP_RENDER_PX="${MAP_RENDER_PX:-448}"
MAP_MASK_METHOD="${MAP_MASK_METHOD:-dilate20}"

# ---------- Map-memory render cache ----------
# Caches rendered (global, local) PNG pairs on disk to eliminate rasterio
# re-rendering cost across epochs. Default ("auto"): the Python layer uses
#   {dataset_root}/map_cache
# (i.e. co-located with ver_260418). Override with any absolute path, or set
# to one of {off,false,none,0,disable,disabled,no} to disable caching.
# Only has effect when MEMORY_METHOD=map.
MAP_CACHE_DIR="${MAP_CACHE_DIR:-auto}"

# ---------- Embedding enhancement ----------
# Pixel coordinate embedding enhancement (Fourier + MLP)
# - false: disable (default)
# - true: enable and train pixel embedding module
USE_PIXEL_EMBED="${USE_PIXEL_EMBED:-false}"

# Pose embedding enhancement (MLP, per-image pose injection)
# - false: disable (default)
# - true: enable and train pose embedding module
USE_POSE_EMBED="${USE_POSE_EMBED:-false}"

# Stage-A UAV adapter enhancement
# - false: disable (default)
# - true: enable and optionally load from an external s2r checkpoint
USE_UAV_ADAPTER="${USE_UAV_ADAPTER:-false}"
UAV_ADAPTER_PATH="${UAV_ADAPTER_PATH:-}"
UAV_ADAPTER_TYPE="${UAV_ADAPTER_TYPE:-transformer_v1}"
UAV_ADAPTER_APPLY_SCOPE="${UAV_ADAPTER_APPLY_SCOPE:-all_images}"

# Pose fusion method: "additive" (default) or "film"
POSE_FUSION_METHOD="${POSE_FUSION_METHOD:-additive}"

# Pose normalization scale for tanh(pos/scale), default 100.0
POSE_NORM_SCALE="${POSE_NORM_SCALE:-100.0}"

# ============================================================================
# Training Parameters
# ============================================================================
TRAIN_TYPE="${TRAIN_TYPE:-full}"
TRAIN_MODE_ARGS=(--tuner_type "$TRAIN_TYPE")
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-32768}"

# Model Freezing
FREEZE_VIT="${FREEZE_VIT:-false}"
FREEZE_LLM="${FREEZE_LLM:-false}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"

# Optimization
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
TF32="${TF32:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"

# Learning Rate Schedule
WARMUP_RATIO="${WARMUP_RATIO:-0.075}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine_with_min_lr}"
LR_SCHEDULER_KWARGS="${LR_SCHEDULER_KWARGS:-{\"min_lr\":1.85e-05}}"

# Attention Implementation
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

# ============================================================================
# Performance Acceleration
# ============================================================================
# NOTE: padding_free must be FALSE for OverlapVLN (custom tokens incompatible)
PADDING_FREE="${PADDING_FREE:-false}"

USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-10}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * GPUS_PER_NODE))
WINDOW_STRIDE=$((NUM_FRAMES - NUM_OVERLAP))

if [ "$MEMORY_METHOD" = "map" ]; then
    if [ "$VLN_ENV_TYPE" != "satnav" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently supports only VLN_ENV_TYPE=satnav."
        exit 1
    fi
    if [ "$HISTORY_PROCESSOR_TYPE" != "per_frame" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently requires HISTORY_PROCESSOR_TYPE=per_frame."
        exit 1
    fi
    if [ "$USE_TOME" = true ] || [ "$USE_TOME" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map currently requires USE_TOME=false."
        exit 1
    fi
    # Map images are synthesized top-down views, so RGB-frame embed
    # enhancements (pixel / pose / uav_adapter) are not meaningful and must
    # stay disabled to avoid silent semantic mismatches.
    if [ "$USE_PIXEL_EMBED" = true ] || [ "$USE_PIXEL_EMBED" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map requires USE_PIXEL_EMBED=false."
        exit 1
    fi
    if [ "$USE_POSE_EMBED" = true ] || [ "$USE_POSE_EMBED" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map requires USE_POSE_EMBED=false."
        exit 1
    fi
    if [ "$USE_UAV_ADAPTER" = true ] || [ "$USE_UAV_ADAPTER" = "true" ]; then
        echo "[ERROR] MEMORY_METHOD=map requires USE_UAV_ADAPTER=false."
        exit 1
    fi
fi

# Build experiment name based on memory method
MEMORY_SUFFIX=""
if [ "$MEMORY_METHOD" = "map" ]; then
    MAP_GLOBAL_TAG=$(printf '%g' "$MAP_GLOBAL_SIDE_M")
    MAP_LOCAL_TAG=$(printf '%g' "$MAP_LOCAL_SIDE_M")
    MAP_MASK_TAG="$MAP_MASK_METHOD"
    if [[ "$MAP_MASK_METHOD" =~ ^dilate([0-9]+([.][0-9]+)?)$ ]]; then
        MAP_MASK_TAG="d$(printf '%g' "${BASH_REMATCH[1]}")"
    fi
    MEMORY_SUFFIX="map-g${MAP_GLOBAL_TAG}-l${MAP_LOCAL_TAG}-r${MAP_RENDER_PX}-${MAP_MASK_TAG}-s${COMPRESS_STRIDE}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    # Per-frame: include history count, log_base, method, stride.
    # NUM_HISTORY=0 is the supported no-memory configuration:
    # the dataset will sample zero history frames and omit <history_memory>.
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = true ] && COMPRESS_METHOD="tome"
    NO_MEMORY_SUFFIX=""
    RANDOM_SUFFIX=""
    if [ "$NUM_HISTORY" = "0" ]; then
        NO_MEMORY_SUFFIX="-nomem"
    elif [ "$USE_RANDOM" = true ] || [ "$USE_RANDOM" = "true" ]; then
        # Only tag random when it actually changes per-frame history sampling.
        RANDOM_SUFFIX="-random"
    fi
    MEMORY_SUFFIX="pf-h${NUM_HISTORY}${NO_MEMORY_SUFFIX}${RANDOM_SUFFIX}-b${LOG_BASE}-${COMPRESS_METHOD}-s${COMPRESS_STRIDE}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    # GTC: include output tokens
    MEMORY_SUFFIX="gtc-k${GTC_OUTPUT_TOKENS}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    # Segment GTC: include output tokens (8 segments, chronological order by default)
    MEMORY_SUFFIX="sgtc-k${GTC_OUTPUT_TOKENS}"
fi

# Add QA suffix if mixed training is enabled
QA_SUFFIX=""
if [ "$USE_QA_MIXED_TRAINING" = true ]; then
    # Convert ratio to percentage (e.g., 0.15 -> 15)
    QA_PCT=$(awk "BEGIN {printf \"%.0f\", ${QA_RATIO} * 100}")
    QA_SUFFIX="-qa${QA_PCT}"
fi

# Add system prompt setting suffix (vanilla = no suffix, others = -<setting>)
PROMPT_SUFFIX=""
if [ "$SYSTEM_PROMPT_SETTING" != "vanilla" ]; then
    PROMPT_SUFFIX="-${SYSTEM_PROMPT_SETTING}"
fi

# Embedding enhancement suffix (pixel + pose combined in one slot)
EMBED_SUFFIX="-noembed"
_EMBED_PARTS=()
if [ "$USE_PIXEL_EMBED" = true ] || [ "$USE_PIXEL_EMBED" = "true" ]; then
    _EMBED_PARTS+=("pixel")
fi
if [ "$USE_POSE_EMBED" = true ] || [ "$USE_POSE_EMBED" = "true" ]; then
    if [ "$POSE_FUSION_METHOD" = "film" ]; then
        _EMBED_PARTS+=("posefilm")
    else
        _EMBED_PARTS+=("pose")
    fi
fi
if [ "$USE_UAV_ADAPTER" = true ] || [ "$USE_UAV_ADAPTER" = "true" ]; then
    _EMBED_PARTS+=("uav")
fi
if [ ${#_EMBED_PARTS[@]} -gt 0 ]; then
    EMBED_SUFFIX="-$(IFS='+'; echo "${_EMBED_PARTS[*]}")"
fi

# Add data version suffix for SatNav (extract version from path, e.g., ver_260202 -> data260202)
DATA_VERSION_SUFFIX=""
if [ "$VLN_ENV_TYPE" = "satnav" ]; then
    # Extract version from first selected SatNav data path (e.g., /path/ver_260202/... -> 260202)
    VERSION_STR=$(echo "${VLN_DATA_PATHS[0]}" | grep -oP 'ver_\K\d+' | head -1)
    if [ -n "$VERSION_STR" ] && [ ${#VERSION_STR} -eq 6 ]; then
        DATA_VERSION_SUFFIX="-data${VERSION_STR}"
    else
        echo "[WARNING] SatNav data version not found in path: ${VLN_DATA_PATHS[0]}"
    fi
fi

_OVERLAP_TAIL_WINDOW_ADJUST_NORMALIZED="$(echo "$OVERLAP_TAIL_WINDOW_ADJUST" | tr '[:upper:]' '[:lower:]')"
_OVERLAP_TAIL_WINDOW_ADJUST_ENABLED=false
case "$_OVERLAP_TAIL_WINDOW_ADJUST_NORMALIZED" in
    true|1|yes|y|on)
        _OVERLAP_TAIL_WINDOW_ADJUST_ENABLED=true
        ;;
esac

TAIL_WINDOW_SUFFIX=""
if [ "$NUM_OVERLAP" -gt 0 ] && [ "$_OVERLAP_TAIL_WINDOW_ADJUST_ENABLED" != "true" ]; then
    TAIL_WINDOW_SUFFIX="-notailadj"
fi

EXP_NAME="overlapvln-${VLN_ENV_TYPE}-${TRAIN_STAGE}-${MODEL_SIZE}-${NUM_EPOCHS}ep-f${NUM_FRAMES}s${NUM_FUTURE_STEPS}-overlap${NUM_OVERLAP}-${MEMORY_SUFFIX}${PROMPT_SUFFIX}${EMBED_SUFFIX}${TAIL_WINDOW_SUFFIX}${DATA_VERSION_SUFFIX}${QA_SUFFIX}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/overlapvln/${EXP_NAME}"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
fi

# Checkpoint Management
# Default to production-style checkpoint cadence. Smoke tests must override this
# explicitly if they need per-step checkpointing.
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Note: MAX_SAMPLES is a soft cap - if actual samples < MAX_SAMPLES, all available samples are used.
# Training is controlled by num_train_epochs, not max_steps, so the trainer will iterate
# over the actual dataset size. This avoids over-iteration when actual samples < MAX_SAMPLES.
MAX_STEPS="${MAX_STEPS:-}"
MAX_STEPS_ARG=""
if [[ -n "$MAX_STEPS" && "$MAX_STEPS" -gt 0 ]]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES set as upper limit. Training controlled by num_epochs=$NUM_EPOCHS"
    echo "[INFO] If actual samples < MAX_SAMPLES, all available samples will be used."
fi

# ============================================================================
# SwanLab Configuration
# ============================================================================
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-StreamVLN}"
SWANLAB_EXP_NAME="${EXP_NAME}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

# WXWork Notification
USE_WXWORK_NOTIFICATION="${USE_WXWORK_NOTIFICATION:-false}"
SWANLAB_NOTIFICATION_METHOD="${SWANLAB_NOTIFICATION_METHOD:-wxwork}"
SWANLAB_WEBHOOK_URL="${SWANLAB_WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d78d3128-7b16-4bf1-a6a7-403bf0915fe0}"
SWANLAB_SECRET="${SWANLAB_SECRET:-}"

# ============================================================================
# Environment Setup
# ============================================================================
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=4
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Map-memory render cache: forward MAP_CACHE_DIR to the Python layer via the
# OVERLAPVLN_MAP_CACHE_DIR env var. "auto" keeps the code default (dataset_root
# /map_cache); explicit paths or "off"-family sentinels are passed through.
if [ "$MEMORY_METHOD" = "map" ]; then
    if [ -n "$MAP_CACHE_DIR" ] && [ "$MAP_CACHE_DIR" != "auto" ]; then
        export OVERLAPVLN_MAP_CACHE_DIR="$MAP_CACHE_DIR"
    fi
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "=========================================="
echo "OverlapVLN Training"
echo "=========================================="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Environment: $VLN_ENV_TYPE"
echo "Data: $VLN_DATA_PATH"
if [ "$VLN_ENV_TYPE" = "satnav" ]; then
    echo "Data Version Tag: ${DATA_VERSION_SUFFIX:-[MISSING]}"
fi
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
if [[ -n "$TRAIN_CUDA_DEVICES" ]]; then
    echo "GPU Override: TRAIN_CUDA_DEVICES=$TRAIN_CUDA_DEVICES"
elif [[ -n "$TRAIN_NUM_GPUS" ]]; then
    echo "GPU Override: TRAIN_NUM_GPUS=$TRAIN_NUM_GPUS"
else
    echo "GPU Override: auto-detect visible GPUs"
fi
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR: $LEARNING_RATE | Epochs: $NUM_EPOCHS"
echo "Attention: $ATTN_IMPL"
echo "------------------------------------------"
echo "Memory Method: $MEMORY_METHOD"
if [ "$MEMORY_METHOD" = "map" ]; then
    echo "  Map: global=${MAP_GLOBAL_SIDE_M}m, local=${MAP_LOCAL_SIDE_M}m, render=${MAP_RENDER_PX}px, mask=${MAP_MASK_METHOD}"
    echo "  Compression: per_frame stride=$COMPRESS_STRIDE ($((COMPRESS_STRIDE * COMPRESS_STRIDE))x), method=pool"
    echo "  Render cache: MAP_CACHE_DIR=${MAP_CACHE_DIR} (env OVERLAPVLN_MAP_CACHE_DIR=${OVERLAPVLN_MAP_CACHE_DIR:-<unset, will use dataset_root/map_cache>})"
else
    echo "History Processor: $HISTORY_PROCESSOR_TYPE"
fi
if [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = true ] && COMPRESS_METHOD="tome"
    if [ "$NUM_HISTORY" = "0" ]; then
        echo "  Sampling: disabled (no-memory, NUM_HISTORY=0; log_base/use_random ignored)"
        echo "  Compression: stride=$COMPRESS_STRIDE ($((COMPRESS_STRIDE * COMPRESS_STRIDE))x), method=$COMPRESS_METHOD [unused while no-memory is active]"
    else
        if [ "$USE_RANDOM" = true ]; then
            SAMPLING_TYPE="random"
        else
            SAMPLING_TYPE="uniform"
            [ "$LOG_BASE" != "1.0" ] && [ "$LOG_BASE" != "1" ] && SAMPLING_TYPE="logarithmic (b=$LOG_BASE)"
        fi
        echo "  Sampling: $SAMPLING_TYPE, ${NUM_HISTORY} frames"
        echo "  Compression: stride=$COMPRESS_STRIDE ($((COMPRESS_STRIDE * COMPRESS_STRIDE))x), method=$COMPRESS_METHOD"
    fi
elif [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    echo "  Sampling: every ${NUM_FUTURE_STEPS} frames from history"
    echo "  GTC: output_tokens=$GTC_OUTPUT_TOKENS, temperature=$GTC_TEMPERATURE, iterations=$GTC_NUM_ITERATIONS"
elif [ "$MEMORY_METHOD" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    echo "  Sampling: every ${NUM_FUTURE_STEPS} frames from history"
    echo "  SegmentGTC: output_tokens=$GTC_OUTPUT_TOKENS, segments=8, temperature=$GTC_TEMPERATURE, iterations=$GTC_NUM_ITERATIONS"
fi
echo "Overlap: num_overlap=$NUM_OVERLAP, window_stride=$WINDOW_STRIDE"
echo "  First $((NUM_OVERLAP / NUM_FUTURE_STEPS)) turns masked for samples with start_idx > 0"
echo "  Tail window adjust: $OVERLAP_TAIL_WINDOW_ADJUST"
echo "System Prompt: $SYSTEM_PROMPT_SETTING"
echo "Pixel Embed: $USE_PIXEL_EMBED"
echo "Pose Embed:  $USE_POSE_EMBED (fusion=$POSE_FUSION_METHOD, norm_scale=$POSE_NORM_SCALE)"
echo "UAV Adapter: $USE_UAV_ADAPTER (type=$UAV_ADAPTER_TYPE, scope=$UAV_ADAPTER_APPLY_SCOPE)"
[ -n "$UAV_ADAPTER_PATH" ] && echo "  UAV Adapter Path: $UAV_ADAPTER_PATH"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "Resume: $RESUME_FROM_CHECKPOINT (resume_only_model=$RESUME_ONLY_MODEL)"
fi
echo "------------------------------------------"
# Mixed training info
if [ "$USE_QA_MIXED_TRAINING" = true ]; then
    echo "Mixed Training: ENABLED"
    echo "  QA Dataset: $QA_DATASET"
    echo "  QA Ratio: ${QA_RATIO} (QA $(awk "BEGIN {printf \"%.0f\", ${QA_RATIO} * 100}")%, VLN $(awk "BEGIN {printf \"%.0f\", (1 - ${QA_RATIO}) * 100}")%)"
    [ "$QA_MAX_SAMPLES" -gt 0 ] 2>/dev/null && echo "  QA Max Samples: $QA_MAX_SAMPLES"
else
    echo "Mixed Training: DISABLED (VLN only)"
fi
echo "------------------------------------------"
echo "Freeze ViT: $FREEZE_VIT | LLM: $FREEZE_LLM | Aligner: $FREEZE_ALIGNER"
echo "DeepSpeed: $USE_DEEPSPEED ($DEEPSPEED_CONFIG)"
echo "------------------------------------------"
echo "Acceleration:"
echo "  dataloader_num_workers: $DATALOADER_NUM_WORKERS"
echo "  dataloader_pin_memory: $DATALOADER_PIN_MEMORY"
echo "  padding_free: $PADDING_FREE (must be false for OverlapVLN)"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
SWANLAB_DIRECT_NETWORK="${SWANLAB_DIRECT_NETWORK:-true}"

unset_proxy_for_swanlab() {
    local -a proxy_vars=(http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY)
    local proxy_var=""
    local had_proxy="false"

    for proxy_var in "${proxy_vars[@]}"; do
        if [[ -n "${!proxy_var:-}" ]]; then
            unset "$proxy_var"
            had_proxy="true"
        fi
    done

    if [[ "$had_proxy" == "true" ]]; then
        echo "[INFO] SwanLab enabled: unset proxy env vars for direct SwanLab access."
    fi
}

# DeepSpeed argument
DEEPSPEED_ARG=""
[ "$USE_DEEPSPEED" = true ] && DEEPSPEED_ARG="--deepspeed $DEEPSPEED_CONFIG"

# SwanLab arguments
SWANLAB_ARGS=""
if [ "$USE_SWANLAB" = true ]; then
    SWANLAB_ARGS="--report_to swanlab --swanlab_project $SWANLAB_PROJECT --swanlab_exp_name $SWANLAB_EXP_NAME --swanlab_mode $SWANLAB_MODE"
    
    if [ "$USE_WXWORK_NOTIFICATION" = true ]; then
        SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_notification_method $SWANLAB_NOTIFICATION_METHOD --swanlab_webhook_url $SWANLAB_WEBHOOK_URL"
        if [ -n "$SWANLAB_SECRET" ]; then
            SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_secret $SWANLAB_SECRET"
        fi
    fi
fi

# Attention implementation argument
ATTN_ARG=""
[ -n "$ATTN_IMPL" ] && ATTN_ARG="--attn_impl $ATTN_IMPL"

# QA dataset arguments (for mixed training)
QA_ARGS=""
if [ "$USE_QA_MIXED_TRAINING" = true ]; then
    QA_ARGS="--qa_dataset $QA_DATASET --qa_ratio $QA_RATIO"
    [ "$QA_MAX_SAMPLES" -gt 0 ] 2>/dev/null && QA_ARGS="$QA_ARGS --qa_max_samples $QA_MAX_SAMPLES"
fi

# History processor arguments
HISTORY_ARGS="--history_processor_type $HISTORY_PROCESSOR_TYPE"
if [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    # Per-frame: pass log_base for sampling distribution.
    # When NUM_HISTORY=0 this becomes an effective no-memory run, so log_base is inert.
    HISTORY_ARGS="$HISTORY_ARGS --log_base $LOG_BASE"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ] || [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    HISTORY_ARGS="$HISTORY_ARGS --gtc_output_tokens $GTC_OUTPUT_TOKENS --gtc_temperature $GTC_TEMPERATURE --gtc_num_iterations $GTC_NUM_ITERATIONS"
fi

MEMORY_ARGS="--memory_method $MEMORY_METHOD --map_global_side_m $MAP_GLOBAL_SIDE_M --map_local_side_m $MAP_LOCAL_SIDE_M --map_render_px $MAP_RENDER_PX --map_mask_method $MAP_MASK_METHOD"

RESUME_ARGS=""
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    RESUME_ARGS="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
    if [[ "$RESUME_ONLY_MODEL" == "true" ]]; then
        RESUME_ARGS="$RESUME_ARGS --resume_only_model true"
    fi
fi

# ============================================================================
# Run Training
# ============================================================================
cd "$SWIFTVLN_ROOT"

if [[ "$TRAIN_DRY_RUN" == "true" ]]; then
    echo "[INFO] TRAIN_DRY_RUN=true, skip torchrun launch after config validation."
    exit 0
fi

if [[ "$USE_SWANLAB" == "true" && "$SWANLAB_DIRECT_NETWORK" == "true" ]]; then
    unset_proxy_for_swanlab
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    src/swiftvln/model/trainer.py \
    --custom_register_path src/swiftvln/model \
    --model_type $MODEL_TYPE \
    --model $MODEL_PATH \
    --dataset $VLN_DATA_PATH \
    "${TRAIN_MODE_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --save_strategy steps \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --lr_scheduler_kwargs "$LR_SCHEDULER_KWARGS" \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --freeze_vit $FREEZE_VIT \
    --freeze_llm $FREEZE_LLM \
    --freeze_aligner $FREEZE_ALIGNER \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_pin_memory $DATALOADER_PIN_MEMORY \
    --dataloader_drop_last true \
    --dataloader_prefetch_factor $DATALOADER_PREFETCH_FACTOR \
    --dataloader_persistent_workers $DATALOADER_PERSISTENT_WORKERS \
    --dataset_num_proc $DATASET_NUM_PROC \
    --ddp_timeout 3600 \
    --num_frames $NUM_FRAMES \
    --num_history $NUM_HISTORY \
    --num_future_steps $NUM_FUTURE_STEPS \
    --use_random $USE_RANDOM \
    --vln_max_samples $MAX_SAMPLES \
    --vln_env_type $VLN_ENV_TYPE \
    --compress_stride $COMPRESS_STRIDE \
    --num_overlap $NUM_OVERLAP \
    --overlap_tail_window_adjust $OVERLAP_TAIL_WINDOW_ADJUST \
    --system_prompt_setting $SYSTEM_PROMPT_SETTING \
    $MEMORY_ARGS \
    --use_pixel_embed $USE_PIXEL_EMBED \
    --use_pose_embed $USE_POSE_EMBED \
    --use_uav_adapter $USE_UAV_ADAPTER \
    --uav_adapter_path "$UAV_ADAPTER_PATH" \
    --uav_adapter_type $UAV_ADAPTER_TYPE \
    --uav_adapter_apply_scope $UAV_ADAPTER_APPLY_SCOPE \
    --pose_fusion_method $POSE_FUSION_METHOD \
    --pose_norm_scale $POSE_NORM_SCALE \
    --use_tome $USE_TOME \
    --tf32 $TF32 \
    --torch_compile $TORCH_COMPILE \
    --padding_free $PADDING_FREE \
    --use_liger_kernel $USE_LIGER_KERNEL \
    $ATTN_ARG \
    $DEEPSPEED_ARG \
    $SWANLAB_ARGS \
    $QA_ARGS \
    $HISTORY_ARGS \
    $RESUME_ARGS \
    $MAX_STEPS_ARG

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

# Persist training metadata (SwanLab URL etc.) for downstream eval/CSV collection
if [ "$USE_SWANLAB" = true ]; then
    _swanlab_url=""
    _latest_run_dir=$(ls -td "${OUTPUT_DIR}"/v0-* 2>/dev/null | head -1)
    if [ -n "$_latest_run_dir" ] && [ -f "${_latest_run_dir}/logging.jsonl" ]; then
        _swanlab_url=$(grep -oP 'https://swanlab\.cn/@[^\s"]+/runs/[^\s"]+' "${_latest_run_dir}/logging.jsonl" 2>/dev/null | tail -1)
    fi
    if [ -z "$_swanlab_url" ] && [ -n "${TRAIN_LOG_FILE:-}" ] && [ -f "$TRAIN_LOG_FILE" ]; then
        _swanlab_url=$(grep -oP 'https://swanlab\.cn/@[^\s"]+/runs/[^\s"]+' "$TRAIN_LOG_FILE" 2>/dev/null | tail -1)
    fi
    _SWANLAB_URL="$_swanlab_url" \
    _SWANLAB_PROJECT="$SWANLAB_PROJECT" \
    _SWANLAB_EXP="$SWANLAB_EXP_NAME" \
    _OUTPUT_DIR="$OUTPUT_DIR" \
    python3 -c "
import json, pathlib, os
meta = {}
url = os.environ.get('_SWANLAB_URL', '')
if url:
    meta['swanlab_url'] = url
proj = os.environ.get('_SWANLAB_PROJECT', '')
if proj:
    meta['swanlab_project'] = proj
exp = os.environ.get('_SWANLAB_EXP', '')
if exp:
    meta['swanlab_exp_name'] = exp
if meta:
    out = pathlib.Path(os.environ['_OUTPUT_DIR']) / 'train_metadata.json'
    out.write_text(json.dumps(meta, indent=2))
    print(f'Saved train metadata: {out}')
" 2>/dev/null || true
fi
