#!/bin/bash
# SwiftVLN Training Script - Qwen VL families (ms-swift)
# 
# Usage:
#   bash src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh
#
# This script trains SwiftVLN with history frame compression.
# Key difference from StreamVLN: adds compress_stride parameter

set -e  # Exit on error

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
SWIFTVLN_TRAIN_CONDA_ENV="${SWIFTVLN_TRAIN_CONDA_ENV:-swift-vln-train-update}"
conda activate "$SWIFTVLN_TRAIN_CONDA_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"

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
MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5_vl}"  # qwen2_5_vl | qwen3_vl
case "$MODEL_FAMILY" in
    qwen2_5_vl|qwen25|qwen2.5)
        MODEL_FAMILY="qwen2_5_vl"
        DEFAULT_MODEL_TYPE="swiftvln_qwen2_5_vl"
        DEFAULT_BASE_MODEL_PATH="/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct"
        ;;
    qwen3_vl|qwen3)
        MODEL_FAMILY="qwen3_vl"
        DEFAULT_MODEL_TYPE="swiftvln_qwen3_vl"
        DEFAULT_BASE_MODEL_PATH="/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct"
        ;;
    *)
        echo "[ERROR] Unknown MODEL_FAMILY: $MODEL_FAMILY. Available: qwen2_5_vl, qwen3_vl."
        exit 1
        ;;
esac
MODEL_TYPE="${MODEL_TYPE:-$DEFAULT_MODEL_TYPE}"

# Base model path
# Defaults to the local offline cache path to avoid ModelScope hub resolution.
# Default remains the local 3B cache path for Qwen2.5 and 2B for Qwen3.
# For larger models, override BASE_MODEL_PATH or MODEL_PATH
# via env, e.g.:
#   BASE_MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$DEFAULT_BASE_MODEL_PATH}"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"

# Extract model size for experiment naming
MODEL_SIZE=$(echo "$MODEL_PATH" | grep -oE '[0-9]+B' | tr '[:upper:]' '[:lower:]')
MODEL_SIZE=${MODEL_SIZE:-"3b"}

# ============================================================================
# VLN Data Configuration
# ============================================================================
# Environment type: "habitat" (forward=0.25m) or "satnav" (forward=10m)
VLN_ENV_TYPE="${VLN_ENV_TYPE:-satnav}"
VLN_DATA_PATH_OVERRIDE="${VLN_DATA_PATH:-}"

# Define data paths for each environment
HABITAT_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
)
SATNAV_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data"
)

# Select data paths based on VLN_ENV_TYPE (using nameref), unless explicitly
# injected by train_queue or a caller.
if [[ -n "$VLN_DATA_PATH_OVERRIDE" ]]; then
    VLN_DATA_PATH="$VLN_DATA_PATH_OVERRIDE"
else
    declare -n VLN_DATA_PATHS="${VLN_ENV_TYPE^^}_DATA_PATHS"
    if [ ${#VLN_DATA_PATHS[@]} -eq 0 ]; then
        echo "[ERROR] Unknown VLN_ENV_TYPE: $VLN_ENV_TYPE. Available: habitat, satnav"
        exit 1
    fi

    VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")
fi

# VLN-Specific Parameters
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-4}"
USE_RANDOM="${USE_RANDOM:-false}"
# Default to baseline full-data training.
# 0 means "use all available samples".
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ============================================================================
# SwiftVLN-Specific Parameters
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
# (i.e. co-located with SatNav-v0.1). Override with any absolute path, or set
# to one of {off,false,none,0,disable,disabled,no} to disable caching.
# Only has effect when MEMORY_METHOD=map.
MAP_CACHE_DIR="${MAP_CACHE_DIR:-auto}"

# Embedding enhancement is one mutually-exclusive choice:
# none | pose (additive) | posefilm (FiLM) | uav (Stage-A adapter)
for legacy_embedding_var in USE_POSE_EMBED USE_UAV_ADAPTER POSE_FUSION_METHOD; do
    if [[ -v "$legacy_embedding_var" ]]; then
        echo "[ERROR] $legacy_embedding_var was removed. Set EMBEDDING_MODE=none|pose|posefilm|uav instead."
        exit 2
    fi
done
EMBEDDING_MODE="${EMBEDDING_MODE:-none}"
case "$EMBEDDING_MODE" in
    none|pose|posefilm|uav) ;;
    *)
        echo "[ERROR] Invalid EMBEDDING_MODE=$EMBEDDING_MODE. Expected none|pose|posefilm|uav."
        exit 2
        ;;
esac

# Stage-A UAV adapter options (used only when EMBEDDING_MODE=uav)
UAV_ADAPTER_PATH="${UAV_ADAPTER_PATH:-}"
UAV_ADAPTER_TYPE="${UAV_ADAPTER_TYPE:-transformer_v1}"
UAV_ADAPTER_APPLY_SCOPE="${UAV_ADAPTER_APPLY_SCOPE:-all_images}"

# Pose normalization scale (used only for pose/posefilm)
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
# NOTE: padding_free must be FALSE for SwiftVLN (custom tokens incompatible)
PADDING_FREE="${PADDING_FREE:-false}"

# Qwen3-VL note (2026-05-12): keep Liger enabled for the default Qwen2.5 path,
# but set USE_LIGER_KERNEL=false for Qwen3-VL training/smoke. Qwen3 vision RoPE
# can fail inside the Liger/Triton kernel with:
#   ValueError('numel (...) exceeds triton maximum tensor numel (1048576)')
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-10}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * GPUS_PER_NODE))
WINDOW_STRIDE=$((NUM_FRAMES - NUM_OVERLAP))

# Name generation and cross-field validation share the same Python schema as eval.
EXP_NAME=$(
    python -m swiftvln.experiment build-name \
        --env-type "$VLN_ENV_TYPE" \
        --model-family "$MODEL_FAMILY" \
        --model-size "$MODEL_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --num-frames "$NUM_FRAMES" \
        --num-future-steps "$NUM_FUTURE_STEPS" \
        --num-overlap "$NUM_OVERLAP" \
        --memory-method "$MEMORY_METHOD" \
        --history-processor-type "$HISTORY_PROCESSOR_TYPE" \
        --num-history "$NUM_HISTORY" \
        --log-base "$LOG_BASE" \
        --use-random "$USE_RANDOM" \
        --compress-stride "$COMPRESS_STRIDE" \
        --use-tome "$USE_TOME" \
        --gtc-output-tokens "$GTC_OUTPUT_TOKENS" \
        --gtc-temperature "$GTC_TEMPERATURE" \
        --gtc-num-iterations "$GTC_NUM_ITERATIONS" \
        --map-global-side-m "$MAP_GLOBAL_SIDE_M" \
        --map-local-side-m "$MAP_LOCAL_SIDE_M" \
        --map-render-px "$MAP_RENDER_PX" \
        --map-mask-method "$MAP_MASK_METHOD" \
        --system-prompt-setting "$SYSTEM_PROMPT_SETTING" \
        --embedding-mode "$EMBEDDING_MODE" \
        --effective-batch-size "$EFFECTIVE_BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --timestamp "$TIMESTAMP"
)
OUTPUT_DIR="output/swiftvln/${EXP_NAME}"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
fi

# Checkpoint Management
# Default to production-style checkpoint cadence. Smoke tests must override this
# explicitly if they need per-step checkpointing.
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-}"
SAVE_SAFETENSORS_ARG=""
if [[ -n "$SAVE_SAFETENSORS" ]]; then
    SAVE_SAFETENSORS_ARG="--save_safetensors $SAVE_SAFETENSORS"
fi
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
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SatNav}"
SWANLAB_EXP_NAME="${EXP_NAME}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

# ============================================================================
# Environment Setup
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=4
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Map-memory render cache: forward MAP_CACHE_DIR to the Python layer via the
# SWIFTVLN_MAP_CACHE_DIR env var. "auto" keeps the code default (dataset_root
# /map_cache); explicit paths or "off"-family sentinels are passed through.
if [ "$MEMORY_METHOD" = "map" ]; then
    if [ -n "$MAP_CACHE_DIR" ] && [ "$MAP_CACHE_DIR" != "auto" ]; then
        export SWIFTVLN_MAP_CACHE_DIR="$MAP_CACHE_DIR"
    fi
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "=========================================="
echo "SwiftVLN Training"
echo "=========================================="
echo "Model Family: $MODEL_FAMILY"
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Environment: $VLN_ENV_TYPE"
echo "Data: $VLN_DATA_PATH"
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
    echo "  Render cache: MAP_CACHE_DIR=${MAP_CACHE_DIR} (env SWIFTVLN_MAP_CACHE_DIR=${SWIFTVLN_MAP_CACHE_DIR:-<unset, will use dataset_root/map_cache>})"
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
echo "System Prompt: $SYSTEM_PROMPT_SETTING"
echo "Embedding Mode: $EMBEDDING_MODE"
if [[ "$EMBEDDING_MODE" == "pose" || "$EMBEDDING_MODE" == "posefilm" ]]; then
    echo "  Pose norm scale: $POSE_NORM_SCALE"
elif [[ "$EMBEDDING_MODE" == "uav" ]]; then
    echo "  UAV Adapter: type=$UAV_ADAPTER_TYPE, scope=$UAV_ADAPTER_APPLY_SCOPE"
    [ -n "$UAV_ADAPTER_PATH" ] && echo "  UAV Adapter Path: $UAV_ADAPTER_PATH"
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "Resume: $RESUME_FROM_CHECKPOINT (resume_only_model=$RESUME_ONLY_MODEL)"
fi
echo "------------------------------------------"
echo "Freeze ViT: $FREEZE_VIT | LLM: $FREEZE_LLM | Aligner: $FREEZE_ALIGNER"
echo "DeepSpeed: $USE_DEEPSPEED ($DEEPSPEED_CONFIG)"
echo "------------------------------------------"
echo "Acceleration:"
echo "  dataloader_num_workers: $DATALOADER_NUM_WORKERS"
echo "  dataloader_pin_memory: $DATALOADER_PIN_MEMORY"
echo "  padding_free: $PADDING_FREE (must be false for SwiftVLN)"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
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
fi

# Attention implementation argument
ATTN_ARG=""
[ -n "$ATTN_IMPL" ] && ATTN_ARG="--attn_impl $ATTN_IMPL"

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

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

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
    --torch_dtype $TORCH_DTYPE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    $SAVE_SAFETENSORS_ARG \
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
    --system_prompt_setting $SYSTEM_PROMPT_SETTING \
    $MEMORY_ARGS \
    --embedding_mode $EMBEDDING_MODE \
    --uav_adapter_path "$UAV_ADAPTER_PATH" \
    --uav_adapter_type $UAV_ADAPTER_TYPE \
    --uav_adapter_apply_scope $UAV_ADAPTER_APPLY_SCOPE \
    --pose_norm_scale $POSE_NORM_SCALE \
    --use_tome $USE_TOME \
    --tf32 $TF32 \
    --torch_compile $TORCH_COMPILE \
    --padding_free $PADDING_FREE \
    --use_liger_kernel $USE_LIGER_KERNEL \
    $ATTN_ARG \
    $DEEPSPEED_ARG \
    $SWANLAB_ARGS \
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
    python3 "${SWIFTVLN_ROOT}/src/swiftvln/scripts/train/_write_train_metadata.py" 2>/dev/null || true
fi
