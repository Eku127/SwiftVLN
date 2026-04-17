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
conda activate swift-vln-train

# ============================================================================
# GPU Configuration
# ============================================================================
CUDA_DEVICES="0,1,2,3,4,5,6,7"    # GPUs to use (comma-separated)
MASTER_PORT=29500                  # Master port for distributed training

# Auto-detect GPU count from CUDA_DEVICES
GPUS_PER_NODE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_TYPE="overlapvln_qwen2_5_vl"

# Training stage: "stage1" (from base Qwen) or "stage2" (from trained VLN model)
TRAIN_STAGE="stage1"

# Model paths for each stage
STAGE1_MODEL_PATH="/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct"
STAGE2_MODEL_PATH="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-3b-1ep-f32h8s4-overlap16-stride2-bs64-lr2e-5-20260124-214153/v0-20260124-214234/checkpoint-2239"

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
VLN_ENV_TYPE="satnav"

# Define data paths for each environment
HABITAT_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
)
SATNAV_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data"
)

# Select data paths based on VLN_ENV_TYPE (using nameref)
declare -n VLN_DATA_PATHS="${VLN_ENV_TYPE^^}_DATA_PATHS"
if [ ${#VLN_DATA_PATHS[@]} -eq 0 ]; then
    echo "[ERROR] Unknown VLN_ENV_TYPE: $VLN_ENV_TYPE. Available: habitat, satnav"
    exit 1
fi

VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")

# VLN-Specific Parameters
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
USE_RANDOM=false
MAX_SAMPLES="600000"  # Max samples cap (0 = use all). If actual < this, uses all available.

# ============================================================================
# Mixed Training: QA Dataset Configuration (Optional)
# ============================================================================
# Set USE_QA_MIXED_TRAINING=true to enable mixed training with VLN + QA data
USE_QA_MIXED_TRAINING=true
QA_DATASET="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/data/qa_swift.jsonl"
QA_RATIO=0.10
QA_MAX_SAMPLES=0          # Max QA samples (0 = use all available)

# ============================================================================
# OverlapVLN-Specific Parameters
# ============================================================================
# History Processor Type: "per_frame", "gtc", or "segment_gtc"
# - per_frame: Per-frame compression (default), each frame compressed independently
#   - log_base=1.0: Uniform sampling
#   - log_base>1.0: Logarithmic sampling (more recent frames preserved)
# - gtc: Global Token Clustering, cross-frame clustering to fixed tokens
# - segment_gtc: Segment-wise GTC, splits history into 8 segments
HISTORY_PROCESSOR_TYPE="per_frame"

# ---------- Per-frame parameters (used when HISTORY_PROCESSOR_TYPE="per_frame") ----------
# Compression stride (2 = 4x compression, 3 = 9x, 4 = 16x)
COMPRESS_STRIDE=2

# Compression method: false = average pooling, true = GridToMe
# GridToMe provides better semantic preservation for small objects
USE_TOME=false

# Sampling distribution: 1.0 = uniform, >1.0 = logarithmic (more recent frames)
# 2.0 = moderate, 3.0+ = aggressive concentration on recent frames
LOG_BASE=1.0

# ---------- GTC/SegmentGTC parameters (used when HISTORY_PROCESSOR_TYPE="gtc" or "segment_gtc") ----------
# Fixed number of output tokens for Global Token Clustering / Segment GTC
GTC_OUTPUT_TOKENS=512

# Temperature for soft assignment in Soft K-Means (lower = sharper)
GTC_TEMPERATURE=0.1

# Number of Soft K-Means iterations (1-2 usually sufficient)
GTC_NUM_ITERATIONS=1

# ---------- Sliding window overlap configuration ----------
# num_overlap: Number of overlapping actions between consecutive windows
# When num_overlap > 0, stride = num_frames - num_overlap
# First (num_overlap / num_future_steps) turns in non-first samples have loss masked
NUM_OVERLAP=16  # 16 = 50% overlap with num_frames=32, stride=16

# ---------- System prompt setting ----------
# System prompt strategy: "vanilla" (default, no initial view) or "initial"
# - vanilla: Standard prompt without initial view image
# - initial: Add the first frame of the episode (uncompressed) to the system prompt
#   as the initial observation at the starting point of the journey
SYSTEM_PROMPT_SETTING="vanilla"

# ---------- Embedding enhancement ----------
# Pixel coordinate embedding enhancement (Fourier + MLP)
# - false: disable (default)
# - true: enable and train pixel embedding module
USE_PIXEL_EMBED=false

# Pose embedding enhancement (MLP, per-image pose injection)
# - false: disable (default)
# - true: enable and train pose embedding module
USE_POSE_EMBED=false

# Pose fusion method: "additive" (default) or "film"
POSE_FUSION_METHOD="additive"

# Pose normalization scale for tanh(pos/scale), default 100.0
POSE_NORM_SCALE=100.0

# ============================================================================
# Training Parameters
# ============================================================================
TRAIN_TYPE="full"
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=8
GRAD_ACCUM_STEPS=1
MAX_LENGTH=32768

# Model Freezing
FREEZE_VIT=false
FREEZE_LLM=false
FREEZE_ALIGNER=false

# Optimization
USE_DEEPSPEED=true
DEEPSPEED_CONFIG="zero2"
GRADIENT_CHECKPOINTING=true
TF32=true
TORCH_COMPILE=false

# Learning Rate Schedule
WARMUP_RATIO=0.075
WEIGHT_DECAY=0.
LR_SCHEDULER_TYPE="cosine_with_min_lr"
LR_SCHEDULER_KWARGS='{"min_lr":1.85e-05}'

# Attention Implementation
ATTN_IMPL="flash_attn"

# ============================================================================
# Performance Acceleration
# ============================================================================
# NOTE: padding_free must be FALSE for OverlapVLN (custom tokens incompatible)
PADDING_FREE=false

USE_LIGER_KERNEL=true
DATALOADER_PREFETCH_FACTOR=10
DATALOADER_PERSISTENT_WORKERS=true
DATASET_NUM_PROC=2

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * GPUS_PER_NODE))
WINDOW_STRIDE=$((NUM_FRAMES - NUM_OVERLAP))

# Build experiment name based on history processor type
# Format: pf-h{history}-b{log_base}-{method}-s{stride}
HISTORY_SUFFIX=""
if [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    # Per-frame: include history count, log_base, method, stride
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = true ] && COMPRESS_METHOD="tome"
    HISTORY_SUFFIX="pf-h${NUM_HISTORY}-b${LOG_BASE}-${COMPRESS_METHOD}-s${COMPRESS_STRIDE}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    # GTC: include output tokens
    HISTORY_SUFFIX="gtc-k${GTC_OUTPUT_TOKENS}"
elif [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    # Segment GTC: include output tokens (8 segments, chronological order by default)
    HISTORY_SUFFIX="sgtc-k${GTC_OUTPUT_TOKENS}"
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

EXP_NAME="overlapvln-${VLN_ENV_TYPE}-${TRAIN_STAGE}-${MODEL_SIZE}-${NUM_EPOCHS}ep-f${NUM_FRAMES}s${NUM_FUTURE_STEPS}-overlap${NUM_OVERLAP}-${HISTORY_SUFFIX}${PROMPT_SUFFIX}${EMBED_SUFFIX}${DATA_VERSION_SUFFIX}${QA_SUFFIX}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/overlapvln/${EXP_NAME}"

# Checkpoint Management
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=10

# Note: MAX_SAMPLES is a soft cap - if actual samples < MAX_SAMPLES, all available samples are used.
# Training is controlled by num_train_epochs, not max_steps, so the trainer will iterate
# over the actual dataset size. This avoids over-iteration when actual samples < MAX_SAMPLES.
MAX_STEPS_ARG=""
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES set as upper limit. Training controlled by num_epochs=$NUM_EPOCHS"
    echo "[INFO] If actual samples < MAX_SAMPLES, all available samples will be used."
fi

# ============================================================================
# SwanLab Configuration
# ============================================================================
USE_SWANLAB=true
SWANLAB_PROJECT="StreamVLN"
SWANLAB_EXP_NAME="${EXP_NAME}"
SWANLAB_MODE="cloud"

# WXWork Notification
USE_WXWORK_NOTIFICATION=true
SWANLAB_NOTIFICATION_METHOD="wxwork"
SWANLAB_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d78d3128-7b16-4bf1-a6a7-403bf0915fe0"
SWANLAB_SECRET=""

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
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR: $LEARNING_RATE | Epochs: $NUM_EPOCHS"
echo "Attention: $ATTN_IMPL"
echo "------------------------------------------"
echo "History Processor: $HISTORY_PROCESSOR_TYPE"
if [ "$HISTORY_PROCESSOR_TYPE" = "per_frame" ]; then
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = true ] && COMPRESS_METHOD="tome"
    SAMPLING_TYPE="uniform"
    [ "$LOG_BASE" != "1.0" ] && [ "$LOG_BASE" != "1" ] && SAMPLING_TYPE="logarithmic (b=$LOG_BASE)"
    echo "  Sampling: $SAMPLING_TYPE, ${NUM_HISTORY} frames"
    echo "  Compression: stride=$COMPRESS_STRIDE ($((COMPRESS_STRIDE * COMPRESS_STRIDE))x), method=$COMPRESS_METHOD"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ]; then
    echo "  Sampling: every ${NUM_FUTURE_STEPS} frames from history"
    echo "  GTC: output_tokens=$GTC_OUTPUT_TOKENS, temperature=$GTC_TEMPERATURE, iterations=$GTC_NUM_ITERATIONS"
elif [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    echo "  Sampling: every ${NUM_FUTURE_STEPS} frames from history"
    echo "  SegmentGTC: output_tokens=$GTC_OUTPUT_TOKENS, segments=8, temperature=$GTC_TEMPERATURE, iterations=$GTC_NUM_ITERATIONS"
fi
echo "Overlap: num_overlap=$NUM_OVERLAP, window_stride=$WINDOW_STRIDE"
echo "  First $((NUM_OVERLAP / NUM_FUTURE_STEPS)) turns masked for samples with start_idx > 0"
echo "System Prompt: $SYSTEM_PROMPT_SETTING"
echo "Pixel Embed: $USE_PIXEL_EMBED"
echo "Pose Embed:  $USE_POSE_EMBED (fusion=$POSE_FUSION_METHOD, norm_scale=$POSE_NORM_SCALE)"
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
echo "  padding_free: $PADDING_FREE (must be false for OverlapVLN)"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"

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
    # Per-frame: pass log_base for sampling distribution
    HISTORY_ARGS="$HISTORY_ARGS --log_base $LOG_BASE"
elif [ "$HISTORY_PROCESSOR_TYPE" = "gtc" ] || [ "$HISTORY_PROCESSOR_TYPE" = "segment_gtc" ]; then
    HISTORY_ARGS="$HISTORY_ARGS --gtc_output_tokens $GTC_OUTPUT_TOKENS --gtc_temperature $GTC_TEMPERATURE --gtc_num_iterations $GTC_NUM_ITERATIONS"
fi

# ============================================================================
# Run Training
# ============================================================================
cd "$SWIFTVLN_ROOT"

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
    --train_type $TRAIN_TYPE \
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
    --dataloader_num_workers 8 \
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
    --use_pixel_embed $USE_PIXEL_EMBED \
    --use_pose_embed $USE_POSE_EMBED \
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
    $MAX_STEPS_ARG

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
