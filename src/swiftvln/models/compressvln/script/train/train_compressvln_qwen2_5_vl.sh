#!/bin/bash
# CompressVLN Training Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   bash src/swiftvln/models/compressvln/script/train/train_compressvln_qwen2_5_vl.sh
#
# This script trains CompressVLN with history frame compression.
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
MODEL_TYPE="compressvln_qwen2_5_vl"

# Training stage: "stage1" (from base Qwen) or "stage2" (from trained VLN model)
TRAIN_STAGE="stage1"

# Model paths for each stage
STAGE1_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
STAGE2_MODEL_PATH="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/compressvln/compressvln-3b-1ep-f32h8s4-stride2-bs64-lr2e-5-20260127-101351/v0-20260127-101426/checkpoint-1480"

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
VLN_ENV_TYPE="habitat"

# Define data paths for each environment
HABITAT_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
)
SATNAV_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260228/trajectory_data"
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
MAX_SAMPLES="0"  # 0 = use all samples

# ============================================================================
# Mixed Training: QA Dataset Configuration (Optional)
# ============================================================================
# Set USE_QA_MIXED_TRAINING=true to enable mixed training with VLN + QA data
USE_QA_MIXED_TRAINING=false
QA_DATASET="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260228/data/qa_swift.jsonl"
QA_RATIO=0.15             # Ratio of QA samples (0.15 = 15% QA, 85% VLN)
QA_MAX_SAMPLES=0          # Max QA samples (0 = use all available)

# ============================================================================
# CompressVLN-Specific Parameters
# ============================================================================
COMPRESS_STRIDE=2  # 2 = 4x compression, 3 = 9x, 4 = 16x

# ============================================================================
# Precomputed Features (Optional - requires running precompute_features.py first)
# ============================================================================
# Set to true to use precomputed ViT features (skips ViT forward, saves memory)
USE_PRECOMPUTED_FEATURES=false
# LRU cache size for episode features (number of episodes in CPU memory)
FEATURE_CACHE_SIZE=100

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

# Auto-enforce: precomputed features requires frozen ViT
if [ "$USE_PRECOMPUTED_FEATURES" = true ]; then
    FREEZE_VIT=true
fi

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
# NOTE: padding_free must be FALSE for CompressVLN (custom tokens incompatible)
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

# Add QA suffix if mixed training is enabled
QA_SUFFIX=""
if [ "$USE_QA_MIXED_TRAINING" = true ]; then
    # Convert ratio to percentage (e.g., 0.15 -> 15)
    QA_PCT=$(echo "$QA_RATIO * 100" | bc | cut -d'.' -f1)
    QA_SUFFIX="-qa${QA_PCT}"
fi

EXP_NAME="compressvln-${VLN_ENV_TYPE}-${TRAIN_STAGE}-${MODEL_SIZE}-${NUM_EPOCHS}ep-f${NUM_FRAMES}h${NUM_HISTORY}s${NUM_FUTURE_STEPS}-stride${COMPRESS_STRIDE}${QA_SUFFIX}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/compressvln/${EXP_NAME}"

# Checkpoint Management
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=10

# Auto-compute max_steps when MAX_SAMPLES is set (required for limited dataset)
MAX_STEPS_ARG=""
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    # Calculate max_steps = ceil(MAX_SAMPLES / EFFECTIVE_BATCH_SIZE) * NUM_EPOCHS
    STEPS_PER_EPOCH=$(( (MAX_SAMPLES + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE ))
    MAX_STEPS=$(( STEPS_PER_EPOCH * NUM_EPOCHS ))
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
    echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES set, computed max_steps=$MAX_STEPS"
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
echo "CompressVLN Training"
echo "=========================================="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Environment: $VLN_ENV_TYPE"
echo "Data: $VLN_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR: $LEARNING_RATE | Epochs: $NUM_EPOCHS"
echo "Attention: $ATTN_IMPL"
echo "------------------------------------------"
echo "Compression: stride=$COMPRESS_STRIDE (${COMPRESS_STRIDE}x${COMPRESS_STRIDE} = $((COMPRESS_STRIDE * COMPRESS_STRIDE))x)"
echo "------------------------------------------"
echo "Freeze ViT: $FREEZE_VIT | LLM: $FREEZE_LLM | Aligner: $FREEZE_ALIGNER"
echo "DeepSpeed: $USE_DEEPSPEED ($DEEPSPEED_CONFIG)"
echo "------------------------------------------"
echo "Acceleration:"
echo "  padding_free: $PADDING_FREE (must be false for CompressVLN)"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "------------------------------------------"
# Mixed training info
if [ "$USE_QA_MIXED_TRAINING" = true ]; then
    echo "Mixed Training: ENABLED"
    echo "  QA Dataset: $QA_DATASET"
    echo "  QA Ratio: ${QA_RATIO} (QA $(echo "scale=0; $QA_RATIO * 100" | bc)%, VLN $(echo "scale=0; (1 - $QA_RATIO) * 100" | bc)%)"
    [ "$QA_MAX_SAMPLES" -gt 0 ] 2>/dev/null && echo "  QA Max Samples: $QA_MAX_SAMPLES"
else
    echo "Mixed Training: DISABLED (VLN only)"
fi
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
    src/swiftvln/models/compressvln/trainer.py \
    --custom_register_path src/swiftvln/models/compressvln \
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
    --use_precomputed_features $USE_PRECOMPUTED_FEATURES \
    --feature_cache_size $FEATURE_CACHE_SIZE \
    --tf32 $TF32 \
    --torch_compile $TORCH_COMPILE \
    --padding_free $PADDING_FREE \
    --use_liger_kernel $USE_LIGER_KERNEL \
    $ATTN_ARG \
    $DEEPSPEED_ARG \
    $SWANLAB_ARGS \
    $QA_ARGS \
    $MAX_STEPS_ARG

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
