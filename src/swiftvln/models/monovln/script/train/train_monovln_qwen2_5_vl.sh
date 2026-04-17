#!/bin/bash
# MonoVLN Training Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   bash src/swiftvln/models/monovln/script/train/train_monovln_qwen2_5_vl.sh
#
# This script trains MonoVLN with single-turn dialogue and history frame compression.
# Key features:
# - Single-turn dialogue format (user + assistant)
# - History frame compression (configurable stride)
# - Episode-based uniform sampling (samples_per_episode)

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
MODEL_TYPE="monovln_qwen2_5_vl"

# Base model (from scratch)
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Or use a finetuned checkpoint
# MODEL_PATH="/path/to/checkpoint"

# Extract model size for experiment naming
MODEL_SIZE=$(echo "$MODEL_PATH" | grep -oE '[0-9]+B' | tr '[:upper:]' '[:lower:]')
MODEL_SIZE=${MODEL_SIZE:-"3b"}

# ============================================================================
# VLN Data Configuration
# ============================================================================
# Habitat datasets (default)
VLN_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
)

# SatNav datasets
# VLN_DATA_PATHS=(
#     "/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260128/trajectory_data"
# )

VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")

# ============================================================================
# MonoVLN Parameters
# ============================================================================
# History and action prediction
NUM_HISTORY=8               # Max history frames to sample (uniform if > available)
NUM_FUTURE_STEPS=4          # Actions to predict per step

# Episode sampling
SAMPLES_PER_EPISODE=5       # Samples per episode (first + random middle + last)

# History compression
COMPRESS_STRIDE=2           # Pooling stride: 2=4x, 3=9x, 4=16x compression

# Dataset limit (0 = use all)
MAX_SAMPLES="20"

# ============================================================================
# Training Parameters
# ============================================================================
TRAIN_TYPE="full"
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=16
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
# LR_SCHEDULER_KWARGS='{"min_lr":3.7e-05}'

# Attention Implementation
ATTN_IMPL="flash_attn"

# ============================================================================
# Performance Acceleration
# ============================================================================
# NOTE: padding_free must be FALSE for MonoVLN (custom tokens incompatible)
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
EXP_NAME="monovln-${MODEL_SIZE}-${NUM_EPOCHS}ep-h${NUM_HISTORY}s${NUM_FUTURE_STEPS}-spe${SAMPLES_PER_EPISODE}-stride${COMPRESS_STRIDE}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/monovln/${EXP_NAME}"

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
echo "MonoVLN Training"
echo "=========================================="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Data: $VLN_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR: $LEARNING_RATE | Epochs: $NUM_EPOCHS"
echo "Attention: $ATTN_IMPL"
echo "------------------------------------------"
echo "MonoVLN Settings:"
echo "  samples_per_episode: $SAMPLES_PER_EPISODE"
echo "  num_history: $NUM_HISTORY"
echo "  num_future_steps: $NUM_FUTURE_STEPS"
echo "  compress_stride: $COMPRESS_STRIDE (${COMPRESS_STRIDE}x${COMPRESS_STRIDE} = $((COMPRESS_STRIDE * COMPRESS_STRIDE))x)"
echo "------------------------------------------"
echo "Freeze ViT: $FREEZE_VIT | LLM: $FREEZE_LLM | Aligner: $FREEZE_ALIGNER"
echo "DeepSpeed: $USE_DEEPSPEED ($DEEPSPEED_CONFIG)"
echo "------------------------------------------"
echo "Acceleration:"
echo "  padding_free: $PADDING_FREE (must be false for MonoVLN)"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
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

# ============================================================================
# Run Training
# ============================================================================
cd "$SWIFTVLN_ROOT"

if [[ "$USE_SWANLAB" == "true" && "$SWANLAB_DIRECT_NETWORK" == "true" ]]; then
    unset_proxy_for_swanlab
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    src/swiftvln/models/monovln/trainer.py \
    --custom_register_path src/swiftvln/models/monovln \
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
    --num_history $NUM_HISTORY \
    --num_future_steps $NUM_FUTURE_STEPS \
    --samples_per_episode $SAMPLES_PER_EPISODE \
    --vln_max_samples $MAX_SAMPLES \
    --compress_stride $COMPRESS_STRIDE \
    --tf32 $TF32 \
    --torch_compile $TORCH_COMPILE \
    --padding_free $PADDING_FREE \
    --use_liger_kernel $USE_LIGER_KERNEL \
    $ATTN_ARG \
    $DEEPSPEED_ARG \
    $SWANLAB_ARGS \
    $MAX_STEPS_ARG

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
