#!/bin/bash
# OverlapVLN Debug Training Script - Multi-GPU test
# 
# Usage:
#   # Test GTC with initial strategy (default)
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh
#
#   # Test different configurations:
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh gtc initial
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh sgtc initial
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh pf initial
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh pf vanilla
#   bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_gtc_debug.sh gtc vanilla
#
# Arguments:
#   $1: History processor type: "gtc", "sgtc", "pf" (per_frame). Default: "gtc"
#   $2: System prompt setting: "initial" or "vanilla". Default: "initial"
#
# This script tests multi-GPU training with:
# - Different history processors (GTC, Segment-GTC, per_frame)
# - Initial view strategy verification (OVERLAPVLN_DEBUG enabled)
# - Minimal data and steps for quick validation

set -e  # Exit on error

# ============================================================================
# Parse arguments
# ============================================================================
HISTORY_PROCESSOR_TYPE="${1:-gtc}"    # gtc, sgtc, pf
SYSTEM_PROMPT_SETTING="${2:-initial}" # initial, vanilla

# Map short names to actual parameter names
case "$HISTORY_PROCESSOR_TYPE" in
    gtc)
        ACTUAL_PROCESSOR_TYPE="gtc"
        ;;
    sgtc)
        ACTUAL_PROCESSOR_TYPE="segment_gtc"
        ;;
    pf)
        ACTUAL_PROCESSOR_TYPE="per_frame"
        ;;
    *)
        echo "[ERROR] Invalid history processor type: $HISTORY_PROCESSOR_TYPE"
        echo "Usage: $0 [gtc|sgtc|pf] [initial|vanilla]"
        exit 1
        ;;
esac

if [ "$SYSTEM_PROMPT_SETTING" != "initial" ] && [ "$SYSTEM_PROMPT_SETTING" != "vanilla" ]; then
    echo "[ERROR] Invalid system_prompt_setting: $SYSTEM_PROMPT_SETTING"
    echo "Usage: $0 [gtc|sgtc|pf] [initial|vanilla]"
    exit 1
fi

echo "=========================================="
echo "Testing: processor=$ACTUAL_PROCESSOR_TYPE, prompt=$SYSTEM_PROMPT_SETTING"
echo "=========================================="

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train

# ============================================================================
# GPU Configuration - default uses all 8 GPUs; can override via env
# ============================================================================
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29501}"
GPUS_PER_NODE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_TYPE="overlapvln_qwen2_5_vl"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"

# ============================================================================
# VLN Data Configuration - Use small dataset for debugging
# ============================================================================
VLN_ENV_TYPE="satnav"
VLN_DATA_PATH="${VLN_DATA_PATH:-/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260202/trajectory_data}"

# VLN-Specific Parameters
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
USE_RANDOM=false
MAX_SAMPLES="${MAX_SAMPLES:-200}"  # Very small for quick testing

# ============================================================================
# History Processor Configuration
# ============================================================================
# Per-frame parameters
COMPRESS_STRIDE=2
USE_TOME=false
LOG_BASE=1.0

# GTC/SegmentGTC parameters
GTC_OUTPUT_TOKENS=512
GTC_TEMPERATURE=0.1
GTC_NUM_ITERATIONS=1

# Overlap configuration
NUM_OVERLAP=16

# Embedding enhancement
USE_PIXEL_EMBED="${USE_PIXEL_EMBED:-false}"
USE_POSE_EMBED="${USE_POSE_EMBED:-false}"
POSE_FUSION_METHOD="${POSE_FUSION_METHOD:-additive}"
POSE_NORM_SCALE="${POSE_NORM_SCALE:-100.0}"

# ============================================================================
# Training Parameters - Minimal for debugging
# ============================================================================
TRAIN_TYPE="full"
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE="${BATCH_SIZE:-2}"  # Small batch for debugging
GRAD_ACCUM_STEPS=1
MAX_LENGTH=32768
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Model Freezing
FREEZE_VIT=false
FREEZE_LLM=false
FREEZE_ALIGNER=false

# Optimization
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
GRADIENT_CHECKPOINTING=true
TF32="${TF32:-true}"
TORCH_COMPILE=false

# Learning Rate Schedule
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.
LR_SCHEDULER_TYPE="cosine"

# Attention Implementation
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

# ============================================================================
# Performance - Minimal for debugging
# ============================================================================
PADDING_FREE=false
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
DATALOADER_PREFETCH_FACTOR=2
DATALOADER_PERSISTENT_WORKERS=false
DATASET_NUM_PROC=2
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * GPUS_PER_NODE))
WINDOW_STRIDE=$((NUM_FRAMES - NUM_OVERLAP))

# Build experiment name
PROMPT_TAG=""
[ "$SYSTEM_PROMPT_SETTING" != "vanilla" ] && PROMPT_TAG="-${SYSTEM_PROMPT_SETTING}"
EXP_NAME="debug-${HISTORY_PROCESSOR_TYPE}${PROMPT_TAG}-${TIMESTAMP}"
OUTPUT_DIR="output/overlapvln_debug/${EXP_NAME}"

# Checkpoint Management
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=1

# Limit to 20 steps for quick test
MAX_STEPS="${MAX_STEPS:-20}"

# ============================================================================
# SwanLab Configuration - Disabled for debug
# ============================================================================
USE_SWANLAB=false

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

# *** Enable OverlapVLN debug output ***
# This triggers debug logging in dataset.py, template.py, trainer.py
# to verify initial strategy is working correctly across all ranks
export OVERLAPVLN_DEBUG=1

# ============================================================================
# Print Configuration
# ============================================================================
echo "=========================================="
echo "OverlapVLN Debug Training"
echo "=========================================="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "History Processor: $ACTUAL_PROCESSOR_TYPE"
if [ "$ACTUAL_PROCESSOR_TYPE" = "per_frame" ]; then
    COMPRESS_METHOD="pool"
    [ "$USE_TOME" = "true" ] && COMPRESS_METHOD="tome"
    echo "  Per-frame: h=$NUM_HISTORY, b=$LOG_BASE, method=$COMPRESS_METHOD, stride=$COMPRESS_STRIDE"
else
    echo "  Output Tokens: $GTC_OUTPUT_TOKENS"
    echo "  Temperature: $GTC_TEMPERATURE"
    echo "  Iterations: $GTC_NUM_ITERATIONS"
fi
echo "------------------------------------------"
echo "System Prompt: $SYSTEM_PROMPT_SETTING"
if [ "$SYSTEM_PROMPT_SETTING" = "initial" ]; then
    echo "  [INITIAL] First frame (uncompressed) will be added to system prompt"
fi
echo "Pixel Embed: $USE_PIXEL_EMBED"
echo "Pose Embed:  $USE_POSE_EMBED (fusion=$POSE_FUSION_METHOD)"
echo "------------------------------------------"
echo "Environment: $VLN_ENV_TYPE"
echo "Data: $VLN_DATA_PATH"
echo "Max Samples: $MAX_SAMPLES (debug mode)"
echo "Max Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "------------------------------------------"
echo "VLN Config:"
echo "  num_frames=$NUM_FRAMES, num_history=$NUM_HISTORY, num_future_steps=$NUM_FUTURE_STEPS"
echo "  num_overlap=$NUM_OVERLAP, window_stride=$WINDOW_STRIDE"
echo "------------------------------------------"
echo "Debug: OVERLAPVLN_DEBUG=$OVERLAPVLN_DEBUG"
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"

# DeepSpeed argument
DEEPSPEED_ARG=""
[ "$USE_DEEPSPEED" = true ] && DEEPSPEED_ARG="--deepspeed $DEEPSPEED_CONFIG"

# Attention implementation argument
ATTN_ARG=""
[ -n "$ATTN_IMPL" ] && ATTN_ARG="--attn_impl $ATTN_IMPL"

# History processor arguments
HISTORY_ARGS="--history_processor_type $ACTUAL_PROCESSOR_TYPE"
if [ "$ACTUAL_PROCESSOR_TYPE" = "per_frame" ]; then
    HISTORY_ARGS="$HISTORY_ARGS --log_base $LOG_BASE"
elif [ "$ACTUAL_PROCESSOR_TYPE" = "gtc" ] || [ "$ACTUAL_PROCESSOR_TYPE" = "segment_gtc" ]; then
    HISTORY_ARGS="$HISTORY_ARGS --gtc_output_tokens $GTC_OUTPUT_TOKENS --gtc_temperature $GTC_TEMPERATURE --gtc_num_iterations $GTC_NUM_ITERATIONS"
fi

# ============================================================================
# Run Training
# ============================================================================
cd "$SWIFTVLN_ROOT"

echo ""
echo "[DEBUG] Starting torchrun with $GPUS_PER_NODE GPUs..."
echo "[DEBUG] OVERLAPVLN_DEBUG=$OVERLAPVLN_DEBUG (debug logging enabled)"
echo "[DEBUG] system_prompt_setting=$SYSTEM_PROMPT_SETTING"
RUN_SINGLE_PROCESS="${RUN_SINGLE_PROCESS:-false}"
echo ""

if [ "$RUN_SINGLE_PROCESS" = "true" ]; then
    python3 src/swiftvln/models/overlapvln/trainer.py \
        --custom_register_path src/swiftvln/models/overlapvln \
        --model_type $MODEL_TYPE \
        --model $MODEL_PATH \
        --dataset $VLN_DATA_PATH \
        --train_type $TRAIN_TYPE \
        --torch_dtype $TORCH_DTYPE \
        --num_train_epochs $NUM_EPOCHS \
        --max_steps $MAX_STEPS \
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
        --gradient_checkpointing $GRADIENT_CHECKPOINTING \
        --freeze_vit $FREEZE_VIT \
        --freeze_llm $FREEZE_LLM \
        --freeze_aligner $FREEZE_ALIGNER \
        --dataloader_num_workers $DATALOADER_NUM_WORKERS \
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
        --log_base $LOG_BASE \
        --tf32 $TF32 \
        --torch_compile $TORCH_COMPILE \
        --padding_free $PADDING_FREE \
        --use_liger_kernel $USE_LIGER_KERNEL \
        $ATTN_ARG \
        $DEEPSPEED_ARG \
        $HISTORY_ARGS
else
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=localhost \
        --master_port=$MASTER_PORT \
        src/swiftvln/models/overlapvln/trainer.py \
        --custom_register_path src/swiftvln/models/overlapvln \
        --model_type $MODEL_TYPE \
        --model $MODEL_PATH \
        --dataset $VLN_DATA_PATH \
        --train_type $TRAIN_TYPE \
        --torch_dtype $TORCH_DTYPE \
        --num_train_epochs $NUM_EPOCHS \
        --max_steps $MAX_STEPS \
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
        --gradient_checkpointing $GRADIENT_CHECKPOINTING \
        --freeze_vit $FREEZE_VIT \
        --freeze_llm $FREEZE_LLM \
        --freeze_aligner $FREEZE_ALIGNER \
        --dataloader_num_workers $DATALOADER_NUM_WORKERS \
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
        --log_base $LOG_BASE \
        --tf32 $TF32 \
        --torch_compile $TORCH_COMPILE \
        --padding_free $PADDING_FREE \
        --use_liger_kernel $USE_LIGER_KERNEL \
        $ATTN_ARG \
        $DEEPSPEED_ARG \
        $HISTORY_ARGS
fi

echo ""
echo "=========================================="
echo "Debug training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "To verify initial strategy worked, check the log for lines containing:"
echo "  [INITIAL DEBUG] - Debug output from dataset/template/trainer"
echo "  [INITIAL]       - Summary from dataset initialization"
echo ""
