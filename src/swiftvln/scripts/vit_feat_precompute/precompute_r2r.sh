#!/bin/bash
# UniNaVid Feature Precomputation Script
#
# Precomputes ViT features for all frames in VLN datasets.
# Features are saved as .pt files for use with use_precomputed_features=true.
#
# Usage:
#   bash src/swiftvln/scripts/vit_feat_precompute/precompute_r2r.sh

set -e  # Exit on error

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train-update

# ============================================================================
# GPU Configuration
# ============================================================================
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"    # GPUs to use
MASTER_PORT="${MASTER_PORT:-29501}"                 # Master port

# Auto-detect GPU count
GPUS_PER_NODE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# ============================================================================
# Model Configuration
# ============================================================================
# Base model for ViT feature extraction
MODEL_PATH="${MODEL_PATH:-/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct}"

# ============================================================================
# Data Configuration
# ============================================================================
# List of datasets to precompute features for
DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    # Add more datasets as needed
)

# Processing parameters
BATCH_SIZE="${BATCH_SIZE:-8}"  # Frames per batch (adjust based on GPU memory)

# ============================================================================
# Environment Setup
# ============================================================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope

# ============================================================================
# Main Script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
cd "$SWIFTVLN_ROOT"

echo "=========================================="
echo "UniNaVid Feature Precomputation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

for DATA_PATH in "${DATA_PATHS[@]}"; do
    echo ""
    echo "Processing: $DATA_PATH"
    echo "------------------------------------------"
    
    OUTPUT_DIR="${DATA_PATH}/features"
    
    # Check if already processed
    if [ -d "$OUTPUT_DIR" ]; then
        EXISTING_COUNT=$(find "$OUTPUT_DIR" -name "*.pt" | wc -l)
        echo "Found $EXISTING_COUNT existing feature files"
    fi
    
    if [ "$GPUS_PER_NODE" -gt 1 ]; then
        # Multi-GPU distributed processing
        torchrun \
            --nnodes=1 \
            --nproc_per_node=$GPUS_PER_NODE \
            --master_addr=localhost \
            --master_port=$MASTER_PORT \
            src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
            --data_path "$DATA_PATH" \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE
    else
        # Single GPU processing
        python src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
            --data_path "$DATA_PATH" \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE
    fi
    
    echo "Completed: $DATA_PATH"
done

echo ""
echo "=========================================="
echo "Feature precomputation complete!"
echo "=========================================="
