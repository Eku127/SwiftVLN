#!/bin/bash
# Debug Landmark Evaluation Script
# 
# This script runs evaluation on a small set of representative landmark episodes
# with detailed debug output (frames, trajectory analysis, model outputs).
#
# Usage:
#   bash src/swiftvln/models/overlapvln/script/eval/eval_debug_landmark.sh
#
# Output will be saved to results/eval/overlapvln/debug_landmark_analysis/

set -e

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval

# ============================================================================
# Configuration
# ============================================================================
# Best performing model (highest overall SR)
MODEL_PATH="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-pf-h8-b2.0-pool-s2-bs64-lr2e-5-20260204-230157/v0-20260204-230218/checkpoint-508"

# Use debug landmark episodes
DEBUG_EPISODES_PATH="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260202/episodes/eval/debug_landmark_episodes.json"

# Use single GPU for debug (easier to read output)
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NUM_GPUS=1
MASTER_PORT="${MASTER_PORT:-29700}"

# Model parameters (must match training)
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
NUM_OVERLAP=16
COMPRESS_STRIDE=2
LOG_BASE=2.0
HISTORY_PROCESSOR_TYPE="per_frame"

# Output
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results/eval/overlapvln/debug_landmark_analysis/${TIMESTAMP}"

# ============================================================================
# Environment Setup
# ============================================================================
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=^docker0,lo
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
VLN_DIR="${SWIFTVLN_ROOT}/src/swiftvln"
CONFIG_PATH="configs/satnav_task.yaml"

# ============================================================================
# Temporarily patch config to use debug episodes
# ============================================================================
# Create a temporary config that points to debug episodes
TEMP_CONFIG="${OUTPUT_DIR}/satnav_debug_config.yaml"
mkdir -p "${OUTPUT_DIR}"
cp "${VLN_DIR}/${CONFIG_PATH}" "${TEMP_CONFIG}"

# Replace DATA_PATH in the temp config
python3 -c "
import yaml
with open('${TEMP_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
config['DATASET']['DATA_PATH'] = '${DEBUG_EPISODES_PATH}'
with open('${TEMP_CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Config updated with debug episodes path')
"

# ============================================================================
# Print Configuration
# ============================================================================
echo "=============================================="
echo "Debug Landmark Evaluation"
echo "=============================================="
echo "Model Path:      ${MODEL_PATH}"
echo "Debug Episodes:  ${DEBUG_EPISODES_PATH}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "GPU:             ${CUDA_DEVICES}"
echo "Config:          ${TEMP_CONFIG}"
echo ""
echo "Parameters:"
echo "  num_frames=${NUM_FRAMES}, num_history=${NUM_HISTORY}"
echo "  compress_stride=${COMPRESS_STRIDE}, log_base=${LOG_BASE}"
echo "  num_overlap=${NUM_OVERLAP}"
echo "  history_processor=${HISTORY_PROCESSOR_TYPE}"
echo "=============================================="

# ============================================================================
# Run Debug Evaluation
# ============================================================================
cd "$SWIFTVLN_ROOT"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m swiftvln.models.overlapvln.eval \
    --model_path "${MODEL_PATH}" \
    --env-type satnav \
    --habitat_config_path "${TEMP_CONFIG}" \
    --satnav-config "${TEMP_CONFIG}" \
    --eval_split val_unseen \
    --num_frames "${NUM_FRAMES}" \
    --num_history "${NUM_HISTORY}" \
    --num_future_steps "${NUM_FUTURE_STEPS}" \
    --num_overlap "${NUM_OVERLAP}" \
    --history_processor_type "${HISTORY_PROCESSOR_TYPE}" \
    --compress_stride "${COMPRESS_STRIDE}" \
    --log_base "${LOG_BASE}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_video \
    --debug_landmark \
    --verbose

echo "=============================================="
echo "Debug Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Check these directories:"
echo "  ${OUTPUT_DIR}/debug_landmark/  - Per-episode debug reports and frames"
echo "  ${OUTPUT_DIR}/videos/          - Navigation videos"
echo "=============================================="
