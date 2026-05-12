#!/bin/bash

set -euo pipefail

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train

REPO_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor"
cd "${REPO_ROOT}"

MANIFEST_PATH="${MANIFEST_PATH:-runtime/s2r/manifests/manifest_v1.jsonl}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-output/swiftvln/swiftvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-noembed-data260317-bs64-lr2e-5-20260318-202149/v0-20260318-202212/checkpoint-3957}"
OUTPUT_DIR="${OUTPUT_DIR:-output/s2r/$(date +%Y%m%d-%H%M%S)}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python src/swiftvln/s2r/trainer.py \
  --manifest_path "${MANIFEST_PATH}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
