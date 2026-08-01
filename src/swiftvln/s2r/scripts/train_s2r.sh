#!/bin/bash

set -euo pipefail

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train-update

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST_PATH="${MANIFEST_PATH:-runtime/s2r/manifests/manifest_v1.jsonl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$BASE_MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-output/s2r/$(date +%Y%m%d-%H%M%S)}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python src/swiftvln/s2r/trainer.py \
  --manifest_path "${MANIFEST_PATH}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
