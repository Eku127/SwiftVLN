#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SWIFTVLN_ROOT="${REPO_ROOT}"
# shellcheck source=../../scripts/lib/local_env.sh
source "${REPO_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env
swiftvln_activate_conda "${SWIFTVLN_TRAIN_CONDA_ENV:-swift-vln-train-update}"
cd "${REPO_ROOT}"

MANIFEST_PATH="${MANIFEST_PATH:-runtime/s2r/manifests/manifest_v1.jsonl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${SWIFTVLN_QWEN25_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$BASE_MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-output/s2r/$(date +%Y%m%d-%H%M%S)}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python src/swiftvln/s2r/trainer.py \
  --manifest_path "${MANIFEST_PATH}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
