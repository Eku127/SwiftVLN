#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-IPEC-COMMUNITY/openfly-agent-7b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
TARGET_DIR="${TARGET_DIR:-${BASELINE_DIR}/model/openfly-agent-7b}"

mkdir -p "${TARGET_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install huggingface_hub in the target env first." >&2
    exit 2
fi

huggingface-cli download "${MODEL_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False

echo "Downloaded ${MODEL_ID} -> ${TARGET_DIR}"
