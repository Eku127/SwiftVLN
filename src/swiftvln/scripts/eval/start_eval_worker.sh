#!/usr/bin/env bash
set -euo pipefail

# Start a non-interactive eval worker that keeps consuming eval_todo.txt.
# Intended host: 98 or 73 (not 17).
#
# Usage:
#   bash src/swiftvln/scripts/eval/start_eval_worker.sh
#   EVAL_SPLIT=val_seen CUDA_DEVICES=0,1,2,3 bash src/swiftvln/scripts/eval/start_eval_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DYNAMIC_TODO=true
export AUTO_TODO=true
export WAIT_FOR_NEW_TASKS=true
export TODO_POLL_INTERVAL="${TODO_POLL_INTERVAL:-60}"
export EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export SAVE_VIDEO="${SAVE_VIDEO:-false}"

echo "[INFO] Starting eval worker with settings:"
echo "  EVAL_SPLIT=${EVAL_SPLIT}"
echo "  CUDA_DEVICES=${CUDA_DEVICES}"
echo "  TODO_POLL_INTERVAL=${TODO_POLL_INTERVAL}"
echo "  SAVE_VIDEO=${SAVE_VIDEO}"

bash "${SCRIPT_DIR}/eval_queue.sh"

