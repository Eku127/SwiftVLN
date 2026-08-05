#!/usr/bin/env bash
set -euo pipefail

# Append one model name into eval_todo.txt (deduplicated, with optional checkpoint check).
# Usage:
#   bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name>
#   bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name> --skip-checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
TODO_FILE="${EVAL_QUEUE_DIR}/eval_todo.txt"
LOCK_FILE="${TODO_FILE}.lock"
SKIP_CHECKPOINT=false
MODEL_NAME="${1:-}"

print_info() { echo "[INFO] $1"; }
print_ok() { echo "[OK] $1"; }
print_err() { echo "[ERROR] $1" >&2; }

if [[ -z "$MODEL_NAME" ]]; then
    print_err "Usage: bash $0 <model_name> [--skip-checkpoint]"
    exit 1
fi

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-checkpoint)
            SKIP_CHECKPOINT=true
            shift
            ;;
        *)
            print_err "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If input is a full path, normalize to basename model name.
if [[ "$MODEL_NAME" == */* ]]; then
    MODEL_NAME="$(basename "$MODEL_NAME")"
fi

find_latest_checkpoint() {
    local model_dir="$1"
    local latest=""
    for version_dir in "$model_dir"/v*; do
        if [[ -d "$version_dir" ]]; then
            local ckpt
            ckpt=$(ls -d "$version_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
            if [[ -n "${ckpt:-}" ]]; then
                latest="$ckpt"
            fi
        fi
    done
    if [[ -z "$latest" ]]; then
        latest=$(ls -d "$model_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    fi
    echo "$latest"
}

if ! python -m swiftvln.experiment parse-name "$MODEL_NAME" >/dev/null; then
    print_err "Invalid SwiftVLN model name: $MODEL_NAME"
    exit 2
fi
MODEL_ARCH="swiftvln"

if [[ "$SKIP_CHECKPOINT" != "true" ]]; then
    MODEL_DIR="${SWIFTVLN_ROOT}/output/${MODEL_ARCH}/${MODEL_NAME}"
    if [[ ! -d "$MODEL_DIR" ]]; then
        print_err "Model directory not found: $MODEL_DIR"
        exit 3
    fi
    CHECKPOINT_PATH="$(find_latest_checkpoint "$MODEL_DIR")"
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        print_err "No checkpoint found under: $MODEL_DIR"
        exit 4
    fi
    print_info "Checkpoint verified: $CHECKPOINT_PATH"
fi

mkdir -p "$(dirname "$TODO_FILE")"
touch "$TODO_FILE"

exec 200>"$LOCK_FILE"
flock -w "${EVAL_TODO_LOCK_TIMEOUT:-30}" 200 || {
    print_err "Timed out waiting for queue lock: $LOCK_FILE"
    exit 5
}

if grep -Fxq "$MODEL_NAME" "$TODO_FILE"; then
    print_info "Already queued: $MODEL_NAME"
else
    echo "$MODEL_NAME" >> "$TODO_FILE"
    print_ok "Queued for eval: $MODEL_NAME"
fi
