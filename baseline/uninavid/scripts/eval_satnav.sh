#!/bin/bash
# ==============================================================================
# Evaluate Uni-NaVid Baseline on SatNav task.
#
# Supports two calling modes:
#
#   1. Eval by name (recommended):
#      bash scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]
#      - Looks for checkpoint in output/uninavid-baseline/<exp_name_or_subpath>/
#      - Extracts data version from exp_name (data{XXXXXX} -> ver_XXXXXX)
#
#   2. Eval by checkpoint path:
#      bash scripts/eval_satnav.sh /path/to/checkpoint [split] [gpus] [max_episodes]
#
# Arguments:
#   exp_name_or_subpath / path  First argument
#   split            Evaluation split: val_seen / val_unseen / test
#                    If omitted, SatNav runs both val_seen and val_unseen
#   gpus             Number of GPUs (default: 8)
#   max_episodes     Limit episodes (for debug/smoke, optional)
#
# Environment variables:
#   SATNAV_VERSION   — Override data version (default: auto from exp name or latest)
#   MODEL_BASE       — Base model path for adapter-only checkpoints (optional)
#   LOCAL_CACHE_DIR  — Local disk dir to cache checkpoint (avoids NFS D-state).
#                      Auto-detected: uses /mnt/data4/jiangjiajun/uninavid_ckpt_cache
#                      if checkpoint is on NFS. Set to "" to disable caching.
#
# Output:
#   results/uninavid-baseline/<exp_name_or_subpath>/<split>/   (eval by name)
#   results/uninavid-baseline/by-path/<ckpt_name>/<split>/     (eval by path)
#
# Environment: conda env uninavid-baseline
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Args ----
INPUT="${1:-}"
SPLIT_ARG="${2:-}"
NUM_GPUS="${3:-8}"
MAX_EPISODES="${4:-}"
SATNAV_VERSION="${SATNAV_VERSION:-}"
MODEL_BASE="${MODEL_BASE:-}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/mnt/data4/jiangjiajun/uninavid_ckpt_cache}"

if [ -z "$INPUT" ]; then
    print_error "Usage: bash scripts/eval_satnav.sh <exp_name_or_subpath | checkpoint_path> [split] [gpus] [max_episodes]"
    echo ""
    echo "Examples:"
    echo "  # Eval by name"
    echo "  bash scripts/eval_satnav.sh uninavid-baseline-data260306-bs16-lr2e-5-20260311-120000"
    echo ""
    echo "  # Eval smoke test by subpath"
    echo "  bash scripts/eval_satnav.sh smoketest/uninavid-baseline-smoke-20260311-120000 val_seen 1 5"
    echo ""
    echo "  # Eval by checkpoint path"
    echo "  bash scripts/eval_satnav.sh /path/to/checkpoint val_unseen 8"
    exit 1
fi

if [ -n "$SPLIT_ARG" ]; then
    SPLITS_LIST="$SPLIT_ARG"
else
    SPLITS_LIST="val_seen val_unseen"
fi

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"

# ---- Detect mode: eval-by-name vs eval-by-path ----
EVAL_MODE="by_name"
if [[ "$INPUT" = /* ]] || [ -d "$INPUT" ]; then
    EVAL_MODE="by_path"
fi

# ---- Resolve checkpoint and output paths ----
if [ "$EVAL_MODE" = "by_name" ]; then
    EXP_NAME="$INPUT"
    MODEL_DIR="${REPO_ROOT}/output/uninavid-baseline/${EXP_NAME}"

    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Experiment directory not found: output/uninavid-baseline/${EXP_NAME}"
        exit 1
    fi

    # Find latest checkpoint (checkpoint-N sorted by N descending)
    CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

    if [ -z "$CHECKPOINT_DIR" ]; then
        if [ -f "${MODEL_DIR}/config.json" ]; then
            CHECKPOINT_DIR="$MODEL_DIR"
            print_info "Using merged model dir as checkpoint"
        else
            print_error "No checkpoint found in: ${MODEL_DIR}"
            exit 1
        fi
    fi

    # Extract data version from EXP_NAME: data{XXXXXX} -> ver_XXXXXX
    if [ -z "$SATNAV_VERSION" ]; then
        PARSED_VER=$(echo "$EXP_NAME" | grep -oP 'data\K\d+' | head -1 || true)
        if [ -n "$PARSED_VER" ]; then
            SATNAV_VERSION="ver_${PARSED_VER}"
            print_info "Parsed data version from exp name: ${SATNAV_VERSION}"
        fi
    fi

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/uninavid-baseline/${EXP_NAME}"

else
    CHECKPOINT_DIR="$INPUT"
    EXP_NAME="$(basename "${CHECKPOINT_DIR}")"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "Checkpoint directory not found: ${CHECKPOINT_DIR}"
        exit 1
    fi

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/uninavid-baseline/by-path/${EXP_NAME}"
fi

# ---- Resolve SatNav version ----
if [ -z "$SATNAV_VERSION" ]; then
    SATNAV_VERSION=$(ls -d "${SATNAV_DATA_ROOT}"/ver_* 2>/dev/null | sort | tail -1 | xargs basename || true)
    if [ -z "$SATNAV_VERSION" ]; then
        print_error "No SatNav data versions found in ${SATNAV_DATA_ROOT}"
        exit 1
    fi
    print_info "Auto-detected SatNav version: ${SATNAV_VERSION}"
else
    print_info "Using SatNav version: ${SATNAV_VERSION}"
fi

SATNAV_SCENES="${SATNAV_DATA_ROOT}/scenes"

# ---- NFS checkpoint cache (avoid D-state from 8 processes reading NFS simultaneously) ----
# If checkpoint is on an NFS mount and LOCAL_CACHE_DIR is set, rsync model weights
# (excluding DeepSpeed optimizer states) to local disk before loading.
maybe_cache_checkpoint() {
    local src="$1"
    if [ -z "$LOCAL_CACHE_DIR" ]; then
        echo "$src"
        return
    fi

    # Check if the path is on an NFS filesystem
    local mount_type
    mount_type=$(stat -f -c "%T" "$src" 2>/dev/null || echo "unknown")
    if [ "$mount_type" = "nfs" ] || findmnt -n -o FSTYPE --target "$src" 2>/dev/null | grep -q "^nfs"; then
        local ckpt_name
        ckpt_name=$(basename "$src")
        local local_ckpt="${LOCAL_CACHE_DIR}/${ckpt_name}"

        # Check if already cached (use sentinel file to mark complete cache)
        if [ -f "${local_ckpt}/.cache_complete" ]; then
            print_info "Using existing local checkpoint cache: ${local_ckpt}" >&2
            echo "$local_ckpt"
            return
        fi

        # Important: this function is used inside command substitution, so only
        # the resolved checkpoint path may go to stdout. All logs must go to stderr.
        print_warning "Checkpoint is on NFS (${mount_type}). Caching model weights to local disk to avoid I/O D-state..." >&2
        print_info "Source      : ${src}" >&2
        print_info "Destination : ${local_ckpt}" >&2
        mkdir -p "$local_ckpt"

        # Rsync model files only — exclude DeepSpeed optimizer states (global_step*)
        # which can be 80-100G and are not needed for inference
        rsync -ah --progress \
            --exclude="global_step*" \
            "${src}/" "${local_ckpt}/" >&2

        touch "${local_ckpt}/.cache_complete"
        local cached_size
        cached_size=$(du -sh "$local_ckpt" | cut -f1)
        print_success "Checkpoint cached locally (${cached_size}): ${local_ckpt}" >&2
        echo "$local_ckpt"
    else
        # Not NFS, use as-is
        echo "$src"
    fi
}

CHECKPOINT_DIR=$(maybe_cache_checkpoint "$CHECKPOINT_DIR")

# ---- PYTHONPATH ----
export PYTHONPATH="${BASELINE_DIR}/src:${BASELINE_DIR}:${PYTHONPATH:-}"

# ---- Launch ----
if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD=(torchrun)
else
    print_warning "torchrun not found, fallback to: python -m torch.distributed.run"
    TORCHRUN_CMD=(python -m torch.distributed.run)
fi

run_single_split() {
    local split="$1"
    local satnav_episodes="${SATNAV_DATA_ROOT}/${SATNAV_VERSION}/episodes/eval/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"

    if [ ! -f "$satnav_episodes" ]; then
        print_error "Episodes file not found: ${satnav_episodes}"
        return 1
    fi

    cp "$SATNAV_CONFIG_TEMPLATE" "$satnav_config"
    sed -i "s|DATA_PATH:.*|DATA_PATH: ${satnav_episodes}|" "$satnav_config"
    sed -i "s|SCENES_DIR:.*|SCENES_DIR: ${SATNAV_SCENES}|" "$satnav_config"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo "Uni-NaVid Baseline Evaluation"
    echo "=========================================="
    echo "  Eval mode  : ${EVAL_MODE}"
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Config     : ${satnav_config}"
    echo "  Data ver   : ${SATNAV_VERSION}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    [ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ -n "$MODEL_BASE"   ] && echo "  Model Base  : ${MODEL_BASE}"
    echo "=========================================="

    COMMON_ARGS=(
        --model_path "${CHECKPOINT_DIR}"
        --satnav_config_path "${satnav_config}"
        --eval_split "${split}"
        --output_path "${output_dir}"
    )

    if [ -n "$MAX_EPISODES" ]; then
        COMMON_ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi

    if [ -n "$MODEL_BASE" ]; then
        COMMON_ARGS+=(--model_base "${MODEL_BASE}")
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        "${TORCHRUN_CMD[@]}" \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port=$((RANDOM % 10000 + 20000)) \
            "${EVAL_SCRIPT}" \
            "${COMMON_ARGS[@]}" \
            --world_size "${NUM_GPUS}" \
            2>&1 | tee "${output_dir}/eval.log"
    else
        python "${EVAL_SCRIPT}" \
            "${COMMON_ARGS[@]}" \
            --world_size 1 \
            --rank 0 \
            2>&1 | tee "${output_dir}/eval.log"
    fi

    rm -f "${satnav_config}"
    print_success "Evaluation completed for split=${split}!"
    echo "  Results: ${output_dir}"
}

for SPLIT in ${SPLITS_LIST}; do
    run_single_split "${SPLIT}"
done
