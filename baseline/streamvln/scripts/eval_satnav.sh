#!/bin/bash
# ==============================================================================
# Evaluate StreamVLN Baseline on SatNav task.
#
# Supports two calling modes:
#
#   1. Eval by name (recommended):
#      bash scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]
#      - Looks for checkpoint in output/streamvln-baseline/<exp_name_or_subpath>/ (fallback: results/)
#      - Extracts data version from exp_name (data{XXXXXX} -> ver_XXXXXX)
#
#   2. Eval by checkpoint path (backward compatible):
#      bash scripts/eval_satnav.sh /path/to/checkpoint [split] [gpus] [max_episodes]
#
# Arguments:
#   exp_name_or_subpath / path  First argument: experiment name/subpath or checkpoint path
#   split            Evaluation split: val_unseen (default), val_seen, test
#   gpus             Number of GPUs (default: 8)
#   max_episodes     Limit episodes for debugging (optional)
#
# Environment variables:
#   SATNAV_VERSION   — Override data version (default: auto from exp name or latest)
#
# Output:
#   results/streamvln-baseline/<exp_name_or_subpath>/<split>/   (eval by name)
#   results/streamvln-baseline/by-path/<ckpt_name>/<split>/     (eval by path)
#
# Environment: conda env streamvln-baseline
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
SPLIT="${2:-val_unseen}"
NUM_GPUS="${3:-8}"
MAX_EPISODES="${4:-}"

if [ -z "$INPUT" ]; then
    print_error "Usage: bash scripts/eval_satnav.sh <exp_name_or_subpath | checkpoint_path> [split] [gpus] [max_episodes]"
    echo ""
    echo "Examples:"
    echo "  # Eval by name"
    echo "  bash scripts/eval_satnav.sh streamvln-baseline-continue-1ep-f32h8s4-data260306-bs32-lr2e-5-20260309-143000"
    echo ""
    echo "  # Eval smoke test by subpath"
    echo "  bash scripts/eval_satnav.sh smoketest/streamvln-baseline-continue-1ep-f32h8s4-data260306-bs32-lr2e-5-20260309-143000 val_seen 8"
    echo ""
    echo "  # Eval by checkpoint path (legacy)"
    echo "  bash scripts/eval_satnav.sh /path/to/checkpoint val_unseen 8"
    exit 1
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
    MODEL_DIR="${REPO_ROOT}/output/streamvln-baseline/${EXP_NAME}"
    if [ ! -d "$MODEL_DIR" ]; then
        LEGACY_MODEL_DIR="${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}"
        if [ -d "$LEGACY_MODEL_DIR" ]; then
            MODEL_DIR="$LEGACY_MODEL_DIR"
            print_warning "Model dir found in legacy path: ${MODEL_DIR}"
        fi
    fi

    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Experiment directory not found in output/ or results/: ${EXP_NAME}"
        exit 1
    fi

    # Find latest checkpoint (checkpoint-N sorted by N descending)
    CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

    # If no checkpoint-* subdir, check if the model dir itself is a merged model
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
    if [ -z "${SATNAV_VERSION:-}" ]; then
        PARSED_VER=$(echo "$EXP_NAME" | grep -oP 'data\K\d+' | head -1)
        if [ -n "$PARSED_VER" ]; then
            SATNAV_VERSION="ver_${PARSED_VER}"
            print_info "Parsed data version from exp name: ${SATNAV_VERSION}"
        fi
    fi

    OUTPUT_DIR="${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}/${SPLIT}"

else
    CHECKPOINT_DIR="$INPUT"
    EXP_NAME="$(basename "${CHECKPOINT_DIR}")"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "Checkpoint directory not found: ${CHECKPOINT_DIR}"
        exit 1
    fi

    OUTPUT_DIR="${REPO_ROOT}/results/streamvln-baseline/by-path/${EXP_NAME}/${SPLIT}"
fi

# ---- Resolve SatNav version ----
if [ -z "${SATNAV_VERSION:-}" ]; then
    SATNAV_VERSION=$(ls -d "${SATNAV_DATA_ROOT}"/ver_* 2>/dev/null | sort | tail -1 | xargs basename)
    if [ -z "$SATNAV_VERSION" ]; then
        print_error "No SatNav data versions found in ${SATNAV_DATA_ROOT}"
        exit 1
    fi
    print_info "Auto-detected SatNav version: ${SATNAV_VERSION}"
else
    print_info "Using SatNav version: ${SATNAV_VERSION}"
fi

SATNAV_EPISODES="${SATNAV_DATA_ROOT}/${SATNAV_VERSION}/episodes/eval/all_episodes.json"
SATNAV_SCENES="${SATNAV_DATA_ROOT}/scenes"

if [ ! -f "$SATNAV_EPISODES" ]; then
    print_error "Episodes file not found: ${SATNAV_EPISODES}"
    exit 1
fi

# ---- Generate version-specific config ----
SATNAV_CONFIG="${BASELINE_DIR}/configs/.satnav_task_eval_tmp.yaml"
trap 'rm -f "$SATNAV_CONFIG"' EXIT
cp "$SATNAV_CONFIG_TEMPLATE" "$SATNAV_CONFIG"
# Patch DATA_PATH and SCENES_DIR to match selected version
sed -i "s|DATA_PATH:.*|DATA_PATH: ${SATNAV_EPISODES}|" "$SATNAV_CONFIG"
sed -i "s|SCENES_DIR:.*|SCENES_DIR: ${SATNAV_SCENES}|" "$SATNAV_CONFIG"

# ---- Tokenizer ----
TOKENIZER_PATH="${CHECKPOINT_DIR}"
if [ ! -f "${CHECKPOINT_DIR}/tokenizer_config.json" ]; then
    TOKENIZER_PATH="${BASELINE_DIR}/model/LLaVA-Video-7B-Qwen2"
    print_info "No tokenizer in checkpoint, using local base model: ${TOKENIZER_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "StreamVLN Baseline Evaluation"
echo "=========================================="
echo "  Eval mode  : ${EVAL_MODE}"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  Checkpoint : ${CHECKPOINT_DIR}"
echo "  Tokenizer  : ${TOKENIZER_PATH}"
echo "  Config     : ${SATNAV_CONFIG}"
echo "  Data ver   : ${SATNAV_VERSION}"
echo "  Split      : ${SPLIT}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  GPUs       : ${NUM_GPUS}"
[ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
echo "=========================================="

# ---- PYTHONPATH ----
export PYTHONPATH="/mnt/data1/home/jiangjiajun/workspace/StreamVLN:\
/mnt/data1/home/jiangjiajun/workspace/StreamVLN/streamvln:\
${BASELINE_DIR}:${PYTHONPATH:-}"

# ---- Build common args ----
COMMON_ARGS=(
    --model_path "${CHECKPOINT_DIR}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --satnav_config_path "${SATNAV_CONFIG}"
    --eval_split "${SPLIT}"
    --output_path "${OUTPUT_DIR}"
    --num_frames 32
    --num_future_steps 4
    --num_history 8
    --model_max_length 32768
)

if [ -n "$MAX_EPISODES" ]; then
    COMMON_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

# ---- Launch ----
if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD=(torchrun)
else
    print_warning "torchrun not found, fallback to: python -m torch.distributed.run"
    TORCHRUN_CMD=(python -m torch.distributed.run)
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    "${TORCHRUN_CMD[@]}" \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=$((RANDOM % 10000 + 20000)) \
        "${EVAL_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --world_size "${NUM_GPUS}" \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
    python "${EVAL_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --world_size 1 \
        --rank 0 \
        --gpu 0 \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
fi

echo ""
print_success "Evaluation completed!"
echo "  Results: ${OUTPUT_DIR}"
echo ""
