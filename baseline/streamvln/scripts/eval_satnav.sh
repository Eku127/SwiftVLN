#!/bin/bash
# ==============================================================================
# Evaluate StreamVLN Baseline on SatNav task.
#
# Supports two calling modes:
#
#   1. Eval by name (recommended):
#      bash scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]
#      bash scripts/eval_satnav.sh --model_name <exp_name> [--model_dir <dir>] [--split <split>] [--gpus <n>]
#      - Looks for checkpoint in <model_dir>/<exp_name>/, default model_dir is output/streamvln-baseline
#      - Extracts data version and f/h/s eval params from exp_name
#
#   2. Eval by checkpoint path (backward compatible):
#      bash scripts/eval_satnav.sh /path/to/checkpoint [split] [gpus] [max_episodes]
#
# Arguments:
#   exp_name_or_subpath / path  First argument: experiment name/subpath or checkpoint path
#   split            Evaluation split: val_seen / val_unseen / test
#                    If omitted, SatNav runs both val_seen and val_unseen
#   gpus             Number of GPUs (default: 8)
#   max_episodes     Limit episodes for debugging (optional)
#
# Environment variables:
#   SATNAV_VERSION   — Override data version (default: auto from exp name or ver_260418)
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

usage() {
    echo "Usage:"
    echo ""
    echo "  # Eval by name from default output/streamvln-baseline"
    echo "  bash baseline/streamvln/scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]"
    echo ""
    echo "  # Eval by name from a custom model root"
    echo "  bash baseline/streamvln/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name streamvln-baseline-continue-1ep-f32h8s4-data260418p80-bs64-lr2e-5-20260420-153328 \\"
    echo "    --split val_seen --gpus 8"
    echo ""
    echo "  # Eval by checkpoint path"
    echo "  bash baseline/streamvln/scripts/eval_satnav.sh --checkpoint_path /path/to/checkpoint --split val_unseen --gpus 8"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/streamvln-baseline"

# ---- Args ----
MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
CHECKPOINT_PATH_ARG=""
SPLIT_ARG=""
NUM_GPUS="8"
MAX_EPISODES=""
DRY_RUN="false"
POSITIONAL=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_dir|--model-dir)
            MODEL_DIR_INPUT="${2:-}"
            shift 2
            ;;
        --model_name|--model-name)
            MODEL_NAME_ARG="${2:-}"
            shift 2
            ;;
        --checkpoint_path|--checkpoint-path)
            CHECKPOINT_PATH_ARG="${2:-}"
            shift 2
            ;;
        --split)
            SPLIT_ARG="${2:-}"
            shift 2
            ;;
        --gpus|--num_gpus|--num-gpus)
            NUM_GPUS="${2:-8}"
            shift 2
            ;;
        --max_episodes|--max-episodes)
            MAX_EPISODES="${2:-}"
            shift 2
            ;;
        --dry_run|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        --satnav_version|--satnav-version)
            SATNAV_VERSION="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        --*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ -n "$MODEL_NAME_ARG" ] && [ -n "$CHECKPOINT_PATH_ARG" ]; then
    print_error "--model_name and --checkpoint_path are mutually exclusive"
    exit 1
fi

if [ -n "$MODEL_NAME_ARG" ]; then
    INPUT="$MODEL_NAME_ARG"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[0]:-}"
    [ "${NUM_GPUS}" = "8" ] && NUM_GPUS="${POSITIONAL[1]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
elif [ -n "$CHECKPOINT_PATH_ARG" ]; then
    INPUT="$CHECKPOINT_PATH_ARG"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[0]:-}"
    [ "${NUM_GPUS}" = "8" ] && NUM_GPUS="${POSITIONAL[1]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
else
    INPUT="${POSITIONAL[0]:-}"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[1]:-}"
    [ "${NUM_GPUS}" = "8" ] && NUM_GPUS="${POSITIONAL[2]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[3]:-}"
fi

if [ -z "$INPUT" ]; then
    print_error "Missing model name or checkpoint path"
    usage
    exit 1
fi

if [ -n "$SPLIT_ARG" ]; then
    SPLITS_LIST="$SPLIT_ARG"
else
    SPLITS_LIST="val_seen val_unseen"
fi

if [ -z "$MODEL_DIR_INPUT" ]; then
    MODEL_ROOT_DIR="$DEFAULT_MODEL_DIR"
elif [[ "$MODEL_DIR_INPUT" = /* ]]; then
    MODEL_ROOT_DIR="$MODEL_DIR_INPUT"
else
    MODEL_ROOT_DIR="${REPO_ROOT}/${MODEL_DIR_INPUT}"
fi

NUM_FRAMES="32"
NUM_HISTORY="8"
NUM_FUTURE_STEPS="4"

# ---- Detect mode: eval-by-name vs eval-by-path ----
EVAL_MODE="by_name"
if [ -n "$CHECKPOINT_PATH_ARG" ] || [[ "$INPUT" = /* ]] || [ -d "$INPUT" ]; then
    EVAL_MODE="by_path"
fi

# ---- Resolve checkpoint and output paths ----
if [ "$EVAL_MODE" = "by_name" ]; then
    EXP_NAME="$INPUT"
    MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"
    if [ ! -d "$MODEL_DIR" ]; then
        LEGACY_MODEL_DIR="${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}"
        if [ -d "$LEGACY_MODEL_DIR" ]; then
            MODEL_DIR="$LEGACY_MODEL_DIR"
            print_warning "Model dir found in legacy path: ${MODEL_DIR}"
        fi
    fi

    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Experiment directory not found: ${MODEL_ROOT_DIR}/${EXP_NAME}"
        exit 1
    fi

    # Find latest checkpoint (checkpoint-N sorted by N descending)
    CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

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

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}"

else
    INPUT_PATH="${INPUT%/}"

    if [ ! -d "$INPUT_PATH" ]; then
        print_error "Checkpoint directory not found: ${INPUT_PATH}"
        exit 1
    fi

    if ls -d "${INPUT_PATH}"/checkpoint-* >/dev/null 2>&1; then
        MODEL_DIR="$INPUT_PATH"
        CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        EXP_NAME="$(basename "${MODEL_DIR}")"
    else
        CHECKPOINT_DIR="$INPUT_PATH"
        if [[ "$(basename "${CHECKPOINT_DIR}")" == checkpoint-* ]]; then
            EXP_NAME="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
        else
            EXP_NAME="$(basename "${CHECKPOINT_DIR}")"
        fi
    fi

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/streamvln-baseline/by-path/${EXP_NAME}"
fi

if [[ "$EXP_NAME" =~ f([0-9]+)h([0-9]+)s([0-9]+) ]]; then
    NUM_FRAMES="${BASH_REMATCH[1]}"
    NUM_HISTORY="${BASH_REMATCH[2]}"
    NUM_FUTURE_STEPS="${BASH_REMATCH[3]}"
    print_info "Parsed eval params from model name: frames=${NUM_FRAMES}, history=${NUM_HISTORY}, future_steps=${NUM_FUTURE_STEPS}"
else
    print_warning "Unable to parse f/h/s from model name, using defaults: frames=${NUM_FRAMES}, history=${NUM_HISTORY}, future_steps=${NUM_FUTURE_STEPS}"
fi

if [ -z "${SATNAV_VERSION:-}" ]; then
    PARSED_VER=$(echo "$EXP_NAME" | grep -oP 'data\K\d+' | head -1 || true)
    if [ -n "$PARSED_VER" ]; then
        SATNAV_VERSION="ver_${PARSED_VER}"
        print_info "Parsed data version from model name: ${SATNAV_VERSION}"
    fi
fi

# ---- Resolve SatNav version ----
if [ -z "${SATNAV_VERSION:-}" ]; then
    SATNAV_VERSION="ver_260418"
    print_info "Using default SatNav version: ${SATNAV_VERSION}"
else
    print_info "Using SatNav version: ${SATNAV_VERSION}"
fi

SATNAV_SCENES="${SATNAV_DATA_ROOT}/scenes"

# ---- Tokenizer ----
TOKENIZER_PATH="${CHECKPOINT_DIR}"
if [ ! -f "${CHECKPOINT_DIR}/tokenizer_config.json" ]; then
    TOKENIZER_PATH="${BASELINE_DIR}/model/LLaVA-Video-7B-Qwen2"
    print_info "No tokenizer in checkpoint, using local base model: ${TOKENIZER_PATH}"
fi

# ---- PYTHONPATH ----
export PYTHONPATH="/mnt/data1/home/jiangjiajun/workspace/StreamVLN:\
/mnt/data1/home/jiangjiajun/workspace/StreamVLN/streamvln:\
${BASELINE_DIR}:${PYTHONPATH:-}"

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
    echo "StreamVLN Baseline Evaluation"
    echo "=========================================="
    echo "  Eval mode  : ${EVAL_MODE}"
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Tokenizer  : ${TOKENIZER_PATH}"
    echo "  Config     : ${satnav_config}"
    echo "  Data ver   : ${SATNAV_VERSION}"
    echo "  Frames     : ${NUM_FRAMES}"
    echo "  History    : ${NUM_HISTORY}"
    echo "  Future     : ${NUM_FUTURE_STEPS}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    [ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ "$DRY_RUN" = "true" ] && echo "  Dry Run    : true"
    echo "=========================================="

    if [ "$DRY_RUN" = "true" ]; then
        rm -f "${satnav_config}"
        print_success "Dry run completed for split=${split}."
        return 0
    fi

    COMMON_ARGS=(
        --model_path "${CHECKPOINT_DIR}"
        --tokenizer_path "${TOKENIZER_PATH}"
        --satnav_config_path "${satnav_config}"
        --eval_split "${split}"
        --output_path "${output_dir}"
        --num_frames "${NUM_FRAMES}"
        --num_future_steps "${NUM_FUTURE_STEPS}"
        --num_history "${NUM_HISTORY}"
        --model_max_length 32768
    )

    if [ -n "$MAX_EPISODES" ]; then
        COMMON_ARGS+=(--max_episodes "${MAX_EPISODES}")
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
            --gpu 0 \
            2>&1 | tee "${output_dir}/eval.log"
    fi

    rm -f "${satnav_config}"
    print_success "Evaluation completed for split=${split}!"
    echo "  Results: ${output_dir}"
}

for SPLIT in ${SPLITS_LIST}; do
    run_single_split "${SPLIT}"
done
