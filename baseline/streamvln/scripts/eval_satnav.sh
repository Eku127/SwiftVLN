#!/bin/bash
# ==============================================================================
# Evaluate StreamVLN Baseline on SatNav task.
#
# Eval data and split are controlled by:
#   baseline/streamvln/configs/satnav_task.yaml
#
# Required:
#   --model_name   Model directory name under --model_dir
#
# Optional:
#   --model_dir    Model root directory (default: output/streamvln-baseline)
#   --gpus         Number of GPUs (default: 8)
#   --max_episodes Limit episodes for debugging
#   --dry_run      Resolve paths and print launch config without running eval
#
# Output:
#   results/streamvln-baseline/<model_name>/<split>/
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
    echo "  bash baseline/streamvln/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name streamvln-baseline-continue-1ep-f32h8s4-lr2e-5 \\"
    echo "    --gpus 8"
    echo ""
    echo "Notes:"
    echo "  - Eval data and split come from baseline/streamvln/configs/satnav_task.yaml."
    echo "  - DATASET.DATA_PATH must be the eval split parent dir, e.g. .../episodes/eval."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/streamvln-baseline"

read_config_value() {
    local key="$1"
    awk -v key="$key" '
        $1 == key ":" {
            sub(/^[^:]+:[[:space:]]*/, "")
            gsub(/^["'\''"]|["'\''"]$/, "")
            print
            exit
        }
    ' "$SATNAV_CONFIG_TEMPLATE"
}

# ---- Args ----
MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
NUM_GPUS="8"
MAX_EPISODES=""
DRY_RUN="false"

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
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            print_error "Positional arguments are not supported: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME_ARG" ]; then
    print_error "Missing required --model_name"
    usage
    exit 1
fi

EXP_NAME="$MODEL_NAME_ARG"

CONFIG_SPLIT="$(read_config_value SPLIT)"

if [ -z "$CONFIG_SPLIT" ]; then
    print_error "DATASET.SPLIT not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
elif [ "$CONFIG_SPLIT" = "all" ]; then
    SPLITS_LIST="val_seen val_unseen"
else
    SPLITS_LIST="$CONFIG_SPLIT"
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

# ---- Resolve checkpoint and output paths ----
MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"
if [ ! -d "$MODEL_DIR" ]; then
    print_error "Model directory not found: ${MODEL_DIR}"
    exit 1
fi

CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CHECKPOINT_DIR" ]; then
    print_error "No checkpoint-* directory found in: ${MODEL_DIR}"
    exit 1
fi

OUTPUT_BASE_DIR="${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}"

if [[ "$EXP_NAME" =~ f([0-9]+)h([0-9]+)s([0-9]+) ]]; then
    NUM_FRAMES="${BASH_REMATCH[1]}"
    NUM_HISTORY="${BASH_REMATCH[2]}"
    NUM_FUTURE_STEPS="${BASH_REMATCH[3]}"
    print_info "Parsed eval params from model name: frames=${NUM_FRAMES}, history=${NUM_HISTORY}, future_steps=${NUM_FUTURE_STEPS}"
else
    print_warning "Unable to parse f/h/s from model name, using defaults: frames=${NUM_FRAMES}, history=${NUM_HISTORY}, future_steps=${NUM_FUTURE_STEPS}"
fi

# ---- Resolve SatNav data from config ----
CONFIG_DATA_PATH="$(read_config_value DATA_PATH)"
CONFIG_SCENES_DIR="$(read_config_value SCENES_DIR)"

if [ -z "$CONFIG_DATA_PATH" ]; then
    print_error "DATASET.DATA_PATH not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
fi

if [ -z "$CONFIG_SCENES_DIR" ]; then
    print_error "DATASET.SCENES_DIR not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
fi

if [ ! -d "$CONFIG_DATA_PATH" ]; then
    print_error "DATASET.DATA_PATH must be an eval split parent directory: ${CONFIG_DATA_PATH}"
    exit 1
fi

SATNAV_SCENES="${CONFIG_SCENES_DIR}"
print_info "Using SatNav eval data root from config: ${CONFIG_DATA_PATH}"

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
    local satnav_episodes="${CONFIG_DATA_PATH%/}/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"

    if [ ! -f "$satnav_episodes" ]; then
        print_error "Episodes file not found: ${satnav_episodes}"
        return 1
    fi

    cp "$SATNAV_CONFIG_TEMPLATE" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)SPLIT:.*|\\1SPLIT: ${split}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)DATA_PATH:.*|\\1DATA_PATH: ${satnav_episodes}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)SCENES_DIR:.*|\\1SCENES_DIR: ${SATNAV_SCENES}|" "$satnav_config"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo "StreamVLN Baseline Evaluation"
    echo "=========================================="
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Tokenizer  : ${TOKENIZER_PATH}"
    echo "  Config     : ${satnav_config}"
    echo "  Episodes   : ${satnav_episodes}"
    echo "  Scenes     : ${SATNAV_SCENES}"
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
