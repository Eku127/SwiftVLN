#!/usr/bin/env bash
# ==============================================================================
# Evaluate OpenFly Baseline on SatNav task.
#
# Eval data and split are controlled by:
#   baseline/openfly/configs/satnav_task.yaml
#
# Required:
#   --model_name   Model directory name under --model_dir
#
# Optional:
#   --model_dir    Model root directory (default: output/openfly-baseline)
#   --gpus         Number of GPUs (default: 8)
#   --max_episodes Limit episodes for debugging
#   --action_format auto|compact|original (default: auto)
#   --action_history_limit N (default: parsed from model name or 16)
#   --processor_path Optional processor/tokenizer source
#   --dry_run      Resolve paths and print launch config without running eval
#
# Output:
#   results/openfly-baseline/<model_name>/<split>/
#
# Environment: conda env openfly-baseline
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
    echo "  bash baseline/openfly/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name openfly-satnav-continue-1ep-actcompact-hist16-lr2e-5 \\"
    echo "    --gpus 8"
    echo ""
    echo "Notes:"
    echo "  - Eval data and split come from baseline/openfly/configs/satnav_task.yaml."
    echo "  - DATASET.DATA_PATH must be the eval split parent dir, e.g. .../episodes/eval."
    echo "  - action format and history may be parsed from -act<format> and -hist<N>."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
# shellcheck source=../../../src/swiftvln/scripts/lib/local_env.sh
source "${REPO_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env "${BASELINE_DIR}"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/openfly-baseline"

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

has_model_weights() {
    local path="$1"
    find "$path" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) | grep -q .
}

resolve_model_load_dir() {
    local model_dir="$1"
    local checkpoint_dir

    if [ -f "${model_dir}/config.json" ] && has_model_weights "$model_dir"; then
        echo "$model_dir"
        return 0
    fi

    checkpoint_dir="$(ls -d "${model_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [ -n "$checkpoint_dir" ] && [ -f "${checkpoint_dir}/config.json" ] && has_model_weights "$checkpoint_dir"; then
        echo "$checkpoint_dir"
        return 0
    fi

    return 1
}

MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
NUM_GPUS="8"
MAX_EPISODES=""
DRY_RUN="false"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
OPENFLY_ACTION_FORMAT="${OPENFLY_ACTION_FORMAT:-auto}"
OPENFLY_ACTION_HISTORY_LIMIT="${OPENFLY_ACTION_HISTORY_LIMIT:-16}"
OPENFLY_UNNORM_KEY="${OPENFLY_UNNORM_KEY:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"

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
        --action_format|--action-format)
            OPENFLY_ACTION_FORMAT="${2:-auto}"
            shift 2
            ;;
        --action_history_limit|--action-history-limit)
            OPENFLY_ACTION_HISTORY_LIMIT="${2:-16}"
            shift 2
            ;;
        --processor_path|--processor-path)
            PROCESSOR_PATH="${2:-}"
            shift 2
            ;;
        --dry_run|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        --checkpoint_path|--checkpoint-path)
            print_error "--checkpoint_path is no longer supported. Use --model_dir + --model_name."
            exit 1
            ;;
        --split|--satnav_version|--satnav-version)
            print_error "$1 is no longer supported. Set DATASET.SPLIT/DATA_PATH in ${SATNAV_CONFIG_TEMPLATE}."
            exit 1
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

if [ -z "$MODEL_DIR_INPUT" ]; then
    MODEL_ROOT_DIR="$DEFAULT_MODEL_DIR"
elif [[ "$MODEL_DIR_INPUT" = /* ]]; then
    MODEL_ROOT_DIR="$MODEL_DIR_INPUT"
else
    MODEL_ROOT_DIR="${REPO_ROOT}/${MODEL_DIR_INPUT}"
fi

EXP_NAME="$MODEL_NAME_ARG"
MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"
OUTPUT_BASE_DIR="${REPO_ROOT}/results/openfly-baseline/${EXP_NAME}"

if [ ! -d "$MODEL_DIR" ]; then
    print_error "Model directory not found: ${MODEL_DIR}"
    exit 1
fi

if ! CHECKPOINT_DIR="$(resolve_model_load_dir "$MODEL_DIR")"; then
    print_error "No loadable OpenFly model root or checkpoint found in: ${MODEL_DIR}"
    exit 1
fi

if [[ "$OPENFLY_ACTION_FORMAT" = "auto" && "$EXP_NAME" =~ -act(compact|original) ]]; then
    OPENFLY_ACTION_FORMAT="${BASH_REMATCH[1]}"
    print_info "Parsed action format from model name: ${OPENFLY_ACTION_FORMAT}"
fi

if [[ "$OPENFLY_ACTION_HISTORY_LIMIT" = "16" && "$EXP_NAME" =~ -hist([0-9]+) ]]; then
    OPENFLY_ACTION_HISTORY_LIMIT="${BASH_REMATCH[1]}"
    print_info "Parsed action history limit from model name: ${OPENFLY_ACTION_HISTORY_LIMIT}"
fi

CONFIG_SPLIT="$(read_config_value SPLIT)"
CONFIG_DATA_PATH="$(read_config_value DATA_PATH)"
CONFIG_SCENES_DIR="$(read_config_value SCENES_DIR)"
CONFIG_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT:-${CONFIG_DATA_PATH}}"
CONFIG_SCENES_DIR="${SWIFTVLN_SATNAV_SCENES_DIR:-${CONFIG_SCENES_DIR}}"

if [ -z "$CONFIG_SPLIT" ]; then
    print_error "DATASET.SPLIT not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
elif [ "$CONFIG_SPLIT" = "all" ]; then
    SPLITS_LIST="val_seen val_unseen"
else
    SPLITS_LIST="$CONFIG_SPLIT"
fi

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

SATNAV_SCENES="$CONFIG_SCENES_DIR"
print_info "Using SatNav eval data root from config: ${CONFIG_DATA_PATH}"
print_info "Using OpenFly model load dir: ${CHECKPOINT_DIR}"

if [ "$DRY_RUN" != "true" ]; then
    swiftvln_activate_conda openfly-baseline
fi

export PYTHONPATH="${BASELINE_DIR}/src:${PYTHONPATH:-}"
export OPENFLY_ACTION_HISTORY_LIMIT

run_single_split() {
    local split="$1"
    local satnav_episodes="${CONFIG_DATA_PATH%/}/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"
    local run_id="${OPENFLY_EVAL_RUN_ID:-openfly_eval_${split}_$(date +%Y%m%d_%H%M%S)}"

    if [ ! -f "$satnav_episodes" ]; then
        print_error "Episodes file not found: ${satnav_episodes}"
        return 1
    fi

    cp "$SATNAV_CONFIG_TEMPLATE" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)SPLIT:.*|\\1SPLIT: ${split}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)DATA_PATH:.*|\\1DATA_PATH: ${satnav_episodes}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)SCENES_DIR:.*|\\1SCENES_DIR: ${SATNAV_SCENES}|" "$satnav_config"

    mkdir -p "$output_dir"

    echo ""
    echo "=========================================="
    echo "OpenFly Baseline Evaluation"
    echo "=========================================="
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  ModelRoot  : ${MODEL_DIR}"
    echo "  ModelLoad  : ${CHECKPOINT_DIR}"
    echo "  Config     : ${satnav_config}"
    echo "  EvalRoot   : ${CONFIG_DATA_PATH}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    echo "  Action fmt : ${OPENFLY_ACTION_FORMAT}"
    echo "  History    : ${OPENFLY_ACTION_HISTORY_LIMIT}"
    [ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ -n "$PROCESSOR_PATH" ] && echo "  Processor  : ${PROCESSOR_PATH}"
    [ "$DRY_RUN" = "true" ] && echo "  Dry Run    : true"
    echo "=========================================="

    if [ "$DRY_RUN" = "true" ]; then
        rm -f "$satnav_config"
        print_success "Dry run completed for split=${split}."
        return 0
    fi

    ARGS=(
        --model_path "$CHECKPOINT_DIR"
        --satnav_config_path "$satnav_config"
        --eval_split "$split"
        --output_path "$output_dir"
        --torch_dtype "$TORCH_DTYPE"
        --action_format "$OPENFLY_ACTION_FORMAT"
        --run_id "$run_id"
    )

    if [ -n "$PROCESSOR_PATH" ]; then
        ARGS+=(--processor_path "$PROCESSOR_PATH")
    fi

    if [ -n "$OPENFLY_UNNORM_KEY" ]; then
        ARGS+=(--unnorm_key "$OPENFLY_UNNORM_KEY")
    fi

    if [ -n "$MAX_EPISODES" ]; then
        ARGS+=(--max_episodes "$MAX_EPISODES")
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        local _master_addr="${MASTER_ADDR}"
        local _master_port="${MASTER_PORT}"
        unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT
        export MASTER_ADDR="${_master_addr}"
        export MASTER_PORT="${_master_port}"
        torchrun --nnodes 1 --nproc-per-node "$NUM_GPUS" --master_addr "$_master_addr" --master_port "$_master_port" \
            "$EVAL_SCRIPT" "${ARGS[@]}" 2>&1 | tee "${output_dir}/eval.log"
    else
        python "$EVAL_SCRIPT" "${ARGS[@]}" 2>&1 | tee "${output_dir}/eval.log"
    fi

    rm -f "$satnav_config"
    print_success "Evaluation completed for split=${split}!"
    echo "  Results: ${output_dir}"
}

for SPLIT in ${SPLITS_LIST}; do
    run_single_split "$SPLIT"
done
