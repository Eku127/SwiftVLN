#!/usr/bin/env bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/openfly-baseline"

usage() {
    echo "Usage:"
    echo ""
    echo "  # Eval by name from default output/openfly-baseline"
    echo "  bash baseline/openfly/scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]"
    echo ""
    echo "  # Eval by name from a custom model root"
    echo "  bash baseline/openfly/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357 \\"
    echo "    --split val_seen --gpus 8"
    echo ""
    echo "  # Eval by checkpoint path"
    echo "  bash baseline/openfly/scripts/eval_satnav.sh --checkpoint_path /path/to/checkpoint --split val_unseen --gpus 8"
}

MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
CHECKPOINT_PATH_ARG=""
SPLIT_ARG=""
NUM_GPUS="1"
MAX_EPISODES=""
DRY_RUN="false"
SATNAV_VERSION="${SATNAV_VERSION:-}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
OPENFLY_ACTION_FORMAT="${OPENFLY_ACTION_FORMAT:-auto}"
OPENFLY_UNNORM_KEY="${OPENFLY_UNNORM_KEY:-}"
OPENFLY_ACTION_HISTORY_LIMIT="${OPENFLY_ACTION_HISTORY_LIMIT:-16}"
RESULTS_BASE_OVERRIDE="${RESULTS_BASE_OVERRIDE:-}"
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
            NUM_GPUS="${2:-1}"
            shift 2
            ;;
        --max_episodes|--max-episodes)
            MAX_EPISODES="${2:-}"
            shift 2
            ;;
        --satnav_version|--satnav-version)
            SATNAV_VERSION="${2:-}"
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
        --results_base|--results-base)
            RESULTS_BASE_OVERRIDE="${2:-}"
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
    [ "$NUM_GPUS" = "1" ] && NUM_GPUS="${POSITIONAL[1]:-1}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
elif [ -n "$CHECKPOINT_PATH_ARG" ]; then
    INPUT="$CHECKPOINT_PATH_ARG"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[0]:-}"
    [ "$NUM_GPUS" = "1" ] && NUM_GPUS="${POSITIONAL[1]:-1}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
else
    INPUT="${POSITIONAL[0]:-}"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[1]:-}"
    [ "$NUM_GPUS" = "1" ] && NUM_GPUS="${POSITIONAL[2]:-1}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[3]:-}"
fi

if [ -z "${INPUT}" ]; then
    print_error "Missing model name or checkpoint path"
    usage
    exit 1
fi

if [ -n "${SPLIT_ARG}" ]; then
    SPLITS_LIST="${SPLIT_ARG}"
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

EVAL_MODE="by_name"
if [ -n "$CHECKPOINT_PATH_ARG" ] || [[ "${INPUT}" = /* ]] || [ -d "${INPUT}" ]; then
    EVAL_MODE="by_path"
fi

if [ "${EVAL_MODE}" = "by_name" ]; then
    EXP_NAME="${INPUT}"
    MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"
    if [ ! -d "${MODEL_DIR}" ]; then
        print_error "Experiment directory not found: ${MODEL_ROOT_DIR}/${EXP_NAME}"
        exit 1
    fi
    CHECKPOINT_DIR="$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [ -z "${CHECKPOINT_DIR}" ]; then
        CHECKPOINT_DIR="${MODEL_DIR}"
    fi
    OUTPUT_BASE_DIR="${REPO_ROOT}/results/openfly-baseline/${EXP_NAME}"
else
    INPUT_PATH="${INPUT%/}"
    if [ ! -d "${INPUT_PATH}" ]; then
        print_error "Checkpoint directory not found: ${INPUT_PATH}"
        exit 1
    fi
    if ls -d "${INPUT_PATH}"/checkpoint-* >/dev/null 2>&1; then
        MODEL_DIR="${INPUT_PATH}"
        CHECKPOINT_DIR="$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
        [ -z "${CHECKPOINT_DIR}" ] && CHECKPOINT_DIR="${MODEL_DIR}"
        EXP_NAME="$(basename "${MODEL_DIR}")"
    else
        CHECKPOINT_DIR="${INPUT_PATH}"
        if [[ "$(basename "${CHECKPOINT_DIR}")" == checkpoint-* ]]; then
            EXP_NAME="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
        else
            EXP_NAME="$(basename "${CHECKPOINT_DIR}")"
        fi
    fi
    OUTPUT_BASE_DIR="${REPO_ROOT}/results/openfly-baseline/by-path/${EXP_NAME}"
fi

if [ -z "${SATNAV_VERSION}" ]; then
    PARSED_VER=$(echo "${EXP_NAME}" | grep -oP 'data\K\d+' | head -1 || true)
    if [ -n "${PARSED_VER}" ]; then
        SATNAV_VERSION="ver_${PARSED_VER}"
        print_info "Parsed data version from model name: ${SATNAV_VERSION}"
    fi
fi

if [[ "${OPENFLY_ACTION_FORMAT}" = "auto" && "${EXP_NAME}" =~ -act(compact|original) ]]; then
    OPENFLY_ACTION_FORMAT="${BASH_REMATCH[1]}"
    print_info "Parsed action format from model name: ${OPENFLY_ACTION_FORMAT}"
fi

if [[ "${OPENFLY_ACTION_HISTORY_LIMIT}" = "16" && "${EXP_NAME}" =~ -hist([0-9]+) ]]; then
    OPENFLY_ACTION_HISTORY_LIMIT="${BASH_REMATCH[1]}"
    print_info "Parsed action history limit from model name: ${OPENFLY_ACTION_HISTORY_LIMIT}"
fi

if [ -n "${RESULTS_BASE_OVERRIDE}" ]; then
    OUTPUT_BASE_DIR="${RESULTS_BASE_OVERRIDE}"
fi

if [ -z "${SATNAV_VERSION}" ]; then
    SATNAV_VERSION="ver_260418"
    print_info "Using default SatNav version: ${SATNAV_VERSION}"
else
    print_info "Using SatNav version: ${SATNAV_VERSION}"
fi

if [ "${DRY_RUN}" != "true" ]; then
    source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
    conda activate openfly-baseline
fi
export PYTHONPATH="${BASELINE_DIR}/src:${PYTHONPATH:-}"
export OPENFLY_ACTION_HISTORY_LIMIT

run_single_split() {
    local split="$1"
    local satnav_episodes="${SATNAV_DATA_ROOT}/${SATNAV_VERSION}/episodes/eval/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"
    local run_id="${OPENFLY_EVAL_RUN_ID:-openfly_eval_${split}_$(date +%Y%m%d_%H%M%S)}"

    if [ ! -f "${satnav_episodes}" ]; then
        print_error "Episodes file not found: ${satnav_episodes}"
        return 1
    fi

    cp "${SATNAV_CONFIG_TEMPLATE}" "${satnav_config}"
    sed -i "s|DATA_PATH:.*|DATA_PATH: ${satnav_episodes}|" "${satnav_config}"
    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo "OpenFly Baseline Evaluation"
    echo "=========================================="
    echo "  Eval mode  : ${EVAL_MODE}"
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Config     : ${satnav_config}"
    echo "  Data ver   : ${SATNAV_VERSION}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    echo "  Action fmt : ${OPENFLY_ACTION_FORMAT}"
    echo "  History    : ${OPENFLY_ACTION_HISTORY_LIMIT}"
    [ -n "${MAX_EPISODES}" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ "${DRY_RUN}" = "true" ] && echo "  Dry Run    : true"
    echo "=========================================="

    if [ "${DRY_RUN}" = "true" ]; then
        rm -f "${satnav_config}"
        print_success "Dry run completed for split=${split}."
        return 0
    fi

    ARGS=(
        --model_path "${CHECKPOINT_DIR}"
        --satnav_config_path "${satnav_config}"
        --eval_split "${split}"
        --output_path "${output_dir}"
        --torch_dtype "${TORCH_DTYPE}"
        --action_format "${OPENFLY_ACTION_FORMAT}"
        --run_id "${run_id}"
    )

    if [ -n "${OPENFLY_UNNORM_KEY}" ]; then
        ARGS+=(--unnorm_key "${OPENFLY_UNNORM_KEY}")
    fi

    if [ -n "${MAX_EPISODES}" ]; then
        ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi

    if [ "${NUM_GPUS}" -gt 1 ]; then
        local _master_addr="${MASTER_ADDR}"
        local _master_port="${MASTER_PORT}"
        unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT
        export MASTER_ADDR="${_master_addr}"
        export MASTER_PORT="${_master_port}"
        torchrun --nnodes 1 --nproc-per-node "${NUM_GPUS}" --master_addr "${_master_addr}" --master_port "${_master_port}" \
            "${EVAL_SCRIPT}" "${ARGS[@]}"
    else
        python "${EVAL_SCRIPT}" "${ARGS[@]}"
    fi

    rm -f "${satnav_config}"
}

for split in ${SPLITS_LIST}; do
    print_info "Running OpenFly SatNav eval: split=${split} model=${CHECKPOINT_DIR}"
    run_single_split "${split}"
done
