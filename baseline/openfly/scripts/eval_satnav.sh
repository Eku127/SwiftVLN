#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

INPUT="${1:-}"
SPLIT_ARG="${2:-}"
NUM_GPUS="${3:-1}"
MAX_EPISODES="${4:-}"
SATNAV_VERSION="${SATNAV_VERSION:-}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
OPENFLY_ACTION_FORMAT="${OPENFLY_ACTION_FORMAT:-auto}"
OPENFLY_UNNORM_KEY="${OPENFLY_UNNORM_KEY:-}"
RESULTS_BASE_OVERRIDE="${RESULTS_BASE_OVERRIDE:-}"

if [ -z "${INPUT}" ]; then
    print_error "Usage: bash scripts/eval_satnav.sh <exp_name_or_checkpoint_path> [split] [gpus] [max_episodes]"
    exit 1
fi

if [ -n "${SPLIT_ARG}" ]; then
    SPLITS_LIST="${SPLIT_ARG}"
else
    SPLITS_LIST="val_seen val_unseen"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"

EVAL_MODE="by_name"
if [[ "${INPUT}" = /* ]] || [ -d "${INPUT}" ]; then
    EVAL_MODE="by_path"
fi

if [ "${EVAL_MODE}" = "by_name" ]; then
    EXP_NAME="${INPUT}"
    MODEL_DIR="${REPO_ROOT}/output/openfly-baseline/${EXP_NAME}"
    if [ ! -d "${MODEL_DIR}" ]; then
        print_error "Experiment directory not found: ${MODEL_DIR}"
        exit 1
    fi
    CHECKPOINT_DIR="$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [ -z "${CHECKPOINT_DIR}" ]; then
        CHECKPOINT_DIR="${MODEL_DIR}"
    fi
    OUTPUT_BASE_DIR="${REPO_ROOT}/results/openfly-baseline/${EXP_NAME}"
else
    CHECKPOINT_DIR="${INPUT}"
    EXP_NAME="$(basename "${CHECKPOINT_DIR}")"
    OUTPUT_BASE_DIR="${REPO_ROOT}/results/openfly-baseline/by-path/${EXP_NAME}"
fi

if [ -n "${RESULTS_BASE_OVERRIDE}" ]; then
    OUTPUT_BASE_DIR="${RESULTS_BASE_OVERRIDE}"
fi

if [ -z "${SATNAV_VERSION}" ]; then
    SATNAV_VERSION=$(ls -d "${SATNAV_DATA_ROOT}"/ver_* 2>/dev/null | sort | tail -1 | xargs basename || true)
fi

if [ -z "${SATNAV_VERSION}" ]; then
    print_error "Unable to determine SATNAV_VERSION"
    exit 1
fi

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate openfly-baseline
export PYTHONPATH="${BASELINE_DIR}/src:${PYTHONPATH:-}"

run_single_split() {
    local split="$1"
    local satnav_episodes="${SATNAV_DATA_ROOT}/${SATNAV_VERSION}/episodes/eval/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"
    local run_id="${OPENFLY_EVAL_RUN_ID:-openfly_eval_${split}_$(date +%Y%m%d_%H%M%S)}"

    cp "${SATNAV_CONFIG_TEMPLATE}" "${satnav_config}"
    sed -i "s|DATA_PATH:.*|DATA_PATH: ${satnav_episodes}|" "${satnav_config}"
    mkdir -p "${output_dir}"

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
