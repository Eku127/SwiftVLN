#!/bin/bash
# ==============================================================================
# One-click pipeline: train StreamVLN baseline on SatNav, then auto-eval.
#
# Usage:
#   bash baseline/streamvln/scripts/train_eval_satnav.sh [continue|scratch]
#
# Defaults:
#   mode = continue
#   eval split/data are controlled by baseline/streamvln/configs/satnav_task.yaml
#
# Env (optional):
#   SATNAV_DATASET   Data dir name under satnav_datasets (default: SatNav-v0.1)
#   SATNAV_VERSION   Deprecated alias for SATNAV_DATASET, kept for old launchers
#   SATNAV_TRAIN_DATA_DIR Explicit trajectory_data dir override
#   TRAIN_GPUS       GPU count for training (default: 8)
#   EVAL_GPUS        GPU count for eval (default: 8)
#   EVAL_MODEL_DIR   Model root for eval (default: output/streamvln-baseline)
#   EVAL_MAX_EPISODES  Limit eval episodes for smoke/debug (default: unset)
#   CLEAN_EVAL_FIRST true/false, clear target eval output before run (default: true)
# ==============================================================================

set -euo pipefail

MODE="${1:-continue}"

if [ $# -ge 2 ] && [ -n "${2:-}" ]; then
    echo "[ERROR] Positional eval split is no longer supported."
    echo "[ERROR] Set DATASET.SPLIT in baseline/streamvln/configs/satnav_task.yaml instead."
    exit 1
fi

if [ "$MODE" != "continue" ] && [ "$MODE" != "scratch" ]; then
    echo "[ERROR] Unknown mode: ${MODE}. Must be continue|scratch"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
# shellcheck source=../../../src/swiftvln/scripts/lib/local_env.sh
source "${REPO_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env "${BASELINE_DIR}"

TRAIN_GPUS="${TRAIN_GPUS:-8}"
EVAL_GPUS="${EVAL_GPUS:-8}"
EVAL_MAX_EPISODES="${EVAL_MAX_EPISODES:-}"
CLEAN_EVAL_FIRST="${CLEAN_EVAL_FIRST:-true}"
SMOKE_TEST="${SMOKE_TEST:-false}"
SATNAV_DATASET="${SATNAV_DATASET:-${SATNAV_VERSION:-${SWIFTVLN_SATNAV_DATASET:-SatNav-v0.1}}}"
EVAL_MODEL_DIR="${EVAL_MODEL_DIR:-${REPO_ROOT}/output/streamvln-baseline}"
SATNAV_CONFIG="${BASELINE_DIR}/configs/satnav_task.yaml"

read_config_value() {
    local key="$1"
    awk -v key="$key" '
        $1 == key ":" {
            sub(/^[^:]+:[[:space:]]*/, "")
            gsub(/^["'\''"]|["'\''"]$/, "")
            print
            exit
        }
    ' "$SATNAV_CONFIG"
}

CONFIG_SPLIT="$(read_config_value SPLIT)"
CONFIG_DATA_PATH="$(read_config_value DATA_PATH)"
CONFIG_DATA_PATH="${SWIFTVLN_SATNAV_EVAL_ROOT:-${CONFIG_DATA_PATH}}"

PIPE_TS="$(date +%Y%m%d-%H%M%S)"
PIPE_LOG="/tmp/streamvln_baseline_train_eval_${PIPE_TS}.log"
HOSTNAME_STR="$(hostname)"

extract_exp_name_from_log() {
    local log_path="$1"
    local exp_name
    if command -v rg >/dev/null 2>&1; then
        exp_name="$(rg 'EXP_NAME' "$log_path" 2>/dev/null | tail -1 | sed -E 's/.*EXP_NAME[[:space:]]*:[[:space:]]*//')"
    else
        exp_name="$(grep -E 'EXP_NAME' "$log_path" 2>/dev/null | tail -1 | sed -E 's/.*EXP_NAME[[:space:]]*:[[:space:]]*//')"
    fi
    if [ -n "$exp_name" ]; then
        echo "$exp_name"
        return 0
    fi
    if [ "${SMOKE_TEST}" = "true" ]; then
        exp_name="$(ls -1dt "${REPO_ROOT}"/output/streamvln-baseline/smoketest/streamvln-baseline-"${MODE}"-* 2>/dev/null | head -1 | xargs -r basename)"
    else
        exp_name="$(ls -1dt "${REPO_ROOT}"/output/streamvln-baseline/streamvln-baseline-"${MODE}"-* 2>/dev/null | head -1 | xargs -r basename)"
    fi
    if [ -z "$exp_name" ]; then
        exp_name="$(ls -1dt "${REPO_ROOT}"/results/streamvln-baseline/streamvln-baseline-"${MODE}"-* 2>/dev/null | head -1 | xargs -r basename)"
    fi
    if [ -n "$exp_name" ]; then
        echo "$exp_name"
        return 0
    fi
    return 1
}

TRAIN_START_TS="$(date +%s)"
echo "[INFO] Pipeline log: ${PIPE_LOG}" | tee -a "${PIPE_LOG}"
echo "[INFO] Host: ${HOSTNAME_STR}" | tee -a "${PIPE_LOG}"
echo "[INFO] Mode: ${MODE}, train_gpus=${TRAIN_GPUS}, satnav_dataset=${SATNAV_DATASET}, eval_split_config=${CONFIG_SPLIT:-unknown}" | tee -a "${PIPE_LOG}"
echo "[INFO] Start training..." | tee -a "${PIPE_LOG}"

set +e
if [ -n "${SATNAV_TRAIN_DATA_DIR:-}" ]; then
    SATNAV_DATASET="${SATNAV_DATASET}" SATNAV_TRAIN_DATA_DIR="${SATNAV_TRAIN_DATA_DIR}" GPUS_PER_NODE="${TRAIN_GPUS}" \
      bash "${BASELINE_DIR}/scripts/train_satnav.sh" "${MODE}" 2>&1 | tee -a "${PIPE_LOG}"
    train_rc=${PIPESTATUS[0]}
else
    SATNAV_DATASET="${SATNAV_DATASET}" GPUS_PER_NODE="${TRAIN_GPUS}" \
      bash "${BASELINE_DIR}/scripts/train_satnav.sh" "${MODE}" 2>&1 | tee -a "${PIPE_LOG}"
    train_rc=${PIPESTATUS[0]}
fi
set -e

TRAIN_END_TS="$(date +%s)"
TRAIN_DURATION_SEC=$((TRAIN_END_TS - TRAIN_START_TS))
TRAIN_STATUS="SUCCESS"
[ "${train_rc}" -ne 0 ] && TRAIN_STATUS="FAILED"

EXP_NAME=""
if ! EXP_NAME="$(extract_exp_name_from_log "${PIPE_LOG}")"; then
    EXP_NAME=""
fi
echo "[INFO] Post-train summary: train_rc=${train_rc}, exp_name=${EXP_NAME:-unknown}" | tee -a "${PIPE_LOG}"

EXP_SUBPATH="${EXP_NAME}"
if [ -n "${EXP_NAME}" ] && [ "${SMOKE_TEST}" = "true" ]; then
    EXP_SUBPATH="smoketest/${EXP_NAME}"
fi
echo "[INFO] Eval target subpath: ${EXP_SUBPATH:-unknown}" | tee -a "${PIPE_LOG}"

echo "[INFO] Train finished: status=${TRAIN_STATUS}, duration_sec=${TRAIN_DURATION_SEC}, exp_name=${EXP_NAME:-unknown}" | tee -a "${PIPE_LOG}"

if [ "${train_rc}" -ne 0 ]; then
    echo "[ERROR] Training failed (rc=${train_rc}). Skip eval." | tee -a "${PIPE_LOG}"
    exit "${train_rc}"
fi

if [ -z "${EXP_NAME}" ]; then
    echo "[ERROR] Training succeeded but EXP_NAME not found. Skip eval." | tee -a "${PIPE_LOG}"
    exit 2
fi

if [ "${CLEAN_EVAL_FIRST}" = "true" ]; then
    echo "[INFO] Cleaning previous eval dir: ${REPO_ROOT}/results/streamvln-baseline/${EXP_SUBPATH}" | tee -a "${PIPE_LOG}"
    rm -rf "${REPO_ROOT}/results/streamvln-baseline/${EXP_SUBPATH}"
fi

EVAL_START_TS="$(date +%s)"
echo "[INFO] Start eval by name..." | tee -a "${PIPE_LOG}"
echo "[INFO] Eval target: model_dir=${EVAL_MODEL_DIR}, model_name=${EXP_SUBPATH}, split=${CONFIG_SPLIT:-unknown}, data_path=${CONFIG_DATA_PATH:-unknown}, gpus=${EVAL_GPUS}, max_episodes=${EVAL_MAX_EPISODES:-full}" | tee -a "${PIPE_LOG}"
EVAL_CMD=(
    bash "${BASELINE_DIR}/scripts/eval_satnav.sh"
    --model_dir "${EVAL_MODEL_DIR}"
    --model_name "${EXP_SUBPATH}"
    --gpus "${EVAL_GPUS}"
)
if [ -n "${EVAL_MAX_EPISODES}" ]; then
    EVAL_CMD+=(--max_episodes "${EVAL_MAX_EPISODES}")
fi

set +e
"${EVAL_CMD[@]}" 2>&1 | tee -a "${PIPE_LOG}"
eval_rc=${PIPESTATUS[0]}
set -e

EVAL_END_TS="$(date +%s)"
EVAL_DURATION_SEC=$((EVAL_END_TS - EVAL_START_TS))
EVAL_STATUS="SUCCESS"
[ "${eval_rc}" -ne 0 ] && EVAL_STATUS="FAILED"

echo "[INFO] Eval finished: status=${EVAL_STATUS}, duration_sec=${EVAL_DURATION_SEC}, eval_root=results/streamvln-baseline/${EXP_SUBPATH}" | tee -a "${PIPE_LOG}"

if [ "${eval_rc}" -ne 0 ]; then
    echo "[ERROR] Eval failed (rc=${eval_rc})." | tee -a "${PIPE_LOG}"
    exit "${eval_rc}"
fi

echo "[OK] Pipeline finished successfully." | tee -a "${PIPE_LOG}"
echo "[OK] EXP_NAME=${EXP_NAME}" | tee -a "${PIPE_LOG}"
echo "[OK] PIPE_LOG=${PIPE_LOG}" | tee -a "${PIPE_LOG}"
