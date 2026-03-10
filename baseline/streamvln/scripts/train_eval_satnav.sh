#!/bin/bash
# ==============================================================================
# One-click pipeline: train StreamVLN baseline on SatNav, then auto-eval.
#
# Usage:
#   bash baseline/streamvln/scripts/train_eval_satnav.sh [continue|scratch] [split]
#
# Defaults:
#   mode  = continue
#   split = val_unseen
#
# Env (optional):
#   SATNAV_VERSION   Data version, e.g. ver_260306 (default: latest via train/eval scripts)
#   TRAIN_GPUS       GPU count for training (default: 8)
#   EVAL_GPUS        GPU count for eval (default: 8)
#   CLEAN_EVAL_FIRST true/false, clear target eval output before run (default: true)
#   TRAIN_WEBHOOK_URL  Webhook for train start/end
#   EVAL_WEBHOOK_URL   Webhook for eval start/end
# ==============================================================================

set -euo pipefail

MODE="${1:-continue}"
SPLIT="${2:-val_unseen}"

if [ "$MODE" != "continue" ] && [ "$MODE" != "scratch" ]; then
    echo "[ERROR] Unknown mode: ${MODE}. Must be continue|scratch"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"

TRAIN_GPUS="${TRAIN_GPUS:-8}"
EVAL_GPUS="${EVAL_GPUS:-8}"
CLEAN_EVAL_FIRST="${CLEAN_EVAL_FIRST:-true}"

TRAIN_WEBHOOK_URL="${TRAIN_WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=503b5488-4d70-455d-a5b9-29fc8d7fb797}"
EVAL_WEBHOOK_URL="${EVAL_WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=87cd9c07-52f0-4cec-a7a8-9586a9dc68c8}"

PIPE_TS="$(date +%Y%m%d-%H%M%S)"
PIPE_LOG="/tmp/streamvln_baseline_train_eval_${PIPE_TS}.log"
HOSTNAME_STR="$(hostname)"

send_wecom_markdown() {
    local webhook_url="$1"
    local content="$2"
    curl -sS -X POST "$webhook_url" \
      -H "Content-Type: application/json" \
      -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"${content//$'\n'/\\n}\"}}" >/dev/null || true
}

extract_exp_name_from_log() {
    local log_path="$1"
    local exp_name
    exp_name="$(rg 'EXP_NAME' "$log_path" | tail -1 | sed -E 's/.*EXP_NAME[[:space:]]*:[[:space:]]*//')"
    if [ -n "$exp_name" ]; then
        echo "$exp_name"
        return 0
    fi
    exp_name="$(ls -1dt "${REPO_ROOT}"/output/streamvln-baseline/streamvln-baseline-"${MODE}"-* 2>/dev/null | head -1 | xargs -r basename)"
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
TRAIN_START_MSG="## StreamVLN Baseline Train Started
host: ${HOSTNAME_STR}
mode: ${MODE}
split_after_train: ${SPLIT}
train_gpus: ${TRAIN_GPUS}
satnav_version: ${SATNAV_VERSION:-auto}
time: $(date '+%Y-%m-%d %H:%M:%S')"
send_wecom_markdown "${TRAIN_WEBHOOK_URL}" "${TRAIN_START_MSG}"

echo "[INFO] Pipeline log: ${PIPE_LOG}" | tee -a "${PIPE_LOG}"
echo "[INFO] Start training..." | tee -a "${PIPE_LOG}"

set +e
if [ -n "${SATNAV_VERSION:-}" ]; then
    SATNAV_VERSION="${SATNAV_VERSION}" GPUS_PER_NODE="${TRAIN_GPUS}" \
      bash "${BASELINE_DIR}/scripts/train_satnav.sh" "${MODE}" 2>&1 | tee -a "${PIPE_LOG}"
    train_rc=${PIPESTATUS[0]}
else
    GPUS_PER_NODE="${TRAIN_GPUS}" \
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

TRAIN_END_MSG="## StreamVLN Baseline Train Finished
status: ${TRAIN_STATUS}
mode: ${MODE}
exp_name: ${EXP_NAME:-unknown}
duration_sec: ${TRAIN_DURATION_SEC}
time: $(date '+%Y-%m-%d %H:%M:%S')"
send_wecom_markdown "${TRAIN_WEBHOOK_URL}" "${TRAIN_END_MSG}"

if [ "${train_rc}" -ne 0 ]; then
    echo "[ERROR] Training failed (rc=${train_rc}). Skip eval." | tee -a "${PIPE_LOG}"
    exit "${train_rc}"
fi

if [ -z "${EXP_NAME}" ]; then
    echo "[ERROR] Training succeeded but EXP_NAME not found. Skip eval." | tee -a "${PIPE_LOG}"
    exit 2
fi

if [ "${CLEAN_EVAL_FIRST}" = "true" ]; then
    rm -rf "${REPO_ROOT}/results/streamvln-baseline/${EXP_NAME}/${SPLIT}"
fi

EVAL_START_TS="$(date +%s)"
EVAL_START_MSG="## StreamVLN Baseline Eval Started
host: ${HOSTNAME_STR}
exp_name: ${EXP_NAME}
split: ${SPLIT}
eval_gpus: ${EVAL_GPUS}
satnav_version: ${SATNAV_VERSION:-auto}
time: $(date '+%Y-%m-%d %H:%M:%S')"
send_wecom_markdown "${EVAL_WEBHOOK_URL}" "${EVAL_START_MSG}"

echo "[INFO] Start eval..." | tee -a "${PIPE_LOG}"
set +e
if [ -n "${SATNAV_VERSION:-}" ]; then
    SATNAV_VERSION="${SATNAV_VERSION}" \
      bash "${BASELINE_DIR}/scripts/eval_satnav.sh" "${EXP_NAME}" "${SPLIT}" "${EVAL_GPUS}" 2>&1 | tee -a "${PIPE_LOG}"
    eval_rc=${PIPESTATUS[0]}
else
    bash "${BASELINE_DIR}/scripts/eval_satnav.sh" "${EXP_NAME}" "${SPLIT}" "${EVAL_GPUS}" 2>&1 | tee -a "${PIPE_LOG}"
    eval_rc=${PIPESTATUS[0]}
fi
set -e

EVAL_END_TS="$(date +%s)"
EVAL_DURATION_SEC=$((EVAL_END_TS - EVAL_START_TS))
EVAL_STATUS="SUCCESS"
[ "${eval_rc}" -ne 0 ] && EVAL_STATUS="FAILED"

EVAL_END_MSG="## StreamVLN Baseline Eval Finished
status: ${EVAL_STATUS}
exp_name: ${EXP_NAME}
split: ${SPLIT}
duration_sec: ${EVAL_DURATION_SEC}
eval_log: results/streamvln-baseline/${EXP_NAME}/${SPLIT}/eval.log
time: $(date '+%Y-%m-%d %H:%M:%S')"
send_wecom_markdown "${EVAL_WEBHOOK_URL}" "${EVAL_END_MSG}"

if [ "${eval_rc}" -ne 0 ]; then
    echo "[ERROR] Eval failed (rc=${eval_rc})." | tee -a "${PIPE_LOG}"
    exit "${eval_rc}"
fi

echo "[OK] Pipeline finished successfully." | tee -a "${PIPE_LOG}"
echo "[OK] EXP_NAME=${EXP_NAME}" | tee -a "${PIPE_LOG}"
echo "[OK] PIPE_LOG=${PIPE_LOG}" | tee -a "${PIPE_LOG}"
