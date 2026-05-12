#!/usr/bin/env bash
set -euo pipefail

STATUS="${1:-SUCCESS}"
EXP_NAME="${2:-swiftvln-train}"
DURATION="${3:-N/A}"
SUCCESS_RATE="${4:-N/A}"
OUTPUT_DIR="${5:-N/A}"
LOG_PATH="${6:-N/A}"
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=503b5488-4d70-455d-a5b9-29fc8d7fb797}"
NOW="$(date '+%Y-%m-%d %H:%M:%S')"

CONTENT="## VLN Training Notification
status: ${STATUS}
exp_name: ${EXP_NAME}
duration: ${DURATION}
success_rate: ${SUCCESS_RATE}
output_dir: ${OUTPUT_DIR}
log: ${LOG_PATH}
time: ${NOW}"

curl -sS -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"${CONTENT//$'\n'/\\n}\"}}"
