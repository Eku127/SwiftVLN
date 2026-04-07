#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

HOST_LABEL="$(hostname)"
CHECK_INTERVAL=30
TEMP_THRESHOLD=85
ALERT_COOLDOWN=600
USE_WEBHOOK=true
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=48e434da-fb2d-453c-a180-c4041b4c7f1e}"
EXPECTED_GPU_COUNT=""
STATE_ROOT="${SWIFTVLN_ROOT}/runtime/gpu_health_monitor"
ONESHOT=false
SEND_STARTUP_WEBHOOK=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-label) HOST_LABEL="$2"; shift 2 ;;
        --expected-gpus) EXPECTED_GPU_COUNT="$2"; shift 2 ;;
        --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
        --temp-threshold) TEMP_THRESHOLD="$2"; shift 2 ;;
        --alert-cooldown) ALERT_COOLDOWN="$2"; shift 2 ;;
        --webhook) USE_WEBHOOK="$2"; shift 2 ;;
        --webhook-url) WEBHOOK_URL="$2"; shift 2 ;;
        --state-root) STATE_ROOT="$2"; shift 2 ;;
        --oneshot) ONESHOT="$2"; shift 2 ;;
        --startup-webhook) SEND_STARTUP_WEBHOOK="$2"; shift 2 ;;
        *)
            echo "[gpu-health] Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

RUN_DIR="${STATE_ROOT}/${HOST_LABEL}"
mkdir -p "$RUN_DIR"
LOG_FILE="${RUN_DIR}/monitor.log"
STATE_FILE="${RUN_DIR}/state.env"
UUIDS_FILE="${RUN_DIR}/expected_uuids.txt"

log() {
    echo "[gpu-health] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"
}

send_webhook() {
    local title="$1"
    local body="$2"
    [[ "$USE_WEBHOOK" != "true" ]] && return 0
    local content="${title}\n${body}\ntime: $(date '+%Y-%m-%d %H:%M:%S')"
    curl -sS -m 8 -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"${content//$'\n'/\\n}\"}}" \
        >/dev/null 2>&1 || true
}

count_gpus() {
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
}

query_gpu_rows() {
    nvidia-smi \
        --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null
}

init_baseline() {
    local current_count
    current_count="$(count_gpus)"
    if [[ -z "$EXPECTED_GPU_COUNT" ]]; then
        EXPECTED_GPU_COUNT="$current_count"
    fi
    if [[ "$current_count" -lt 1 ]]; then
        log "ERROR: no GPUs visible at startup"
        exit 1
    fi
    query_gpu_rows | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' > "$UUIDS_FILE"
    log "Started host=${HOST_LABEL} expected_gpus=${EXPECTED_GPU_COUNT} temp_threshold=${TEMP_THRESHOLD}C cooldown=${ALERT_COOLDOWN}s"
    log "Expected UUIDs: $(paste -sd, "$UUIDS_FILE")"
    if [[ "$SEND_STARTUP_WEBHOOK" == "true" ]]; then
        send_webhook \
            "[GPU Monitor Started] ${HOST_LABEL}" \
            "host=${HOST_LABEL}\nexpected_gpus=${EXPECTED_GPU_COUNT}\ntemp_threshold=${TEMP_THRESHOLD}C\nlog=${LOG_FILE}"
    fi
}

load_state() {
    LAST_STATUS="healthy"
    LAST_ALERT_TS=0
    LAST_ALERT_KEY=""
    if [[ -f "$STATE_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$STATE_FILE"
    fi
}

save_state() {
    cat > "$STATE_FILE" <<EOF
LAST_STATUS="${LAST_STATUS}"
LAST_ALERT_TS=${LAST_ALERT_TS}
LAST_ALERT_KEY="${LAST_ALERT_KEY}"
EOF
}

evaluate_health() {
    CURRENT_STATUS="healthy"
    CURRENT_KEY="healthy"
    CURRENT_DETAILS=""

    local rows
    if ! rows="$(query_gpu_rows)"; then
        CURRENT_STATUS="alert"
        CURRENT_KEY="nvidia_smi_failed"
        CURRENT_DETAILS="nvidia-smi query failed"
        return 0
    fi

    local current_count
    current_count="$(echo "$rows" | sed '/^\s*$/d' | wc -l | awk '{print $1}')"
    if [[ "$current_count" != "$EXPECTED_GPU_COUNT" ]]; then
        CURRENT_STATUS="alert"
        CURRENT_KEY="gpu_count_mismatch"
        CURRENT_DETAILS="expected_gpus=${EXPECTED_GPU_COUNT}, current_gpus=${current_count}"
    fi

    local missing_uuids=()
    while IFS= read -r expected_uuid; do
        [[ -z "$expected_uuid" ]] && continue
        if ! echo "$rows" | grep -Fq "$expected_uuid"; then
            missing_uuids+=("$expected_uuid")
        fi
    done < "$UUIDS_FILE"

    if [[ ${#missing_uuids[@]} -gt 0 ]]; then
        CURRENT_STATUS="alert"
        CURRENT_KEY="gpu_uuid_missing"
        CURRENT_DETAILS="${CURRENT_DETAILS:+${CURRENT_DETAILS}; }missing_uuids=$(IFS=,; echo "${missing_uuids[*]}")"
    fi

    local hot_entries=()
    local summary_entries=()
    while IFS=',' read -r gpu_index gpu_uuid gpu_name gpu_temp gpu_util mem_used mem_total; do
        gpu_index="$(echo "$gpu_index" | xargs)"
        gpu_temp="$(echo "$gpu_temp" | xargs)"
        gpu_util="$(echo "$gpu_util" | xargs)"
        mem_used="$(echo "$mem_used" | xargs)"
        mem_total="$(echo "$mem_total" | xargs)"
        [[ -z "$gpu_index" ]] && continue

        summary_entries+=("gpu${gpu_index}:temp=${gpu_temp}C,util=${gpu_util}%,mem=${mem_used}/${mem_total}MiB")

        if [[ "$gpu_temp" == "N/A" || "$gpu_temp" == "[Not Supported]" || -z "$gpu_temp" ]]; then
            CURRENT_STATUS="alert"
            CURRENT_KEY="gpu_temp_unavailable"
            CURRENT_DETAILS="${CURRENT_DETAILS:+${CURRENT_DETAILS}; }gpu${gpu_index}_temp=${gpu_temp}"
            continue
        fi

        if (( gpu_temp >= TEMP_THRESHOLD )); then
            hot_entries+=("gpu${gpu_index}=${gpu_temp}C")
        fi
    done <<< "$rows"

    LAST_SUMMARY="$(IFS=' | '; echo "${summary_entries[*]}")"

    if [[ ${#hot_entries[@]} -gt 0 ]]; then
        CURRENT_STATUS="alert"
        CURRENT_KEY="gpu_temp_high"
        CURRENT_DETAILS="${CURRENT_DETAILS:+${CURRENT_DETAILS}; }hot=$(IFS=,; echo "${hot_entries[*]}")"
    fi

    if [[ "$CURRENT_STATUS" == "healthy" ]]; then
        CURRENT_DETAILS="all_ok; ${LAST_SUMMARY}"
    else
        CURRENT_DETAILS="${CURRENT_DETAILS}; ${LAST_SUMMARY}"
    fi
}

maybe_send_alert() {
    local now_ts
    now_ts="$(date +%s)"

    if [[ "$CURRENT_STATUS" == "healthy" ]]; then
        if [[ "$LAST_STATUS" != "healthy" ]]; then
            send_webhook \
                "[GPU Monitor Recovered] ${HOST_LABEL}" \
                "host=${HOST_LABEL}\nstatus=healthy\ndetails=${CURRENT_DETAILS}"
        fi
        LAST_STATUS="healthy"
        LAST_ALERT_KEY="healthy"
        LAST_ALERT_TS=0
        return 0
    fi

    if [[ "$CURRENT_KEY" != "$LAST_ALERT_KEY" || $((now_ts - LAST_ALERT_TS)) -ge $ALERT_COOLDOWN ]]; then
        send_webhook \
            "[GPU Monitor Alert] ${HOST_LABEL}" \
            "host=${HOST_LABEL}\nkey=${CURRENT_KEY}\ndetails=${CURRENT_DETAILS}"
        LAST_ALERT_TS="$now_ts"
        LAST_ALERT_KEY="$CURRENT_KEY"
    fi
    LAST_STATUS="alert"
}

init_baseline
load_state

while true; do
    evaluate_health
    log "status=${CURRENT_STATUS} key=${CURRENT_KEY} details=${CURRENT_DETAILS}"
    maybe_send_alert
    save_state
    [[ "$ONESHOT" == "true" ]] && exit 0
    sleep "$CHECK_INTERVAL"
done
