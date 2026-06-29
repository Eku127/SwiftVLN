#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
MONITOR_SCRIPT="${SWIFTVLN_ROOT}/src/swiftvln/scripts/monitor/gpu_health_monitor.sh"

SESSION_98="gpu_health_98"
SESSION_73="gpu_health_73"
CHECK_INTERVAL=30
TEMP_THRESHOLD=85
ALERT_COOLDOWN=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
        --temp-threshold) TEMP_THRESHOLD="$2"; shift 2 ;;
        --alert-cooldown) ALERT_COOLDOWN="$2"; shift 2 ;;
        *)
            echo "[gpu-health-start] Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "$MONITOR_SCRIPT" ]]; then
    echo "[gpu-health-start] monitor script not executable: $MONITOR_SCRIPT" >&2
    exit 1
fi

start_local_session() {
    local session_name="$1"
    local host_label="$2"
    local expected_gpus="$3"
    tmux has-session -t "$session_name" 2>/dev/null && tmux kill-session -t "$session_name"
    tmux new-session -d -s "$session_name" \
        "bash '$MONITOR_SCRIPT' \
            --host-label '$host_label' \
            --expected-gpus '$expected_gpus' \
            --check-interval '$CHECK_INTERVAL' \
            --temp-threshold '$TEMP_THRESHOLD' \
            --alert-cooldown '$ALERT_COOLDOWN'"
}

start_remote_session() {
    local session_name="$1"
    local host_label="$2"
    local expected_gpus="$3"
    ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73 "\
        tmux has-session -t '$session_name' 2>/dev/null && tmux kill-session -t '$session_name' || true; \
        tmux new-session -d -s '$session_name' \
            \"bash '$MONITOR_SCRIPT' \
                --host-label '$host_label' \
                --expected-gpus '$expected_gpus' \
                --check-interval '$CHECK_INTERVAL' \
                --temp-threshold '$TEMP_THRESHOLD' \
                --alert-cooldown '$ALERT_COOLDOWN'\""
}

start_local_session "$SESSION_98" "98" "8"
start_remote_session "$SESSION_73" "73" "7"

echo "LOCAL_SESSION=${SESSION_98}"
echo "REMOTE_SESSION=${SESSION_73}"
echo "LOG_98=${SWIFTVLN_ROOT}/runtime/gpu_health_monitor/98/monitor.log"
echo "LOG_73=${SWIFTVLN_ROOT}/runtime/gpu_health_monitor/73/monitor.log"
