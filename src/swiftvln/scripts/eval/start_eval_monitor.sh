#!/usr/bin/env bash
set -euo pipefail

# Long-running eval monitor:
# - Processes current eval_todo in rounds (one round = consume current queue and exit).
# - After each round, checks 98/73/17 training activity + GPU usage.
# - If all three hosts are idle, stops monitoring.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
EVAL_QUEUE_SCRIPT="${SCRIPT_DIR}/eval_queue.sh"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
TODO_FILE="${EVAL_QUEUE_DIR}/eval_todo.txt"
mkdir -p "${EVAL_QUEUE_DIR}"
touch "${TODO_FILE}"

HOST_98="${HOST_98:-10.246.152.98}"
HOST_73="${HOST_73:-10.246.152.73}"
HOST_17="${HOST_17:-10.246.132.17}"
SSH_USER="${SSH_USER:-jiangjiajun}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-8}"
SSH_RELAX_HOSTKEY="${SSH_RELAX_HOSTKEY:-true}"
# true: SSH不可达时按idle处理，避免monitor卡死；false: 按busy处理，更保守。
SSH_FAILURE_AS_IDLE="${SSH_FAILURE_AS_IDLE:-true}"

CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
TODO_POLL_INTERVAL="${TODO_POLL_INTERVAL:-60}"
IDLE_GPU_UTIL_MAX="${IDLE_GPU_UTIL_MAX:-5}"
IDLE_GPU_MEM_MAX_MIB="${IDLE_GPU_MEM_MAX_MIB:-1024}"

# Heuristic process patterns indicating active training.
TRAIN_REGEX="${TRAIN_REGEX:-train_queue\\.sh|train_swiftvln|deepspeed|swift sft|swift pt|torchrun.*src/swiftvln|python.*src/swiftvln.*train}"

log() {
    echo "[eval-monitor] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

ssh_exec() {
    local host="$1"
    shift
    if [[ "$SSH_RELAX_HOSTKEY" == "true" ]]; then
        ssh -o BatchMode=yes \
            -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${SSH_USER}@${host}" "$@"
    else
        ssh -o BatchMode=yes \
            -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
            "${SSH_USER}@${host}" "$@"
    fi
}

is_gpu_busy_from_stats() {
    local stats="$1"
    if [[ "$stats" == "__SSH_UNREACHABLE__" ]]; then
        [[ "$SSH_FAILURE_AS_IDLE" != "true" ]]
        return
    fi

    if [[ -z "${stats// }" ]]; then
        return 1
    fi

    awk -F, -v util_max="$IDLE_GPU_UTIL_MAX" -v mem_max="$IDLE_GPU_MEM_MAX_MIB" '
        {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            if (($1 + 0) > util_max || ($2 + 0) > mem_max) {
                busy = 1;
            }
        }
        END { exit busy ? 0 : 1 }
    ' <<<"$stats"
}

has_local_training_proc() {
    pgrep -af -f "$TRAIN_REGEX" >/dev/null 2>&1
}

has_remote_training_proc() {
    local host="$1"
    ssh_exec "$host" \
        "pgrep -af -f '$TRAIN_REGEX' >/dev/null 2>&1"
}

has_remote_training_proc_in_container_17() {
    ssh_exec "$HOST_17" \
        "docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'streamvln-container' && \
         docker exec streamvln-container bash -lc \"pgrep -af -f '$TRAIN_REGEX' >/dev/null 2>&1\""
}

get_local_gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || true
}

get_remote_gpu_stats() {
    local host="$1"
    local out=""
    if ! out="$(ssh_exec "$host" \
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" 2>/dev/null)"; then
        echo "__SSH_UNREACHABLE__"
        return 0
    fi
    echo "$out"
}

check_host_idle() {
    local host_label="$1"
    local host_addr="$2"
    local train_active=false
    local gpu_busy=false
    local stats=""

    if [[ "$host_label" == "98" || "$host_label" == "73" ]]; then
        local current
        current="$(hostname)"
        if [[ "$current" == *"$host_label"* ]]; then
            if has_local_training_proc; then
                train_active=true
            fi
            stats="$(get_local_gpu_stats)"
        else
            if has_remote_training_proc "$host_addr"; then
                train_active=true
            fi
            stats="$(get_remote_gpu_stats "$host_addr")"
        fi
    else
        if has_remote_training_proc "$host_addr"; then
            train_active=true
        fi
        if has_remote_training_proc_in_container_17; then
            train_active=true
        fi
        stats="$(get_remote_gpu_stats "$host_addr")"
    fi

    if [[ "$stats" == "__SSH_UNREACHABLE__" ]]; then
        log "host=${host_label} ssh_unreachable=true policy(SSH_FAILURE_AS_IDLE)=${SSH_FAILURE_AS_IDLE}"
    fi

    if is_gpu_busy_from_stats "$stats"; then
        gpu_busy=true
    fi

    log "host=${host_label} train_active=${train_active} gpu_busy=${gpu_busy}"
    if [[ "$train_active" == "true" || "$gpu_busy" == "true" ]]; then
        return 1
    fi
    return 0
}

all_hosts_idle() {
    local idle_98=false
    local idle_73=false
    local idle_17=false

    if check_host_idle "98" "$HOST_98"; then idle_98=true; fi
    if check_host_idle "73" "$HOST_73"; then idle_73=true; fi
    if check_host_idle "17" "$HOST_17"; then idle_17=true; fi

    if [[ "$idle_98" == "true" && "$idle_73" == "true" && "$idle_17" == "true" ]]; then
        return 0
    fi
    return 1
}

run_eval_round() {
    AUTO_TODO=true \
    DYNAMIC_TODO=true \
    WAIT_FOR_NEW_TASKS=false \
    TODO_POLL_INTERVAL="${TODO_POLL_INTERVAL}" \
    bash "$EVAL_QUEUE_SCRIPT"
}

# 检查本机是否已经可以跑 eval：无训练进程 + GPU 空闲
local_host_ready() {
    if has_local_training_proc; then
        log "local host has active training processes, waiting..."
        return 1
    fi
    local stats
    stats="$(get_local_gpu_stats)"
    if is_gpu_busy_from_stats "$stats"; then
        log "local GPU is busy (util or memory above threshold), waiting..."
        return 1
    fi
    return 0
}

main() {
    log "start monitor: CHECK_INTERVAL=${CHECK_INTERVAL}s IDLE_GPU_UTIL_MAX=${IDLE_GPU_UTIL_MAX} IDLE_GPU_MEM_MAX_MIB=${IDLE_GPU_MEM_MAX_MIB} SSH_FAILURE_AS_IDLE=${SSH_FAILURE_AS_IDLE} SSH_RELAX_HOSTKEY=${SSH_RELAX_HOSTKEY}"
    while true; do
        if [[ -s "$TODO_FILE" ]]; then
            if local_host_ready; then
                log "todo detected and local GPU ready, starting eval round"
                run_eval_round
                log "eval round finished, checking cluster idle status"
                if all_hosts_idle; then
                    log "all hosts idle, stop monitor"
                    break
                fi
            fi
            # else: already logged "waiting..." inside local_host_ready
        fi
        sleep "$CHECK_INTERVAL"
    done
}

main "$@"
