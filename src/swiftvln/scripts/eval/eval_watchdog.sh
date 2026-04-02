#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Eval Watchdog — 异步监控 eval tmux 会话，完成/失败时回调 Codex
# ============================================================================
#
# 后台运行，监控指定 tmux session 中的评测进程。
# 每次运行创建独立的 per-run 目录，多服务器并发安全。
#
# 当评测完成或失败时：
#   1. 写入 per-run watchdog 状态文件
#   2. 发送 webhook 通知
#   3. 可选：通过 `codex exec resume` 回调 Codex 进行智能处理
#
# 用法:
#   nohup bash src/swiftvln/scripts/eval/eval_watchdog.sh [OPTIONS] &
#
# 选项:
#   --tmux-session NAME      要监控的 tmux session 名称（必需）
#   --codex-session UUID     Codex CLI session ID，用于完成后回调（可选）
#   --eval-log PATH          eval 日志路径，用于错误分析和进度检测（可选）
#   --check-interval SECS    轮询间隔秒数（默认 30）
#   --max-wait SECS          最大等待时间（默认 21600 = 6 小时）
#   --stall-threshold N      连续无进度 N 轮后告警（默认 20，即 ~10 分钟）
#   --webhook true|false     是否发 webhook 通知（默认 true）
#   --codex-model MODEL      codex exec resume 使用的模型（可选）
#   --cleanup-days N         启动时清理 N 天前的 run 目录和日志（默认 7，0=不清理）
#   --remote-host HOST       远程主机 IP（如 10.246.152.73），tmux 操作走 SSH，codex 回调在本机执行
#                            适用于：watchdog 在 98 上跑，监控 73 上的 tmux session
#
# 环境变量:
#   WEBHOOK_URL              webhook 地址（默认企业微信机器人）
#   EVAL_QUEUE_DIR           队列目录（默认 runtime/eval_queue）
#
# Per-run 目录结构:
#   runtime/eval_queue/runs/<hostname>_<session_name>/
#     ├── watchdog_result.json     watchdog 最终结果
#     ├── watchdog.log             watchdog 日志
#     ├── codex_response.txt       Codex 回调响应（如有）
#     └── eval_queue_status.json   eval_queue.sh 写入的完成状态（如设置了 EVAL_RUN_DIR）
#
# 示例:
#   # 仅监控 + webhook
#   nohup bash src/swiftvln/scripts/eval/eval_watchdog.sh \
#     --tmux-session eval_queue_153025 &
#
#   # 完整：监控 + Codex 回调
#   nohup bash src/swiftvln/scripts/eval/eval_watchdog.sh \
#     --tmux-session eval_queue_153025 \
#     --codex-session 019cfad1-71ee-7053-b767-08be324daa07 \
#     --eval-log logs/train_launch/eval_queue_153025.log &
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
TMUX_SESSION=""
CODEX_SESSION=""
EVAL_LOG=""
CHECK_INTERVAL=30
MAX_WAIT=21600
STALL_THRESHOLD=20
USE_WEBHOOK=true
CODEX_MODEL=""
CLEANUP_DAYS=7
CODEX_HOST=""   # 运行 codex CLI 的主机（默认本地；远程 eval 时设为安装了 codex 的服务器 IP）
CODEX_BIN_PATH="/mnt/data1/home/jiangjiajun/.nvm/versions/node/v24.13.0/bin/codex"
NODE_BIN_PATH="/mnt/data1/home/jiangjiajun/.nvm/versions/node/v24.13.0/bin/node"
REMOTE_HOST=""
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=87cd9c07-52f0-4cec-a7a8-9586a9dc68c8}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
RUNS_DIR="${EVAL_QUEUE_DIR}/runs"
HOSTNAME_SAFE="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tmux-session)    TMUX_SESSION="$2";    shift 2 ;;
        --codex-session)   CODEX_SESSION="$2";   shift 2 ;;
        --eval-log)        EVAL_LOG="$2";        shift 2 ;;
        --check-interval)  CHECK_INTERVAL="$2";  shift 2 ;;
        --max-wait)        MAX_WAIT="$2";        shift 2 ;;
        --stall-threshold) STALL_THRESHOLD="$2"; shift 2 ;;
        --webhook)         USE_WEBHOOK="$2";     shift 2 ;;
        --codex-model)     CODEX_MODEL="$2";     shift 2 ;;
        --cleanup-days)    CLEANUP_DAYS="$2";    shift 2 ;;
        --remote-host)     REMOTE_HOST="$2";     shift 2 ;;
        --codex-host)      CODEX_HOST="$2";      shift 2 ;;
        --codex-bin-path)  CODEX_BIN_PATH="$2";  shift 2 ;;
        --node-bin-path)   NODE_BIN_PATH="$2";   shift 2 ;;
        *) echo "[watchdog] Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TMUX_SESSION" ]]; then
    echo "[watchdog] ERROR: --tmux-session is required" >&2
    exit 1
fi

# ── Per-run directory setup ─────────────────────────────────────────────────
RUN_DIR="${RUNS_DIR}/${HOSTNAME_SAFE}_${TMUX_SESSION}"
mkdir -p "$RUN_DIR"

WATCHDOG_LOG="${RUN_DIR}/watchdog.log"
WATCHDOG_RESULT="${RUN_DIR}/watchdog_result.json"
CODEX_RESPONSE="${RUN_DIR}/codex_response.txt"

# Export EVAL_RUN_DIR so eval_queue.sh can also write status here
export EVAL_RUN_DIR="$RUN_DIR"

log() { echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WATCHDOG_LOG"; }

# ── Remote tmux helpers ──────────────────────────────────────────────────────
# 当设置了 REMOTE_HOST 时，tmux 操作走 SSH；否则本地执行
tmux_has_session() {
    if [[ -n "$REMOTE_HOST" ]]; then
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST" \
            "tmux has-session -t '$TMUX_SESSION'" 2>/dev/null
    else
        tmux has-session -t "$TMUX_SESSION" 2>/dev/null
    fi
}

tmux_capture_pane() {
    local lines="${1:-80}"
    if [[ -n "$REMOTE_HOST" ]]; then
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST" \
            "tmux capture-pane -pt '$TMUX_SESSION' -S -${lines}" 2>/dev/null || echo "(无法获取远程 tmux 输出)"
    else
        tmux capture-pane -pt "$TMUX_SESSION" -S -"$lines" 2>/dev/null || echo "(无法获取 tmux 输出)"
    fi
}

send_webhook() {
    local title="$1" body="$2"
    [[ "$USE_WEBHOOK" != "true" ]] && return 0
    local content="## ${title}\n${body}\nhost: ${HOSTNAME_SAFE}\ntime: $(date '+%Y-%m-%d %H:%M:%S')"
    curl -sS -m 8 -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"${content//$'\n'/\\n}\"}}" \
        >/dev/null 2>&1 || true
}

# ── Cleanup old runs ────────────────────────────────────────────────────────
cleanup_old_runs() {
    local days="$1"
    [[ "$days" -le 0 ]] && return 0
    local count=0

    # Clean old per-run directories
    if [[ -d "$RUNS_DIR" ]]; then
        while IFS= read -r -d '' dir; do
            rm -rf "$dir"
            ((count++))
        done < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$days" -print0 2>/dev/null)
    fi

    # Clean old watchdog logs (legacy flat location)
    local legacy_log_dir="${SWIFTVLN_ROOT}/logs/eval_watchdog"
    if [[ -d "$legacy_log_dir" ]]; then
        find "$legacy_log_dir" -name "watchdog_*.log" -mtime +"$days" -delete 2>/dev/null || true
        find "$legacy_log_dir" -name "codex_response_*.txt" -mtime +"$days" -delete 2>/dev/null || true
    fi

    # Clean old eval queue logs
    local queue_log_dir="${SWIFTVLN_ROOT}/logs"
    find "$queue_log_dir" -maxdepth 1 -name "eval_queue_*.log" -mtime +"$days" -delete 2>/dev/null || true

    if [[ $count -gt 0 ]]; then
        log "Cleaned up $count run directories older than ${days} days"
    fi
}

# ── Helper: 获取错误上下文 ──────────────────────────────────────────────────
get_error_context() {
    local lines=80
    if [[ -n "$EVAL_LOG" && -f "$EVAL_LOG" ]]; then
        tail -"$lines" "$EVAL_LOG" 2>/dev/null || echo "(无法读取日志)"
        return
    fi
    # fallback: tmux capture-pane (only if session still exists)
    if tmux_has_session; then
        tmux_capture_pane "$lines"
    else
        # session 已不在，尝试从最近的 eval_queue log 获取
        local log_candidate
        log_candidate=$(ls -t "${SWIFTVLN_ROOT}"/logs/eval_queue_*.log 2>/dev/null | head -1 || echo "")
        if [[ -n "$log_candidate" ]]; then
            tail -"$lines" "$log_candidate" 2>/dev/null || echo "(无法读取日志)"
        else
            echo "(未找到评测日志)"
        fi
    fi
}

# ── Helper: 从队列文件构建摘要 ──────────────────────────────────────────────
build_summary() {
    local done_count=0 failed_count=0
    local done_models="" failed_models=""

    if [[ -f "${EVAL_QUEUE_DIR}/eval_done.txt" ]]; then
        done_count=$(grep -c . "${EVAL_QUEUE_DIR}/eval_done.txt" 2>/dev/null || echo 0)
        done_models=$(tr '\n' ',' < "${EVAL_QUEUE_DIR}/eval_done.txt" 2>/dev/null | sed 's/,$//')
    fi
    if [[ -f "${EVAL_QUEUE_DIR}/eval_failed_todo.txt" ]]; then
        failed_count=$(grep -c . "${EVAL_QUEUE_DIR}/eval_failed_todo.txt" 2>/dev/null || echo 0)
        failed_models=$(tr '\n' ',' < "${EVAL_QUEUE_DIR}/eval_failed_todo.txt" 2>/dev/null | sed 's/,$//')
    fi

    echo "success_count=${done_count}"
    echo "fail_count=${failed_count}"
    echo "done_models=${done_models}"
    echo "failed_models=${failed_models}"
}

# ── Helper: 写入 watchdog 结果文件 ──────────────────────────────────────────
write_watchdog_result() {
    local outcome="$1" elapsed="$2"
    eval "$(build_summary)"

    cat > "$WATCHDOG_RESULT" <<EOF
{
    "watchdog_completed_at": "$(date -Iseconds)",
    "hostname": "${HOSTNAME_SAFE}",
    "tmux_session": "${TMUX_SESSION}",
    "outcome": "${outcome}",
    "elapsed_seconds": ${elapsed},
    "success_count": ${success_count},
    "fail_count": ${fail_count},
    "done_models": "${done_models}",
    "failed_models": "${failed_models}",
    "codex_session": "${CODEX_SESSION:-null}",
    "run_dir": "${RUN_DIR}",
    "eval_log": "${EVAL_LOG:-null}"
}
EOF
    log "Wrote watchdog result: $WATCHDOG_RESULT"
}

# ── Helper: 检测 eval_queue.sh 的完成状态文件 ───────────────────────────────
find_completion_status() {
    # 优先：per-run 目录里的文件
    if [[ -f "${RUN_DIR}/eval_queue_status.json" ]]; then
        local age=$(( $(date +%s) - $(stat -c %Y "${RUN_DIR}/eval_queue_status.json" 2>/dev/null || echo 0) ))
        if [[ $age -lt 300 ]]; then
            echo "${RUN_DIR}/eval_queue_status.json"
            return 0
        fi
    fi
    # 回退：per-host 全局文件
    local host_status="${EVAL_QUEUE_DIR}/eval_queue_last_run_${HOSTNAME_SAFE}.json"
    if [[ -f "$host_status" ]]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$host_status" 2>/dev/null || echo 0) ))
        if [[ $age -lt 300 ]]; then
            echo "$host_status"
            return 0
        fi
    fi
    return 1
}

# ── Helper: 解析 codex 可执行文件路径 ────────────────────────────────────────
_resolve_codex_bin() {
    command -v codex 2>/dev/null || \
    { [[ -x "$CODEX_BIN_PATH" ]] && echo "$CODEX_BIN_PATH"; } || \
    echo ""
}

# ── Helper: 触发 Codex 回调 ─────────────────────────────────────────────────
trigger_codex_resume() {
    local prompt="$1"
    if [[ -z "$CODEX_SESSION" ]]; then
        log "No Codex session ID, skipping resume callback"
        return 0
    fi

    local model_arg=""
    [[ -n "$CODEX_MODEL" ]] && model_arg="-m ${CODEX_MODEL}"

    if [[ -n "$CODEX_HOST" ]]; then
        # 远程执行：SSH 到 98，用绝对路径 node + codex 执行回调
        # SSH 非交互式 shell 没有 nvm PATH，必须用绝对路径
        log "Remote Codex resume via SSH to ${CODEX_HOST}: session=${CODEX_SESSION}"
        printf '%s' "$prompt" | \
            ssh -o BatchMode=yes -o ConnectTimeout=15 "$CODEX_HOST" \
            "cd ${SWIFTVLN_ROOT} && ${NODE_BIN_PATH} ${CODEX_BIN_PATH} exec resume --full-auto ${model_arg} -o ${CODEX_RESPONSE} ${CODEX_SESSION} -" \
            >> "$WATCHDOG_LOG" 2>&1 && {
            log "Remote Codex resume completed, response at: $CODEX_RESPONSE"
        } || {
            log "WARNING: remote codex exec resume exited $?"
        }
    else
        local codex_bin
        codex_bin=$(_resolve_codex_bin)
        if [[ -z "$codex_bin" ]]; then
            log "WARNING: codex CLI not found locally and no --codex-host set, skipping resume"
            return 1
        fi

        log "Triggering: codex exec resume --full-auto ${model_arg} ... ${CODEX_SESSION}"
        if "$codex_bin" exec resume --full-auto $model_arg -o "$CODEX_RESPONSE" "$CODEX_SESSION" - \
                <<< "$prompt" >> "$WATCHDOG_LOG" 2>&1; then
            log "Codex resume completed, response at: $CODEX_RESPONSE"
        else
            local rc=$?
            log "WARNING: codex exec resume exited with code $rc"
        fi
    fi
}

# ── Pre-flight ──────────────────────────────────────────────────────────────
if ! tmux_has_session; then
    echo "[watchdog] ERROR: tmux session '$TMUX_SESSION' does not exist at startup" >&2
    [[ -n "$REMOTE_HOST" ]] && echo "[watchdog]   (checked on remote host: $REMOTE_HOST)" >&2
    exit 1
fi

# Cleanup old artifacts at startup
cleanup_old_runs "$CLEANUP_DAYS"

log "=========================================="
log "Eval Watchdog started"
log "  hostname      = $HOSTNAME_SAFE"
log "  tmux_session  = $TMUX_SESSION"
log "  remote_host   = ${REMOTE_HOST:-(local)}"
log "  run_dir       = $RUN_DIR"
log "  codex_session = ${CODEX_SESSION:-(none)}"
log "  codex_host    = ${CODEX_HOST:-(local)}"
log "  codex_bin     = ${CODEX_BIN_PATH}"
log "  eval_log      = ${EVAL_LOG:-(auto-detect)}"
log "  check_interval= ${CHECK_INTERVAL}s"
log "  max_wait      = ${MAX_WAIT}s"
log "  stall_threshold= ${STALL_THRESHOLD} cycles"
log "  webhook       = $USE_WEBHOOK"
log "  cleanup_days  = $CLEANUP_DAYS"
log "  pid           = $$"
log "=========================================="

# ── Main monitoring loop ────────────────────────────────────────────────────
ELAPSED=0
STALL_CYCLES=0
LAST_PROGRESS=""
WARNED_STALL=false

while true; do
    sleep "$CHECK_INTERVAL"
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))

    # ── Safety timeout ──────────────────────────────────────────────────
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        log "Max wait (${MAX_WAIT}s) reached. Eval may still be running."
        write_watchdog_result "timeout" "$ELAPSED"
        send_webhook "Eval Watchdog Timeout" "tmux_session=${TMUX_SESSION}\nelapsed=$((ELAPSED/60))min\neval 可能仍在运行"
        if [[ -n "$CODEX_SESSION" ]]; then
            trigger_codex_resume "评测已运行超过 $((MAX_WAIT/3600)) 小时（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}），仍未结束。请检查评测进度，判断是否需要干预。"
        fi
        break
    fi

    # ── Check tmux session alive ────────────────────────────────────────
    if ! tmux_has_session; then
        log "tmux session '$TMUX_SESSION' ended after ${ELAPSED}s${REMOTE_HOST:+ (remote: $REMOTE_HOST)}"

        eval "$(build_summary)"

        # Determine: normal completion vs crash
        local_outcome="completed"
        if ! find_completion_status >/dev/null 2>&1; then
            if [[ $success_count -eq 0 && $fail_count -eq 0 ]]; then
                local_outcome="crash"
                log "No completion status and no queue outcomes — likely crashed"
            fi
        fi

        write_watchdog_result "$local_outcome" "$ELAPSED"

        if [[ "$local_outcome" == "crash" ]]; then
            # ── Crash / 异常退出 ──
            error_ctx=$(get_error_context)
            send_webhook "Eval Crashed" "host=${HOSTNAME_SAFE}\ntmux_session=${TMUX_SESSION}\nelapsed=$((ELAPSED/60))min\nsuccess=${success_count} failed=${fail_count}"
            if [[ -n "$CODEX_SESSION" ]]; then
                trigger_codex_resume "$(cat <<PROMPT
评测进程异常退出（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}，运行 $((ELAPSED/60)) 分钟后退出）。

队列状态: 成功=${success_count}, 失败=${fail_count}
成功模型: ${done_models:-无}
失败模型: ${failed_models:-无}
Run 目录: ${RUN_DIR}

最后 80 行输出:
\`\`\`
${error_ctx}
\`\`\`

请分析崩溃原因并修复。如果是可恢复错误，请重新启动评测。
PROMPT
)"
            fi

        elif [[ $fail_count -gt 0 ]]; then
            # ── 正常结束但有失败 ──
            error_ctx=$(get_error_context)
            send_webhook "Eval Done (with failures)" "host=${HOSTNAME_SAFE}\ntmux_session=${TMUX_SESSION}\nsuccess=${success_count}\nfailed=${fail_count}\nfailed_models=${failed_models}\nelapsed=$((ELAPSED/60))min"
            if [[ -n "$CODEX_SESSION" ]]; then
                trigger_codex_resume "$(cat <<PROMPT
评测队列已结束（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}），但存在失败模型。

成功: ${success_count} 个 (${done_models:-无})
失败: ${fail_count} 个 (${failed_models})
Run 目录: ${RUN_DIR}

最后 80 行日志:
\`\`\`
${error_ctx}
\`\`\`

请分析失败原因。如果可修复，请修复后将失败模型重新加入 runtime/eval_queue/eval_todo.txt 并重启评测。
PROMPT
)"
            fi

        else
            # ── 全部成功 ──
            send_webhook "Eval All Success" "host=${HOSTNAME_SAFE}\ntmux_session=${TMUX_SESSION}\nsuccess=${success_count}\nmodels=${done_models}\nelapsed=$((ELAPSED/60))min"
            if [[ -n "$CODEX_SESSION" ]]; then
                trigger_codex_resume "$(cat <<PROMPT
评测队列已全部成功完成（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}）。

成功: ${success_count} 个模型 (${done_models})
耗时: $((ELAPSED/60)) 分钟
Run 目录: ${RUN_DIR}

请汇总评测结果并向用户报告。如有需要，执行 CSV 结果收集。
PROMPT
)"
            fi
        fi

        break
    fi

    # ── Progress / stall detection ──────────────────────────────────────
    # eval log 是共享文件系统，直接本地读即可（无需 SSH）
    if [[ -n "$EVAL_LOG" && -f "$EVAL_LOG" ]]; then
        current_progress=$(grep -oP 'Rank 0.*\d+/\d+' "$EVAL_LOG" 2>/dev/null | tail -1 || echo "")
        if [[ -n "$current_progress" ]]; then
            if [[ "$current_progress" == "$LAST_PROGRESS" ]]; then
                STALL_CYCLES=$((STALL_CYCLES + 1))
                if [[ $STALL_CYCLES -ge $STALL_THRESHOLD && "$WARNED_STALL" != "true" ]]; then
                    log "WARNING: Progress stalled for $((STALL_CYCLES * CHECK_INTERVAL))s at: $current_progress"
                    send_webhook "Eval Stall Warning" "host=${HOSTNAME_SAFE}\ntmux_session=${TMUX_SESSION}\nstalled_at=${current_progress}\nstall_duration=$((STALL_CYCLES * CHECK_INTERVAL))s"
                    WARNED_STALL=true
                fi
            else
                STALL_CYCLES=0
                WARNED_STALL=false
                LAST_PROGRESS="$current_progress"
            fi
        fi
    fi

    # Periodic heartbeat (every 5 minutes)
    if (( ELAPSED % 300 == 0 )); then
        log "Heartbeat: tmux alive, elapsed=${ELAPSED}s ($((ELAPSED/60))min)"
    fi
done

log "Watchdog exiting (pid=$$, run_dir=${RUN_DIR})"
