#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Train Watchdog — 事件驱动监控训练 tmux 会话
# ============================================================================
#
# 后台运行，监控指定 tmux session 中的训练进程。
# 事件驱动（非 Codex 轮询），仅在关键时刻回调 Codex，最小化 token 消耗。
#
# 触发事件:
#   1. 单个实验完成 → resume Codex 检查队列衔接（可选）
#   2. 实验失败且脚本内建修复失败 → resume Codex 做智能修复
#   3. 全部完成 → enqueue eval + 启动新 Codex session 按 eval skill 执行
#   4. 进程崩溃 → resume Codex 诊断并恢复
#   5. 进度停滞 → webhook 告警
#
# 用法:
#   nohup bash src/swiftvln/scripts/train/train_watchdog.sh [OPTIONS] &
#
# 选项:
#   --tmux-session NAME      要监控的 tmux session 名称（必需）
#   --codex-session UUID     Codex CLI session ID，用于错误/衔接回调（可选）
#   --train-log PATH         训练日志路径（可选，用于事件检测）
#   --check-interval SECS    轮询间隔（默认 30）
#   --max-wait SECS          最大等待时间（默认 86400 = 24 小时）
#   --stall-threshold N      连续无进度 N 轮后告警（默认 40，即 ~20 分钟）
#   --on-experiment-fail resume|skip  单实验失败时是否 resume Codex（默认 resume）
#   --on-all-done eval|notify        全部完成时触发 eval 还是仅通知（默认 eval）
#   --webhook true|false     是否发 webhook（默认 true）
#   --codex-model MODEL      codex exec resume/exec 使用的模型（可选）
#   --cleanup-days N         启动时清理 N 天前的 run 目录（默认 7）
#   --remote-host HOST       远程主机 IP（如 10.246.152.73），tmux 操作走 SSH，codex 回调在本机执行
#                            适用于：watchdog 在 98 上跑，监控 73 上的 tmux session
#
# 环境变量:
#   WEBHOOK_URL              webhook 地址
#   TRAIN_QUEUE_DIR          训练队列目录（默认 runtime/train_queue）
#
# Per-run 目录:
#   runtime/train_queue/runs/<hostname>_<session_name>/
#     ├── watchdog_result.json
#     ├── watchdog.log
#     ├── codex_response_*.txt
#     ├── train_queue_status.json   (train_queue.sh 写入)
#     └── train_events.log          (train_queue.sh 写入)
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
TMUX_SESSION=""
CODEX_SESSION=""
TRAIN_LOG=""
CHECK_INTERVAL=30
MAX_WAIT=86400
STALL_THRESHOLD=40
ON_EXPERIMENT_FAIL="resume"
ON_ALL_DONE="eval"
USE_WEBHOOK=true
CODEX_MODEL=""
CLEANUP_DAYS=7
REMOTE_HOST=""
CODEX_HOST=""   # 运行 codex CLI 的主机（默认本地；远程训练时设为安装了 codex 的服务器 IP）
CODEX_BIN_PATH="/mnt/data1/home/jiangjiajun/.nvm/versions/node/v24.13.0/bin/codex"
NODE_BIN_PATH="/mnt/data1/home/jiangjiajun/.nvm/versions/node/v24.13.0/bin/node"
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=503b5488-4d70-455d-a5b9-29fc8d7fb797}"
TRAIN_QUEUE_DIR="${TRAIN_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/train_queue}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
RUNS_DIR="${TRAIN_QUEUE_DIR}/runs"
HOSTNAME_SAFE="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tmux-session)      TMUX_SESSION="$2";      shift 2 ;;
        --codex-session)     CODEX_SESSION="$2";     shift 2 ;;
        --train-log)         TRAIN_LOG="$2";         shift 2 ;;
        --check-interval)    CHECK_INTERVAL="$2";    shift 2 ;;
        --max-wait)          MAX_WAIT="$2";          shift 2 ;;
        --stall-threshold)   STALL_THRESHOLD="$2";   shift 2 ;;
        --on-experiment-fail) ON_EXPERIMENT_FAIL="$2"; shift 2 ;;
        --on-all-done)       ON_ALL_DONE="$2";       shift 2 ;;
        --webhook)           USE_WEBHOOK="$2";       shift 2 ;;
        --codex-model)       CODEX_MODEL="$2";       shift 2 ;;
        --cleanup-days)      CLEANUP_DAYS="$2";      shift 2 ;;
        --remote-host)       REMOTE_HOST="$2";       shift 2 ;;
        --codex-host)        CODEX_HOST="$2";        shift 2 ;;
        --codex-bin-path)    CODEX_BIN_PATH="$2";    shift 2 ;;
        --node-bin-path)     NODE_BIN_PATH="$2";     shift 2 ;;
        *) echo "[train-watchdog] Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TMUX_SESSION" ]]; then
    echo "[train-watchdog] ERROR: --tmux-session is required" >&2
    exit 1
fi

# ── Per-run directory ───────────────────────────────────────────────────────
RUN_DIR="${RUNS_DIR}/${HOSTNAME_SAFE}_${TMUX_SESSION}"
mkdir -p "$RUN_DIR"

WATCHDOG_LOG="${RUN_DIR}/watchdog.log"
WATCHDOG_RESULT="${RUN_DIR}/watchdog_result.json"
EVENTS_FILE="${RUN_DIR}/train_events.log"
touch "$EVENTS_FILE"

export TRAIN_RUN_DIR="$RUN_DIR"
export TRAIN_EVENTS_FILE="$EVENTS_FILE"

log() { echo "[train-watchdog] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WATCHDOG_LOG"; }

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

# ── Cleanup ─────────────────────────────────────────────────────────────────
cleanup_old_runs() {
    local days="$1"
    [[ "$days" -le 0 ]] && return 0
    local count=0
    if [[ -d "$RUNS_DIR" ]]; then
        while IFS= read -r -d '' dir; do
            rm -rf "$dir"
            ((count += 1))
        done < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$days" -print0 2>/dev/null)
    fi
    # Legacy flat logs
    find "${SWIFTVLN_ROOT}/logs" -maxdepth 1 -name "train_queue_*.log" -mtime +"$days" -delete 2>/dev/null || true
    if [[ $count -gt 0 ]]; then
        log "Cleaned $count old run dirs (>${days} days)"
    fi
    return 0
}

# ── Helpers ─────────────────────────────────────────────────────────────────
get_error_context() {
    local lines=80
    if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
        tail -"$lines" "$TRAIN_LOG" 2>/dev/null
        return
    fi
    if tmux_has_session; then
        tmux_capture_pane "$lines"
    else
        local cand
        cand=$(ls -t "${SWIFTVLN_ROOT}"/logs/train_queue_*.log 2>/dev/null | head -1 || echo "")
        [[ -n "$cand" ]] && tail -"$lines" "$cand" 2>/dev/null || echo "(未找到训练日志)"
    fi
}

read_completion_status() {
    local status_file="${RUN_DIR}/train_queue_status.json"
    [[ -f "$status_file" ]] || status_file="${TRAIN_QUEUE_DIR}/train_queue_last_run_${HOSTNAME_SAFE}.json"
    if [[ -f "$status_file" ]]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$status_file" 2>/dev/null || echo 0) ))
        if [[ $age -lt 300 ]]; then
            cat "$status_file"
            return 0
        fi
    fi
    return 1
}

# Parse the events file for the latest experiment status
read_latest_event() {
    local event_type="$1"
    grep "^${event_type}|" "$EVENTS_FILE" 2>/dev/null | tail -1 || echo ""
}

count_events() {
    local event_type="$1"
    grep -c "^${event_type}|" "$EVENTS_FILE" 2>/dev/null || echo 0
}

write_watchdog_result() {
    local outcome="$1" elapsed="$2"
    local success_n=$(count_events "EXPERIMENT_SUCCESS")
    local fail_n=$(count_events "EXPERIMENT_FAILED")
    local total=$((success_n + fail_n))

    cat > "$WATCHDOG_RESULT" <<EOF
{
    "watchdog_completed_at": "$(date -Iseconds)",
    "hostname": "${HOSTNAME_SAFE}",
    "tmux_session": "${TMUX_SESSION}",
    "outcome": "${outcome}",
    "elapsed_seconds": ${elapsed},
    "success_count": ${success_n},
    "fail_count": ${fail_n},
    "total_experiments": ${total},
    "codex_session": "${CODEX_SESSION:-null}",
    "run_dir": "${RUN_DIR}"
}
EOF
    log "Wrote result: $WATCHDOG_RESULT"
}

_resolve_codex_bin() {
    # 优先 PATH 中查找，其次用绝对路径
    command -v codex 2>/dev/null || \
    { [[ -x "$CODEX_BIN_PATH" ]] && echo "$CODEX_BIN_PATH"; } || \
    echo ""
}

trigger_codex_resume() {
    local prompt="$1"
    [[ -z "$CODEX_SESSION" ]] && { log "No Codex session, skip resume"; return 0; }

    local resp_file="${RUN_DIR}/codex_response_$(date +%Y%m%d_%H%M%S).txt"
    local model_arg=""
    [[ -n "$CODEX_MODEL" ]] && model_arg="-m ${CODEX_MODEL}"

    if [[ -n "$CODEX_HOST" ]]; then
        # 远程执行：SSH 到 98，用绝对路径 node + codex 执行回调
        # SSH 非交互式 shell 没有 nvm PATH，必须用绝对路径
        log "Remote Codex resume via SSH to ${CODEX_HOST}: session=${CODEX_SESSION}"
        printf '%s' "$prompt" | \
            ssh -o BatchMode=yes -o ConnectTimeout=15 "$CODEX_HOST" \
            "cd ${SWIFTVLN_ROOT} && ${NODE_BIN_PATH} ${CODEX_BIN_PATH} exec resume --full-auto ${model_arg} -o ${resp_file} ${CODEX_SESSION} -" \
            >> "$WATCHDOG_LOG" 2>&1 || {
            log "WARNING: remote codex exec resume exited $?"
        }
    else
        local codex_bin
        codex_bin=$(_resolve_codex_bin)
        [[ -z "$codex_bin" ]] && { log "WARNING: codex CLI not found locally and no --codex-host set"; return 1; }

        log "Resume Codex: session=$CODEX_SESSION"
        "$codex_bin" exec resume --full-auto $model_arg -o "$resp_file" "$CODEX_SESSION" - \
            <<< "$prompt" >> "$WATCHDOG_LOG" 2>&1 || {
            log "WARNING: codex exec resume exited $?"
        }
    fi
}

trigger_codex_new_session() {
    local prompt="$1"
    local resp_file="${RUN_DIR}/codex_eval_trigger_$(date +%Y%m%d_%H%M%S).txt"
    local model_arg=""
    [[ -n "$CODEX_MODEL" ]] && model_arg="-m ${CODEX_MODEL}"

    if [[ -n "$CODEX_HOST" ]]; then
        log "Remote new Codex session via SSH to ${CODEX_HOST}"
        printf '%s' "$prompt" | \
            ssh -o BatchMode=yes -o ConnectTimeout=15 "$CODEX_HOST" \
            "cd ${SWIFTVLN_ROOT} && ${NODE_BIN_PATH} ${CODEX_BIN_PATH} exec --full-auto ${model_arg} -o ${resp_file} -" \
            >> "$WATCHDOG_LOG" 2>&1 || {
            log "WARNING: remote codex exec new session exited $?"
        }
    else
        local codex_bin
        codex_bin=$(_resolve_codex_bin)
        [[ -z "$codex_bin" ]] && { log "WARNING: codex CLI not found locally and no --codex-host set"; return 1; }

        log "Starting new Codex session for eval"
        "$codex_bin" exec --full-auto $model_arg -o "$resp_file" - \
            <<< "$prompt" >> "$WATCHDOG_LOG" 2>&1 || {
            log "WARNING: codex exec for eval exited $?"
        }
    fi
}

# ── Event detection from events file ────────────────────────────────────────
LAST_SEEN_LINE=0

check_new_events() {
    [[ ! -f "$EVENTS_FILE" ]] && return
    local total_lines
    total_lines=$(wc -l < "$EVENTS_FILE" 2>/dev/null || echo 0)
    [[ $total_lines -le $LAST_SEEN_LINE ]] && return

    # Read new lines
    local new_lines
    new_lines=$(tail -n +"$((LAST_SEEN_LINE + 1))" "$EVENTS_FILE")
    LAST_SEEN_LINE=$total_lines

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local event_type="${line%%|*}"

        case "$event_type" in
            EXPERIMENT_FAILED)
                # Format: EXPERIMENT_FAILED|idx|total|model|exp_name|error_summary|log_path|timestamp
                IFS='|' read -r _ idx total model exp_name error_summary log_path ts <<< "$line"
                log "EVENT: experiment ${idx}/${total} FAILED: ${model} (${exp_name})"

                if [[ "$ON_EXPERIMENT_FAIL" == "resume" && -n "$CODEX_SESSION" ]]; then
                    local err_ctx=""
                    [[ -n "$log_path" && -f "$log_path" ]] && err_ctx=$(tail -60 "$log_path" 2>/dev/null)
                    trigger_codex_resume "$(cat <<PROMPT
训练实验 ${idx}/${total} 失败（host: ${HOSTNAME_SAFE}，tmux: ${TMUX_SESSION}）。

模型: ${model}
实验名: ${exp_name}
错误摘要: ${error_summary}
日志: ${log_path}
Run 目录: ${RUN_DIR}

最后 60 行日志:
\`\`\`
${err_ctx}
\`\`\`

train_queue.sh 已跳过此实验并继续队列。请分析错误原因。
如果是可修复问题（配置/代码），请修复并记录，后续可重新入队。
如果是资源问题（OOM/NCCL），请记录建议。
PROMPT
)"
                fi
                ;;

            EXPERIMENT_SUCCESS)
                IFS='|' read -r _ idx total model exp_name output_path ts <<< "$line"
                log "EVENT: experiment ${idx}/${total} SUCCESS: ${model} (${exp_name})"
                ;;

            QUEUE_DONE)
                # Handled separately in main loop when tmux exits
                log "EVENT: QUEUE_DONE detected"
                ;;
        esac
    done <<< "$new_lines"
}

# ── Pre-flight ──────────────────────────────────────────────────────────────
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "[train-watchdog] ERROR: tmux session '$TMUX_SESSION' not found" >&2
    exit 1
fi

cleanup_old_runs "$CLEANUP_DAYS"

log "=========================================="
log "Train Watchdog started"
log "  hostname        = $HOSTNAME_SAFE"
log "  tmux_session    = $TMUX_SESSION"
log "  run_dir         = $RUN_DIR"
log "  codex_session   = ${CODEX_SESSION:-(none)}"
log "  codex_host      = ${CODEX_HOST:-(local)}"
log "  codex_bin_path  = ${CODEX_BIN_PATH}"
log "  train_log       = ${TRAIN_LOG:-(auto)}"
log "  check_interval  = ${CHECK_INTERVAL}s"
log "  max_wait        = ${MAX_WAIT}s (${MAX_WAIT_H:=$((MAX_WAIT/3600))}h)"
log "  stall_threshold = ${STALL_THRESHOLD} cycles"
log "  on_exp_fail     = $ON_EXPERIMENT_FAIL"
log "  on_all_done     = $ON_ALL_DONE"
log "  cleanup_days    = $CLEANUP_DAYS"
log "  pid             = $$"
log "=========================================="

# ── Main loop ───────────────────────────────────────────────────────────────
ELAPSED=0
STALL_CYCLES=0
LAST_PROGRESS=""
WARNED_STALL=false

while true; do
    sleep "$CHECK_INTERVAL"
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))

    # ── Safety timeout ──────────────────────────────────────────────────
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        log "Max wait ${MAX_WAIT}s reached"
        write_watchdog_result "timeout" "$ELAPSED"
        send_webhook "Train Watchdog Timeout" "tmux=${TMUX_SESSION}\nhost=${HOSTNAME_SAFE}\nelapsed=$((ELAPSED/3600))h"
        if [[ -n "$CODEX_SESSION" ]]; then
            trigger_codex_resume "训练已运行超过 $((MAX_WAIT/3600)) 小时（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}），仍未结束。请检查训练进度，判断是否需要干预。"
        fi
        break
    fi

    # ── Process event file for mid-flight events ────────────────────────
    check_new_events

    # ── Check tmux session alive ────────────────────────────────────────
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "tmux session '$TMUX_SESSION' ended after ${ELAPSED}s ($((ELAPSED/60))min)"

        # Re-process any remaining events
        check_new_events

        success_n=$(count_events "EXPERIMENT_SUCCESS")
        fail_n=$(count_events "EXPERIMENT_FAILED")
        has_queue_done=$(count_events "QUEUE_DONE")

        # Determine outcome
        outcome="completed"
        if [[ $has_queue_done -eq 0 && $success_n -eq 0 && $fail_n -eq 0 ]]; then
            outcome="crash"
            log "No events recorded — likely crashed before any experiment ran"
        elif [[ $has_queue_done -eq 0 ]]; then
            outcome="crash"
            log "No QUEUE_DONE event — process likely crashed mid-queue"
        fi

        write_watchdog_result "$outcome" "$ELAPSED"

        if [[ "$outcome" == "crash" ]]; then
            # ── Crash ──
            err_ctx=""
            err_ctx=$(get_error_context)
            send_webhook "Train Crashed" "host=${HOSTNAME_SAFE}\ntmux=${TMUX_SESSION}\nsuccess=${success_n}\nfailed=${fail_n}\nelapsed=$((ELAPSED/60))min"
            if [[ -n "$CODEX_SESSION" ]]; then
                trigger_codex_resume "$(cat <<PROMPT
训练进程异常退出（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}，运行 $((ELAPSED/60)) 分钟）。

成功实验: ${success_n}，失败实验: ${fail_n}
Run 目录: ${RUN_DIR}

最后 80 行输出:
\`\`\`
${err_ctx}
\`\`\`

请分析崩溃原因并修复。如果是可恢复错误，请重新启动训练。
PROMPT
)"
            fi

        elif [[ $fail_n -gt 0 && "$ON_ALL_DONE" == "eval" ]]; then
            # ── Completed with some failures ──
            send_webhook "Train Done (partial)" "host=${HOSTNAME_SAFE}\ntmux=${TMUX_SESSION}\nsuccess=${success_n}\nfailed=${fail_n}\nelapsed=$((ELAPSED/60))min"

            # Still trigger eval for successful models
            if [[ $success_n -gt 0 ]]; then
                log "Triggering eval for $success_n successful models"
                trigger_eval_for_completed_models
            fi

            if [[ -n "$CODEX_SESSION" ]]; then
                trigger_codex_resume "$(cat <<PROMPT
训练队列已结束（tmux: ${TMUX_SESSION}，host: ${HOSTNAME_SAFE}）。

成功: ${success_n} 个实验，失败: ${fail_n} 个实验
Run 目录: ${RUN_DIR}

成功的模型已自动入 eval 队列并触发了 eval skill。
请检查失败实验的原因，判断是否需要重新训练。
PROMPT
)"
            fi

        elif [[ "$ON_ALL_DONE" == "eval" && $success_n -gt 0 ]]; then
            # ── All success → trigger eval ──
            send_webhook "Train All Success" "host=${HOSTNAME_SAFE}\ntmux=${TMUX_SESSION}\nsuccess=${success_n}\nelapsed=$((ELAPSED/60))min"
            trigger_eval_for_completed_models

        else
            # ── notify only ──
            send_webhook "Train Done" "host=${HOSTNAME_SAFE}\ntmux=${TMUX_SESSION}\nsuccess=${success_n}\nfailed=${fail_n}\nelapsed=$((ELAPSED/60))min"
        fi

        break
    fi

    # ── Progress / stall detection ──────────────────────────────────────
    if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
        current_progress=""
        current_progress=$(grep -oP '(?:train_loss|global_step|epoch)\s*[=:]\s*[0-9.]+' "$TRAIN_LOG" 2>/dev/null | tail -1 || echo "")
        if [[ -n "$current_progress" ]]; then
            if [[ "$current_progress" == "$LAST_PROGRESS" ]]; then
                STALL_CYCLES=$((STALL_CYCLES + 1))
                if [[ $STALL_CYCLES -ge $STALL_THRESHOLD && "$WARNED_STALL" != "true" ]]; then
                    log "WARNING: stalled for $((STALL_CYCLES * CHECK_INTERVAL))s at: $current_progress"
                    send_webhook "Train Stall Warning" "host=${HOSTNAME_SAFE}\ntmux=${TMUX_SESSION}\nstalled_at=${current_progress}\nduration=$((STALL_CYCLES * CHECK_INTERVAL))s"
                    WARNED_STALL=true
                fi
            else
                STALL_CYCLES=0
                WARNED_STALL=false
                LAST_PROGRESS="$current_progress"
            fi
        fi
    fi

    # Heartbeat every 10 minutes
    if (( ELAPSED % 600 == 0 )); then
        log "Heartbeat: tmux alive, elapsed=$((ELAPSED/60))min, success=$(count_events EXPERIMENT_SUCCESS), failed=$(count_events EXPERIMENT_FAILED)"
    fi
done

log "Watchdog exiting (pid=$$)"

# ── eval trigger function ───────────────────────────────────────────────────
trigger_eval_for_completed_models() {
    # Collect successful model names from events file
    local models=""
    while IFS='|' read -r _ _ _ _ exp_name _ _; do
        [[ -n "$exp_name" && "$exp_name" != "unknown" ]] && {
            [[ -n "$models" ]] && models="${models}, "
            models="${models}${exp_name}"
        }
    done < <(grep "^EXPERIMENT_SUCCESS|" "$EVENTS_FILE" 2>/dev/null)

    if [[ -z "$models" ]]; then
        log "No successful models to trigger eval for"
        return 0
    fi

    log "Triggering eval Codex session for models: $models"

    # The new Codex session follows the eval skill — it does NOT run eval scripts directly
    trigger_codex_new_session "$(cat <<PROMPT
训练已完成，以下模型已自动入队 runtime/eval_queue/eval_todo.txt：
${models}

请按照 overlapvln-eval skill 的流程执行评测。具体步骤：

1. 检查服务器 98 和 73 的 GPU 可用性（nvidia-smi，检查是否有 torchrun/train 进程）
2. 如果找到空闲服务器（8 卡均空闲）：
   - 在该服务器上 tmux 启动评测 + eval_watchdog
3. 如果两台服务器都在忙（仍有训练在跑）：
   - 不要强行启动 eval（会 OOM）
   - 模型已经在 eval_todo.txt 中，不会丢失
   - 在空闲率最高的服务器上启动 start_eval_worker.sh（在 tmux 中）
   - worker 会持续轮询 eval_todo.txt，训练结束 GPU 释放后自动开始 eval
   - 注册 eval_watchdog 监控这个 worker 的 tmux session
4. 确认评测已启动或 worker 已注册后，报告状态

关键信息：
- 评测 conda env: swift-vln-eval
- eval_todo.txt: runtime/eval_queue/eval_todo.txt
- 不要在 GPU 被占用时直接运行 eval 脚本
PROMPT
)"
}
