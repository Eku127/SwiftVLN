#!/bin/bash
# ============================================================================
# SwiftVLN file-backed serial evaluation queue
# ============================================================================
#
# 功能:
#   - 支持从 eval_todo.txt 读取待评估模型
#   - 串行执行评估，避免资源竞争
#   - 自动记录评估结果和错误
#   - 评估完成后显示汇总表格（包含结果路径）
#   - 动态模式下持续消费运行期间追加的新任务
#
# 使用方法:
#   DYNAMIC_TODO=true WAIT_FOR_NEW_TASKS=true \
#     bash scripts/queue/eval_queue.sh
#
# 环境变量:
#   EVAL_SPLIT   - SatNav 默认 val_seen / Habitat 默认 val_unseen (可手动覆盖)
#   CUDA_DEVICES - GPU设备 (default: 0,1,2,3,4,5,6,7)
#   SAVE_VIDEO   - 保存视频 (true/false)
#   MAX_EPISODES - 限制episode数量 (用于调试)
#   EVAL_QUEUE_DIR - 评测队列状态目录 (default: ${SWIFTVLN_ROOT}/runtime/eval_queue)
#   DYNAMIC_TODO - 启用动态模式 (true/false)
#   AUTO_TODO    - 兼容旧启动器；true 时同时启用 DYNAMIC_TODO
#   WAIT_FOR_NEW_TASKS - 动态模式下队列空时持续等待新任务 (true/false)
#   TODO_POLL_INTERVAL - 空队列轮询间隔秒数 (default: 60)
#
# 注意: ENV_TYPE 现在自动从模型名中解析 (habitat/satnav)
#       当前主线仅支持 swiftvln-* 模型名
#
# ============================================================================

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "\n${BOLD}${CYAN}$1${NC}\n"; }

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
source "${SWIFTVLN_ROOT}/scripts/eval/eval_lib.sh"
EVAL_BY_NAME_SCRIPT="${SWIFTVLN_ROOT}/scripts/eval/eval_by_name.sh"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
TODO_FILE="${EVAL_QUEUE_DIR}/eval_todo.txt"
DONE_FILE="${EVAL_QUEUE_DIR}/eval_done.txt"
FAILED_FILE="${EVAL_QUEUE_DIR}/eval_failed_todo.txt"
TODO_LOCK_FILE="${TODO_FILE}.lock"

initialize_queue_files() {
    mkdir -p "$EVAL_QUEUE_DIR"
    touch "$TODO_FILE" "$DONE_FILE" "$FAILED_FILE"
}

# ============================================================================
# 全局变量
# ============================================================================
declare -a MODELS=()           # 模型列表
declare -a EVALUATED=()        # 已评估的模型列表
declare -a EXP_RESULTS=()      # 评估结果
declare -a EXP_ERRORS=()       # 错误记录
declare -a RESULT_PATHS=()     # 结果路径
SLEEP_BETWEEN_EVALS=30         # 评估间隔（秒）
DYNAMIC_TODO="${DYNAMIC_TODO:-false}"
WAIT_FOR_NEW_TASKS="${WAIT_FOR_NEW_TASKS:-false}"
TODO_POLL_INTERVAL="${TODO_POLL_INTERVAL:-60}"

# ============================================================================
# Todo Queue File Helpers
# ============================================================================
append_unique_line() {
    local file="$1"
    local line="$2"
    mkdir -p "$(dirname "$file")"
    touch "$file"
    if ! grep -Fxq "$line" "$file"; then
        echo "$line" >> "$file"
    fi
}

remove_line_from_todo() {
    local model="$1"
    mkdir -p "$(dirname "$TODO_FILE")"
    touch "$TODO_FILE"
    (
        flock -w "${EVAL_TODO_LOCK_TIMEOUT:-30}" 201 || exit 1
        grep -Fxv "$model" "$TODO_FILE" > "${TODO_FILE}.tmp" || true
        mv "${TODO_FILE}.tmp" "$TODO_FILE"
    ) 201>"$TODO_LOCK_FILE" || {
        print_error "更新 todo 队列失败: 无法获取锁 ${TODO_LOCK_FILE}"
        return 1
    }
}

mark_model_done() {
    local model="$1"
    remove_line_from_todo "$model"
    append_unique_line "$DONE_FILE" "$model"
}

mark_model_failed() {
    local model="$1"
    remove_line_from_todo "$model"
    append_unique_line "$FAILED_FILE" "$model"
}

# ============================================================================
# 检查依赖
# ============================================================================
if [ ! -f "$EVAL_BY_NAME_SCRIPT" ]; then
    print_error "找不到 eval_by_name.sh: $EVAL_BY_NAME_SCRIPT"
    exit 1
fi

# ============================================================================
# 从 TODO 文件读取模型列表
# ============================================================================
read_models_from_todo() {
    if [ ! -f "$TODO_FILE" ]; then
        return 1
    fi
    
    local temp_models=()
    # 使用 || [ -n "$line" ] 确保读取最后一行（即使没有换行符）
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | xargs)
        if [[ -n "$line" && ! "$line" =~ ^# ]]; then
            temp_models+=("$line")
        fi
    done < "$TODO_FILE"
    
    if [[ ${#temp_models[@]} -eq 0 ]]; then
        return 1
    fi
    
    MODELS=("${temp_models[@]}")
    return 0
}

# ============================================================================
# 检查模型是否已评估
# ============================================================================
is_evaluated() {
    local model=$1
    for evaluated_model in "${EVALUATED[@]}"; do
        if [[ "$evaluated_model" == "$model" ]]; then
            return 0
        fi
    done
    return 1
}

# ============================================================================
# 动态刷新模型列表（从 todo 文件读取新增的模型）
# ============================================================================
refresh_models_from_todo() {
    if [ ! -f "$TODO_FILE" ]; then
        return 1
    fi
    
    local new_models_added=0
    
    # 读取 todo 文件中的所有模型
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | xargs)
        if [[ -n "$line" && ! "$line" =~ ^# ]]; then
            # 检查是否已在队列中
            local already_in_queue=false
            for existing_model in "${MODELS[@]}"; do
                if [[ "$existing_model" == "$line" ]]; then
                    already_in_queue=true
                    break
                fi
            done
            
            # 如果不在队列中，添加到队列
            if [ "$already_in_queue" = false ]; then
                MODELS+=("$line")
                ((new_models_added++))
                print_info "🆕 发现新模型: $line"
            fi
        fi
    done < "$TODO_FILE"
    
    if [[ $new_models_added -gt 0 ]]; then
        print_success "从 todo 文件中添加了 $new_models_added 个新模型"
    fi
    
    return 0
}

# ============================================================================
# File-backed queue protocol
# ============================================================================
show_usage() {
    cat <<EOF
Usage:
  [DYNAMIC_TODO=true] [WAIT_FOR_NEW_TASKS=true] \\
    bash scripts/queue/eval_queue.sh [--check-queue]

Models are read only from:
  $TODO_FILE

Use enqueue_eval.sh to add validated model names. Positional model lists and the
interactive wizard are no longer supported.
EOF
}

configure_queue() {
    if [[ "${AUTO_TODO:-false}" == "true" ]]; then
        DYNAMIC_TODO=true
    fi
    unset ENV_TYPE
    if [[ -n "${EVAL_SPLIT:-}" ]]; then
        export EVAL_SPLIT
    else
        unset EVAL_SPLIT
    fi
    export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    export SAVE_VIDEO="${SAVE_VIDEO:-false}"
    if [[ "$SAVE_VIDEO" == "true" ]]; then
        export VIDEO_COMPRESSION=true
    else
        export VIDEO_COMPRESSION=false
    fi

    if ! read_models_from_todo; then
        MODELS=()
    fi

    local model
    for model in "${MODELS[@]}"; do
        local python_executable="${PYTHON_EXECUTABLE:-python}"
        if ! command -v "$python_executable" >/dev/null 2>&1; then
            python_executable=python3
        fi
        if ! "$python_executable" -m swiftvln.experiment parse-name "$model" >/dev/null; then
            print_error "Invalid SwiftVLN model name in $TODO_FILE: $model"
            return 1
        fi
    done

    if [[ "$WAIT_FOR_NEW_TASKS" == "true" && "$DYNAMIC_TODO" != "true" ]]; then
        print_error "WAIT_FOR_NEW_TASKS=true requires DYNAMIC_TODO=true"
        return 1
    fi
    print_success "Validated ${#MODELS[@]} queued models"
}

# ============================================================================
# 显示评估汇总
# ============================================================================
show_summary() {
    print_header "╔══════════════════════════════════════════════════════════════╗"
    echo -e "                    ${BOLD}评估配置汇总${NC}"
    print_header "╚══════════════════════════════════════════════════════════════╝"
    
    local _split_display="${EVAL_SPLIT:-auto (SatNav: val_seen+val_unseen, Habitat: val_unseen)}"
    echo -e "${BOLD}评估配置:${NC}"
    echo "  ENV_TYPE:     (自动从模型名解析)"
    echo "  EVAL_SPLIT:   ${_split_display}"
    echo "  CUDA_DEVICES: $CUDA_DEVICES"
    echo "  SAVE_VIDEO:   $SAVE_VIDEO"
    echo "  TODO_FILE:    $TODO_FILE"
    echo "  DONE_FILE:    $DONE_FILE"
    echo "  FAILED_FILE:  $FAILED_FILE"
    if [[ "$SAVE_VIDEO" == "true" ]]; then
        echo "  VIDEO_COMPRESSION: true (自动启用)"
    fi
    if [ "$DYNAMIC_TODO" = true ]; then
        echo -e "  DYNAMIC_TODO: ${GREEN}启用${NC} (会在每次评估后检查新任务)"
    else
        echo "  DYNAMIC_TODO: 禁用"
    fi
    echo ""
    
    echo -e "${BOLD}待评估模型 (共 ${#MODELS[@]} 个):${NC}"
    echo "┌────┬──────────────────────────────────────────────────────────────────────────────┬──────────┬────────────────┐"
    echo "│ #  │ 模型名称                                                                     │ 环境类型 │ Embed          │"
    echo "├────┼──────────────────────────────────────────────────────────────────────────────┼──────────┼────────────────┤"
    
    local idx=1
    for model in "${MODELS[@]}"; do
        local env_type=$(parse_env_type_from_model "$model")
        local embed_slot=$(parse_embed_slot_from_model "$model")
        printf "│ %-2d │ %-76s │ %-8s │ %-14s │\n" "$idx" "${model:0:76}" "$env_type" "$embed_slot"
        ((idx++))
    done
    
    echo "└────┴──────────────────────────────────────────────────────────────────────────────┴──────────┴────────────────┘"
}

# ============================================================================
# 运行单个评估
# ============================================================================
run_evaluation() {
    local exp_idx=$1
    local model=$2
    local total=$3
    
    print_header "🚀 评估 $exp_idx/$total: $model"
    
    local start_time=$(date +%s)
    local log_file="${SWIFTVLN_ROOT}/logs/eval_queue_${model}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "日志文件: $log_file"
    print_info "开始评估..."
    
    local eval_status=1
    local run_log_file="$log_file"

    bash "$EVAL_BY_NAME_SCRIPT" "$model" 2>&1 | tee "$run_log_file" || true
    eval_status=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_str=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    # 解析模型架构
    local model_arch=""
    if [[ "$model" == swiftvln-* ]]; then
        model_arch="swiftvln"
    fi

    # 确定本次实际评测的 split 列表（与 eval_by_name.sh 的逻辑保持一致）
    local _model_env_type
    local _actual_splits
    _model_env_type=$(parse_env_type_from_model "$model")
    _actual_splits="$(infer_eval_splits "$_model_env_type" "${EVAL_SPLIT:-}")"

    # 查找各 split 结果路径（格式: results/eval/<arch>/<model>/<split>/<timestamp>/）
    local result_path=""         # 用于摘要展示（取第一个有效 split）
    declare -a _split_result_paths=()
    if [[ -n "$model_arch" ]]; then
        local results_base="${SWIFTVLN_ROOT}/results/eval/${model_arch}/${model}"
        for _s in ${_actual_splits}; do
            local _split_dir="${results_base}/${_s}"
            local _candidate=""
            if [[ -d "$_split_dir" ]]; then
                _candidate=$(ls -td "${_split_dir}/"* 2>/dev/null | head -1)
            fi
            if [[ -n "$_candidate" && -d "$_candidate" ]]; then
                _split_result_paths+=("${_s}|${_candidate}")
                [[ -z "$result_path" ]] && result_path="$_candidate"
            else
                _split_result_paths+=("${_s}|N/A")
            fi
        done
        # 兜底：找 <model>/*/<timestamp>/ 下最新的目录（任意 split）
        if [[ -z "$result_path" && -d "$results_base" ]]; then
            result_path=$(find "$results_base" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | \
                xargs -I{} stat -c '%Y %n' {} 2>/dev/null | sort -rn | awk 'NR==1{print $2}')
        fi
    fi

    if [[ $eval_status -eq 0 ]]; then
        # 尝试从首个 split 结果目录读取评估指标（供摘要展示）
        local sr="--"
        local spl="--"
        local ne="--"
        
        if [[ -n "$result_path" && -f "${result_path}/evaluation_summary.json" ]]; then
            sr=$(grep -oP '"success_rate":\s*\K[0-9.]+' "${result_path}/evaluation_summary.json" 2>/dev/null || echo "--")
            spl=$(grep -oP '"mean_spl":\s*\K[0-9.]+' "${result_path}/evaluation_summary.json" 2>/dev/null || echo "--")
            ne=$(grep -oP '"navigation_error":\s*\K[0-9.]+' "${result_path}/evaluation_summary.json" 2>/dev/null || echo "--")
        fi

        # 构建多 split 结果路径展示字符串
        local _paths_str=""
        for _sp in "${_split_result_paths[@]}"; do
            local _sname="${_sp%%|*}"
            local _spath="${_sp##*|}"
            _paths_str+="[${_sname}] ${_spath}  "
        done

        EXP_RESULTS+=("$exp_idx|$model|SUCCESS|$duration_str|SR:$sr SPL:$spl NE:$ne")
        RESULT_PATHS+=("$exp_idx|$model|${_paths_str:-${result_path:-N/A}}")
        print_success "评估 $exp_idx 完成! 耗时: $duration_str"
        
        return 0
    else
        local error_msg=$(tail -50 "$run_log_file" | grep -iE "(error|oom|cuda|exception)" | head -3 | tr '\n' ' ')
        error_msg=${error_msg:-"未知错误"}
        
        EXP_RESULTS+=("$exp_idx|$model|FAILED|$duration_str|--")
        RESULT_PATHS+=("$exp_idx|$model|${run_log_file}")
        EXP_ERRORS+=("评估 $exp_idx ($model): ${error_msg:0:100}")
        print_error "评估 $exp_idx 失败!"
        
        return 1
    fi
}

# ============================================================================
# 写入机器可读的完成状态（供外部工具使用）
# ============================================================================
write_completion_status() {
    local result_file="${1:-}"
    local success_count="${2:-0}"
    local fail_count="${3:-0}"
    local _hostname
    _hostname="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

    # per-host 状态文件（多服务器并发安全）
    local status_file="${EVAL_QUEUE_DIR}/eval_queue_last_run_${_hostname}.json"

    local success_list=""
    local failed_list=""
    for result in "${EXP_RESULTS[@]}"; do
        IFS='|' read -r _idx model status _dur _metrics <<< "$result"
        if [[ "$status" == "SUCCESS" ]]; then
            [[ -n "$success_list" ]] && success_list="${success_list},"
            success_list="${success_list}\"${model}\""
        else
            [[ -n "$failed_list" ]] && failed_list="${failed_list},"
            failed_list="${failed_list}\"${model}\""
        fi
    done

    local json_body
    json_body=$(cat <<EOF
{
    "completed_at": "$(date -Iseconds)",
    "eval_split": "${EVAL_SPLIT:-val_unseen}",
    "success_count": ${success_count},
    "fail_count": ${fail_count},
    "total_count": ${#EXP_RESULTS[@]},
    "success_models": [${success_list}],
    "failed_models": [${failed_list}],
    "result_file": "${result_file}",
    "hostname": "${_hostname}"
}
EOF
)
    echo "$json_body" > "$status_file"
    print_info "完成状态已写入: $status_file"

}

# ============================================================================
# 显示最终结果
# ============================================================================
show_final_results() {
    local LOGS_DIR="${SWIFTVLN_ROOT}/logs/eval_queue_results"
    mkdir -p "$LOGS_DIR"
    local RESULT_FILE="${LOGS_DIR}/eval_results_$(date +%Y%m%d_%H%M%S).txt"
    
    # 同时输出到终端和文件
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "                                           VLN 评估结果汇总"
        echo "                                        $(date '+%Y-%m-%d %H:%M:%S')"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo ""
        echo "评估配置:"
        echo "  ENV_TYPE:     (自动从模型名解析)"
        echo "  EVAL_SPLIT:   ${EVAL_SPLIT:-auto (SatNav: val_seen+val_unseen, Habitat: val_unseen)}"
        echo "  CUDA_DEVICES: $CUDA_DEVICES"
        echo "  SAVE_VIDEO:   $SAVE_VIDEO"
        if [[ "$SAVE_VIDEO" == "true" ]]; then
            echo "  VIDEO_COMPRESSION: true"
        fi
        if [ "$DYNAMIC_TODO" = true ]; then
            echo "  DYNAMIC_TODO: 启用"
        fi
        echo ""
        echo "┌────┬────────────────────────────────────────────────────────────────────────────┬─────────┬──────────┬──────────────────────────┐"
        echo "│ #  │ 模型名称                                                                   │ 状态    │ 耗时     │ 评估指标                 │"
        echo "├────┼────────────────────────────────────────────────────────────────────────────┼─────────┼──────────┼──────────────────────────┤"
        
        for result in "${EXP_RESULTS[@]}"; do
            IFS='|' read -r idx model status duration metrics <<< "$result"
            printf "│ %-2s │ %-74s │ %-7s │ %-8s │ %-24s │\n" "$idx" "${model:0:74}" "$status" "$duration" "${metrics:0:24}"
        done
        
        echo "└────┴────────────────────────────────────────────────────────────────────────────┴─────────┴──────────┴──────────────────────────┘"
        
        # 统计
        local success_count=0
        local fail_count=0
        for result in "${EXP_RESULTS[@]}"; do
            if [[ "$result" == *"|SUCCESS|"* ]]; then
                ((success_count++))
            else
                ((fail_count++))
            fi
        done
        
        echo ""
        echo "统计: 成功 $success_count / 失败 $fail_count / 总计 ${#EXP_RESULTS[@]}"
        
        # 显示结果路径
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "📁 评估结果路径"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        for path_info in "${RESULT_PATHS[@]}"; do
            IFS='|' read -r idx model path <<< "$path_info"
            local status_icon="✓"
            for result in "${EXP_RESULTS[@]}"; do
                if [[ "$result" == "$idx|$model|FAILED"* ]]; then
                    status_icon="✗"
                    break
                fi
            done
            echo "  [$status_icon] 评估 $idx ($model):"
            echo "      $path"
        done
        
        # 显示错误详情
        if [[ ${#EXP_ERRORS[@]} -gt 0 ]]; then
            echo ""
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            echo "❌ 错误详情"
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            for error in "${EXP_ERRORS[@]}"; do
                echo "  • $error"
            done
        fi
        
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    } | tee "$RESULT_FILE"

    local success_count=0
    local fail_count=0
    for result in "${EXP_RESULTS[@]}"; do
        if [[ "$result" == *"|SUCCESS|"* ]]; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done

    # 写入机器可读的完成状态文件（供外部工具使用）
    write_completion_status "$RESULT_FILE" "$success_count" "$fail_count"

    echo ""
    print_success "结果已保存到: $RESULT_FILE"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    local validate_only="${EVAL_QUEUE_VALIDATE_ONLY:-false}"
    case "${1:-}" in
        --help|-h)
            show_usage
            return 0
            ;;
        --check-queue)
            validate_only=true
            ;;
        "") ;;
        *)
            print_error "Unknown argument: $1"
            show_usage
            return 2
            ;;
    esac

    initialize_queue_files
    configure_queue || return 1
    show_summary
    if [[ "$validate_only" == "true" ]]; then
        print_success "Queue check passed; no evaluation was launched."
        return 0
    fi

    echo ""
    echo -e "${BOLD}${CYAN}"
    echo "  ╦  ╦╦  ╔╗╔  ╔═╗┬  ┬┌─┐┬    ╔═╗ ┬ ┬┌─┐┬ ┬┌─┐"
    echo "  ╚╗╔╝║  ║║║  ║╣ └┐┌┘├─┤│    ║═╬╗│ │├┤ │ │├┤ "
    echo "   ╚╝ ╩═╝╝╚╝  ╚═╝ └┘ ┴ ┴┴─┘  ╚═╝╚└─┘└─┘└─┘└─┘"
    echo -e "${NC}"
    echo ""

    # 开始评估
    print_header "🏃 开始串行评估"
    
    if [ "$DYNAMIC_TODO" = true ]; then
        print_info "动态模式已启用 - 脚本会在每次评估后检查 todo 文件中的新模型"
        if [ "$WAIT_FOR_NEW_TASKS" = true ]; then
            print_info "队列为空时将每 ${TODO_POLL_INTERVAL}s 检查新任务"
        else
            print_info "所有模型评估完成后将自动退出"
        fi
    else
        print_info "串行模式 - 将依次评估 ${#MODELS[@]} 个模型"
    fi
    
    local exp_idx=1
    local model_idx=0
    
    while true; do
        # 动态模式下，每次循环开始时刷新模型列表
        if [ "$DYNAMIC_TODO" = true ]; then
            refresh_models_from_todo
        fi
        
        local total=${#MODELS[@]}
        
        # 检查是否还有未评估的模型
        local found_unevaluated=false
        local current_model=""
        
        for ((i=model_idx; i<total; i++)); do
            if ! is_evaluated "${MODELS[$i]}"; then
                found_unevaluated=true
                current_model="${MODELS[$i]}"
                model_idx=$i
                break
            fi
        done
        
        # 如果没有未评估的模型，退出循环
        if [ "$found_unevaluated" = false ]; then
            if [ "$DYNAMIC_TODO" = true ] && [ "$WAIT_FOR_NEW_TASKS" = true ]; then
                print_info "当前无待评估模型，${TODO_POLL_INTERVAL}s 后检查新任务..."
                sleep "$TODO_POLL_INTERVAL"
                continue
            fi
            if [ "$DYNAMIC_TODO" = true ]; then
                print_info "所有模型评估完成，动态模式退出"
            fi
            break
        fi
        
        # 计算已评估数量
        local evaluated_count=${#EVALUATED[@]}
        local pending_count=$((total - evaluated_count))
        
        echo ""
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
        echo -e "  进度: 已评估 $evaluated_count / 总计 $total (待评估 $pending_count)"
        if [ "$DYNAMIC_TODO" = true ]; then
            echo -e "  ${CYAN}[动态模式] 可随时向 eval_todo.txt 追加新模型${NC}"
        fi
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
        
        # 运行评估
        local eval_rc=0
        run_evaluation "$exp_idx" "$current_model" "$total" || eval_rc=$?
        if [[ $eval_rc -eq 0 ]]; then
            mark_model_done "$current_model"
        else
            mark_model_failed "$current_model"
        fi
        
        # 标记为已评估
        EVALUATED+=("$current_model")
        
        # 移动到下一个模型
        ((model_idx++))
        ((exp_idx++))
        
        # 等待GPU清空
        print_info "等待 ${SLEEP_BETWEEN_EVALS} 秒让GPU清空..."
        sleep $SLEEP_BETWEEN_EVALS
    done
    
    # 显示最终结果
    show_final_results
    
    print_header "🎉 所有评估任务完成!"
}

# ============================================================================
# 执行
# ============================================================================
main "$@"
