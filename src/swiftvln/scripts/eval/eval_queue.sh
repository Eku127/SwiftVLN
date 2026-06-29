#!/bin/bash
# ============================================================================
# VLN 串行评估脚本 - 交互式配置
# ============================================================================
#
# 功能:
#   - 支持从 eval_todo.txt 读取待评估模型
#   - 交互式输入多个模型名称（分号分隔）
#   - 串行执行评估，避免资源竞争
#   - 自动记录评估结果和错误
#   - 评估完成后显示汇总表格（包含结果路径）
#   - 【动态模式】每次评估完成后重新读取 todo 文件，支持运行期间追加新任务
#
# 使用方法:
#   bash src/swiftvln/scripts/eval/eval_queue.sh
#
# 也支持非交互模式:
#   bash src/swiftvln/scripts/eval/eval_queue.sh "model1;model2;model3"
#
# 动态模式使用说明:
#   1. 交互式启动脚本时，选择使用 eval_todo.txt
#   2. 启用"动态读取 todo 文件"选项
#   3. 脚本运行期间，可以随时向 eval_todo.txt 文件追加新的模型名称
#   4. 脚本会在每次评估完成后自动检测并添加新任务到队列
#
# 环境变量:
#   EVAL_SPLIT   - SatNav 默认 val_seen / Habitat 默认 val_unseen (可手动覆盖)
#   CUDA_DEVICES - GPU设备 (default: 0,1,2,3,4,5,6,7)
#   SAVE_VIDEO   - 保存视频 (true/false)
#   MAX_EPISODES - 限制episode数量 (用于调试)
#   EVAL_QUEUE_DIR - 评测队列状态目录 (default: ${SWIFTVLN_ROOT}/runtime/eval_queue)
#   DYNAMIC_TODO - 启用动态模式 (true/false, 非交互模式下使用)
#   AUTO_TODO    - 无交互从 eval_todo.txt 启动 (true/false)
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
MAGENTA='\033[0;35m'
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
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
EVAL_BY_NAME_SCRIPT="${SCRIPT_DIR}/eval_by_name.sh"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_eval_results.py"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-${SWIFTVLN_ROOT}/runtime/eval_queue}"
TODO_FILE="${EVAL_QUEUE_DIR}/eval_todo.txt"
DONE_FILE="${EVAL_QUEUE_DIR}/eval_done.txt"
FAILED_FILE="${EVAL_QUEUE_DIR}/eval_failed_todo.txt"
TODO_LOCK_FILE="${TODO_FILE}.lock"

mkdir -p "$EVAL_QUEUE_DIR"
touch "$TODO_FILE" "$DONE_FILE" "$FAILED_FILE"

# ============================================================================
# 全局变量
# ============================================================================
declare -a MODELS=()           # 模型列表
declare -a EVALUATED=()        # 已评估的模型列表
declare -a EXP_RESULTS=()      # 评估结果
declare -a EXP_ERRORS=()       # 错误记录
declare -a RESULT_PATHS=()     # 结果路径
SLEEP_BETWEEN_EVALS=30         # 评估间隔（秒）
DYNAMIC_TODO=false             # 是否动态读取 todo 文件
WAIT_FOR_NEW_TASKS=false       # 动态模式下空队列是否持续等待
TODO_POLL_INTERVAL="${TODO_POLL_INTERVAL:-60}"
MAX_EVAL_AUTO_FIX_RETRIES="${MAX_EVAL_AUTO_FIX_RETRIES:-2}"

# Webhook 通知配置
USE_WEBHOOK_NOTIFICATION="${USE_WEBHOOK_NOTIFICATION:-true}"
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=87cd9c07-52f0-4cec-a7a8-9586a9dc68c8}"
LAST_AUTO_FIX_ACTIONS=""

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

send_webhook() {
    local title="$1"
    local body="$2"
    if [[ "$USE_WEBHOOK_NOTIFICATION" != "true" ]]; then
        return 0
    fi
    local content="## ${title}\n${body}\ntime: $(date '+%Y-%m-%d %H:%M:%S')"
    curl -sS -m 8 -X POST "$WEBHOOK_URL" \
      -H "Content-Type: application/json" \
      -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"${content//$'\n'/\\n}\"}}" >/dev/null 2>&1 || true
}

apply_auto_fix_for_eval_failure() {
    local log_file="$1"
    local fixed=false
    local actions=()

    if grep -qiE "address already in use|Address already in use" "$log_file"; then
        export MASTER_PORT=$((29600 + RANDOM % 1000))
        fixed=true
        actions+=("set MASTER_PORT=$MASTER_PORT")
        print_warning "自动修复: 更换 MASTER_PORT=$MASTER_PORT"
    fi

    if grep -qi "Your setup doesn't support bf16/gpu" "$log_file"; then
        export TORCH_DTYPE="float16"
        fixed=true
        actions+=("set TORCH_DTYPE=float16")
        print_warning "自动修复: bf16 -> float16"
    fi

    LAST_AUTO_FIX_ACTIONS="$(IFS='; '; echo "${actions[*]}")"

    [[ "$fixed" == "true" ]]
}

# ============================================================================
# 检查依赖
# ============================================================================
if [ ! -f "$EVAL_BY_NAME_SCRIPT" ]; then
    print_error "找不到 eval_by_name.sh: $EVAL_BY_NAME_SCRIPT"
    exit 1
fi
if [ ! -f "$COLLECT_SCRIPT" ]; then
    print_warning "找不到收集脚本: $COLLECT_SCRIPT (将跳过CSV收集)"
fi

# ============================================================================
# 从模型名解析环境类型（和 eval_by_name.sh 保持一致）
# ============================================================================
parse_env_type_from_model() {
    local name="$1"
    
    # 新格式: {arch}-{env_type}-{model_size}-...
    # 检测第二个字段是否是 habitat 或 satnav
    local second_field=$(echo "$name" | cut -d'-' -f2)
    
    if [[ "$second_field" == "habitat" ]] || [[ "$second_field" == "satnav" ]]; then
        echo "$second_field"
    else
        # 旧格式，默认 habitat
        echo "habitat"
    fi
}

# ============================================================================
# 从模型名解析 embedding 增强开关（和 swiftvln exp_name 保持一致）
# ============================================================================
parse_embed_slot_from_model() {
    local name="$1"
    if [[ "$name" != swiftvln-* ]]; then
        echo "-"
        return
    fi
    if [[ "$name" == *"-posefilm-"* ]]; then
        echo "posefilm"
    elif [[ "$name" == *"-pose-"* ]]; then
        echo "pose"
    elif [[ "$name" == *"-noembed-"* ]]; then
        echo "noembed"
    else
        echo "noembed"
    fi
}

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

# 显示 TODO 文件中的模型
show_todo_models() {
    echo ""
    echo -e "${BOLD}eval_todo.txt 中的模型列表:${NC}"
    echo "┌────┬────────────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│ #  │ 模型名称                                                                                   │"
    echo "├────┼────────────────────────────────────────────────────────────────────────────────────────────┤"
    
    local idx=1
    for model in "${MODELS[@]}"; do
        printf "│ %-2d │ %-90s │\n" "$idx" "${model:0:90}"
        idx=$((idx + 1))
    done
    
    echo "└────┴────────────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
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
# 交互式配置
# ============================================================================
interactive_setup() {
    print_header "╔══════════════════════════════════════════════════════════════╗"
    echo -e "         ${BOLD}VLN 串行评估配置向导${NC}"
    print_header "╚══════════════════════════════════════════════════════════════╝"
    
    # 0. 检查是否要使用 eval_todo.txt
    local use_todo=false
    if [ -f "$TODO_FILE" ]; then
        # 检查 TODO 文件是否有内容
        if read_models_from_todo; then
            echo ""
            echo -e "${CYAN}检测到 eval_todo.txt 文件，包含 ${#MODELS[@]} 个待评估模型${NC}"
            show_todo_models
            
            read -p "是否使用 eval_todo.txt 中的模型列表? [Y/n]: " use_todo_input
            if [[ "$use_todo_input" =~ ^[Yy]?$ ]]; then
                use_todo=true
                print_success "将使用 eval_todo.txt 中的 ${#MODELS[@]} 个模型"
                
                # 询问是否启用动态读取
                echo ""
                echo -e "${CYAN}提示: 动态模式下，脚本会在每次评估完成后重新读取 todo 文件${NC}"
                echo -e "${CYAN}      你可以在脚本运行期间向 eval_todo.txt 追加新的模型名称${NC}"
                echo -e "${CYAN}      所有模型评估完成后将自动退出${NC}"
                read -p "是否启用动态读取 todo 文件? [Y/n]: " dynamic_input
                if [[ "$dynamic_input" =~ ^[Yy]|^[Yy][Ee][Ss]|^$ ]]; then
                    DYNAMIC_TODO=true
                    print_success "已启用动态读取模式"
                else
                    print_info "动态模式未启用"
                fi
            else
                MODELS=()  # 清空，让用户手动输入
            fi
        fi
    fi
    
    # 1. 输入模型名称 (如果没有使用 todo 文件)
    if [ "$use_todo" = false ]; then
        print_header "📝 Step 1: 输入待评估的模型名称"
        echo "格式: 多个模型名用分号(;)分隔"
        echo ""
        echo "示例:"
        echo "  swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-123456"
        echo "  swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-123456  # per_frame, random"
        echo "  swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-123456  # per_frame, no embed"
        echo "  swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-initial-pose-bs64-lr2e-5-123456  # initial + pose"
        echo "  swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b2.0-tome-s2-pose-bs64-lr2e-5-123456  # pose"
        echo "  swiftvln-satnav-3b-1ep-f32s4-overlap16-gtc-k512-noembed-bs64-lr2e-5-123456  # GTC, no embed"
        echo "  swiftvln-satnav-3b-1ep-f32s4-overlap16-sgtc-k512-noembed-bs64-lr2e-5-123456  # SegmentGTC"
        echo ""
        read -p "请输入模型名称: " model_input
        
        if [[ -z "$model_input" ]]; then
            print_error "未输入任何模型名称!"
            exit 1
        fi
        
        # 解析模型列表
        IFS=';' read -ra MODELS <<< "$model_input"
        
        # 去除首尾空格
        for i in "${!MODELS[@]}"; do
            MODELS[$i]=$(echo "${MODELS[$i]}" | xargs)
        done
        
        # 过滤空项
        local temp_models=()
        for model in "${MODELS[@]}"; do
            if [[ -n "$model" ]]; then
                temp_models+=("$model")
            fi
        done
        MODELS=("${temp_models[@]}")
        
        if [[ ${#MODELS[@]} -eq 0 ]]; then
            print_error "未输入任何有效的模型名称!"
            exit 1
        fi
        
        print_success "已添加 ${#MODELS[@]} 个模型"
    fi
    
    # 2. 显示模型解析结果（env_type 自动从模型名解析）
    print_header "🔍 Step 2: 模型环境类型解析结果"
    
    # 解析每个模型的 env_type
    declare -a MODEL_ENV_TYPES=()
    local habitat_count=0
    local satnav_count=0
    
    echo "┌────┬──────────────────────────────────────────────────────────────────────────────┬──────────┬────────────────┐"
    echo "│ #  │ 模型名称                                                                     │ 环境类型 │ Embed          │"
    echo "├────┼──────────────────────────────────────────────────────────────────────────────┼──────────┼────────────────┤"
    
    local idx=1
    for model in "${MODELS[@]}"; do
        local env_type=$(parse_env_type_from_model "$model")
        local embed_slot=$(parse_embed_slot_from_model "$model")
        MODEL_ENV_TYPES+=("$env_type")
        
        if [[ "$env_type" == "habitat" ]]; then
            ((habitat_count++))
        else
            ((satnav_count++))
        fi
        
        printf "│ %-2d │ %-76s │ %-8s │ %-14s │\n" "$idx" "${model:0:76}" "$env_type" "$embed_slot"
        ((idx++))
    done
    
    echo "└────┴──────────────────────────────────────────────────────────────────────────────┴──────────┴────────────────┘"
    echo ""
    
    # 显示汇总
    if [[ $habitat_count -gt 0 && $satnav_count -gt 0 ]]; then
        print_warning "注意：模型列表包含不同环境类型 (habitat: $habitat_count, satnav: $satnav_count)"
        echo "每个模型将使用其对应的环境类型进行评估"
    elif [[ $habitat_count -gt 0 ]]; then
        print_info "所有模型使用 habitat 环境评估"
    else
        print_info "所有模型使用 satnav 环境评估"
    fi
    echo ""
    
    read -p "解析结果是否正确? [Y/n]: " confirm_parse
    if [[ ! "$confirm_parse" =~ ^[Yy]?$ ]]; then
        print_warning "请检查模型名称格式或手动设置 ENV_TYPE 环境变量后重新运行"
        exit 0
    fi
    
    # 3. 其他评估配置
    print_header "⚙️  Step 3: 其他评估配置"
    echo "当前配置:"
    echo "  EVAL_SPLIT:   ${EVAL_SPLIT:-auto (SatNav: val_seen+val_unseen, Habitat: val_unseen)}"
    echo "  CUDA_DEVICES: ${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    echo "  SAVE_VIDEO:   ${SAVE_VIDEO:-false}"
    echo ""
    read -p "是否修改配置? [y/N]: " modify_env
    
    if [[ "$modify_env" =~ ^[Yy]$ ]]; then
        echo ""
        echo "EVAL_SPLIT 留空 = auto (SatNav 跑 val_seen+val_unseen, Habitat 跑 val_unseen)"
        read -p "EVAL_SPLIT [${EVAL_SPLIT:-}]: " new_eval_split
        EVAL_SPLIT="${new_eval_split:-${EVAL_SPLIT:-}}"
        
        read -p "CUDA_DEVICES [${CUDA_DEVICES:-0,1,2,3,4,5,6,7}]: " new_cuda_devices
        CUDA_DEVICES=${new_cuda_devices:-${CUDA_DEVICES:-0,1,2,3,4,5,6,7}}
        
        read -p "SAVE_VIDEO [${SAVE_VIDEO:-false}]: " new_save_video
        SAVE_VIDEO=${new_save_video:-${SAVE_VIDEO:-false}}
    fi
    
    # 导出环境变量 (不再导出 ENV_TYPE，让 eval_by_name.sh 自动从模型名解析)
    unset ENV_TYPE  # 确保不覆盖模型名中的 env_type
    # EVAL_SPLIT 留空时由 eval_by_name.sh 决定 (SatNav: 两个 split; Habitat: val_unseen)
    [ -n "${EVAL_SPLIT}" ] && export EVAL_SPLIT || unset EVAL_SPLIT
    export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    export SAVE_VIDEO="${SAVE_VIDEO:-false}"
    
    # 如果保存视频，自动启用视频压缩
    if [[ "$SAVE_VIDEO" == "true" ]]; then
        export VIDEO_COMPRESSION="true"
        print_info "SAVE_VIDEO=true, 自动启用视频压缩 (VIDEO_COMPRESSION=true)"
    else
        export VIDEO_COMPRESSION="false"
    fi
    
    # 4. 显示汇总
    show_summary
    
    # 4. 确认执行
    echo ""
    read -p "是否开始评估? [Y/n]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]?$ ]]; then
        print_warning "已取消评估"
        exit 0
    fi
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
    
    local original_cuda_devices="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    local original_master_port="${MASTER_PORT:-}"
    local original_torch_dtype="${TORCH_DTYPE:-}"
    local eval_status=1
    local run_log_file="$log_file"
    local attempt=1
    local max_attempts=$((MAX_EVAL_AUTO_FIX_RETRIES + 1))

    # 运行评估（自动修复重试）
    send_webhook "Eval Started" "model=${model}\nindex=${exp_idx}/${total}\nsplit=${EVAL_SPLIT}"
    while true; do
        if [[ $attempt -eq 1 ]]; then
            run_log_file="$log_file"
        else
            run_log_file="${log_file%.log}_retry${attempt}.log"
            print_warning "开始第 ${attempt} 次评估尝试..."
        fi

        bash "$EVAL_BY_NAME_SCRIPT" "$model" 2>&1 | tee "$run_log_file" || true
        eval_status=${PIPESTATUS[0]}
        if [[ $eval_status -eq 0 ]]; then
            break
        fi

        if [[ $attempt -lt $max_attempts ]] && apply_auto_fix_for_eval_failure "$run_log_file"; then
            ((attempt++))
            continue
        fi
        break
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_str=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    # 解析模型架构
    local model_arch=""
    if [[ "$model" == swiftvln-* ]]; then
        model_arch="swiftvln"
    fi

    # 确定本次实际评测的 split 列表（与 eval_by_name.sh 的逻辑保持一致）
    local _actual_splits
    if [[ -n "${EVAL_SPLIT:-}" ]]; then
        _actual_splits="${EVAL_SPLIT}"
    else
        local _model_env_type
        _model_env_type=$(parse_env_type_from_model "$model")
        if [[ "$_model_env_type" == "satnav" ]]; then
            _actual_splits="val_seen val_unseen"
        else
            _actual_splits="val_unseen"
        fi
    fi

    # 查找各 split 结果路径（格式: results/eval/<arch>/<model>/<split>/<timestamp>/）
    local result_path=""         # 用于 webhook/摘要展示（取第一个有效 split）
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
        send_webhook "Eval Finished" "model=${model}\nindex=${exp_idx}/${total}\nstatus=SUCCESS\nduration=${duration_str}\nmetrics=SR:${sr} SPL:${spl} NE:${ne}\nsplits=${_actual_splits}"

        # 自动收集到 results/eval_collected/<split>/eval_results_data<version>.csv
        # 对每个实际评测的 split 分别收集
        if [[ -f "$COLLECT_SCRIPT" ]]; then
            for _sp in "${_split_result_paths[@]}"; do
                local _sname="${_sp%%|*}"
                local _spath="${_sp##*|}"
                if [[ "$_spath" != "N/A" && -d "$_spath" ]]; then
                    python3 "$COLLECT_SCRIPT" \
                        --model-name "$model" \
                        --result-path "$_spath" \
                        --eval-split "$_sname" \
                        --output-dir "${SWIFTVLN_ROOT}/results/eval_collected" >/dev/null 2>&1 || \
                        print_warning "CSV收集失败 [${_sname}]: $model"
                fi
            done
        fi
        
        export CUDA_DEVICES="$original_cuda_devices"
        if [[ -n "$original_master_port" ]]; then export MASTER_PORT="$original_master_port"; else unset MASTER_PORT; fi
        if [[ -n "$original_torch_dtype" ]]; then export TORCH_DTYPE="$original_torch_dtype"; else unset TORCH_DTYPE; fi
        return 0
    else
        local error_msg=$(tail -50 "$run_log_file" | grep -iE "(error|oom|cuda|exception)" | head -3 | tr '\n' ' ')
        error_msg=${error_msg:-"未知错误"}
        
        EXP_RESULTS+=("$exp_idx|$model|FAILED|$duration_str|--")
        RESULT_PATHS+=("$exp_idx|$model|${run_log_file}")
        EXP_ERRORS+=("评估 $exp_idx ($model): ${error_msg:0:100}")
        print_error "评估 $exp_idx 失败!"
        send_webhook "Eval Finished" "model=${model}\nindex=${exp_idx}/${total}\nstatus=FAILED\nduration=${duration_str}"

        export CUDA_DEVICES="$original_cuda_devices"
        if [[ -n "$original_master_port" ]]; then export MASTER_PORT="$original_master_port"; else unset MASTER_PORT; fi
        if [[ -n "$original_torch_dtype" ]]; then export TORCH_DTYPE="$original_torch_dtype"; else unset TORCH_DTYPE; fi
        
        return 1
    fi
}

# ============================================================================
# 写入机器可读的完成状态（供 eval_watchdog / 外部工具使用）
# ============================================================================
write_completion_status() {
    local result_file="${1:-}"
    local success_count="${2:-0}"
    local fail_count="${3:-0}"
    local _hostname
    _hostname="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

    # per-host 状态文件（多服务器并发安全）
    local status_file="${EVAL_QUEUE_DIR}/eval_queue_last_run_${_hostname}.json"

    # 如果调用方设置了 EVAL_RUN_DIR，同时写入 per-run 目录
    local run_status_file=""
    if [[ -n "${EVAL_RUN_DIR:-}" && -d "${EVAL_RUN_DIR}" ]]; then
        run_status_file="${EVAL_RUN_DIR}/eval_queue_status.json"
    fi

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

    if [[ -n "$run_status_file" ]]; then
        echo "$json_body" > "$run_status_file"
        print_info "Per-run 状态已写入: $run_status_file"
    fi
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

    send_webhook "Eval Queue Finished" "success=${success_count}\nfailed=${fail_count}\ntotal=${#EXP_RESULTS[@]}"

    # 写入机器可读的完成状态文件（供 eval_watchdog 等外部工具使用）
    write_completion_status "$RESULT_FILE" "$success_count" "$fail_count"

    echo ""
    print_success "结果已保存到: $RESULT_FILE"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    echo ""
    echo -e "${BOLD}${CYAN}"
    echo "  ╦  ╦╦  ╔╗╔  ╔═╗┬  ┬┌─┐┬    ╔═╗ ┬ ┬┌─┐┬ ┬┌─┐"
    echo "  ╚╗╔╝║  ║║║  ║╣ └┐┌┘├─┤│    ║═╬╗│ │├┤ │ │├┤ "
    echo "   ╚╝ ╩═╝╝╚╝  ╚═╝ └┘ ┴ ┴┴─┘  ╚═╝╚└─┘└─┘└─┘└─┘"
    echo -e "${NC}"
    echo ""
    
    # AUTO_TODO: 无交互从 eval_todo.txt 启动（适合作为常驻 worker）
    if [[ "${AUTO_TODO:-false}" == "true" ]]; then
        print_info "AUTO_TODO 模式: 从 eval_todo.txt 读取模型列表"
        if read_models_from_todo; then
            print_success "从 todo 文件加载 ${#MODELS[@]} 个模型"
        else
            MODELS=()
            print_warning "todo 文件当前为空，将等待新任务"
        fi

        unset ENV_TYPE
        # 若用户未显式设置 EVAL_SPLIT，留空由 eval_by_name.sh 决定（SatNav: 两个 split；Habitat: val_unseen）
        [ -n "${EVAL_SPLIT}" ] && export EVAL_SPLIT
        export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
        export SAVE_VIDEO="${SAVE_VIDEO:-false}"
        if [[ "$SAVE_VIDEO" == "true" ]]; then
            export VIDEO_COMPRESSION="true"
            print_info "SAVE_VIDEO=true, 自动启用视频压缩"
        else
            export VIDEO_COMPRESSION="false"
        fi

        DYNAMIC_TODO=true
        if [[ "${WAIT_FOR_NEW_TASKS:-false}" == "true" ]]; then
            WAIT_FOR_NEW_TASKS=true
            print_info "WAIT_FOR_NEW_TASKS 已启用，空队列将持续轮询"
        fi
        print_info "TODO_POLL_INTERVAL=${TODO_POLL_INTERVAL}s"
        show_summary

    # 检查是否有命令行参数（非交互模式）
    elif [[ $# -ge 1 && -n "$1" ]]; then
        print_info "非交互模式: 使用命令行参数"
        IFS=';' read -ra MODELS <<< "$1"
        
        # 去除首尾空格并过滤空项
        local temp_models=()
        for model in "${MODELS[@]}"; do
            model=$(echo "$model" | xargs)
            if [[ -n "$model" ]]; then
                temp_models+=("$model")
            fi
        done
        MODELS=("${temp_models[@]}")
        
        if [[ ${#MODELS[@]} -eq 0 ]]; then
            print_error "未提供任何有效的模型名称!"
            exit 1
        fi
        
        # 导出环境变量 (不再导出 ENV_TYPE，让 eval_by_name.sh 自动从模型名解析)
        unset ENV_TYPE  # 确保不覆盖模型名中的 env_type
        # 若用户未显式设置 EVAL_SPLIT，留空由 eval_by_name.sh 决定（SatNav: 两个 split；Habitat: val_unseen）
        [ -n "${EVAL_SPLIT}" ] && export EVAL_SPLIT
        export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
        export SAVE_VIDEO="${SAVE_VIDEO:-false}"
        
        # 如果保存视频，自动启用视频压缩
        if [[ "$SAVE_VIDEO" == "true" ]]; then
            export VIDEO_COMPRESSION="true"
            print_info "SAVE_VIDEO=true, 自动启用视频压缩"
        else
            export VIDEO_COMPRESSION="false"
        fi
        
        # 非交互模式下支持 DYNAMIC_TODO 环境变量
        if [[ "${DYNAMIC_TODO}" == "true" ]]; then
            DYNAMIC_TODO=true
            print_info "动态模式已通过环境变量启用"
        fi
        if [[ "${WAIT_FOR_NEW_TASKS:-false}" == "true" ]]; then
            WAIT_FOR_NEW_TASKS=true
            print_info "空队列等待模式已启用"
        fi
        
        show_summary
    else
        # 交互式配置
        interactive_setup
    fi
    
    # 开始评估
    print_header "🏃 开始串行评估"
    
    if [ "$DYNAMIC_TODO" = true ]; then
        print_info "动态模式已启用 - 脚本会在每次评估后检查 todo 文件中的新模型"
        print_info "注意: 所有模型评估完成后将自动退出"
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
