#!/bin/bash
# ============================================================================
# SwiftVLN config-driven serial training queue
# ============================================================================
#
# 功能:
#   - 从 TRAIN_EXPERIMENTS_FILE 加载结构化实验列表
#   - 串行执行训练，避免资源竞争
#   - 自动记录实验结果和错误
#   - 训练完成后显示汇总表格
#
# 使用方法:
#   TRAIN_EXPERIMENTS_FILE=/path/to/experiments.sh \
#     bash scripts/queue/train_queue.sh
#
# 支持的模型: swiftvln
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
VLN_ROOT="${SWIFTVLN_ROOT}/src/swiftvln"

# ============================================================================
# 全局变量
# ============================================================================
declare -a EXPERIMENTS=()      # 实验列表
declare -a EXP_RESULTS=()      # 实验结果
declare -a EXP_ERRORS=()       # 错误记录
SLEEP_BETWEEN_EXPERIMENTS=30   # 实验间隔（秒）
MAX_AUTO_FIX_RETRIES="${MAX_AUTO_FIX_RETRIES:-2}"

LAST_AUTO_FIX_ACTIONS=""
AUTO_ENQUEUE_EVAL="${AUTO_ENQUEUE_EVAL:-true}"
EVAL_ENQUEUE_SKIP_CHECKPOINT_LOCAL="${EVAL_ENQUEUE_SKIP_CHECKPOINT_LOCAL:-false}"
EVAL_ENQUEUE_RETRIES="${EVAL_ENQUEUE_RETRIES:-3}"
EVAL_ENQUEUE_RETRY_SLEEP="${EVAL_ENQUEUE_RETRY_SLEEP:-3}"
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SatNav}"
SWANLAB_DIRECT_NETWORK="${SWANLAB_DIRECT_NETWORK:-true}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-}"
TRAIN_DRY_RUN="${TRAIN_DRY_RUN:-false}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"
MEMORY_METHOD="${MEMORY_METHOD:-history}"
MAP_GLOBAL_SIDE_M="${MAP_GLOBAL_SIDE_M:-1000}"
MAP_LOCAL_SIDE_M="${MAP_LOCAL_SIDE_M:-400}"
MAP_RENDER_PX="${MAP_RENDER_PX:-448}"
MAP_MASK_METHOD="${MAP_MASK_METHOD:-dilate20}"

enqueue_model_for_eval() {
    local model_name="$1"
    local enqueue_opt_local=""
    local enqueue_cmd_output=""
    local attempt=1
    local retries="${EVAL_ENQUEUE_RETRIES}"
    if [[ -z "$model_name" || "$model_name" == "unknown" ]]; then
        print_warning "无法解析模型名，跳过自动入评测队列"
        return 1
    fi
    if [[ "${AUTO_ENQUEUE_EVAL}" != "true" ]]; then
        print_info "AUTO_ENQUEUE_EVAL=false，跳过自动入队"
        return 0
    fi

    if [[ "${EVAL_ENQUEUE_SKIP_CHECKPOINT_LOCAL}" == "true" ]]; then
        enqueue_opt_local="--skip-checkpoint"
    fi

    print_info "准备自动入评测队列(共享工作区本地文件): model=${model_name}"
    while [[ $attempt -le $retries ]]; do
        enqueue_cmd_output="$(bash "${SWIFTVLN_ROOT}/scripts/queue/enqueue_eval.sh" "$model_name" ${enqueue_opt_local} 2>&1)" && {
            print_success "已自动入评测队列(本地共享路径): ${model_name}"
            [[ -n "${enqueue_cmd_output}" ]] && print_info "入队输出: ${enqueue_cmd_output}"
            return 0
        }
        print_warning "自动入评测队列失败(本地) attempt=${attempt}/${retries}: ${model_name}"
        [[ -n "${enqueue_cmd_output}" ]] && print_warning "失败详情: ${enqueue_cmd_output}"
        ((attempt++))
        [[ $attempt -le $retries ]] && sleep "${EVAL_ENQUEUE_RETRY_SLEEP}"
    done

    print_warning "自动入评测队列失败: ${model_name}"
    return 1
}

set_train_env() {
    local env_name="$1"
    local key="$2"
    local value="$3"
    local -n env_ref="$env_name"
    env_ref["$key"]="$value"
}

resolve_master_port_from_env() {
    local env_name="$1"
    local -n env_ref="$env_name"
    echo "${env_ref[MASTER_PORT]:-${MASTER_PORT:-29500}}"
}

train_env_to_args() {
    local env_name="$1"
    local out_name="$2"
    local -n env_ref="$env_name"
    local -n out_ref="$out_name"
    local key=""

    out_ref=()
    while IFS= read -r key; do
        [[ -z "$key" ]] && continue
        out_ref+=("${key}=${env_ref[$key]}")
    done < <(printf '%s\n' "${!env_ref[@]}" | sort)
}

is_local_tcp_port_free() {
    local port="$1"
    python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

find_available_master_port() {
    local preferred_port="${1:-29500}"
    python3 - "$preferred_port" <<'PY'
import socket
import sys

preferred = int(sys.argv[1])
candidates = [preferred]
candidates.extend(range(29500, 30000))
candidates.extend(range(29000, 29500))
candidates.extend(range(30000, 31000))

seen = set()
for port in candidates:
    if port in seen:
        continue
    seen.add(port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        continue
    else:
        print(port)
        sys.exit(0)
    finally:
        s.close()

sys.exit(1)
PY
}

ensure_available_master_port() {
    local env_name="$1"
    local -n env_ref="$env_name"
    local current_port=""
    local new_port=""

    current_port="$(resolve_master_port_from_env "$env_name")"
    [[ -z "$current_port" ]] && return 0

    if is_local_tcp_port_free "$current_port"; then
        return 0
    fi

    new_port="$(find_available_master_port "$current_port")" || {
        print_warning "启动前检测到 MASTER_PORT=${current_port} 已占用，但未找到可用替代端口"
        return 1
    }

    if [[ "$new_port" != "$current_port" ]]; then
        env_ref[MASTER_PORT]="$new_port"
        print_warning "启动前检测到 MASTER_PORT=${current_port} 已占用，切换为 MASTER_PORT=${new_port}"
    fi

    return 0
}

apply_auto_fix_for_train_failure() {
    local log_file="$1"
    local env_name="$2"
    local -n env_ref="$env_name"
    local fixed=false
    local actions=()

    if grep -qi "dataloader_prefetch_factor can only be set.*dataloader_num_workers > 1" "$log_file"; then
        env_ref[DATALOADER_NUM_WORKERS]="2"
        env_ref[DATALOADER_PREFETCH_FACTOR]="2"
        fixed=true
        actions+=("set DATALOADER_NUM_WORKERS=2, DATALOADER_PREFETCH_FACTOR=2")
        print_warning "自动修复: 调整 dataloader workers/prefetch"
    fi

    if grep -qi "Your setup doesn't support bf16/gpu" "$log_file"; then
        env_ref[TORCH_DTYPE]="float16"
        fixed=true
        actions+=("set TORCH_DTYPE=float16")
        print_warning "自动修复: bf16 -> float16"
    fi

    if grep -qiE "address already in use|Address already in use" "$log_file"; then
        local current_port=""
        local new_port=""
        current_port="$(resolve_master_port_from_env "$env_name")"
        new_port="$(find_available_master_port "${current_port:-29500}")" || new_port=""
        if [[ -n "$new_port" ]]; then
            env_ref[MASTER_PORT]="$new_port"
            fixed=true
            actions+=("set MASTER_PORT=${new_port}")
            print_warning "自动修复: 更换 MASTER_PORT=${new_port}"
        else
            print_warning "自动修复失败: 未找到可用 MASTER_PORT"
        fi
    fi

    if grep -qiE "out of memory|CUDA out of memory" "$log_file"; then
        env_ref[BATCH_SIZE]="1"
        env_ref[GRAD_ACCUM_STEPS]="1"
        fixed=true
        actions+=("set BATCH_SIZE=1, GRAD_ACCUM_STEPS=1")
        print_warning "自动修复: 降低 batch 配置"
    fi

    if grep -qi "weights trying to be saved contained shared tensors" "$log_file"; then
        env_ref[SAVE_SAFETENSORS]="false"
        fixed=true
        actions+=("set SAVE_SAFETENSORS=false")
        print_warning "自动修复: 设置 --save_safetensors false"
    fi

    LAST_AUTO_FIX_ACTIONS="$(IFS='; '; echo "${actions[*]}")"

    [[ "$fixed" == "true" ]]
}

# ============================================================================
# Config-only queue protocol
# ============================================================================
show_usage() {
    cat <<EOF
Usage:
  TRAIN_EXPERIMENTS_FILE=/path/to/experiments.sh \\
    bash scripts/queue/train_queue.sh [--check-config]

The config file must define:
  ENV_TYPE="satnav"  # or habitat
  EXPERIMENTS=(
    "swiftvln|default|baseline|SatNav|/path/to/trajectory_data"
  )

Each experiment is: model|comma-separated KEY=VALUE overrides|description|dataset name|dataset path.
EOF
}

load_experiment_config() {
    if [[ -z "${TRAIN_EXPERIMENTS_FILE:-}" ]]; then
        print_error "TRAIN_EXPERIMENTS_FILE is required; the interactive wizard was removed."
        show_usage
        return 1
    fi
    if [[ ! -f "$TRAIN_EXPERIMENTS_FILE" ]]; then
        print_error "TRAIN_EXPERIMENTS_FILE does not exist: $TRAIN_EXPERIMENTS_FILE"
        return 1
    fi

    print_info "Loading experiment config: $TRAIN_EXPERIMENTS_FILE"
    ENV_TYPE=""
    EXPERIMENTS=()
    # shellcheck source=/dev/null
    source "$TRAIN_EXPERIMENTS_FILE"

    case "${ENV_TYPE:-}" in
        satnav|habitat) ;;
        *)
            print_error "ENV_TYPE must be satnav or habitat in TRAIN_EXPERIMENTS_FILE"
            return 1
            ;;
    esac
    if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
        print_error "EXPERIMENTS must contain at least one entry"
        return 1
    fi

    local experiment model config changes dataset_names dataset_paths extra item
    local -a config_items=()
    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r model config changes dataset_names dataset_paths extra <<< "$experiment"
        if [[ -n "$extra" || -z "$model" || -z "$config" || -z "$changes" || -z "$dataset_names" || -z "$dataset_paths" ]]; then
            print_error "Invalid experiment entry (expected 5 non-empty fields): $experiment"
            return 1
        fi
        if [[ "$model" != "swiftvln" ]]; then
            print_error "Only swiftvln is supported, got: $model"
            return 1
        fi
        if [[ "$config" != "default" ]]; then
            IFS=',' read -ra config_items <<< "$config"
            for item in "${config_items[@]}"; do
                if [[ ! "$item" =~ ^[A-Z_][A-Z0-9_]*=.+$ ]]; then
                    print_error "Invalid KEY=VALUE override: $item"
                    return 1
                fi
            done
        fi
    done

    USE_SWANLAB="${USE_SWANLAB:-false}"
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-SatNav}"
    print_success "Validated ${#EXPERIMENTS[@]} experiments (env=$ENV_TYPE)"
}

# ============================================================================
# 显示实验汇总
# ============================================================================
show_summary() {
    print_header "╔══════════════════════════════════════════════════════════════╗"
    echo -e "                    ${BOLD}实验配置汇总${NC}"
    print_header "╚══════════════════════════════════════════════════════════════╝"

    echo -e "${BOLD}基础配置:${NC}"
    echo "  SwanLab:    $([ "$USE_SWANLAB" = true ] && echo "启用 ($SWANLAB_PROJECT)" || echo "禁用")"
    echo "  环境类型:   $ENV_TYPE"
    echo ""

    echo -e "${BOLD}实验列表 (共 ${#EXPERIMENTS[@]} 个实验):${NC}"

    echo "┌────┬──────────────┬──────────────────────────────────────┬──────────────────────┐"
    echo "│ #  │ 模型         │ 配置改动                             │ 数据集               │"
    echo "├────┼──────────────┼──────────────────────────────────────┼──────────────────────┤"

    local idx=1
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r model config changes ds_names ds_paths <<< "$exp"
        printf "│ %-2d │ %-12s │ %-36s │ %-20s │\n" "$idx" "$model" "${changes:0:36}" "${ds_names:0:20}"
        ((idx++))
    done

    echo "└────┴──────────────┴──────────────────────────────────────┴──────────────────────┘"
}

# ============================================================================
# 运行单个实验
# ============================================================================
run_experiment() {
    local exp_idx=$1
    local model=$2
    local config=$3
    local changes=$4
    local ds_names=$5
    local ds_paths=$6

    if [[ "$model" != "swiftvln" ]]; then
        print_error "当前主线 train_queue 仅支持 swiftvln，收到不受支持的模型: $model"
        return 1
    fi

    print_header "🚀 实验 $exp_idx: $model"
    echo "配置: $changes"
    echo "数据集: $ds_names"
    echo "环境: $ENV_TYPE"
    if [[ -n "$TRAIN_CUDA_DEVICES" ]]; then
        echo "GPU 配置: TRAIN_CUDA_DEVICES=$TRAIN_CUDA_DEVICES"
    elif [[ -n "$TRAIN_NUM_GPUS" ]]; then
        echo "GPU 配置: TRAIN_NUM_GPUS=$TRAIN_NUM_GPUS"
    else
        echo "GPU 配置: auto (使用当前可见 GPU)"
    fi
    if [[ "$TRAIN_DRY_RUN" == "true" ]]; then
        echo "Dry Run: true"
    fi
    echo ""

    # 获取训练脚本路径
    local train_script="${SWIFTVLN_ROOT}/scripts/train/train_swiftvln_qwen_vl.sh"

    if [[ ! -f "$train_script" ]]; then
        print_error "找不到训练脚本: $train_script"
        return 1
    fi

    # Build per-run environment overrides and call the original train script.
    # Keep this data-driven instead of rewriting a temporary shell script.
    declare -A train_env=()
    set_train_env train_env "VLN_ENV_TYPE" "$ENV_TYPE"
    set_train_env train_env "VLN_DATA_PATH" "$ds_paths"
    set_train_env train_env "USE_SWANLAB" "$USE_SWANLAB"
    set_train_env train_env "SWANLAB_PROJECT" "$SWANLAB_PROJECT"
    set_train_env train_env "SWANLAB_DIRECT_NETWORK" "$SWANLAB_DIRECT_NETWORK"
    set_train_env train_env "MEMORY_METHOD" "$MEMORY_METHOD"
    set_train_env train_env "MAP_GLOBAL_SIDE_M" "$MAP_GLOBAL_SIDE_M"
    set_train_env train_env "MAP_LOCAL_SIDE_M" "$MAP_LOCAL_SIDE_M"
    set_train_env train_env "MAP_RENDER_PX" "$MAP_RENDER_PX"
    set_train_env train_env "MAP_MASK_METHOD" "$MAP_MASK_METHOD"
    set_train_env train_env "TRAIN_CUDA_DEVICES" "$TRAIN_CUDA_DEVICES"
    set_train_env train_env "TRAIN_NUM_GPUS" "$TRAIN_NUM_GPUS"
    set_train_env train_env "TRAIN_DRY_RUN" "$TRAIN_DRY_RUN"

    if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
        set_train_env train_env "RESUME_FROM_CHECKPOINT" "$RESUME_FROM_CHECKPOINT"
        set_train_env train_env "RESUME_ONLY_MODEL" "$RESUME_ONLY_MODEL"
        print_info "恢复训练: $RESUME_FROM_CHECKPOINT (resume_only_model=$RESUME_ONLY_MODEL)"
    fi

    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        set_train_env train_env "OUTPUT_DIR_OVERRIDE" "$OUTPUT_DIR_OVERRIDE"
        print_info "输出目录覆盖: $OUTPUT_DIR_OVERRIDE"
    fi

    # 注入自定义配置
    if [[ "$config" != "default" ]]; then
        IFS=',' read -ra config_items <<< "$config"
        for item in "${config_items[@]}"; do
            if [[ -n "$item" && "$item" == *"="* ]]; then
                local var_name="${item%%=*}"
                local var_value="${item#*=}"
                set_train_env train_env "$var_name" "$var_value"
                case "$var_name" in
                    MEMORY_METHOD) set_train_env train_env "MEMORY_METHOD" "$var_value" ;;
                    MAP_GLOBAL_SIDE_M) set_train_env train_env "MAP_GLOBAL_SIDE_M" "$var_value" ;;
                    MAP_LOCAL_SIDE_M) set_train_env train_env "MAP_LOCAL_SIDE_M" "$var_value" ;;
                    MAP_RENDER_PX) set_train_env train_env "MAP_RENDER_PX" "$var_value" ;;
                    MAP_MASK_METHOD) set_train_env train_env "MAP_MASK_METHOD" "$var_value" ;;
                esac
                print_info "配置覆盖: ${var_name}=${var_value}"
            fi
        done
    fi

    print_info "环境类型: $ENV_TYPE"
    print_info "Memory 配置: method=${train_env[MEMORY_METHOD]}, global=${train_env[MAP_GLOBAL_SIDE_M]}, local=${train_env[MAP_LOCAL_SIDE_M]}, render=${train_env[MAP_RENDER_PX]}, mask=${train_env[MAP_MASK_METHOD]}"
    print_info "数据路径: $ds_paths"

    # 运行训练
    local start_time=$(date +%s)
    local log_file="${SWIFTVLN_ROOT}/logs/train_queue_${model}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    print_info "日志文件: $log_file"
    print_info "开始训练..."

    local attempt=1
    local max_attempts=$((MAX_AUTO_FIX_RETRIES + 1))
    local run_log_file="$log_file"
    local attempted_fixes=""
    local -a train_env_args=()

    # 执行训练脚本（带自动修复重试）
    while true; do
        if [[ $attempt -eq 1 ]]; then
            run_log_file="$log_file"
        else
            run_log_file="${log_file%.log}_retry${attempt}.log"
            print_warning "开始第 ${attempt} 次尝试..."
        fi

        ensure_available_master_port train_env || true
        set_train_env train_env "TRAIN_LOG_FILE" "$run_log_file"
        train_env_to_args train_env train_env_args

        env "${train_env_args[@]}" bash "$train_script" 2>&1 | tee "$run_log_file"
        local train_exit_code=${PIPESTATUS[0]}

        if [[ $train_exit_code -eq 0 ]]; then
            break
        fi

        print_warning "训练脚本退出码: ${train_exit_code}"

        if [[ $attempt -lt $max_attempts ]] && apply_auto_fix_for_train_failure "$run_log_file" train_env; then
            attempted_fixes="${attempted_fixes}\n- attempt ${attempt}: ${LAST_AUTO_FIX_ACTIONS}"
            ((attempt++))
            continue
        fi
        break
    done

    local dry_run_completed=false
    if [[ "$TRAIN_DRY_RUN" == "true" ]] && [[ -f "$run_log_file" ]] && grep -q "TRAIN_DRY_RUN=true, skip torchrun launch after config validation." "$run_log_file"; then
        dry_run_completed=true
    fi

    if [[ -f "$run_log_file" ]] && { grep -q "Training completed!" "$run_log_file" || [[ "$dry_run_completed" == "true" ]]; }; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_str=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))

        # 从日志中提取实际的输出目录（从 "Model saved to:" 行）
        local output_path=$(grep -oP 'Model saved to: \K.*' "$run_log_file" | tail -1)
        local exp_name=$(basename "$output_path" 2>/dev/null)
        if [[ "$dry_run_completed" == "true" ]]; then
            exp_name=${exp_name:-"dry-run"}
        else
            exp_name=${exp_name:-"unknown"}
        fi

        # Write train_metadata.json if not already created by the training script
        if [[ -n "$output_path" && -d "$output_path" && ! -f "${output_path}/train_metadata.json" ]]; then
            local _swanlab_url=""
            if [[ "$USE_SWANLAB" == true ]]; then
                _swanlab_url=$(grep -oP 'https://swanlab\.cn/@[^\s"]+/runs/[^\s"]+' "$run_log_file" 2>/dev/null | tail -1)
            fi
            _SWANLAB_URL="$_swanlab_url" \
            _SWANLAB_PROJECT="${SWANLAB_PROJECT:-}" \
            _SWANLAB_EXP="$exp_name" \
            _OUTPUT_DIR="$output_path" \
            python3 "${SWIFTVLN_ROOT}/scripts/train/_write_train_metadata.py" 2>/dev/null || true
        fi

        # 格式: idx|model|changes|ds_names|status|duration|exp_name
        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|SUCCESS|$duration_str|$exp_name")
        if [[ "$dry_run_completed" == "true" ]]; then
            print_success "实验 $exp_idx Dry Run 完成! 耗时: $duration_str"
        else
            print_success "实验 $exp_idx 完成! 耗时: $duration_str"
            enqueue_model_for_eval "$exp_name" || true
        fi

        return 0
    else
        local error_msg=$(tail -50 "$run_log_file" | grep -iE "(error|oom|cuda|exception)" | head -5)
        error_msg=${error_msg:-"未知错误"}

        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|FAILED|--|--")
        EXP_ERRORS+=("实验 $exp_idx ($model): $error_msg")
        print_error "实验 $exp_idx 失败!"

        return 1
    fi
}

# ============================================================================
# 显示最终结果
# ============================================================================
show_final_results() {
    local LOGS_DIR="${SWIFTVLN_ROOT}/logs/train_queue_results"
    mkdir -p "$LOGS_DIR"
    local RESULT_FILE="${LOGS_DIR}/train_results_$(date +%Y%m%d_%H%M%S).txt"

    local success_count=0
    local fail_count=0
    for result in "${EXP_RESULTS[@]}"; do
        if [[ "$result" == *"|SUCCESS|"* ]]; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done

    {
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "                                               训练结果汇总"
        echo "                                            $(date '+%Y-%m-%d %H:%M:%S')"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo ""
        echo "基础配置:"
        echo "  SwanLab:    $([ "$USE_SWANLAB" = true ] && echo "启用 ($SWANLAB_PROJECT)" || echo "禁用")"
        echo "  环境类型:   $ENV_TYPE"
        echo ""
        echo "┌────┬──────────────┬──────────────────────────────────────┬──────────────────────┬─────────┬──────────┐"
        echo "│ #  │ 模型         │ 配置改动                             │ 数据集               │ 状态    │ 耗时     │"
        echo "├────┼──────────────┼──────────────────────────────────────┼──────────────────────┼─────────┼──────────┤"

        for result in "${EXP_RESULTS[@]}"; do
            IFS="|" read -r idx model changes ds_names status duration exp_name <<< "$result"
            printf "│ %-2s │ %-12s │ %-36s │ %-20s │ %-7s │ %-8s │\n" "$idx" "$model" "${changes:0:36}" "${ds_names:0:20}" "$status" "$duration"
        done

        echo "└────┴──────────────┴──────────────────────────────────────┴──────────────────────┴─────────┴──────────┘"
        echo ""
        echo "统计: 成功 $success_count / 失败 $fail_count / 总计 ${#EXP_RESULTS[@]}"

        if [[ ${#EXP_ERRORS[@]} -gt 0 ]]; then
            echo ""
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            echo "错误详情"
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            for error in "${EXP_ERRORS[@]}"; do
                echo "  • $error"
            done
        fi

        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "成功实验输出目录"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        for result in "${EXP_RESULTS[@]}"; do
            IFS="|" read -r idx model changes ds_names status duration exp_name <<< "$result"
            if [[ "$status" == "SUCCESS" ]]; then
                echo "  • 实验 $idx ($model, $ds_names): output/${model}/${exp_name}"
            fi
        done

        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    } | tee "$RESULT_FILE"

    _write_train_completion_status "$RESULT_FILE" "$success_count" "$fail_count"

    echo ""
    print_success "结果已保存到: $RESULT_FILE"
}

_write_train_completion_status() {
    local result_file="${1:-}" success_count="${2:-0}" fail_count="${3:-0}"
    local _hostname
    _hostname="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

    local success_list="" failed_list=""
    for result in "${EXP_RESULTS[@]}"; do
        IFS='|' read -r _idx _model _changes _ds _status _dur exp_name <<< "$result"
        if [[ "$_status" == "SUCCESS" ]]; then
            [[ -n "$success_list" ]] && success_list="${success_list},"
            success_list="${success_list}\"${exp_name}\""
        else
            [[ -n "$failed_list" ]] && failed_list="${failed_list},"
            failed_list="${failed_list}\"${_model}\""
        fi
    done

    local json_body
    json_body=$(cat <<EOF
{
    "completed_at": "$(date -Iseconds)",
    "env_type": "${ENV_TYPE}",
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

    local host_file="${SWIFTVLN_ROOT}/runtime/train_queue/train_queue_last_run_${_hostname}.json"
    mkdir -p "$(dirname "$host_file")"
    echo "$json_body" > "$host_file"
    print_info "完成状态已写入: $host_file"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    local validate_only="${TRAIN_QUEUE_VALIDATE_ONLY:-false}"
    case "${1:-}" in
        --help|-h)
            show_usage
            return 0
            ;;
        --check-config)
            validate_only=true
            ;;
        "") ;;
        *)
            print_error "Unknown argument: $1"
            show_usage
            return 2
            ;;
    esac

    load_experiment_config || return 1
    show_summary
    if [[ "$validate_only" == "true" ]]; then
        print_success "Configuration check passed; no training was launched."
        return 0
    fi

    echo ""
    echo -e "${BOLD}${CYAN}"
    echo "  ╦  ╦╦  ╔╗╔  ╔╦╗┬─┐┌─┐┬┌┐┌  ╔═╗ ┬ ┬┌─┐┬ ┬┌─┐"
    echo "  ╚╗╔╝║  ║║║   ║ ├┬┘├─┤││││  ║═╬╗│ │├┤ │ │├┤ "
    echo "   ╚╝ ╩═╝╝╚╝   ╩ ┴└─┴ ┴┴┘└┘  ╚═╝╚└─┘└─┘└─┘└─┘"
    echo -e "${NC}"
    echo ""

    # 开始训练
    print_header "🏃 开始串行训练 ($ENV_TYPE)"

    local exp_idx=1
    local total=${#EXPERIMENTS[@]}

    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r model config changes ds_names ds_paths <<< "$exp"

        echo ""
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
        echo -e "  进度: $exp_idx / $total"
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"

        # 运行实验
        run_experiment "$exp_idx" "$model" "$config" "$changes" "$ds_names" "$ds_paths" || true

        # 如果不是最后一个实验，等待GPU清空
        if [[ $exp_idx -lt $total ]]; then
            print_info "等待 ${SLEEP_BETWEEN_EXPERIMENTS} 秒让GPU清空..."
            sleep $SLEEP_BETWEEN_EXPERIMENTS
        fi

        ((exp_idx++))
    done

    # 显示最终结果
    show_final_results

    print_header "🎉 所有训练任务完成!"
}

# ============================================================================
# 执行
# ============================================================================
main "$@"
exit $?
