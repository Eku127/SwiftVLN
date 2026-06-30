#!/bin/bash
# ============================================================================
# VLN 串行训练脚本 - 交互式配置
# ============================================================================
#
# 功能:
#   - 交互式配置多个模型的训练参数
#   - 支持多组实验配置（分号分隔）
#   - 串行执行训练，避免资源竞争
#   - 自动记录实验结果和错误
#   - 训练完成后显示汇总表格
#
# 使用方法:
#   bash src/swiftvln/scripts/train/train_queue.sh
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

# ── 事件日志（供 train_watchdog 消费）──────────────────────────────────────
# 写入 TRAIN_EVENTS_FILE（由 watchdog export），回退到 TRAIN_RUN_DIR 下的文件
_emit_train_event() {
    local event_file="${TRAIN_EVENTS_FILE:-${TRAIN_RUN_DIR:+${TRAIN_RUN_DIR}/train_events.log}}"
    [[ -z "$event_file" ]] && return 0
    echo "$*" >> "$event_file"
}

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
        enqueue_cmd_output="$(bash "${SWIFTVLN_ROOT}/src/swiftvln/scripts/eval/enqueue_eval.sh" "$model_name" ${enqueue_opt_local} 2>&1)" && {
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
# 查找模型目录下最新的 checkpoint (参考 eval_by_name.sh)
# ============================================================================
find_latest_checkpoint() {
    local model_dir="$1"
    local latest_checkpoint=""

    # 首先在 v0-* 子目录中查找
    for version_dir in "$model_dir"/v*; do
        if [ -d "$version_dir" ]; then
            # 查找 checkpoint-* 目录，取最大的
            for ckpt in "$version_dir"/checkpoint-*; do
                if [ -d "$ckpt" ]; then
                    latest_checkpoint="$ckpt"
                fi
            done
        fi
    done

    # 如果没有找到，直接在模型目录下查找
    if [ -z "$latest_checkpoint" ]; then
        for ckpt in "$model_dir"/checkpoint-*; do
            if [ -d "$ckpt" ]; then
                latest_checkpoint="$ckpt"
            fi
        done
    fi

    echo "$latest_checkpoint"
}

# ============================================================================
# 模型默认配置
# ============================================================================
get_default_config() {
    local model="$1"

    case "$model" in
        swiftvln)
            cat << 'EOF'
# SwiftVLN 可配置参数 (代号=默认值) - 滑动窗口重叠压缩
a) NUM_FRAMES=32           # 视频帧数
b) NUM_HISTORY=8           # 历史帧数 (per_frame模式有效)
c) NUM_FUTURE_STEPS=4      # 预测动作步数
d) COMPRESS_STRIDE=2       # 压缩步长 (per_frame: 2=4x, 3=9x, 4=16x)
e) NUM_OVERLAP=0           # 滑动窗口重叠帧数 (0=禁用; stride = num_frames - num_overlap)
f) NUM_EPOCHS=1            # 训练轮数
g) LEARNING_RATE=2e-5      # 学习率
h) BATCH_SIZE=8            # 批量大小
i) FREEZE_VIT=false        # 冻结ViT
j) FREEZE_LLM=false        # 冻结LLM
k) FREEZE_ALIGNER=false    # 冻结Aligner
l) USE_TOME=false          # 使用GridToMe压缩 (per_frame模式: true=ToMe, false=AvgPool)
m) HISTORY_PROCESSOR_TYPE=per_frame  # 历史处理方式: per_frame(默认), gtc 或 sgtc(segment_gtc)
n) GTC_OUTPUT_TOKENS=512   # GTC/SegmentGTC输出tokens数 (gtc/segment_gtc模式有效)
o) LOG_BASE=1.0            # 历史采样分布 (per_frame: 1.0=均匀, >1.0=对数/更多近帧; NUM_HISTORY=0时忽略)
p) SYSTEM_PROMPT_SETTING=vanilla  # System prompt策略: vanilla(默认) 或 initial
q) USE_POSE_EMBED=false       # Pose增强: true(开启) 或 false(关闭)
r) POSE_FUSION_METHOD=additive  # Pose融合方式: additive(默认) 或 film
# 说明: SwiftVLN 没有单独的 USE_MEMORY 开关；如需 no-memory，请用
#       HISTORY_PROCESSOR_TYPE=per_frame + NUM_HISTORY=0
EOF
            ;;
    esac
}

# ============================================================================
# 展开代号为完整参数名
# ============================================================================
expand_shortcodes() {
    local model="$1"
    local config="$2"

    # 如果是default或空，直接返回
    if [[ -z "$config" || "$config" == "default" ]]; then
        echo "$config"
        return
    fi

    # 定义映射
    declare -A mapping
    case "$model" in
        swiftvln)
            mapping=([a]="NUM_FRAMES" [b]="NUM_HISTORY" [c]="NUM_FUTURE_STEPS" [d]="COMPRESS_STRIDE" [e]="NUM_OVERLAP" [f]="NUM_EPOCHS" [g]="LEARNING_RATE" [h]="BATCH_SIZE" [i]="FREEZE_VIT" [j]="FREEZE_LLM" [k]="FREEZE_ALIGNER" [l]="USE_TOME" [m]="HISTORY_PROCESSOR_TYPE" [n]="GTC_OUTPUT_TOKENS" [o]="LOG_BASE" [p]="SYSTEM_PROMPT_SETTING" [q]="USE_POSE_EMBED" [r]="POSE_FUSION_METHOD")
            ;;
    esac

    local result="$config"

    # 替换代号为完整参数名
    for code in "${!mapping[@]}"; do
        local param="${mapping[$code]}"
        # 替换开头的 代号= 和 ,代号=
        result=$(echo "$result" | sed "s/^${code}=/${param}=/I" | sed "s/,${code}=/,${param}=/gI")
    done

    # 值的简写转换: sgtc -> segment_gtc
    result=$(echo "$result" | sed 's/HISTORY_PROCESSOR_TYPE=sgtc/HISTORY_PROCESSOR_TYPE=segment_gtc/g')

    echo "$result"
}

# ============================================================================
# 解析配置字符串
# ============================================================================
parse_config_string() {
    local config_str="$1"
    # 将逗号分隔的配置转换为export语句
    echo "$config_str" | tr ',' '\n' | while read -r item; do
        if [[ -n "$item" ]]; then
            echo "export $item"
        fi
    done
}

# ============================================================================
# 生成实验描述（改动项）
# ============================================================================
get_experiment_changes() {
    local config_str="$1"
    if [[ -z "$config_str" || "$config_str" == "default" ]]; then
        echo "默认配置"
    else
        echo "$config_str" | tr ',' ' '
    fi
}

# ============================================================================
# 交互式配置
# ============================================================================
interactive_setup() {
    # ── Non-interactive mode ──────────────────────────────────────────────────
    # If TRAIN_EXPERIMENTS_FILE is set, source it to load all config variables
    # and skip the interactive wizard entirely.
    #
    # The file must define (at minimum):
    #   EXPERIMENTS=("model|config|changes|ds_names|ds_paths" ...)
    #   ENV_TYPE="satnav"      (or "habitat")
    #
    # Optional:
    #   USE_SWANLAB="true"
    #   SWANLAB_PROJECT="YourProject"
    if [[ -n "${TRAIN_EXPERIMENTS_FILE:-}" ]]; then
        if [[ ! -f "$TRAIN_EXPERIMENTS_FILE" ]]; then
            print_error "TRAIN_EXPERIMENTS_FILE 指定的文件不存在: $TRAIN_EXPERIMENTS_FILE"
            exit 1
        fi
        print_info "非交互模式：从文件加载实验配置 → $TRAIN_EXPERIMENTS_FILE"
        # shellcheck source=/dev/null
        source "$TRAIN_EXPERIMENTS_FILE"
        USE_SWANLAB="${USE_SWANLAB:-false}"
        SWANLAB_PROJECT="${SWANLAB_PROJECT:-SatNav}"
        if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
            print_error "TRAIN_EXPERIMENTS_FILE 加载后 EXPERIMENTS 数组为空，请检查文件内容"
            exit 1
        fi
        print_success "已加载 ${#EXPERIMENTS[@]} 个实验，env=${ENV_TYPE}, SwanLab=${SWANLAB_PROJECT}"
        return 0
    fi
    # ─────────────────────────────────────────────────────────────────────────

    print_header "╔══════════════════════════════════════════════════════════════╗"
    echo -e "         ${BOLD}VLN 串行训练配置向导${NC}"
    print_header "╚══════════════════════════════════════════════════════════════╝"

    # 1. SwanLab 配置
    print_header "📊 Step 1: SwanLab 配置"
    print_info "SwanLab 默认状态: $([ "$USE_SWANLAB" = true ] && echo "启用" || echo "禁用")"
    local swanlab_default="n"
    [[ "$USE_SWANLAB" == true ]] && swanlab_default="Y"
    read -p "启用 SwanLab 记录实验? [${swanlab_default}]: " swanlab_enable_input
    swanlab_enable_input=${swanlab_enable_input:-$swanlab_default}
    if [[ "$swanlab_enable_input" =~ ^[Yy]$ ]]; then
        USE_SWANLAB=true
        read -p "SwanLab Project 名称 [${SWANLAB_PROJECT}]: " swanlab_project
        SWANLAB_PROJECT=${swanlab_project:-$SWANLAB_PROJECT}
        print_success "SwanLab: 启用, Project: $SWANLAB_PROJECT"
    else
        USE_SWANLAB=false
        print_success "SwanLab: 禁用"
    fi

    # 2. 选择模型
    print_header "🤖 Step 2: 选择训练模型"
    echo "可选模型:"
    echo "  a) swiftvln"
    echo ""
    echo "示例: a 或 swiftvln"
    read -p "请选择模型 [a]: " models_input
    models_input=${models_input:-a}

    if [[ -z "$models_input" ]]; then
        print_error "未选择任何模型!"
        exit 1
    fi

    # 展开模型代号
    declare -A model_mapping=([a]="swiftvln")
    SELECTED_MODELS=()
    IFS=',' read -ra model_codes <<< "$models_input"
    for code in "${model_codes[@]}"; do
        code=$(echo "$code" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
        if [[ -n "${model_mapping[$code]}" ]]; then
            SELECTED_MODELS+=("${model_mapping[$code]}")
        elif [[ "$code" =~ ^(swiftvln)$ ]]; then
            # 也支持直接输入模型名
            SELECTED_MODELS+=("$code")
        else
            print_warning "未知模型代号: $code (已跳过)"
        fi
    done

    if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
        print_error "未选择任何有效模型!"
        exit 1
    fi

    print_success "已选择模型: ${SELECTED_MODELS[*]}"

    # 3. 环境类型
    print_header "🌍 Step 3: 环境类型"
    echo "可选环境:"
    echo "  a) satnav  (卫星导航, forward=10m) [默认]"
    echo "  b) habitat (室内导航, forward=0.25m)"
    echo ""
    read -p "选择环境类型 [a]: " env_input
    env_input=${env_input:-a}

    # 展开环境代号
    case "$env_input" in
        a|A|satnav|SatNav|SATNAV)
            ENV_TYPE="satnav"
            ;;
        b|B|habitat|Habitat|HABITAT)
            ENV_TYPE="habitat"
            ;;
        *)
            print_warning "未知环境类型: $env_input, 使用默认 satnav"
            ENV_TYPE="satnav"
            ;;
    esac
    print_success "环境类型: $ENV_TYPE"

    # 4. 数据集配置
    print_header "📁 Step 4: 数据集配置"

    # 数据集映射（全局）
    declare -A DATASET_MAPPING=([a]="R2R" [b]="RxR" [c]="EnvDrop" [d]="ScaleVLN")
    declare -A DATASET_PATHS_MAP=(
        [R2R]="/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
        [RxR]="/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
        [EnvDrop]="/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
        [ScaleVLN]="/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/ScaleVLN"
    )

    if [[ "$ENV_TYPE" == "habitat" ]]; then
        echo "可用数据集:"
        echo "  a) R2R"
        echo "  b) RxR"
        echo "  c) EnvDrop"
        echo "  d) ScaleVLN"
        echo ""
        echo "示例: a,b 或 a,b,c"
        echo "多组数据集用分号分隔: a,b;a,b,c (每组独立跑一轮实验)"
        read -p "选择数据集 (逗号分隔) [a,b]: " datasets_input
        datasets_input=${datasets_input:-a,b}

        # 解析多组数据集配置（分号分隔）
        DATASET_CONFIGS=()  # 存储所有数据集配置组
        IFS=';' read -ra ds_groups <<< "$datasets_input"

        for ds_group in "${ds_groups[@]}"; do
            ds_group=$(echo "$ds_group" | xargs)  # trim

            # 解析单组数据集
            local group_paths=()
            local group_names=()

            IFS=',' read -ra ds_codes <<< "$ds_group"
            for code in "${ds_codes[@]}"; do
                code=$(echo "$code" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
                local ds_name=""

                # 检查是否是代号
                if [[ -n "${DATASET_MAPPING[$code]}" ]]; then
                    ds_name="${DATASET_MAPPING[$code]}"
                elif [[ "$code" =~ ^(R2R|RxR|EnvDrop|ScaleVLN)$ ]]; then
                    ds_name="$code"
                elif [[ "$code" =~ ^(r2r|rxr|envdrop|scalevln)$ ]]; then
                    case "$code" in
                        r2r) ds_name="R2R" ;;
                        rxr) ds_name="RxR" ;;
                        envdrop) ds_name="EnvDrop" ;;
                        scalevln) ds_name="ScaleVLN" ;;
                    esac
                fi

                if [[ -n "$ds_name" && -n "${DATASET_PATHS_MAP[$ds_name]}" ]]; then
                    group_paths+=("${DATASET_PATHS_MAP[$ds_name]}")
                    group_names+=("$ds_name")
                else
                    print_warning "未知数据集: $code (已跳过)"
                fi
            done

            if [[ ${#group_paths[@]} -gt 0 ]]; then
                # 存储格式: "名称1,名称2|路径1,路径2"
                local names_str=$(IFS=','; echo "${group_names[*]}")
                local paths_str=$(IFS=','; echo "${group_paths[*]}")
                DATASET_CONFIGS+=("${names_str}|${paths_str}")
            fi
        done

        if [[ ${#DATASET_CONFIGS[@]} -eq 0 ]]; then
            print_error "未选择任何有效数据集!"
            exit 1
        fi

        # 显示数据集配置
        if [[ ${#DATASET_CONFIGS[@]} -eq 1 ]]; then
            IFS='|' read -r names paths <<< "${DATASET_CONFIGS[0]}"
            print_success "数据集: $names"
        else
            print_success "数据集配置组: ${#DATASET_CONFIGS[@]} 组"
            local group_idx=1
            for ds_config in "${DATASET_CONFIGS[@]}"; do
                IFS='|' read -r names paths <<< "$ds_config"
                echo "  组 $group_idx: $names"
                ((group_idx++))
            done
        fi
    else
        # SatNav 环境
        local default_satnav_path="/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data"
        echo "默认 SatNav 数据路径:"
        echo "  $default_satnav_path"
        echo ""
        read -p "使用默认路径? [Y/n]: " use_default_satnav
        use_default_satnav=${use_default_satnav:-Y}

        if [[ "$use_default_satnav" =~ ^[Yy]$ ]]; then
            DATASET_CONFIGS=("SatNav|${default_satnav_path}")
        else
            read -p "请输入自定义 SatNav 数据路径: " custom_satnav_path
            if [[ -z "$custom_satnav_path" ]]; then
                print_warning "未输入路径，使用默认路径"
                DATASET_CONFIGS=("SatNav|${default_satnav_path}")
            else
                DATASET_CONFIGS=("SatNav|${custom_satnav_path}")
            fi
        fi
        IFS='|' read -r _ satnav_path <<< "${DATASET_CONFIGS[0]}"
        print_success "数据集: SatNav ($satnav_path)"
    fi

    # 配置实验参数
    print_header "⚙️ Step 6: 配置各模型实验参数"

        for model in "${SELECTED_MODELS[@]}"; do
            model=$(echo "$model" | tr -d ' ')

            echo ""
            echo -e "${BOLD}${MAGENTA}━━━ $model 配置 ━━━${NC}"
            echo ""
            echo "默认配置:"
            get_default_config "$model"
            echo ""
            echo "输入格式: 代号=值,代号=值  (如: g=16,h=true)"
            echo "也支持完整参数名: BATCH_SIZE=16,FREEZE_VIT=true"
            echo "多组实验用分号分隔: g=16;g=32,e=2"
            echo "直接回车使用默认配置"
            echo ""
            read -p "[$model] 配置 (或按Enter使用默认): " config_input

            # 收集该模型的所有参数配置
            local model_configs=()
            if [[ -z "$config_input" ]]; then
                model_configs+=("default")
            else
                IFS=';' read -ra exp_configs <<< "$config_input"
                for exp_config in "${exp_configs[@]}"; do
                    exp_config=$(echo "$exp_config" | xargs)  # trim
                    exp_config=$(expand_shortcodes "$model" "$exp_config")
                    model_configs+=("$exp_config")
                done
            fi

            # 组合模型配置与数据集配置
            for model_cfg in "${model_configs[@]}"; do
                changes=$(get_experiment_changes "$model_cfg")
                for ds_config in "${DATASET_CONFIGS[@]}"; do
                    IFS='|' read -r ds_names ds_paths <<< "$ds_config"
                    EXPERIMENTS+=("${model}|${model_cfg}|${changes}|${ds_names}|${ds_paths}")
                done
            done
        done
    # 8. 显示汇总
    show_summary

    # 9. 确认执行
    echo ""
    read -p "是否开始训练? [Y/n]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]?$ ]]; then
        print_warning "已取消训练"
        exit 0
    fi
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
    local train_script="${VLN_ROOT}/model/script/train/train_swiftvln_qwen_vl.sh"

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
            python3 "${SWIFTVLN_ROOT}/src/swiftvln/scripts/train/_write_train_metadata.py" 2>/dev/null || true
        fi

        # 格式: idx|model|changes|ds_names|status|duration|exp_name
        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|SUCCESS|$duration_str|$exp_name")
        if [[ "$dry_run_completed" == "true" ]]; then
            print_success "实验 $exp_idx Dry Run 完成! 耗时: $duration_str"
            _emit_train_event "EXPERIMENT_DRY_RUN_SUCCESS|${exp_idx}|${total:-0}|${model}|${exp_name}|${run_log_file}|$(date -Iseconds)"
        else
            print_success "实验 $exp_idx 完成! 耗时: $duration_str"
            enqueue_model_for_eval "$exp_name" || true
            _emit_train_event "EXPERIMENT_SUCCESS|${exp_idx}|${total:-0}|${model}|${exp_name}|${output_path:-N/A}|$(date -Iseconds)"
        fi

        return 0
    else
        local error_msg=$(tail -50 "$run_log_file" | grep -iE "(error|oom|cuda|exception)" | head -5)
        error_msg=${error_msg:-"未知错误"}

        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|FAILED|--|--")
        EXP_ERRORS+=("实验 $exp_idx ($model): $error_msg")
        print_error "实验 $exp_idx 失败!"
        _emit_train_event "EXPERIMENT_FAILED|${exp_idx}|${total:-0}|${model}|unknown|${error_msg:0:200}|${run_log_file}|$(date -Iseconds)"

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

    _emit_train_event "QUEUE_DONE|${success_count}|${fail_count}|${#EXP_RESULTS[@]}|$(date -Iseconds)"
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

    local queue_dir="${TRAIN_RUN_DIR:-${SWIFTVLN_ROOT}/runtime/train_queue}"
    mkdir -p "$queue_dir"

    # Per-host global file
    local host_file="${SWIFTVLN_ROOT}/runtime/train_queue/train_queue_last_run_${_hostname}.json"
    mkdir -p "$(dirname "$host_file")"
    echo "$json_body" > "$host_file"
    print_info "完成状态已写入: $host_file"

    # Per-run file (if TRAIN_RUN_DIR set by watchdog)
    if [[ -n "${TRAIN_RUN_DIR:-}" && -d "${TRAIN_RUN_DIR}" ]]; then
        echo "$json_body" > "${TRAIN_RUN_DIR}/train_queue_status.json"
        print_info "Per-run 状态已写入: ${TRAIN_RUN_DIR}/train_queue_status.json"
    fi
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    echo ""
    echo -e "${BOLD}${CYAN}"
    echo "  ╦  ╦╦  ╔╗╔  ╔╦╗┬─┐┌─┐┬┌┐┌  ╔═╗ ┬ ┬┌─┐┬ ┬┌─┐"
    echo "  ╚╗╔╝║  ║║║   ║ ├┬┘├─┤││││  ║═╬╗│ │├┤ │ │├┤ "
    echo "   ╚╝ ╩═╝╝╚╝   ╩ ┴└─┴ ┴┴┘└┘  ╚═╝╚└─┘└─┘└─┘└─┘"
    echo -e "${NC}"
    echo ""

    # 交互式配置
    interactive_setup

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
