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
# 支持的模型: streamvln, compressvln, overlapvln
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
TRAIN_STAGE="stage1"           # 训练阶段: stage1 或 stage2
MAX_AUTO_FIX_RETRIES="${MAX_AUTO_FIX_RETRIES:-2}"

# Webhook 通知配置
USE_WEBHOOK_NOTIFICATION="${USE_WEBHOOK_NOTIFICATION:-true}"
WEBHOOK_URL="${WEBHOOK_URL:-https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=503b5488-4d70-455d-a5b9-29fc8d7fb797}"
LAST_AUTO_FIX_ACTIONS=""
AUTO_ENQUEUE_EVAL="${AUTO_ENQUEUE_EVAL:-true}"
EVAL_ENQUEUE_SKIP_CHECKPOINT_LOCAL="${EVAL_ENQUEUE_SKIP_CHECKPOINT_LOCAL:-false}"
EVAL_ENQUEUE_RETRIES="${EVAL_ENQUEUE_RETRIES:-3}"
EVAL_ENQUEUE_RETRY_SLEEP="${EVAL_ENQUEUE_RETRY_SLEEP:-3}"
USE_SWANLAB=true
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
MAP_RENDER_PX="${MAP_RENDER_PX:-384}"
MAP_MASK_METHOD="${MAP_MASK_METHOD:-dilate20}"

# QA 混合训练配置
USE_QA_MIXED_TRAINING=false
QA_RATIO=0.15
QA_DATASET="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/data/qa_swift.jsonl"

# Stage2 默认基础模型路径
declare -A STAGE2_DEFAULT_MODELS=(
    [streamvln]="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln/streamvln-3b-1ep-f32h8s4-bs64-lr2e-5-20260126-214851/v0-20260126-214921/checkpoint-1480"
    [compressvln]="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/compressvln/compressvln-3b-1ep-f32h8s4-stride2-bs64-lr2e-5-20260127-101351/v0-20260127-101426/checkpoint-1480"
    [overlapvln]="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-3b-1ep-f32h8s4-overlap16-stride2-bs64-lr2e-5-20260124-214153/v0-20260124-214234/checkpoint-2239"
)

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

resolve_master_port_from_script() {
    local temp_script="$1"
    local port=""

    port="$(sed -n 's/^MASTER_PORT="\${MASTER_PORT:-\([0-9]\+\)}".*/\1/p' "$temp_script" | head -n 1)"
    if [[ -z "$port" ]]; then
        port="$(sed -n 's/^MASTER_PORT=\([0-9]\+\).*/\1/p' "$temp_script" | head -n 1)"
    fi

    echo "$port"
}

is_local_tcp_port_free() {
    local port="$1"
    python - "$port" <<'PY'
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
    python - "$preferred_port" <<'PY'
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
    local temp_script="$1"
    local current_port=""
    local new_port=""

    current_port="$(resolve_master_port_from_script "$temp_script")"
    [[ -z "$current_port" ]] && return 0

    if is_local_tcp_port_free "$current_port"; then
        return 0
    fi

    new_port="$(find_available_master_port "$current_port")" || {
        print_warning "启动前检测到 MASTER_PORT=${current_port} 已占用，但未找到可用替代端口"
        return 1
    }

    if [[ "$new_port" != "$current_port" ]]; then
        sed -i "s/^MASTER_PORT=.*/MASTER_PORT=${new_port}/" "$temp_script" || true
        print_warning "启动前检测到 MASTER_PORT=${current_port} 已占用，切换为 MASTER_PORT=${new_port}"
    fi

    return 0
}

apply_auto_fix_for_train_failure() {
    local log_file="$1"
    local temp_script="$2"
    local fixed=false
    local actions=()

    if grep -qi "dataloader_prefetch_factor can only be set.*dataloader_num_workers > 1" "$log_file"; then
        sed -i "s/^DATALOADER_NUM_WORKERS=.*/DATALOADER_NUM_WORKERS=2/" "$temp_script" || true
        sed -i "s/^DATALOADER_PREFETCH_FACTOR=.*/DATALOADER_PREFETCH_FACTOR=2/" "$temp_script" || true
        fixed=true
        actions+=("set DATALOADER_NUM_WORKERS=2, DATALOADER_PREFETCH_FACTOR=2")
        print_warning "自动修复: 调整 dataloader workers/prefetch"
    fi

    if grep -qi "Your setup doesn't support bf16/gpu" "$log_file"; then
        sed -i "s/--torch_dtype bfloat16/--torch_dtype float16/g" "$temp_script" || true
        fixed=true
        actions+=("replace --torch_dtype bfloat16 -> float16")
        print_warning "自动修复: bf16 -> float16"
    fi

    if grep -qiE "address already in use|Address already in use" "$log_file"; then
        local current_port=""
        local new_port=""
        current_port="$(resolve_master_port_from_script "$temp_script")"
        new_port="$(find_available_master_port "${current_port:-29500}")" || new_port=""
        if [[ -n "$new_port" ]]; then
            sed -i "s/^MASTER_PORT=.*/MASTER_PORT=${new_port}/" "$temp_script" || true
            fixed=true
            actions+=("set MASTER_PORT=${new_port}")
            print_warning "自动修复: 更换 MASTER_PORT=${new_port}"
        else
            print_warning "自动修复失败: 未找到可用 MASTER_PORT"
        fi
    fi

    if grep -qiE "out of memory|CUDA out of memory" "$log_file"; then
        sed -i "s/^BATCH_SIZE=.*/BATCH_SIZE=1/" "$temp_script" || true
        sed -i "s/^GRAD_ACCUM_STEPS=.*/GRAD_ACCUM_STEPS=1/" "$temp_script" || true
        fixed=true
        actions+=("set BATCH_SIZE=1, GRAD_ACCUM_STEPS=1")
        print_warning "自动修复: 降低 batch 配置"
    fi

    if grep -qi "weights trying to be saved contained shared tensors" "$log_file"; then
        if ! grep -q -- "--save_safetensors false" "$temp_script"; then
            sed -i "/\\\$MAX_STEPS_ARG/i\\    --save_safetensors false \\\\" "$temp_script" || true
        fi
        fixed=true
        actions+=("append --save_safetensors false")
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
# 数据集名称映射
# ============================================================================
get_dataset_short_name() {
    local path="$1"
    if [[ "$path" == *"R2R"* ]]; then
        echo "R2R"
    elif [[ "$path" == *"RxR"* ]]; then
        echo "RxR"
    elif [[ "$path" == *"EnvDrop"* ]]; then
        echo "EnvDrop"
    elif [[ "$path" == *"ScaleVLN"* ]]; then
        echo "ScaleVLN"
    elif [[ "$path" == *"satnav"* ]]; then
        echo "SatNav"
    else
        echo "Custom"
    fi
}

# ============================================================================
# 模型默认配置
# ============================================================================
get_default_config() {
    local model="$1"
    
    case "$model" in
        streamvln)
            cat << 'EOF'
# StreamVLN 可配置参数 (代号=默认值)
a) NUM_FRAMES=32           # 视频帧数
b) NUM_HISTORY=8           # 历史帧数
c) NUM_FUTURE_STEPS=4      # 预测动作步数
d) NUM_EPOCHS=1            # 训练轮数
e) LEARNING_RATE=2e-5      # 学习率
f) BATCH_SIZE=8            # 批量大小
g) FREEZE_VIT=false        # 冻结ViT
h) FREEZE_LLM=false        # 冻结LLM
i) FREEZE_ALIGNER=false    # 冻结Aligner
EOF
            ;;
        compressvln)
            cat << 'EOF'
# CompressVLN 可配置参数 (代号=默认值)
a) NUM_FRAMES=32           # 视频帧数
b) NUM_HISTORY=8           # 历史帧数
c) NUM_FUTURE_STEPS=4      # 预测动作步数
d) COMPRESS_STRIDE=2       # 压缩步长 (2=4x, 3=9x)
e) NUM_EPOCHS=1            # 训练轮数
f) LEARNING_RATE=2e-5      # 学习率
g) BATCH_SIZE=8            # 批量大小
h) FREEZE_VIT=false        # 冻结ViT (预计算模式下自动=true)
i) FREEZE_LLM=false        # 冻结LLM
j) FREEZE_ALIGNER=false    # 冻结Aligner
k) USE_PRECOMPUTED_FEATURES=false  # 使用预计算ViT特征 (自动设置FREEZE_VIT=true)
EOF
            ;;
        overlapvln)
            cat << 'EOF'
# OverlapVLN 可配置参数 (代号=默认值) - 滑动窗口重叠压缩
a) NUM_FRAMES=32           # 视频帧数
b) NUM_HISTORY=8           # 历史帧数 (per_frame模式有效)
c) NUM_FUTURE_STEPS=4      # 预测动作步数
d) COMPRESS_STRIDE=2       # 压缩步长 (per_frame: 2=4x, 3=9x, 4=16x)
e) NUM_OVERLAP=16          # 滑动窗口重叠帧数 (stride = num_frames - num_overlap)
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
q) USE_PIXEL_EMBED=false      # 像素坐标增强: true(开启) 或 false(关闭)
r) USE_POSE_EMBED=false       # Pose增强: true(开启) 或 false(关闭)
s) POSE_FUSION_METHOD=additive  # Pose融合方式: additive(默认) 或 film
# 说明: OverlapVLN 没有单独的 USE_MEMORY 开关；如需 no-memory，请用
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
        streamvln)
            mapping=([a]="NUM_FRAMES" [b]="NUM_HISTORY" [c]="NUM_FUTURE_STEPS" [d]="NUM_EPOCHS" [e]="LEARNING_RATE" [f]="BATCH_SIZE" [g]="FREEZE_VIT" [h]="FREEZE_LLM" [i]="FREEZE_ALIGNER")
            ;;
        compressvln)
            mapping=([a]="NUM_FRAMES" [b]="NUM_HISTORY" [c]="NUM_FUTURE_STEPS" [d]="COMPRESS_STRIDE" [e]="NUM_EPOCHS" [f]="LEARNING_RATE" [g]="BATCH_SIZE" [h]="FREEZE_VIT" [i]="FREEZE_LLM" [j]="FREEZE_ALIGNER" [k]="USE_PRECOMPUTED_FEATURES")
            ;;
        overlapvln)
            mapping=([a]="NUM_FRAMES" [b]="NUM_HISTORY" [c]="NUM_FUTURE_STEPS" [d]="COMPRESS_STRIDE" [e]="NUM_OVERLAP" [f]="NUM_EPOCHS" [g]="LEARNING_RATE" [h]="BATCH_SIZE" [i]="FREEZE_VIT" [j]="FREEZE_LLM" [k]="FREEZE_ALIGNER" [l]="USE_TOME" [m]="HISTORY_PROCESSOR_TYPE" [n]="GTC_OUTPUT_TOKENS" [o]="LOG_BASE" [p]="SYSTEM_PROMPT_SETTING" [q]="USE_PIXEL_EMBED" [r]="USE_POSE_EMBED" [s]="POSE_FUSION_METHOD")
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
# 从 Stage1 模型名解析配置参数
# ============================================================================
parse_stage1_config() {
    local model_path="$1"
    local model_name=$(basename "$(dirname "$(dirname "$model_path")")")
    
    # 新格式: f{frames}s{steps} (不含 h)
    # 示例: f32s4-overlap16-pf-h8-b1.0-pool-s2
    # no-memory 示例: f32s4-overlap16-pf-h0-nomem-b1.0-pool-s2
    local frames=$(echo "$model_name" | grep -oP 'f\d+s' | sed 's/f//' | sed 's/s//')
    local steps=$(echo "$model_name" | grep -oP 'f\d+s\d+' | grep -oP 's\d+' | sed 's/s//')
    
    # 解析 overlap (overlapvln)
    local overlap=$(echo "$model_name" | sed -n 's/.*overlap\([0-9]*\).*/\1/p')
    
    # 解析 system_prompt_setting: 检查 -initial 后缀
    local system_prompt_setting="vanilla"
    if [[ "$model_name" == *"-initial-"* ]]; then
        system_prompt_setting="initial"
    fi
    
    # 解析 history_processor_type 和相关参数
    # 新格式: pf-h8-b1.0-pool-s2 或 pf-h8-b2.0-tome-s2
    # no-memory: pf-h0-nomem-b1.0-pool-s2
    # GTC格式: gtc-k512, sgtc-k512
    local history_processor_type="per_frame"
    local history="8"
    local log_base="1.0"
    local stride=""
    local gtc_output_tokens=""
    local use_tome="false"
    local use_pixel_embed="false"
    
    if [[ "$model_name" == *"-sgtc-k"* ]]; then
        # SegmentGTC: sgtc-k512
        history_processor_type="segment_gtc"
        gtc_output_tokens=$(echo "$model_name" | grep -oP 'sgtc-k\d+' | sed 's/sgtc-k//')
    elif [[ "$model_name" == *"-gtc-k"* ]]; then
        # GTC: gtc-k512
        history_processor_type="gtc"
        gtc_output_tokens=$(echo "$model_name" | grep -oP 'gtc-k\d+' | sed 's/gtc-k//')
    elif [[ "$model_name" == *"-pf-h"* ]]; then
        # 新格式 per_frame: pf-h8-b1.0-pool-s2 或 pf-h8-b2.0-tome-s2
        history_processor_type="per_frame"
        
        # 提取 num_history: pf-h{X}-
        history=$(echo "$model_name" | grep -oP 'pf-h\d+' | sed 's/pf-h//')
        
        # 提取 log_base: -b{X.Y}-
        log_base=$(echo "$model_name" | grep -oP '\-b[0-9.]+\-' | sed 's/-b//' | sed 's/-//')
        
        # 提取 compress_stride: -{method}-s{X}
        stride=$(echo "$model_name" | grep -oP '\-(pool|tome)\-s\d+' | grep -oP 's\d+' | sed 's/s//')
        
        # 检查是否使用 tome
        if [[ "$model_name" == *"-tome-s"* ]]; then
            use_tome="true"
        fi
    fi

    # 解析 embedding enhancement slot (统一格式)
    local use_pose_embed="false"
    local pose_fusion_method="additive"
    
    if [[ "$model_name" == *"-pixel+posefilm-"* ]]; then
        use_pixel_embed="true"
        use_pose_embed="true"
        pose_fusion_method="film"
    elif [[ "$model_name" == *"-pixel+pose-"* ]]; then
        use_pixel_embed="true"
        use_pose_embed="true"
        pose_fusion_method="additive"
    elif [[ "$model_name" == *"-posefilm-"* ]]; then
        use_pixel_embed="false"
        use_pose_embed="true"
        pose_fusion_method="film"
    elif [[ "$model_name" == *"-pose-"* ]]; then
        use_pixel_embed="false"
        use_pose_embed="true"
        pose_fusion_method="additive"
    elif [[ "$model_name" == *"-pixel-"* ]]; then
        use_pixel_embed="true"
    elif [[ "$model_name" == *"-noembed-"* ]]; then
        use_pixel_embed="false"
    fi
    
    # 返回解析结果 (格式: frames|history|steps|stride|overlap|use_tome|history_processor_type|gtc_output_tokens|log_base|system_prompt_setting|use_pixel_embed|use_pose_embed|pose_fusion_method)
    echo "${frames}|${history}|${steps}|${stride}|${overlap}|${use_tome}|${history_processor_type}|${gtc_output_tokens}|${log_base}|${system_prompt_setting}|${use_pixel_embed}|${use_pose_embed}|${pose_fusion_method}"
}

# ============================================================================
# 格式化配置显示字符串
# ============================================================================
format_config_display() {
    local frames="$1"
    local history="$2"
    local steps="$3"
    local stride="$4"
    local overlap="$5"
    local use_tome="$6"
    local history_processor_type="${7:-per_frame}"
    local gtc_output_tokens="$8"
    local log_base="${9:-1.0}"
    local system_prompt_setting="${10:-vanilla}"
    local use_pixel_embed="${11:-false}"
    local use_pose_embed="${12:-false}"
    local pose_fusion_method="${13:-additive}"
    
    local config_str=""
    [[ -n "$frames" ]] && config_str+="f${frames}"
    [[ -n "$steps" ]] && config_str+="s${steps}"
    [[ -n "$overlap" ]] && config_str+="-overlap${overlap}"
    
    # 历史处理方式
    if [[ "$history_processor_type" == "segment_gtc" ]]; then
        config_str+="-sgtc-k${gtc_output_tokens:-512}"
    elif [[ "$history_processor_type" == "gtc" ]]; then
        config_str+="-gtc-k${gtc_output_tokens:-512}"
    else
        config_str+="-pf-h${history:-8}"
        if [[ "${history:-8}" == "0" ]]; then
            config_str+="-nomem"
        fi
        config_str+="-b${log_base}"
        if [[ "$use_tome" == "true" ]]; then
            config_str+="-tome"
        else
            config_str+="-pool"
        fi
        [[ -n "$stride" ]] && config_str+="-s${stride}"
    fi
    
    if [[ "$system_prompt_setting" == "initial" ]]; then
        config_str+="-initial"
    fi

    # Combined embedding enhancement slot
    local embed_parts=()
    [[ "$use_pixel_embed" == "true" ]] && embed_parts+=("pixel")
    if [[ "$use_pose_embed" == "true" ]]; then
        if [[ "$pose_fusion_method" == "film" ]]; then
            embed_parts+=("posefilm")
        else
            embed_parts+=("pose")
        fi
    fi
    if [[ ${#embed_parts[@]} -gt 0 ]]; then
        config_str+="-$(IFS='+'; echo "${embed_parts[*]}")"
    else
        config_str+="-noembed"
    fi
    
    echo "$config_str"
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
    #   EXPERIMENTS=("model|config|changes|ds_names|ds_paths||qa_ratio" ...)
    #   TRAIN_STAGE="stage1"   (or "stage2")
    #   ENV_TYPE="satnav"      (or "habitat")
    #
    # Optional:
    #   SWANLAB_PROJECT="YourProject"   # train_queue 默认强制启用 SwanLab
    #   USE_QA_MIXED_TRAINING="false"
    #   QA_DATASET="..."
    if [[ -n "${TRAIN_EXPERIMENTS_FILE:-}" ]]; then
        if [[ ! -f "$TRAIN_EXPERIMENTS_FILE" ]]; then
            print_error "TRAIN_EXPERIMENTS_FILE 指定的文件不存在: $TRAIN_EXPERIMENTS_FILE"
            exit 1
        fi
        print_info "非交互模式：从文件加载实验配置 → $TRAIN_EXPERIMENTS_FILE"
        # shellcheck source=/dev/null
        source "$TRAIN_EXPERIMENTS_FILE"
        if [[ "${USE_SWANLAB:-true}" != "true" ]]; then
            print_warning "TRAIN_EXPERIMENTS_FILE 中的 USE_SWANLAB=${USE_SWANLAB} 将被忽略，train_queue 现统一强制启用 SwanLab"
        fi
        USE_SWANLAB=true
        SWANLAB_PROJECT="${SWANLAB_PROJECT:-SatNav}"
        if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
            print_error "TRAIN_EXPERIMENTS_FILE 加载后 EXPERIMENTS 数组为空，请检查文件内容"
            exit 1
        fi
        print_success "已加载 ${#EXPERIMENTS[@]} 个实验，stage=${TRAIN_STAGE}, env=${ENV_TYPE}, SwanLab=${SWANLAB_PROJECT}"
        return 0
    fi
    # ─────────────────────────────────────────────────────────────────────────

    print_header "╔══════════════════════════════════════════════════════════════╗"
    echo -e "         ${BOLD}VLN 串行训练配置向导${NC}"
    print_header "╚══════════════════════════════════════════════════════════════╝"
    
    # 1. SwanLab 配置
    print_header "📊 Step 1: SwanLab 配置"
    print_info "train_queue 现统一启用 SwanLab 记录实验"
    read -p "SwanLab Project 名称 [${SWANLAB_PROJECT}]: " swanlab_project
    SWANLAB_PROJECT=${swanlab_project:-$SWANLAB_PROJECT}
    print_success "SwanLab: 启用, Project: $SWANLAB_PROJECT"
    
    # 2. 选择模型
    print_header "🤖 Step 2: 选择训练模型"
    echo "可选模型:"
    echo "  a) streamvln"
    echo "  b) compressvln"
    echo "  c) overlapvln"
    echo ""
    echo "示例: a,b 或 c 或 a,b,c"
    read -p "请选择模型 (逗号分隔) [c]: " models_input
    models_input=${models_input:-c}
    
    if [[ -z "$models_input" ]]; then
        print_error "未选择任何模型!"
        exit 1
    fi
    
    # 展开模型代号
    declare -A model_mapping=([a]="streamvln" [b]="compressvln" [c]="overlapvln")
    SELECTED_MODELS=()
    IFS=',' read -ra model_codes <<< "$models_input"
    for code in "${model_codes[@]}"; do
        code=$(echo "$code" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
        if [[ -n "${model_mapping[$code]}" ]]; then
            SELECTED_MODELS+=("${model_mapping[$code]}")
        elif [[ "$code" =~ ^(streamvln|compressvln|overlapvln)$ ]]; then
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
    
    # 3. 训练阶段选择
    print_header "🎯 Step 3: 训练阶段"
    echo "可选阶段:"
    echo "  a) stage1 - 从基础 Qwen 模型训练 (默认)"
    echo "  b) stage2 - 从已训练的 VLN 模型继续训练"
    echo ""
    read -p "选择训练阶段 [a]: " stage_input
    stage_input=${stage_input:-a}
    
    case "$stage_input" in
        a|A|stage1|1)
            TRAIN_STAGE="stage1"
            ;;
        b|B|stage2|2)
            TRAIN_STAGE="stage2"
            ;;
        *)
            print_warning "未知阶段: $stage_input, 使用默认 stage1"
            TRAIN_STAGE="stage1"
            ;;
    esac
    print_success "训练阶段: $TRAIN_STAGE"
    
    # 4. 环境类型
    print_header "🌍 Step 4: 环境类型"
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
    
    # 5. 数据集配置
    print_header "📁 Step 5: 数据集配置"
    
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
        local default_satnav_path="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data"
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
    
    # 5.5. QA 混合训练配置 (仅 SatNav 环境)
    # QA_RATIOS 数组存储所有要测试的比例，0 表示不使用 QA
    QA_RATIOS=()
    
    if [[ "$ENV_TYPE" == "satnav" ]]; then
        print_header "🔀 Step 5.5: QA 混合训练配置"
        echo "QA 数据可以帮助模型更好地理解地标和位置"
        echo "QA 数据路径: $QA_DATASET"
        echo ""
        echo "QA_RATIO 说明: 控制 QA 数据在训练集中的比例"
        echo "  0    = 不使用 QA (仅 VLN)"
        echo "  0.15 = 15% QA + 85% VLN (推荐)"
        echo "  0.20 = 20% QA + 80% VLN"
        echo ""
        echo -e "${YELLOW}提示: 可输入多个比例用分号分隔，将分别训练${NC}"
        echo "  示例: 0;0.15 = 分别训练 [无QA] 和 [15% QA] 两个版本"
        echo ""
        read -p "输入 QA 比例 [0.15]: " qa_ratio_input
        qa_ratio_input=${qa_ratio_input:-0.15}
        
        # 解析多个比例（分号分隔）
        IFS=';' read -ra ratio_inputs <<< "$qa_ratio_input"
        for ratio in "${ratio_inputs[@]}"; do
            # trim 空白字符
            ratio=$(echo "$ratio" | tr -d '[:space:]')
            # 跳过空字符串
            [[ -z "$ratio" ]] && continue
            # 验证输入是有效数字（0, 1, 0.xx, .xx 格式）
            if [[ "$ratio" =~ ^[0-9]*\.?[0-9]+$ ]]; then
                # 检查范围 0-1 (使用 awk 替代 bc)
                if awk "BEGIN {exit !($ratio >= 0 && $ratio <= 1)}"; then
                    QA_RATIOS+=("$ratio")
                else
                    print_warning "无效的比例值: $ratio (应在 0-1 范围内)"
                fi
            else
                print_warning "无效的比例值: $ratio (已跳过)"
            fi
        done
        
        # 如果没有有效的比例，使用默认值
        if [[ ${#QA_RATIOS[@]} -eq 0 ]]; then
            QA_RATIOS=("0.15")
            print_warning "未输入有效比例，使用默认 0.15"
        fi
        
        # 显示配置
        if [[ ${#QA_RATIOS[@]} -eq 1 ]]; then
            local ratio="${QA_RATIOS[0]}"
            if [[ "$ratio" == "0" ]]; then
                print_success "QA 混合训练: 禁用 (仅 VLN)"
            else
                local qa_pct=$(awk "BEGIN {printf \"%.0f\", $ratio * 100}")
                print_success "QA 混合训练: 启用 (${qa_pct}% QA)"
            fi
        else
            print_success "QA 混合训练: ${#QA_RATIOS[@]} 组配置"
            for ratio in "${QA_RATIOS[@]}"; do
                if [[ "$ratio" == "0" ]]; then
                    echo "  - 无 QA (仅 VLN)"
                else
                    local qa_pct=$(awk "BEGIN {printf \"%.0f\", $ratio * 100}")
                    echo "  - ${qa_pct}% QA + $((100 - qa_pct))% VLN"
                fi
            done
        fi
    else
        # 非 satnav 环境不使用 QA
        QA_RATIOS=("0")
    fi
    
    # Stage1: 配置实验参数
    if [[ "$TRAIN_STAGE" == "stage1" ]]; then
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
            
            # 组合模型配置、数据集配置与 QA 比例
            for model_cfg in "${model_configs[@]}"; do
                changes=$(get_experiment_changes "$model_cfg")
                for ds_config in "${DATASET_CONFIGS[@]}"; do
                    IFS='|' read -r ds_names ds_paths <<< "$ds_config"
                    for qa_ratio in "${QA_RATIOS[@]}"; do
                        EXPERIMENTS+=("${model}|${model_cfg}|${changes}|${ds_names}|${ds_paths}||${qa_ratio}")
                    done
                done
            done
        done
    fi
    
    # Stage2: 选择基础模型（参数从模型名自动解析）
    if [[ "$TRAIN_STAGE" == "stage2" ]]; then
        print_header "🔧 Step 6: 选择基础模型"
        echo "Stage2 从已训练的模型继续训练，参数将从模型名自动解析"
        echo -e "${YELLOW}提示: 可以输入多个模型，用分号 ; 分隔${NC}"
        echo ""
        
        for model in "${SELECTED_MODELS[@]}"; do
            model=$(echo "$model" | tr -d ' ')
            local default_path="${STAGE2_DEFAULT_MODELS[$model]}"
            
            echo -e "${BOLD}${MAGENTA}━━━ $model 基础模型选择 ━━━${NC}"
            
            # 列出可用的已训练模型
            local output_dir="${SWIFTVLN_ROOT}/output/${model}"
            declare -a available_models=()
            declare -a available_model_names=()
            
            if [[ -d "$output_dir" ]]; then
                echo "已有模型:"
                local idx=1
                # 直接遍历 output/{model}/ 下的模型目录
                while IFS= read -r -d '' model_dir; do
                    local exp_name=$(basename "$model_dir")
                    # 查找最新的 checkpoint (参考 eval_by_name.sh 的逻辑)
                    local checkpoint=$(find_latest_checkpoint "$model_dir")
                    if [[ -n "$checkpoint" ]]; then
                        local ckpt_name=$(basename "$checkpoint")
                        available_models+=("$checkpoint")
                        available_model_names+=("$exp_name")
                        echo "  $idx) $exp_name ($ckpt_name)"
                        ((idx++))
                    fi
                done < <(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z -r)
                echo ""
            fi
            
            if [[ -n "$default_path" ]]; then
                local default_name=$(basename "$(dirname "$(dirname "$default_path")")")
                local default_ckpt=$(basename "$default_path")
                echo "默认: $default_name ($default_ckpt)"
            fi
            echo ""
            echo "输入方式:"
            echo "  - 直接回车: 使用默认模型"
            echo "  - 数字: 选择上面列出的模型 (如: 1 或 1;2;3)"
            echo "  - 模型名: 输入完整模型名 (如: ${model}-3b-1ep-f32h8s4-...)"
            echo "  - 路径: 输入完整 checkpoint 路径"
            echo ""
            read -p "[$model] 选择基础模型: " base_models_input
            
            # 解析用户输入
            local base_model_paths=()
            if [[ -z "$base_models_input" ]]; then
                # 使用默认
                if [[ -n "$default_path" ]]; then
                    base_model_paths+=("$default_path")
                else
                    print_warning "$model 没有默认基础模型，跳过"
                    continue
                fi
            else
                IFS=';' read -ra inputs <<< "$base_models_input"
                for input in "${inputs[@]}"; do
                    input=$(echo "$input" | xargs)  # trim
                    if [[ "$input" =~ ^[0-9]+$ ]]; then
                        # 数字索引
                        local idx=$((input - 1))
                        if [[ $idx -ge 0 && $idx -lt ${#available_models[@]} ]]; then
                            base_model_paths+=("${available_models[$idx]}")
                        else
                            print_warning "无效索引: $input"
                        fi
                    elif [[ -d "$input" ]]; then
                        # 完整路径 - 检查是模型目录还是 checkpoint 目录
                        if [[ "$input" == *"/checkpoint-"* ]]; then
                            # 已经是 checkpoint 路径
                            base_model_paths+=("$input")
                        else
                            # 是模型目录，查找最新 checkpoint
                            local checkpoint=$(find_latest_checkpoint "$input")
                            if [[ -n "$checkpoint" ]]; then
                                base_model_paths+=("$checkpoint")
                            else
                                print_warning "目录 $input 没有找到 checkpoint"
                            fi
                        fi
                    elif [[ -d "${output_dir}/${input}" ]]; then
                        # 模型名匹配 - 查找最新的 checkpoint
                        local matched_dir="${output_dir}/${input}"
                        local checkpoint=$(find_latest_checkpoint "$matched_dir")
                        if [[ -n "$checkpoint" ]]; then
                            base_model_paths+=("$checkpoint")
                            print_info "找到 checkpoint: $(basename "$checkpoint")"
                        else
                            print_warning "模型 $input 没有找到 checkpoint"
                        fi
                    else
                        print_warning "无效输入: $input"
                    fi
                done
            fi
            
            # 为每个基础模型创建实验
            for base_path in "${base_model_paths[@]}"; do
                # 解析配置
                local parsed_config=$(parse_stage1_config "$base_path")
                IFS='|' read -r frames history steps stride overlap use_tome history_processor_type gtc_output_tokens log_base system_prompt_setting use_pixel_embed use_pose_embed pose_fusion_method <<< "$parsed_config"
                
                # 构建继承的配置字符串
                local inherited_config=""
                [[ -n "$frames" ]] && inherited_config+="NUM_FRAMES=$frames,"
                [[ -n "$history" ]] && inherited_config+="NUM_HISTORY=$history,"
                [[ -n "$steps" ]] && inherited_config+="NUM_FUTURE_STEPS=$steps,"
                [[ -n "$stride" ]] && inherited_config+="COMPRESS_STRIDE=$stride,"
                [[ -n "$overlap" ]] && inherited_config+="NUM_OVERLAP=$overlap,"
                [[ "$use_tome" == "true" ]] && inherited_config+="USE_TOME=true,"
                [[ -n "$history_processor_type" ]] && inherited_config+="HISTORY_PROCESSOR_TYPE=$history_processor_type,"
                [[ -n "$gtc_output_tokens" ]] && inherited_config+="GTC_OUTPUT_TOKENS=$gtc_output_tokens,"
                [[ -n "$log_base" ]] && inherited_config+="LOG_BASE=$log_base,"
                [[ -n "$system_prompt_setting" ]] && inherited_config+="SYSTEM_PROMPT_SETTING=$system_prompt_setting,"
                [[ -n "$use_pixel_embed" ]] && inherited_config+="USE_PIXEL_EMBED=$use_pixel_embed,"
                [[ -n "$use_pose_embed" ]] && inherited_config+="USE_POSE_EMBED=$use_pose_embed,"
                [[ -n "$pose_fusion_method" ]] && inherited_config+="POSE_FUSION_METHOD=$pose_fusion_method,"
                inherited_config="${inherited_config%,}"  # 去掉末尾逗号
                
                local model_display=$(basename "$(dirname "$(dirname "$base_path")")")
                local config_display=$(format_config_display "$frames" "$history" "$steps" "$stride" "$overlap" "$use_tome" "$history_processor_type" "$gtc_output_tokens" "$log_base" "$system_prompt_setting" "$use_pixel_embed" "$use_pose_embed" "$pose_fusion_method")
                print_success "添加: $model_display"
                echo "  解析参数:"
                [[ -n "$frames" ]] && echo "    NUM_FRAMES=$frames"
                [[ -n "$history" ]] && echo "    NUM_HISTORY=$history"
                [[ -n "$steps" ]] && echo "    NUM_FUTURE_STEPS=$steps"
                [[ -n "$stride" ]] && echo "    COMPRESS_STRIDE=$stride"
                [[ -n "$overlap" ]] && echo "    NUM_OVERLAP=$overlap"
                [[ "$use_tome" == "true" ]] && echo "    USE_TOME=true"
                [[ -n "$history_processor_type" ]] && echo "    HISTORY_PROCESSOR_TYPE=$history_processor_type"
                [[ -n "$gtc_output_tokens" ]] && echo "    GTC_OUTPUT_TOKENS=$gtc_output_tokens"
                [[ -n "$log_base" && "$log_base" != "1.0" ]] && echo "    LOG_BASE=$log_base"
                [[ -n "$system_prompt_setting" && "$system_prompt_setting" != "vanilla" ]] && echo "    SYSTEM_PROMPT_SETTING=$system_prompt_setting"
                [[ -n "$use_pixel_embed" ]] && echo "    USE_PIXEL_EMBED=$use_pixel_embed"
                [[ "$use_pose_embed" == "true" ]] && echo "    USE_POSE_EMBED=$use_pose_embed (fusion=$pose_fusion_method)"
                echo "  配置简写: $config_display"
                
                # 组合数据集配置与 QA 比例
                for ds_config in "${DATASET_CONFIGS[@]}"; do
                    IFS='|' read -r ds_names ds_paths <<< "$ds_config"
                    local changes="基于 ${model_display}"
                    for qa_ratio in "${QA_RATIOS[@]}"; do
                        EXPERIMENTS+=("${model}|${inherited_config}|${changes}|${ds_names}|${ds_paths}|${base_path}|${qa_ratio}")
                    done
                done
            done
            echo ""
        done
    fi
    
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
# 格式化 QA 比例显示
# ============================================================================
format_qa_ratio() {
    local ratio="$1"
    if [[ "$ratio" == "0" ]]; then
        echo "无QA"
    else
        local pct=$(awk "BEGIN {printf \"%.0f\", $ratio * 100}")
        echo "QA${pct}%"
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
    echo "  训练阶段:   $TRAIN_STAGE"
    echo "  环境类型:   $ENV_TYPE"
    if [[ "$ENV_TYPE" == "satnav" && ${#QA_RATIOS[@]} -gt 0 ]]; then
        echo -n "  QA配置:     "
        local qa_display=""
        for r in "${QA_RATIOS[@]}"; do
            qa_display+="$(format_qa_ratio "$r"), "
        done
        echo "${qa_display%, }"
    fi
    echo ""
    
    echo -e "${BOLD}实验列表 (共 ${#EXPERIMENTS[@]} 个实验):${NC}"
    
    if [[ "$TRAIN_STAGE" == "stage1" ]]; then
        echo "┌────┬──────────────┬──────────────────────────────────────┬──────────────────────┬────────┐"
        echo "│ #  │ 模型         │ 配置改动                             │ 数据集               │ QA     │"
        echo "├────┼──────────────┼──────────────────────────────────────┼──────────────────────┼────────┤"
        
        local idx=1
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r model config changes ds_names ds_paths stage2_path qa_ratio <<< "$exp"
            local qa_display=$(format_qa_ratio "$qa_ratio")
            printf "│ %-2d │ %-12s │ %-36s │ %-20s │ %-6s │\n" "$idx" "$model" "${changes:0:36}" "${ds_names:0:20}" "$qa_display"
            ((idx++))
        done
        
        echo "└────┴──────────────┴──────────────────────────────────────┴──────────────────────┴────────┘"
    else
        # Stage2: 显示基础模型信息
        echo "┌────┬──────────────┬────────────────────────────────────────────────────────┬──────────────────────┬────────┐"
        echo "│ #  │ 模型         │ 基础模型                                               │ 数据集               │ QA     │"
        echo "├────┼──────────────┼────────────────────────────────────────────────────────┼──────────────────────┼────────┤"
        
        local idx=1
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r model config changes ds_names ds_paths stage2_path qa_ratio <<< "$exp"
            local base_model_name=""
            if [[ -n "$stage2_path" ]]; then
                base_model_name=$(basename "$(dirname "$(dirname "$stage2_path")")")
            fi
            local qa_display=$(format_qa_ratio "$qa_ratio")
            printf "│ %-2d │ %-12s │ %-54s │ %-20s │ %-6s │\n" "$idx" "$model" "${base_model_name:0:54}" "${ds_names:0:20}" "$qa_display"
            ((idx++))
        done
        
        echo "└────┴──────────────┴────────────────────────────────────────────────────────┴──────────────────────┴────────┘"
    fi
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
    local stage2_path=$7
    local qa_ratio=$8
    
    print_header "🚀 实验 $exp_idx: $model ($TRAIN_STAGE)"
    if [[ "$TRAIN_STAGE" == "stage2" && -n "$stage2_path" ]]; then
        local base_model_name=$(basename "$(dirname "$(dirname "$stage2_path")")")
        echo "基础模型: $base_model_name"
    else
        echo "配置: $changes"
    fi
    echo "数据集: $ds_names"
    echo "环境: $ENV_TYPE"
    echo "QA 配置: $(format_qa_ratio "$qa_ratio")"
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
    local train_script="${VLN_ROOT}/models/${model}/script/train/train_${model}_qwen2_5_vl.sh"
    
    # StreamVLN 使用不同的脚本名
    if [[ "$model" == "streamvln" ]]; then
        train_script="${VLN_ROOT}/models/${model}/script/train/train_${model}_qwen2_5_vl_single_node.sh"
    fi
    
    if [[ ! -f "$train_script" ]]; then
        print_error "找不到训练脚本: $train_script"
        return 1
    fi
    
    # 创建临时脚本（注入配置）
    local temp_script=$(mktemp --suffix=.sh)
    
    # 读取原始脚本并修改
    cat "$train_script" > "$temp_script"
    
    # 注入自定义配置 - 直接替换变量值
    if [[ "$config" != "default" ]]; then
        IFS=',' read -ra config_items <<< "$config"
        for item in "${config_items[@]}"; do
            if [[ -n "$item" && "$item" == *"="* ]]; then
                local var_name="${item%%=*}"
                local var_value="${item#*=}"
                # 转义 sed 特殊字符 (/, &, \) 避免替换失败
                local var_value_escaped=$(printf '%s\n' "$var_value" | sed 's/[&/\]/\\&/g')
                # 替换脚本中的变量赋值
                # 匹配: VAR_NAME=value 或 VAR_NAME="value" 或 VAR_NAME='value'
                sed -i "s/^${var_name}=.*/${var_name}=${var_value_escaped}/" "$temp_script"
                case "$var_name" in
                    MEMORY_METHOD) MEMORY_METHOD="$var_value" ;;
                    MAP_GLOBAL_SIDE_M) MAP_GLOBAL_SIDE_M="$var_value" ;;
                    MAP_LOCAL_SIDE_M) MAP_LOCAL_SIDE_M="$var_value" ;;
                    MAP_RENDER_PX) MAP_RENDER_PX="$var_value" ;;
                    MAP_MASK_METHOD) MAP_MASK_METHOD="$var_value" ;;
                esac
                print_info "配置覆盖: ${var_name}=${var_value}"
            fi
        done
    fi
    
    # 修改训练阶段
    sed -i "s/^TRAIN_STAGE=.*/TRAIN_STAGE=\"$TRAIN_STAGE\"/" "$temp_script"
    print_info "训练阶段: $TRAIN_STAGE"
    
    # 修改环境类型
    sed -i "s/^VLN_ENV_TYPE=.*/VLN_ENV_TYPE=\"$ENV_TYPE\"/" "$temp_script"
    print_info "环境类型: $ENV_TYPE"
    
    # Stage2: 修改基础模型路径
    if [[ "$TRAIN_STAGE" == "stage2" && -n "$stage2_path" ]]; then
        # 转义路径中的特殊字符
        local escaped_path=$(echo "$stage2_path" | sed 's/[\/&]/\\&/g')
        sed -i "s|^STAGE2_MODEL_PATH=.*|STAGE2_MODEL_PATH=\"$stage2_path\"|" "$temp_script"
        print_info "Stage2 基础模型: $stage2_path"
    fi
    
    # 修改 SwanLab 配置
    if [[ "$USE_SWANLAB" == true ]]; then
        sed -i "s/^SWANLAB_PROJECT=.*/SWANLAB_PROJECT=\"$SWANLAB_PROJECT\"/" "$temp_script"
        sed -i "s/^SWANLAB_DIRECT_NETWORK=.*/SWANLAB_DIRECT_NETWORK=\"$SWANLAB_DIRECT_NETWORK\"/" "$temp_script"
        sed -i "s/^USE_SWANLAB=.*/USE_SWANLAB=true/" "$temp_script"
    else
        sed -i "s/^USE_SWANLAB=.*/USE_SWANLAB=false/" "$temp_script"
    fi
    
    # 修改 QA 混合训练配置 (根据实验的 qa_ratio)
    if [[ -n "$qa_ratio" && "$qa_ratio" != "0" ]]; then
        sed -i "s/^USE_QA_MIXED_TRAINING=.*/USE_QA_MIXED_TRAINING=true/" "$temp_script"
        sed -i "s/^QA_RATIO=.*/QA_RATIO=$qa_ratio/" "$temp_script"
        print_info "QA 混合训练: 启用 (比例: $(format_qa_ratio "$qa_ratio"))"
    else
        sed -i "s/^USE_QA_MIXED_TRAINING=.*/USE_QA_MIXED_TRAINING=false/" "$temp_script"
        print_info "QA 混合训练: 禁用"
    fi

    if [[ -n "$QA_DATASET" ]]; then
        sed -i "s|^QA_DATASET=.*|QA_DATASET=\"$QA_DATASET\"|" "$temp_script"
        print_info "QA 数据集: $QA_DATASET"
    fi

    if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
        sed -i "s|^RESUME_FROM_CHECKPOINT=.*|RESUME_FROM_CHECKPOINT=\"$RESUME_FROM_CHECKPOINT\"|" "$temp_script"
        sed -i "s|^RESUME_ONLY_MODEL=.*|RESUME_ONLY_MODEL=\"$RESUME_ONLY_MODEL\"|" "$temp_script"
        print_info "恢复训练: $RESUME_FROM_CHECKPOINT (resume_only_model=$RESUME_ONLY_MODEL)"
    fi

    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        sed -i "s|^OUTPUT_DIR_OVERRIDE=.*|OUTPUT_DIR_OVERRIDE=\"$OUTPUT_DIR_OVERRIDE\"|" "$temp_script"
        print_info "输出目录覆盖: $OUTPUT_DIR_OVERRIDE"
    fi

    sed -i "s|^MEMORY_METHOD=.*|MEMORY_METHOD=\"$MEMORY_METHOD\"|" "$temp_script"
    sed -i "s|^MAP_GLOBAL_SIDE_M=.*|MAP_GLOBAL_SIDE_M=\"$MAP_GLOBAL_SIDE_M\"|" "$temp_script"
    sed -i "s|^MAP_LOCAL_SIDE_M=.*|MAP_LOCAL_SIDE_M=\"$MAP_LOCAL_SIDE_M\"|" "$temp_script"
    sed -i "s|^MAP_RENDER_PX=.*|MAP_RENDER_PX=\"$MAP_RENDER_PX\"|" "$temp_script"
    sed -i "s|^MAP_MASK_METHOD=.*|MAP_MASK_METHOD=\"$MAP_MASK_METHOD\"|" "$temp_script"
    print_info "Memory 配置: method=$MEMORY_METHOD, global=$MAP_GLOBAL_SIDE_M, local=$MAP_LOCAL_SIDE_M, render=$MAP_RENDER_PX, mask=$MAP_MASK_METHOD"
    
    # 修改数据路径 - 根据环境类型替换对应的数组
    local data_array_name="HABITAT_DATA_PATHS"
    [[ "$ENV_TYPE" == "satnav" ]] && data_array_name="SATNAV_DATA_PATHS"
    
    local data_paths_content="${data_array_name}=(\n"
    IFS=',' read -ra path_arr <<< "$ds_paths"
    for path in "${path_arr[@]}"; do
        data_paths_content+="    \"$path\"\n"
    done
    data_paths_content+=")"
    
    # 使用 awk 替换对应的数据路径数组
    awk -v new_content="$data_paths_content" -v array_name="$data_array_name" '
        $0 ~ "^"array_name"=\\(" { 
            print new_content
            in_array=1
            next
        }
        in_array && /^\)/ {
            in_array=0
            next
        }
        !in_array { print }
    ' "$temp_script" > "${temp_script}.tmp" && mv "${temp_script}.tmp" "$temp_script"
    
    # 运行训练
    local start_time=$(date +%s)
    local log_file="${SWIFTVLN_ROOT}/logs/train_queue_${model}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "日志文件: $log_file"
    print_info "开始训练..."
    export TRAIN_CUDA_DEVICES TRAIN_NUM_GPUS TRAIN_DRY_RUN
    
    # 修复 SWIFTVLN_ROOT 路径问题
    # 原始脚本使用 BASH_SOURCE 计算 SWIFTVLN_ROOT，但复制到临时文件后路径会错误
    # 直接硬编码 SWIFTVLN_ROOT 为正确的绝对路径
    sed -i "s|^SWIFTVLN_ROOT=.*|SWIFTVLN_ROOT=\"${SWIFTVLN_ROOT}\"|g" "$temp_script"
    
    # 将相对路径改为绝对路径
    sed -i "s|src/swiftvln/models/${model}/trainer.py|${SWIFTVLN_ROOT}/src/swiftvln/models/${model}/trainer.py|g" "$temp_script"
    sed -i "s|--custom_register_path src/swiftvln/models/${model}|--custom_register_path ${SWIFTVLN_ROOT}/src/swiftvln/models/${model}|g" "$temp_script"

    local attempt=1
    local max_attempts=$((MAX_AUTO_FIX_RETRIES + 1))
    local run_log_file="$log_file"
    local attempted_fixes=""

    # 执行训练脚本（带自动修复重试）
    while true; do
        if [[ $attempt -eq 1 ]]; then
            run_log_file="$log_file"
        else
            run_log_file="${log_file%.log}_retry${attempt}.log"
            print_warning "开始第 ${attempt} 次尝试..."
        fi

        ensure_available_master_port "$temp_script" || true

        bash "$temp_script" 2>&1 | tee "$run_log_file"
        local train_exit_code=${PIPESTATUS[0]}

        if [[ $train_exit_code -eq 0 ]]; then
            break
        fi

        print_warning "训练脚本退出码: ${train_exit_code}"

        if [[ $attempt -lt $max_attempts ]] && apply_auto_fix_for_train_failure "$run_log_file" "$temp_script"; then
            attempted_fixes="${attempted_fixes}\n- attempt ${attempt}: ${LAST_AUTO_FIX_ACTIONS}"
            send_webhook "Train Auto-Fix Retry" "experiment=${exp_idx}\nmodel=${model}\nattempt=${attempt}\nissue_log=${run_log_file}\nsolution=${LAST_AUTO_FIX_ACTIONS}\nresult=retrying next attempt"
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
            python3 -c "
import json, pathlib, os
meta = {}
url = os.environ.get('_SWANLAB_URL', '')
if url:
    meta['swanlab_url'] = url
proj = os.environ.get('_SWANLAB_PROJECT', '')
if proj:
    meta['swanlab_project'] = proj
exp = os.environ.get('_SWANLAB_EXP', '')
if exp:
    meta['swanlab_exp_name'] = exp
if meta:
    out = pathlib.Path(os.environ['_OUTPUT_DIR']) / 'train_metadata.json'
    out.write_text(json.dumps(meta, indent=2))
    print(f'Saved train metadata: {out}')
" 2>/dev/null || true
        fi

        # 格式: idx|model|changes|ds_names|status|duration|exp_name|base_model|qa_ratio
        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|SUCCESS|$duration_str|$exp_name|$stage2_path|$qa_ratio")
        if [[ "$dry_run_completed" == "true" ]]; then
            print_success "实验 $exp_idx Dry Run 完成! 耗时: $duration_str"
            _emit_train_event "EXPERIMENT_DRY_RUN_SUCCESS|${exp_idx}|${total:-0}|${model}|${exp_name}|${run_log_file}|$(date -Iseconds)"
        else
            print_success "实验 $exp_idx 完成! 耗时: $duration_str"
            enqueue_model_for_eval "$exp_name" || true
            send_webhook "Train Success" "experiment=${exp_idx}\nmodel=${model}\nduration=${duration_str}\noutput=${output_path:-N/A}\nlog=${run_log_file}"
            _emit_train_event "EXPERIMENT_SUCCESS|${exp_idx}|${total:-0}|${model}|${exp_name}|${output_path:-N/A}|$(date -Iseconds)"
        fi
        
        rm -f "$temp_script"
        return 0
    else
        local error_msg=$(tail -50 "$run_log_file" | grep -iE "(error|oom|cuda|exception)" | head -5)
        error_msg=${error_msg:-"未知错误"}
        
        EXP_RESULTS+=("$exp_idx|$model|$changes|$ds_names|FAILED|--|--|$stage2_path|$qa_ratio")
        EXP_ERRORS+=("实验 $exp_idx ($model): $error_msg")
        print_error "实验 $exp_idx 失败!"
        send_webhook "Train Failed" "experiment=${exp_idx}\nmodel=${model}\nerror=${error_msg}\nlog=${run_log_file}\nattempted_fixes=${attempted_fixes:-none}\nresult=marked FAILED and continue queue"
        _emit_train_event "EXPERIMENT_FAILED|${exp_idx}|${total:-0}|${model}|unknown|${error_msg:0:200}|${run_log_file}|$(date -Iseconds)"
        
        rm -f "$temp_script"
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
    
    # 同时输出到终端和文件
    {
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "                                               训练结果汇总"
        echo "                                            $(date '+%Y-%m-%d %H:%M:%S')"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo ""
        echo "基础配置:"
        echo "  SwanLab:    $([ "$USE_SWANLAB" = true ] && echo "启用 ($SWANLAB_PROJECT)" || echo "禁用")"
        echo "  训练阶段:   $TRAIN_STAGE"
        echo "  环境类型:   $ENV_TYPE"
        if [[ "$ENV_TYPE" == "satnav" && ${#QA_RATIOS[@]} -gt 0 ]]; then
            echo -n "  QA配置:     "
            local qa_display=""
            for r in "${QA_RATIOS[@]}"; do
                qa_display+="$(format_qa_ratio "$r"), "
            done
            echo "${qa_display%, }"
        fi
        echo ""
        if [[ "$TRAIN_STAGE" == "stage1" ]]; then
            echo "┌────┬──────────────┬──────────────────────────────────────┬──────────────────────┬────────┬─────────┬──────────┐"
            echo "│ #  │ 模型         │ 配置改动                             │ 数据集               │ QA     │ 状态    │ 耗时     │"
            echo "├────┼──────────────┼──────────────────────────────────────┼──────────────────────┼────────┼─────────┼──────────┤"
        else
            echo "┌────┬──────────────┬────────────────────────────────────────────────────────┬──────────────────────┬────────┬─────────┬──────────┐"
            echo "│ #  │ 模型         │ 基础模型                                               │ 数据集               │ QA     │ 状态    │ 耗时     │"
            echo "├────┼──────────────┼────────────────────────────────────────────────────────┼──────────────────────┼────────┼─────────┼──────────┤"
        fi
        
        for result in "${EXP_RESULTS[@]}"; do
            # 格式: idx|model|changes|ds_names|status|duration|exp_name|base_model|qa_ratio
            IFS='|' read -r idx model changes ds_names status duration exp_name base_model qa_ratio <<< "$result"
            local qa_display=$(format_qa_ratio "$qa_ratio")
            if [[ "$TRAIN_STAGE" == "stage1" ]]; then
                printf "│ %-2s │ %-12s │ %-36s │ %-20s │ %-6s │ %-7s │ %-8s │\n" "$idx" "$model" "${changes:0:36}" "${ds_names:0:20}" "$qa_display" "$status" "$duration"
            else
                # Stage2: 显示基础模型名
                local base_model_short="${base_model##*/}"  # 只取模型名部分
                base_model_short=$(basename "$(dirname "$(dirname "$base_model")")" 2>/dev/null || echo "$base_model_short")
                printf "│ %-2s │ %-12s │ %-54s │ %-20s │ %-6s │ %-7s │ %-8s │\n" "$idx" "$model" "${base_model_short:0:54}" "${ds_names:0:20}" "$qa_display" "$status" "$duration"
            fi
        done
        
        if [[ "$TRAIN_STAGE" == "stage1" ]]; then
            echo "└────┴──────────────┴──────────────────────────────────────┴──────────────────────┴────────┴─────────┴──────────┘"
        else
            echo "└────┴──────────────┴────────────────────────────────────────────────────────┴──────────────────────┴────────┴─────────┴──────────┘"
        fi
        
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
        
        # 显示错误详情
        if [[ ${#EXP_ERRORS[@]} -gt 0 ]]; then
            echo ""
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            echo "❌ 错误详情"
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            for error in "${EXP_ERRORS[@]}"; do
                echo "  • $error"
            done
        fi
        
        # 显示成功实验的输出路径
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        echo "📁 成功实验输出目录"
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
        for result in "${EXP_RESULTS[@]}"; do
            IFS='|' read -r idx model changes ds_names status duration exp_name base_model <<< "$result"
            if [[ "$status" == "SUCCESS" ]]; then
                echo "  • 实验 $idx ($model, $ds_names): output/${model}/${exp_name}"
                if [[ "$TRAIN_STAGE" == "stage2" && -n "$base_model" ]]; then
                    local base_model_name=$(basename "$(dirname "$(dirname "$base_model")")" 2>/dev/null || echo "$base_model")
                    echo "    └─ 基础模型: $base_model_name"
                fi
            fi
        done
        
        # Stage2: 显示基础模型详情
        if [[ "$TRAIN_STAGE" == "stage2" ]]; then
            echo ""
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            echo "🔧 Stage2 基础模型详情"
            echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
            local shown_base_models=()
            for result in "${EXP_RESULTS[@]}"; do
                IFS='|' read -r idx model changes ds_names status duration exp_name base_model <<< "$result"
                if [[ -n "$base_model" ]]; then
                    local base_model_name=$(basename "$(dirname "$(dirname "$base_model")")" 2>/dev/null || echo "$base_model")
                    # 避免重复显示
                    local already_shown=false
                    for shown in "${shown_base_models[@]}"; do
                        if [[ "$shown" == "$base_model_name" ]]; then
                            already_shown=true
                            break
                        fi
                    done
                    if [[ "$already_shown" == "false" ]]; then
                        shown_base_models+=("$base_model_name")
                        # 解析并显示配置
                        local parsed=$(parse_stage1_config "$base_model")
                        IFS='|' read -r frames history steps stride overlap use_tome history_processor_type gtc_output_tokens log_base system_prompt_setting use_pixel_embed use_pose_embed pose_fusion_method <<< "$parsed"
                        local config_display=$(format_config_display "$frames" "$history" "$steps" "$stride" "$overlap" "$use_tome" "$history_processor_type" "$gtc_output_tokens" "$log_base" "$system_prompt_setting" "$use_pixel_embed" "$use_pose_embed" "$pose_fusion_method")
                        echo "  • $base_model_name"
                        echo "    └─ 配置: $config_display"
                        echo "    └─ 路径: $base_model"
                    fi
                fi
            done
        fi
        
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
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

    send_webhook "Train Queue Finished" "stage=${TRAIN_STAGE}\nenv=${ENV_TYPE}\nsuccess=${success_count}\nfailed=${fail_count}\ntotal=${#EXP_RESULTS[@]}\nreport=${RESULT_FILE}"

    # 事件日志: QUEUE_DONE
    _emit_train_event "QUEUE_DONE|${success_count}|${fail_count}|${#EXP_RESULTS[@]}|$(date -Iseconds)"

    # 写入机器可读完成状态（供 train_watchdog / 外部工具）
    _write_train_completion_status "$RESULT_FILE" "$success_count" "$fail_count"

    # 带颜色输出到终端（额外显示）
    echo ""
    print_success "结果已保存到: $RESULT_FILE"
}

_write_train_completion_status() {
    local result_file="${1:-}" success_count="${2:-0}" fail_count="${3:-0}"
    local _hostname
    _hostname="$(hostname | sed 's/[^a-zA-Z0-9._-]/_/g')"

    local success_list="" failed_list=""
    for result in "${EXP_RESULTS[@]}"; do
        IFS='|' read -r _idx _model _changes _ds _status _dur exp_name _base _qa <<< "$result"
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
    "train_stage": "${TRAIN_STAGE}",
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
    print_header "🏃 开始串行训练 ($TRAIN_STAGE, $ENV_TYPE)"
    
    local exp_idx=1
    local total=${#EXPERIMENTS[@]}
    
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r model config changes ds_names ds_paths stage2_path qa_ratio <<< "$exp"
        
        echo ""
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
        echo -e "  进度: $exp_idx / $total"
        echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
        
        # 运行实验
        run_experiment "$exp_idx" "$model" "$config" "$changes" "$ds_names" "$ds_paths" "$stage2_path" "$qa_ratio" || true
        
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
