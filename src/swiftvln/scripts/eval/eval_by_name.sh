#!/bin/bash
# ============================================================================
# Unified VLN Model Evaluation Script
# ============================================================================
# 
# 支持的模型架构: swiftvln
#
# 使用方法:
#   bash src/swiftvln/scripts/eval/eval_by_name.sh <swiftvln_model_name> [options]
#
# 示例:
#   # SwiftVLN 评估 (per_frame, no embedding)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-123456
#
#   # SwiftVLN 评估 (per_frame with random history sampling)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-123456
#   
#   # SwiftVLN 评估 (per_frame with tome, no embedding)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b2.0-tome-s2-noembed-bs64-lr2e-5-123456
#   
#   # SwiftVLN 评估 (GTC, no embedding)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-satnav-3b-1ep-f32s4-overlap16-gtc-k512-noembed-bs64-lr2e-5-123456
#
#   # SwiftVLN 评估 (Pose Embed, additive)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-pose-bs64-lr2e-5-123456
#
#   # SwiftVLN 评估 (SegmentGTC, no embedding)
#   bash src/swiftvln/scripts/eval/eval_by_name.sh swiftvln-satnav-3b-1ep-f32s4-overlap16-sgtc-k512-noembed-bs64-lr2e-5-123456
#
# 环境变量:
#   ENV_TYPE     - habitat (默认) 或 satnav (如果模型名包含 env_type，会自动解析)
#   EVAL_SPLIT   - satnav 默认 val_seen, habitat 默认 val_unseen (可手动覆盖)
#   CUDA_DEVICES - GPU设备 (default: 0,1,2,3,4,5,6,7)
#   MASTER_PORT  - 分布式端口 (default: 29600)
#   MAX_EPISODES - 限制episode数量 (用于调试)
#   SAVE_VIDEO   - 保存视频 (true/false)
#   DRY_RUN      - 仅解析参数，不运行评估 (true/false)
#
# ============================================================================

set -e

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTVLN_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"
VLN_ROOT="${SWIFTVLN_ROOT}/src/swiftvln"
OUTPUT_ROOT="${SWIFTVLN_ROOT}/output"
source "${SCRIPT_DIR}/eval_lib.sh"

# ============================================================================
# 参数检查
# ============================================================================
if [ $# -lt 1 ]; then
    print_error "请提供模型名称!"
    echo ""
    echo "使用方法: bash $0 <model_name> [options]"
    echo ""
    echo "示例:"
    echo "  bash $0 swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-123456"
    exit 1
fi

MODEL_NAME="$1"

# 检查是否为 dry-run 模式
DRY_RUN="${DRY_RUN:-false}"
CHECK_ONLY="${CHECK_ONLY:-false}"
if [ "$DRY_RUN" == "true" ]; then
    print_info "DRY-RUN 模式: 仅解析参数，不运行评估"
fi
if [ "$CHECK_ONLY" == "true" ]; then
    print_info "CHECK-ONLY 模式: 检查eval脚本和参数配置，不运行评估"
fi

# ============================================================================
# 解析模型架构
# ============================================================================
parse_model_arch() {
    local name="$1"
    
    if [[ "$name" == swiftvln-* ]]; then
        echo "swiftvln"
    else
        echo ""
    fi
}

MODEL_ARCH=$(parse_model_arch "$MODEL_NAME")

if [ -z "$MODEL_ARCH" ]; then
    print_error "无法解析模型架构! 模型名称必须以 swiftvln- 开头"
    print_error "输入的模型名称: $MODEL_NAME"
    exit 1
fi

print_info "检测到模型架构: ${MODEL_ARCH}"

# ============================================================================
# 解析模型参数 (基于EXP_NAME格式)
# ============================================================================
# 新格式 (带 env_type):
# SwiftVLN (per_frame):   swiftvln-{env_type}-[qwen3vl-]{model_size}-{epochs}ep-f{num_frames}s{num_future_steps}-overlap{num_overlap}-pf-h{num_history}[-nomem][-random]-b{log_base}-{method}-s{compress_stride}[-initial]-{embed_slot}-bs{batch_size}-lr{learning_rate}-{timestamp}
# SwiftVLN (gtc):         swiftvln-{env_type}-{model_size}-{epochs}ep-f{num_frames}s{num_future_steps}-overlap{num_overlap}-gtc-k{output_tokens}[-initial]-{embed_slot}-bs{batch_size}-lr{learning_rate}-{timestamp}
# SwiftVLN (segment_gtc): swiftvln-{env_type}-{model_size}-{epochs}ep-f{num_frames}s{num_future_steps}-overlap{num_overlap}-sgtc-k{output_tokens}[-initial]-{embed_slot}-bs{batch_size}-lr{learning_rate}-{timestamp}
#   embed_slot: noembed | pose | posefilm | uav | pose+uav | posefilm+uav

parse_swiftvln_params() {
    local name="$1"
    # 新格式 (map):         swiftvln-satnav-3b-1ep-f32s4-overlap16-map-g1000-l400-r448-d20-s2[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # 新格式 (per_frame):   swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # random 示例:          swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-b1.0-pool-s2[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # no-memory 示例:       swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h0-nomem-b1.0-pool-s2[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # 新格式 (gtc):         swiftvln-satnav-3b-1ep-f32s4-overlap16-gtc-k512[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # 新格式 (segment_gtc): swiftvln-satnav-3b-1ep-f32s4-overlap16-sgtc-k512[-initial]-{embed_slot}-bs64-lr2e-5-123456
    # embed_slot: noembed | pose | posefilm | uav | pose+uav | posefilm+uav
    # 注: -initial 是可选的，vanilla 模式下不显示（默认）
    
    local model_size=$(echo "$name" | grep -oP '\d+[bB](?=-\d+ep)' | head -1)
    local model_family="qwen2_5_vl"
    if [[ "$name" == *"-qwen3vl-"* ]]; then
        model_family="qwen3_vl"
    fi
    local epochs=$(echo "$name" | sed -n 's/.*-\([0-9]*\)ep-.*$/\1/p')
    
    # 新格式: f{num_frames}s{num_future_steps} (不含 h)
    local frames_steps=$(echo "$name" | grep -oP 'f\d+s\d+' | head -1)
    local num_frames=$(echo "$frames_steps" | sed -n 's/f\([0-9]*\)s.*/\1/p')
    local num_future_steps=$(echo "$frames_steps" | sed -n 's/.*s\([0-9]*\)$/\1/p')
    
    local num_overlap=$(echo "$name" | sed -n 's/.*-overlap\([0-9]*\)-.*$/\1/p')
    local batch_size=$(echo "$name" | sed -n 's/.*-bs\([0-9]*\)-.*$/\1/p')
    local learning_rate=$(echo "$name" | grep -oP 'lr\d+e-\d+' | sed 's/lr//')
    
    # 解析 system_prompt_setting: 检查 -initial 后缀
    local system_prompt_setting="vanilla"
    if [[ "$name" == *"-initial-"* ]]; then
        system_prompt_setting="initial"
    fi
    local memory_method="history"
    local map_global_side_m=""
    local map_local_side_m=""
    local map_render_px=""
    local map_mask_method=""
    
    # 解析历史处理器类型和相关参数
    local history_processor_type="per_frame"
    local num_history="8"
    local log_base="1.0"
    local compress_stride="2"
    local use_random="false"
    local use_tome="false"
    local gtc_output_tokens=""
    local use_pose_embed="false"
    local pose_fusion_method="additive"
    
    if [[ "$name" == *"-map-g"* ]]; then
        local map_block
        map_block=$(echo "$name" | grep -oP 'map-g[^-]+-l[^-]+-r\d+-[^-]+-s\d+' | head -1)
        memory_method="map"
        history_processor_type="per_frame"
        use_tome="false"
        if [ -n "$map_block" ]; then
            map_global_side_m=$(echo "$map_block" | sed -n 's/.*map-g\([^-]*\)-l.*/\1/p')
            map_local_side_m=$(echo "$map_block" | sed -n 's/.*-l\([^-]*\)-r.*/\1/p')
            map_render_px=$(echo "$map_block" | sed -n 's/.*-r\([0-9]*\)-.*/\1/p')
            compress_stride=$(echo "$map_block" | sed -n 's/.*-s\([0-9]*\)$/\1/p')
            local map_mask_tag
            map_mask_tag=$(echo "$map_block" | sed -n 's/.*-r[0-9]*-\([^-]*\)-s[0-9]*$/\1/p')
            if [[ "$map_mask_tag" == d* ]]; then
                map_mask_method="dilate${map_mask_tag#d}"
            else
                map_mask_method="$map_mask_tag"
            fi
        fi
    elif [[ "$name" == *"-sgtc-k"* ]]; then
        history_processor_type="segment_gtc"
        gtc_output_tokens=$(echo "$name" | grep -oP 'sgtc-k\d+' | sed 's/sgtc-k//')
    elif [[ "$name" == *"-gtc-k"* ]]; then
        history_processor_type="gtc"
        gtc_output_tokens=$(echo "$name" | grep -oP 'gtc-k\d+' | sed 's/gtc-k//')
    elif [[ "$name" == *"-pf-h"* ]]; then
        history_processor_type="per_frame"
        num_history=$(echo "$name" | grep -oP 'pf-h\d+' | sed 's/pf-h//')
        if [[ "$name" == *"-random-"* ]]; then
            use_random="true"
        fi
        local parsed_log_base=""
        parsed_log_base=$(echo "$name" | grep -oP '\-b[0-9.]+\-' | sed 's/-b//' | sed 's/-//' || true)
        if [ -n "$parsed_log_base" ]; then
            log_base="$parsed_log_base"
        fi
        compress_stride=$(echo "$name" | grep -oP '\-(pool|tome)\-s\d+' | grep -oP 's\d+' | sed 's/s//')
        if [[ "$name" == *"-tome-s"* ]]; then
            use_tome="true"
        fi
    fi

    # 解析 embedding enhancement slot
    # 匹配顺序: posefilm > pose > noembed
    if [[ "$name" == *"-posefilm-"* ]]; then
        use_pose_embed="true"
        pose_fusion_method="film"
    elif [[ "$name" == *"-pose-"* ]]; then
        use_pose_embed="true"
        pose_fusion_method="additive"
    elif [[ "$name" == *"-noembed-"* ]]; then
        use_pose_embed="false"
    fi
    
    echo "MODEL_FAMILY=$model_family"
    echo "MODEL_SIZE=$model_size"
    echo "NUM_EPOCHS=$epochs"
    echo "NUM_FRAMES=$num_frames"
    echo "NUM_HISTORY=$num_history"
    echo "NUM_FUTURE_STEPS=$num_future_steps"
    echo "NUM_OVERLAP=$num_overlap"
    echo "MEMORY_METHOD=$memory_method"
    echo "HISTORY_PROCESSOR_TYPE=$history_processor_type"
    echo "LOG_BASE=$log_base"
    echo "USE_RANDOM=$use_random"
    echo "COMPRESS_STRIDE=$compress_stride"
    echo "USE_TOME=$use_tome"
    echo "GTC_OUTPUT_TOKENS=$gtc_output_tokens"
    echo "MAP_GLOBAL_SIDE_M=$map_global_side_m"
    echo "MAP_LOCAL_SIDE_M=$map_local_side_m"
    echo "MAP_RENDER_PX=$map_render_px"
    echo "MAP_MASK_METHOD=$map_mask_method"
    echo "SYSTEM_PROMPT_SETTING=$system_prompt_setting"
    echo "USE_POSE_EMBED=$use_pose_embed"
    echo "POSE_FUSION_METHOD=$pose_fusion_method"
    echo "BATCH_SIZE=$batch_size"
    echo "LEARNING_RATE=$learning_rate"
}

print_swiftvln_params() {
    if [ "$MODEL_ARCH" != "swiftvln" ]; then
        return
    fi

    echo "NUM_OVERLAP:    ${NUM_OVERLAP:-N/A}"
    echo "MEMORY_METHOD:  ${MEMORY_METHOD:-history}"
    if [ "${MEMORY_METHOD:-history}" == "map" ]; then
        echo "MAP_GLOBAL_SIDE_M: ${MAP_GLOBAL_SIDE_M:-1000}"
        echo "MAP_LOCAL_SIDE_M: ${MAP_LOCAL_SIDE_M:-400}"
        echo "MAP_RENDER_PX: ${MAP_RENDER_PX:-448}"
        echo "MAP_MASK_METHOD: ${MAP_MASK_METHOD:-dilate20}"
        echo "COMPRESS_STRIDE: ${COMPRESS_STRIDE:-2}"
    else
        echo "HISTORY_PROCESSOR_TYPE: ${HISTORY_PROCESSOR_TYPE:-per_frame}"
    fi
    if [ "${MEMORY_METHOD:-history}" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" == "gtc" ]; then
        echo "GTC_OUTPUT_TOKENS: ${GTC_OUTPUT_TOKENS:-512}"
    elif [ "${MEMORY_METHOD:-history}" != "map" ] && [ "$HISTORY_PROCESSOR_TYPE" == "segment_gtc" ]; then
        echo "SGTC_OUTPUT_TOKENS: ${GTC_OUTPUT_TOKENS:-512}"
        echo "SGTC_NUM_SEGMENTS: 8 (fixed)"
    elif [ "${MEMORY_METHOD:-history}" != "map" ]; then
        echo "NUM_HISTORY:    ${NUM_HISTORY:-8}"
        echo "LOG_BASE:       ${LOG_BASE:-1.0}"
        echo "USE_RANDOM:     ${USE_RANDOM:-false}"
        if [ "${USE_RANDOM:-false}" = "true" ]; then
            echo "SAMPLING_MODE:  random (LOG_BASE metadata only)"
        fi
        echo "COMPRESS_STRIDE: ${COMPRESS_STRIDE:-2}"
        echo "USE_TOME:       ${USE_TOME:-false}"
    fi
    echo "SYSTEM_PROMPT:  ${SYSTEM_PROMPT_SETTING:-vanilla}"
    echo "USE_POSE_EMBED: ${USE_POSE_EMBED:-false}"
    if [ "${USE_POSE_EMBED:-false}" = "true" ]; then
        echo "POSE_FUSION_METHOD: ${POSE_FUSION_METHOD:-additive}"
    fi
}

print_swiftvln_env_assignments() {
    for name in NUM_FRAMES NUM_HISTORY NUM_FUTURE_STEPS COMPRESS_STRIDE NUM_OVERLAP MEMORY_METHOD HISTORY_PROCESSOR_TYPE; do
        if [ -n "${!name:-}" ]; then
            echo "${name}=${!name}"
        fi
    done

    if [ "${MEMORY_METHOD:-history}" == "map" ]; then
        for name in MAP_GLOBAL_SIDE_M MAP_LOCAL_SIDE_M MAP_RENDER_PX MAP_MASK_METHOD; do
            if [ -n "${!name:-}" ]; then
                echo "${name}=${!name}"
            fi
        done
    elif [ "${HISTORY_PROCESSOR_TYPE:-per_frame}" == "gtc" ] || [ "${HISTORY_PROCESSOR_TYPE:-per_frame}" == "segment_gtc" ]; then
        if [ -n "${GTC_OUTPUT_TOKENS:-}" ]; then
            echo "GTC_OUTPUT_TOKENS=${GTC_OUTPUT_TOKENS}"
        fi
        if [ "${HISTORY_PROCESSOR_TYPE:-per_frame}" == "segment_gtc" ]; then
            echo "SGTC_NUM_SEGMENTS=8 (fixed)"
        fi
    else
        for name in LOG_BASE USE_RANDOM USE_TOME; do
            if [ -n "${!name:-}" ]; then
                echo "${name}=${!name}"
            fi
        done
    fi

    if [ -n "${SYSTEM_PROMPT_SETTING:-}" ]; then
        echo "SYSTEM_PROMPT_SETTING=${SYSTEM_PROMPT_SETTING}"
    fi
    if [ "$MODEL_ARCH" == "swiftvln" ] && [ "${USE_POSE_EMBED:-false}" = "true" ]; then
        echo "USE_POSE_EMBED=${USE_POSE_EMBED}"
        echo "POSE_FUSION_METHOD=${POSE_FUSION_METHOD:-additive}"
    fi
}

# 解析 swiftvln 参数
eval "$(parse_swiftvln_params "$MODEL_NAME")"

# 解析环境类型 (从模型名中提取，如果用户没有指定 ENV_TYPE)
PARSED_ENV_TYPE=$(parse_env_type_from_model "$MODEL_NAME")

# 如果用户没有指定 ENV_TYPE，则使用从模型名解析出的值
if [ -z "$ENV_TYPE" ]; then
    ENV_TYPE="$PARSED_ENV_TYPE"
    print_info "从模型名解析环境类型: ${ENV_TYPE}"
else
    print_info "使用用户指定的环境类型: ${ENV_TYPE}"
fi

# ============================================================================
# 打印解析结果
# ============================================================================
echo ""
echo "=============================================="
echo "解析的训练参数"
echo "=============================================="
echo "模型架构:       ${MODEL_ARCH}"
echo "模型名称:       ${MODEL_NAME}"
echo "环境类型:       ${ENV_TYPE} (解析自模型名: ${PARSED_ENV_TYPE})"
echo "模型族:         ${MODEL_FAMILY:-qwen2_5_vl}"
echo "模型大小:       ${MODEL_SIZE:-N/A}"
echo "训练轮数:       ${NUM_EPOCHS:-N/A}"

echo "NUM_FRAMES:     ${NUM_FRAMES:-N/A}"
echo "NUM_HISTORY:    ${NUM_HISTORY:-N/A}"
echo "NUM_FUTURE_STEPS: ${NUM_FUTURE_STEPS:-N/A}"

print_swiftvln_params

echo "BATCH_SIZE:     ${BATCH_SIZE:-N/A}"
echo "LEARNING_RATE:  ${LEARNING_RATE:-N/A}"
echo "=============================================="
echo ""

# ============================================================================
# 检查模型目录和checkpoint
# ============================================================================
MODEL_DIR="${OUTPUT_ROOT}/${MODEL_ARCH}/${MODEL_NAME}"

# DRY-RUN 模式下跳过目录检查
if [ "$DRY_RUN" == "true" ]; then
    print_info "预期模型目录: $MODEL_DIR"
    print_success "DRY-RUN 模式完成，参数解析成功!"
    exit 0
fi

# CHECK-ONLY 模式: 跳过模型检查，但验证eval脚本存在
if [ "$CHECK_ONLY" == "true" ]; then
    print_info "预期模型目录: $MODEL_DIR"
    
    # 检查eval脚本是否存在
    EVAL_SCRIPT="${VLN_ROOT}/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh"
    if [ ! -f "$EVAL_SCRIPT" ]; then
        print_error "找不到eval脚本: $EVAL_SCRIPT"
        exit 1
    fi
    print_success "找到eval脚本: $EVAL_SCRIPT"
    
    # 显示将要传递的环境变量
    echo ""
    echo "=============================================="
    echo "将传递给eval脚本的环境变量"
    echo "=============================================="
    echo "MODEL_PATH=<checkpoint_path>"
    echo "MODEL_FAMILY=${MODEL_FAMILY:-qwen2_5_vl}"
    echo "ENV_TYPE=${ENV_TYPE}"
    echo "EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}"
    echo "CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    echo "MASTER_PORT=${MASTER_PORT:-29600} (实际运行时会自动检测端口占用)"
    print_swiftvln_env_assignments
    echo "SAVE_VIDEO=${SAVE_VIDEO:-false}"
    if [ -n "$MAX_EPISODES" ]; then
        echo "MAX_EPISODES=${MAX_EPISODES}"
    fi
    echo "=============================================="
    echo ""
    print_success "CHECK-ONLY 模式完成，配置检查通过!"
    exit 0
fi

if [ ! -d "$MODEL_DIR" ]; then
    print_error "模型目录不存在: $MODEL_DIR"
    exit 1
fi

print_info "模型目录: $MODEL_DIR"

# 查找最新的checkpoint (按 checkpoint 编号数字排序)
find_latest_checkpoint() {
    local model_dir="$1"
    local latest_checkpoint=""
    
    # 首先在 v*-* 子目录中查找
    for version_dir in "$model_dir"/v*; do
        if [ -d "$version_dir" ]; then
            # 查找 checkpoint-* 目录，按数字排序取最大
            local ckpt
            ckpt=$(ls -d "$version_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
            if [ -n "$ckpt" ] && [ -d "$ckpt" ]; then
                latest_checkpoint="$ckpt"
            fi
        fi
    done
    
    # 如果没有找到，直接在模型目录下查找
    if [ -z "$latest_checkpoint" ]; then
        local ckpt
        ckpt=$(ls -d "$model_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "$ckpt" ] && [ -d "$ckpt" ]; then
            latest_checkpoint="$ckpt"
        fi
    fi
    
    echo "$latest_checkpoint"
}

CHECKPOINT_PATH=$(find_latest_checkpoint "$MODEL_DIR")

if [ -z "$CHECKPOINT_PATH" ]; then
    print_error "在模型目录下找不到checkpoint!"
    print_error "模型目录: $MODEL_DIR"
    print_error "请确保模型训练已完成并保存了checkpoint"
    exit 1
fi

print_success "找到checkpoint: $CHECKPOINT_PATH"

# ============================================================================
# 验证checkpoint完整性 (检查必要文件)
# ============================================================================
check_checkpoint_integrity() {
    local ckpt_path="$1"
    local required_files=("config.json")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$ckpt_path/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    # 检查是否有模型权重文件 (可能是 .safetensors 或 .bin)
    if ! ls "$ckpt_path"/*.safetensors >/dev/null 2>&1 && \
       ! ls "$ckpt_path"/*.bin >/dev/null 2>&1; then
        missing_files+=("model weights (.safetensors or .bin)")
    fi
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        print_error "Checkpoint不完整! 缺少以下文件:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    
    return 0
}

if ! check_checkpoint_integrity "$CHECKPOINT_PATH"; then
    exit 1
fi

print_success "Checkpoint完整性检查通过"

# ============================================================================
# 确定eval脚本路径
# ============================================================================
EVAL_SCRIPT="${VLN_ROOT}/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    print_error "找不到eval脚本: $EVAL_SCRIPT"
    exit 1
fi

print_info "Eval脚本: $EVAL_SCRIPT"

# ============================================================================
# 端口检测和自动切换
# ============================================================================
check_port_available() {
    local port=$1
    # 使用 ss 或 netstat 检查端口是否被占用
    if command -v ss &> /dev/null; then
        ! ss -tuln | grep -q ":${port} "
    elif command -v netstat &> /dev/null; then
        ! netstat -tuln | grep -q ":${port} "
    else
        # 如果没有 ss 或 netstat，尝试用 /dev/tcp 检测
        (echo >/dev/tcp/localhost/$port) 2>/dev/null && return 1 || return 0
    fi
}

find_available_port() {
    local start_port=${1:-29600}
    local max_attempts=100
    local port=$start_port
    
    for ((i=0; i<max_attempts; i++)); do
        if check_port_available $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    
    # 如果找不到可用端口，返回原始端口（让后续程序报错）
    echo $start_port
    return 1
}

# 获取初始端口
INITIAL_PORT="${MASTER_PORT:-29600}"

# 检测端口是否可用，如果不可用则自动切换
if ! check_port_available $INITIAL_PORT; then
    print_warning "端口 ${INITIAL_PORT} 已被占用，正在查找可用端口..."
    AVAILABLE_PORT=$(find_available_port $INITIAL_PORT)
    if [ "$AVAILABLE_PORT" != "$INITIAL_PORT" ]; then
        print_success "找到可用端口: ${AVAILABLE_PORT}"
        MASTER_PORT=$AVAILABLE_PORT
    else
        print_error "无法找到可用端口！请手动指定 MASTER_PORT"
        exit 1
    fi
else
    MASTER_PORT=$INITIAL_PORT
    print_info "端口 ${MASTER_PORT} 可用"
fi

# ============================================================================
# 准备环境变量
# ============================================================================
export MODEL_PATH="$CHECKPOINT_PATH"
export ENV_TYPE="$ENV_TYPE"  # 已在前面从模型名解析或使用用户指定值
export MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5_vl}"

# 确定要评测的 split 列表
# 若用户已显式设置 EVAL_SPLIT，仅跑该 split；否则 SatNav 默认同时跑两个 split，Habitat 默认 val_unseen
EVAL_SPLITS_LIST="$(infer_eval_splits "$ENV_TYPE" "${EVAL_SPLIT:-}")"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT
export SAVE_VIDEO="${SAVE_VIDEO:-false}"
export VIDEO_COMPRESSION="${VIDEO_COMPRESSION:-false}"

# 传递模型特定参数
if [ -n "$NUM_FRAMES" ]; then
    export NUM_FRAMES
fi
if [ -n "$NUM_HISTORY" ]; then
    export NUM_HISTORY
fi
if [ -n "$NUM_FUTURE_STEPS" ]; then
    export NUM_FUTURE_STEPS
fi
if [ -n "$COMPRESS_STRIDE" ]; then
    export COMPRESS_STRIDE
fi
if [ -n "$MAX_EPISODES" ]; then
    export MAX_EPISODES
fi
# SwiftVLN 特有参数
if [ -n "$NUM_OVERLAP" ]; then
    export NUM_OVERLAP
fi
if [ -n "$MEMORY_METHOD" ]; then
    export MEMORY_METHOD
fi
if [ -n "$HISTORY_PROCESSOR_TYPE" ]; then
    export HISTORY_PROCESSOR_TYPE
fi
if [ "$MEMORY_METHOD" == "map" ]; then
    [ -n "$MAP_GLOBAL_SIDE_M" ] && export MAP_GLOBAL_SIDE_M
    [ -n "$MAP_LOCAL_SIDE_M" ] && export MAP_LOCAL_SIDE_M
    [ -n "$MAP_RENDER_PX" ] && export MAP_RENDER_PX
    [ -n "$MAP_MASK_METHOD" ] && export MAP_MASK_METHOD
elif [ "$HISTORY_PROCESSOR_TYPE" == "gtc" ] || [ "$HISTORY_PROCESSOR_TYPE" == "segment_gtc" ]; then
    if [ -n "$GTC_OUTPUT_TOKENS" ]; then
        export GTC_OUTPUT_TOKENS
    fi
else
    # per_frame 参数
    if [ -n "$LOG_BASE" ]; then
        export LOG_BASE
    fi
    if [ -n "$USE_RANDOM" ]; then
        export USE_RANDOM
    fi
    if [ -n "$COMPRESS_STRIDE" ]; then
        export COMPRESS_STRIDE
    fi
    if [ -n "$USE_TOME" ]; then
        export USE_TOME
    fi
fi
# SwiftVLN system prompt setting
if [ -n "$SYSTEM_PROMPT_SETTING" ]; then
    export SYSTEM_PROMPT_SETTING
fi
if [ "$MODEL_ARCH" == "swiftvln" ] && [ "${USE_POSE_EMBED:-false}" = "true" ]; then
    export USE_POSE_EMBED
    export POSE_FUSION_METHOD="${POSE_FUSION_METHOD:-additive}"
fi

# ============================================================================
# 打印评估配置
# ============================================================================
echo ""
echo "=============================================="
echo "评估配置"
echo "=============================================="
echo "模型架构:       ${MODEL_ARCH}"
echo "模型名称:       ${MODEL_NAME}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "模型族:         ${MODEL_FAMILY}"
echo "环境类型:       ${ENV_TYPE}"
echo "评估集:         ${EVAL_SPLITS_LIST}"
echo "CUDA设备:       ${CUDA_DEVICES}"
echo "保存视频:       ${SAVE_VIDEO}"
if [ -n "$MAX_EPISODES" ]; then
    echo "最大Episodes:   ${MAX_EPISODES}"
fi
# SwiftVLN 特有参数
if [ "$MODEL_ARCH" == "swiftvln" ]; then
    echo "--- SwiftVLN Parameters ---"
    print_swiftvln_params
fi
echo "=============================================="
echo ""

# ============================================================================
# 运行评估（逐 split 循环）
# ============================================================================
print_info "开始评估... splits: ${EVAL_SPLITS_LIST}"

cd "$SWIFTVLN_ROOT"
OVERALL_STATUS=0

for _SPLIT in ${EVAL_SPLITS_LIST}; do
    export EVAL_SPLIT="${_SPLIT}"

    echo ""
    echo "=============================================="
    echo "评测 split: ${_SPLIT}"
    echo "=============================================="

    bash "$EVAL_SCRIPT"
    EVAL_STATUS=$?

    echo ""
    echo "----------------------------------------------"
    if [ $EVAL_STATUS -eq 0 ]; then
        print_success "评估完成 [split=${_SPLIT}]!"

        # 格式: results/eval/<arch>/<model>/<split>/<timestamp>/
        SPLIT_RESULTS_DIR="${SWIFTVLN_ROOT}/results/eval/${MODEL_ARCH}/${MODEL_NAME}/${_SPLIT}"
        LATEST_RESULT=$(ls -td ${SPLIT_RESULTS_DIR}/* 2>/dev/null | head -1)

        if [ -n "$LATEST_RESULT" ] && [ -d "$LATEST_RESULT" ]; then
            print_success "评估结果保存在: ${LATEST_RESULT}"
            ls -la "$LATEST_RESULT" 2>/dev/null || true
        else
            print_info "评估结果保存在: ${SWIFTVLN_ROOT}/results/eval/${MODEL_ARCH}/${MODEL_NAME}/${_SPLIT}/"
        fi
    else
        print_error "评估失败 [split=${_SPLIT}]! 退出码: $EVAL_STATUS"
        OVERALL_STATUS=$EVAL_STATUS
    fi
    echo "----------------------------------------------"
done

EVAL_STATUS=$OVERALL_STATUS
