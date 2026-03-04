#!/bin/bash
# StreamVLN 评估波动性测试脚本
# 
# 功能：
#   对同一个模型运行多次评估，统计结果的波动性（均值、标准差、最大/最小值）
#
# 使用方法：
#   bash src/swiftvln/models/streamvln/script/eval/eval_variance.sh --model /path/to/checkpoint --runs 5
#   bash src/swiftvln/models/streamvln/script/eval/eval_variance.sh -m /path/to/checkpoint -n 10 --split val_seen
#
# 参数：
#   -m, --model     模型checkpoint路径（必需）
#   -n, --runs      评估运行次数（默认: 3）
#   -s, --split     评估数据集split（默认: val_unseen）
#   -g, --gpus      使用的GPU（默认: 0,1,2,3,4,5,6,7）
#   -o, --output    输出目录（默认: ./results/variance）
#   -h, --help      显示帮助信息

set -e

# ============================================================================
# 默认参数
# ============================================================================
MODEL_PATH=""
NUM_RUNS=3
EVAL_SPLIT="val_unseen"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
OUTPUT_BASE_DIR="./results/variance/streamvln"
BASE_PORT=29600  # 起始端口号
ENV_TYPE="${ENV_TYPE:-habitat}"  # habitat or satnav

# ============================================================================
# 辅助函数：查找可用端口
# ============================================================================
find_available_port() {
    local port=$1
    local max_attempts=100
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # 检查端口是否被占用
        if ! netstat -tuln 2>/dev/null | grep -q ":${port} " && \
           ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        attempt=$((attempt + 1))
    done
    
    # 如果找不到可用端口，返回一个随机端口
    echo $((29600 + RANDOM % 1000))
    return 0
}

# ============================================================================
# 参数解析
# ============================================================================
show_help() {
    echo "StreamVLN 评估波动性测试脚本"
    echo ""
    echo "使用方法："
    echo "  $0 --model /path/to/checkpoint --runs 5"
    echo ""
    echo "参数："
    echo "  -m, --model     模型checkpoint路径（必需）"
    echo "  -n, --runs      评估运行次数（默认: 3）"
    echo "  -s, --split     评估数据集split（默认: val_unseen）"
    echo "  -e, --env-type  环境类型: habitat 或 satnav（默认: habitat）"
    echo "  -g, --gpus      使用的GPU（默认: 0,1,2,3,4,5,6,7）"
    echo "  -o, --output    输出目录（默认: ./results/variance）"
    echo "  -h, --help      显示帮助信息"
    echo ""
    echo "示例："
    echo "  $0 -m /path/to/checkpoint -n 5"
    echo "  $0 --model /path/to/checkpoint --runs 10 --split val_seen"
    echo "  ENV_TYPE=satnav $0 -m /path/to/checkpoint -n 5"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -n|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -s|--split)
            EVAL_SPLIT="$2"
            shift 2
            ;;
        -e|--env-type)
            ENV_TYPE="$2"
            shift 2
            ;;
        -g|--gpus)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 参数验证
# ============================================================================
if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] 必须指定模型路径 (--model)"
    show_help
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 模型路径不存在: $MODEL_PATH"
    exit 1
fi

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [ "$NUM_RUNS" -lt 1 ]; then
    echo "[ERROR] 运行次数必须是正整数: $NUM_RUNS"
    exit 1
fi

# ============================================================================
# 路径设置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_streamvln_qwen2_5_vl_distributed.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] 评估脚本不存在: $EVAL_SCRIPT"
    exit 1
fi

# 提取模型名称
MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/output/streamvln/\([^/]*\)/.*|\1|p')
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

# 创建输出目录
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# 打印配置
# ============================================================================
echo "=============================================="
echo "StreamVLN 评估波动性测试"
echo "=============================================="
echo "模型路径:     $MODEL_PATH"
echo "模型名称:     $MODEL_NAME"
echo "评估次数:     $NUM_RUNS"
echo "数据集Split:  $EVAL_SPLIT"
echo "GPU:          $CUDA_DEVICES"
echo "输出目录:     $OUTPUT_DIR"
echo "=============================================="

# ============================================================================
# 存储结果的数组
# ============================================================================
declare -a SUCCESS_RATES=()
declare -a SPL_VALUES=()
declare -a ORACLE_SUCCESS_VALUES=()
declare -a NAV_ERROR_VALUES=()
declare -a RUN_STATUS=()

# ============================================================================
# 运行多次评估
# ============================================================================
for run_idx in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "=============================================="
    echo "开始第 $run_idx/$NUM_RUNS 次评估"
    echo "=============================================="
    
    # 每次运行使用不同的输出目录
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/run_${run_idx}"
    mkdir -p "$RUN_OUTPUT_DIR"
    
    # 查找可用端口（每次运行递增，避免冲突）
    CURRENT_PORT=$(find_available_port $((BASE_PORT + run_idx - 1)))
    echo "使用端口: $CURRENT_PORT"
    
    # 运行评估
    start_time=$(date +%s)
    
    if ENV_TYPE="$ENV_TYPE" \
       MODEL_PATH="$MODEL_PATH" \
       EVAL_SPLIT="$EVAL_SPLIT" \
       CUDA_DEVICES="$CUDA_DEVICES" \
       OUTPUT_DIR="$RUN_OUTPUT_DIR" \
       MASTER_PORT="$CURRENT_PORT" \
       SAVE_VIDEO="false" \
       bash "$EVAL_SCRIPT" 2>&1 | tee "${RUN_OUTPUT_DIR}/eval.log"; then
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "[SUCCESS] 第 $run_idx 次评估完成，耗时: ${duration}秒"
        RUN_STATUS+=("success")
        
        # 解析结果
        SUMMARY_FILE="${RUN_OUTPUT_DIR}/evaluation_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            # 使用 Python 解析 JSON（更可靠）
            read -r sr spl os ne <<< $(python3 -c "
import json
with open('$SUMMARY_FILE') as f:
    data = json.load(f)
print(data.get('success_rate', 0), data.get('mean_spl', 0), data.get('oracle_success', 0), data.get('navigation_error', 0))
" 2>/dev/null || echo "0 0 0 0")
            
            SUCCESS_RATES+=("$sr")
            SPL_VALUES+=("$spl")
            ORACLE_SUCCESS_VALUES+=("$os")
            NAV_ERROR_VALUES+=("$ne")
            
            echo "  Success Rate:    $sr"
            echo "  SPL:             $spl"
            echo "  Oracle Success:  $os"
            echo "  Nav Error:       $ne"
        else
            echo "[WARNING] 未找到结果文件: $SUMMARY_FILE"
            SUCCESS_RATES+=("0")
            SPL_VALUES+=("0")
            ORACLE_SUCCESS_VALUES+=("0")
            NAV_ERROR_VALUES+=("0")
        fi
    else
        echo "[FAILED] 第 $run_idx 次评估失败"
        RUN_STATUS+=("failed")
        SUCCESS_RATES+=("0")
        SPL_VALUES+=("0")
        ORACLE_SUCCESS_VALUES+=("0")
        NAV_ERROR_VALUES+=("0")
    fi
done

# ============================================================================
# 统计分析
# ============================================================================
echo ""
echo "=============================================="
echo "统计分析"
echo "=============================================="

# 使用 Python 进行统计计算
python3 << EOF
import json
import sys

# 读取数据
success_rates = [float(x) for x in "${SUCCESS_RATES[*]}".split()]
spl_values = [float(x) for x in "${SPL_VALUES[*]}".split()]
oracle_success = [float(x) for x in "${ORACLE_SUCCESS_VALUES[*]}".split()]
nav_errors = [float(x) for x in "${NAV_ERROR_VALUES[*]}".split()]
run_status = "${RUN_STATUS[*]}".split()

num_runs = len(success_rates)
successful_runs = sum(1 for s in run_status if s == "success")

def calc_stats(values, name, is_percentage=False):
    """计算统计量"""
    valid_values = [v for v, s in zip(values, run_status) if s == "success" and v > 0]
    
    if not valid_values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "range": 0}
    
    n = len(valid_values)
    mean = sum(valid_values) / n
    
    if n > 1:
        variance = sum((x - mean) ** 2 for x in valid_values) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    range_val = max_val - min_val
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "values": valid_values
    }

# 计算各指标的统计量
sr_stats = calc_stats(success_rates, "Success Rate", is_percentage=True)
spl_stats = calc_stats(spl_values, "SPL")
os_stats = calc_stats(oracle_success, "Oracle Success", is_percentage=True)
ne_stats = calc_stats(nav_errors, "Navigation Error")

# 打印结果
print(f"\n运行统计: 成功 {successful_runs}/{num_runs} 次")
print("\n" + "="*70)
print(f"{'指标':<20} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12} {'波动范围':>12}")
print("="*70)

def format_val(v, is_pct=False):
    if is_pct:
        return f"{v*100:.2f}%"
    return f"{v:.4f}"

print(f"{'Success Rate':<20} {format_val(sr_stats['mean'], True):>12} {format_val(sr_stats['std'], True):>12} {format_val(sr_stats['min'], True):>12} {format_val(sr_stats['max'], True):>12} {format_val(sr_stats['range'], True):>12}")
print(f"{'SPL':<20} {format_val(spl_stats['mean']):>12} {format_val(spl_stats['std']):>12} {format_val(spl_stats['min']):>12} {format_val(spl_stats['max']):>12} {format_val(spl_stats['range']):>12}")
print(f"{'Oracle Success':<20} {format_val(os_stats['mean'], True):>12} {format_val(os_stats['std'], True):>12} {format_val(os_stats['min'], True):>12} {format_val(os_stats['max'], True):>12} {format_val(os_stats['range'], True):>12}")
print(f"{'Navigation Error':<20} {format_val(ne_stats['mean']):>12} {format_val(ne_stats['std']):>12} {format_val(ne_stats['min']):>12} {format_val(ne_stats['max']):>12} {format_val(ne_stats['range']):>12}")
print("="*70)

# 打印每次运行的详细结果
print("\n各次运行详细结果:")
print("-"*70)
print(f"{'运行':<8} {'状态':<10} {'Success Rate':>14} {'SPL':>12} {'Oracle Succ':>14} {'Nav Error':>12}")
print("-"*70)
for i in range(num_runs):
    status = run_status[i]
    if status == "success":
        print(f"Run {i+1:<4} {'成功':<10} {format_val(success_rates[i], True):>14} {format_val(spl_values[i]):>12} {format_val(oracle_success[i], True):>14} {format_val(nav_errors[i]):>12}")
    else:
        print(f"Run {i+1:<4} {'失败':<10} {'-':>14} {'-':>12} {'-':>14} {'-':>12}")
print("-"*70)

# 波动性评估
print("\n波动性评估:")
if sr_stats['mean'] > 0:
    cv_sr = (sr_stats['std'] / sr_stats['mean']) * 100 if sr_stats['mean'] > 0 else 0
    print(f"  Success Rate 变异系数 (CV): {cv_sr:.2f}%")
    if cv_sr < 1:
        print("  -> 波动性: 极低 (非常稳定)")
    elif cv_sr < 3:
        print("  -> 波动性: 低 (稳定)")
    elif cv_sr < 5:
        print("  -> 波动性: 中等")
    else:
        print("  -> 波动性: 高 (不稳定，建议检查评估设置)")

# 保存结果到 JSON
summary = {
    "model_path": "$MODEL_PATH",
    "model_name": "$MODEL_NAME",
    "eval_split": "$EVAL_SPLIT",
    "num_runs": num_runs,
    "successful_runs": successful_runs,
    "statistics": {
        "success_rate": sr_stats,
        "spl": spl_stats,
        "oracle_success": os_stats,
        "navigation_error": ne_stats
    },
    "all_runs": [
        {
            "run": i + 1,
            "status": run_status[i],
            "success_rate": success_rates[i],
            "spl": spl_values[i],
            "oracle_success": oracle_success[i],
            "navigation_error": nav_errors[i]
        }
        for i in range(num_runs)
    ]
}

with open("${OUTPUT_DIR}/variance_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n详细结果已保存到: ${OUTPUT_DIR}/variance_summary.json")
EOF

echo ""
echo "=============================================="
echo "评估波动性测试完成"
echo "=============================================="
echo "结果目录: $OUTPUT_DIR"
echo "=============================================="
