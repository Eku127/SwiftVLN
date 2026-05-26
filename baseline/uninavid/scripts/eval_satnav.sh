#!/bin/bash
# ==============================================================================
# Evaluate Uni-NaVid Baseline on SatNav task.
#
# Supports two calling modes:
#
#   1. Eval by name (recommended):
#      bash scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]
#      - Looks for checkpoint in output/uninavid-baseline/<exp_name_or_subpath>/
#        (fallback: output/uninavid-baseline/legacy/<exp_name_or_subpath>/)
#      - Extracts data version from exp_name (data{XXXXXX} -> ver_XXXXXX)
#
#   2. Eval by checkpoint path:
#      bash scripts/eval_satnav.sh /path/to/checkpoint [split] [gpus] [max_episodes]
#
# Arguments:
#   exp_name_or_subpath / path  First argument
#   split            Evaluation split: val_seen / val_unseen / test
#                    If omitted, SatNav runs both val_seen and val_unseen
#   gpus             Number of GPUs (default: 8)
#   max_episodes     Limit episodes (for debug/smoke, optional)
#
# Environment variables:
#   SATNAV_VERSION   — Override data version (default: auto from exp name or ver_260418)
#   MODEL_BASE       — Base model path for adapter-only checkpoints (optional)
#   LOCAL_CACHE_DIR  — Local disk dir to cache checkpoint (avoids NFS D-state).
#                      Auto-detected: uses /mnt/data4/jiangjiajun/uninavid_ckpt_cache
#                      if checkpoint is on NFS. Set to "" to disable caching.
#
# Output:
#   results/uninavid-baseline/<exp_name_or_subpath>/<split>/   (eval by name)
#   results/uninavid-baseline/by-path/<ckpt_name>/<split>/     (eval by path)
#
# Environment: conda env uninavid-baseline
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage:"
    echo ""
    echo "  # Eval by name from default output/uninavid-baseline"
    echo "  bash baseline/uninavid/scripts/eval_satnav.sh <exp_name_or_subpath> [split] [gpus] [max_episodes]"
    echo ""
    echo "  # Eval by name from a custom model root"
    echo "  bash baseline/uninavid/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618 \\"
    echo "    --split val_seen --gpus 8"
    echo ""
    echo "  # Eval by checkpoint path"
    echo "  bash baseline/uninavid/scripts/eval_satnav.sh --checkpoint_path /path/to/checkpoint --split val_unseen --gpus 8"
}

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/uninavid-baseline"

# ---- Args ----
MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
CHECKPOINT_PATH_ARG=""
SPLIT_ARG=""
NUM_GPUS="8"
MAX_EPISODES=""
DRY_RUN="false"
SATNAV_VERSION="${SATNAV_VERSION:-}"
MODEL_BASE="${MODEL_BASE:-}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/mnt/data4/jiangjiajun/uninavid_ckpt_cache}"
POSITIONAL=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_dir|--model-dir)
            MODEL_DIR_INPUT="${2:-}"
            shift 2
            ;;
        --model_name|--model-name)
            MODEL_NAME_ARG="${2:-}"
            shift 2
            ;;
        --checkpoint_path|--checkpoint-path)
            CHECKPOINT_PATH_ARG="${2:-}"
            shift 2
            ;;
        --split)
            SPLIT_ARG="${2:-}"
            shift 2
            ;;
        --gpus|--num_gpus|--num-gpus)
            NUM_GPUS="${2:-8}"
            shift 2
            ;;
        --max_episodes|--max-episodes)
            MAX_EPISODES="${2:-}"
            shift 2
            ;;
        --satnav_version|--satnav-version)
            SATNAV_VERSION="${2:-}"
            shift 2
            ;;
        --model_base|--model-base)
            MODEL_BASE="${2:-}"
            shift 2
            ;;
        --dry_run|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        --*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ -n "$MODEL_NAME_ARG" ] && [ -n "$CHECKPOINT_PATH_ARG" ]; then
    print_error "--model_name and --checkpoint_path are mutually exclusive"
    exit 1
fi

if [ -n "$MODEL_NAME_ARG" ]; then
    INPUT="$MODEL_NAME_ARG"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[0]:-}"
    [ "$NUM_GPUS" = "8" ] && NUM_GPUS="${POSITIONAL[1]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
elif [ -n "$CHECKPOINT_PATH_ARG" ]; then
    INPUT="$CHECKPOINT_PATH_ARG"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[0]:-}"
    [ "$NUM_GPUS" = "8" ] && NUM_GPUS="${POSITIONAL[1]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[2]:-}"
else
    INPUT="${POSITIONAL[0]:-}"
    [ -z "$SPLIT_ARG" ] && SPLIT_ARG="${POSITIONAL[1]:-}"
    [ "$NUM_GPUS" = "8" ] && NUM_GPUS="${POSITIONAL[2]:-8}"
    [ -z "$MAX_EPISODES" ] && MAX_EPISODES="${POSITIONAL[3]:-}"
fi

if [ -z "$INPUT" ]; then
    print_error "Missing model name or checkpoint path"
    usage
    exit 1
fi

if [ -n "$SPLIT_ARG" ]; then
    SPLITS_LIST="$SPLIT_ARG"
else
    SPLITS_LIST="val_seen val_unseen"
fi

if [ -z "$MODEL_DIR_INPUT" ]; then
    MODEL_ROOT_DIR="$DEFAULT_MODEL_DIR"
elif [[ "$MODEL_DIR_INPUT" = /* ]]; then
    MODEL_ROOT_DIR="$MODEL_DIR_INPUT"
else
    MODEL_ROOT_DIR="${REPO_ROOT}/${MODEL_DIR_INPUT}"
fi

# ---- Detect mode: eval-by-name vs eval-by-path ----
EVAL_MODE="by_name"
if [ -n "$CHECKPOINT_PATH_ARG" ] || [[ "$INPUT" = /* ]] || [ -d "$INPUT" ]; then
    EVAL_MODE="by_path"
fi

# ---- Resolve checkpoint and output paths ----
if [ "$EVAL_MODE" = "by_name" ]; then
    EXP_NAME="$INPUT"
    MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"

    if [ ! -d "$MODEL_DIR" ]; then
        LEGACY_MODEL_DIR="${REPO_ROOT}/output/uninavid-baseline/legacy/${EXP_NAME}"
        if [ -d "$LEGACY_MODEL_DIR" ]; then
            MODEL_DIR="$LEGACY_MODEL_DIR"
            print_warning "Experiment directory found in legacy path: ${MODEL_DIR}"
        else
            print_error "Experiment directory not found: ${MODEL_ROOT_DIR}/${EXP_NAME}"
            print_error "Legacy path also not found: output/uninavid-baseline/legacy/${EXP_NAME}"
            exit 1
        fi
    fi

    # Find latest checkpoint (checkpoint-N sorted by N descending)
    CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

    if [ -z "$CHECKPOINT_DIR" ]; then
        if [ -f "${MODEL_DIR}/config.json" ]; then
            CHECKPOINT_DIR="$MODEL_DIR"
            print_info "Using merged model dir as checkpoint"
        else
            print_error "No checkpoint found in: ${MODEL_DIR}"
            exit 1
        fi
    fi

    # Extract data version from EXP_NAME: data{XXXXXX} -> ver_XXXXXX
    if [ -z "$SATNAV_VERSION" ]; then
        PARSED_VER=$(echo "$EXP_NAME" | grep -oP 'data\K\d+' | head -1 || true)
        if [ -n "$PARSED_VER" ]; then
            SATNAV_VERSION="ver_${PARSED_VER}"
            print_info "Parsed data version from exp name: ${SATNAV_VERSION}"
        fi
    fi

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/uninavid-baseline/${EXP_NAME}"

else
    INPUT_PATH="${INPUT%/}"

    if [ ! -d "$INPUT_PATH" ]; then
        print_error "Checkpoint directory not found: ${INPUT_PATH}"
        exit 1
    fi

    if ls -d "${INPUT_PATH}"/checkpoint-* >/dev/null 2>&1; then
        MODEL_DIR="$INPUT_PATH"
        CHECKPOINT_DIR=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        EXP_NAME="$(basename "${MODEL_DIR}")"
    else
        CHECKPOINT_DIR="$INPUT_PATH"
        if [[ "$(basename "${CHECKPOINT_DIR}")" == checkpoint-* ]]; then
            EXP_NAME="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
        else
            EXP_NAME="$(basename "${CHECKPOINT_DIR}")"
        fi
    fi

    OUTPUT_BASE_DIR="${REPO_ROOT}/results/uninavid-baseline/by-path/${EXP_NAME}"
fi

if [ -z "$SATNAV_VERSION" ]; then
    PARSED_VER=$(echo "$EXP_NAME" | grep -oP 'data\K\d+' | head -1 || true)
    if [ -n "$PARSED_VER" ]; then
        SATNAV_VERSION="ver_${PARSED_VER}"
        print_info "Parsed data version from model name: ${SATNAV_VERSION}"
    fi
fi

# ---- Resolve SatNav version ----
if [ -z "$SATNAV_VERSION" ]; then
    SATNAV_VERSION="ver_260418"
    print_info "Using default SatNav version: ${SATNAV_VERSION}"
else
    print_info "Using SatNav version: ${SATNAV_VERSION}"
fi

SATNAV_SCENES="${SATNAV_DATA_ROOT}/scenes"

# ---- NFS checkpoint cache (avoid D-state from 8 processes reading NFS simultaneously) ----
# If checkpoint is on an NFS mount and LOCAL_CACHE_DIR is set, rsync model weights
# (excluding DeepSpeed optimizer states) to local disk before loading.
maybe_cache_checkpoint() {
    local src="$1"
    if [ -z "$LOCAL_CACHE_DIR" ]; then
        echo "$src"
        return
    fi

    # Check if the path is on an NFS filesystem
    local mount_type
    mount_type=$(stat -f -c "%T" "$src" 2>/dev/null || echo "unknown")
    if [ "$mount_type" = "nfs" ] || findmnt -n -o FSTYPE --target "$src" 2>/dev/null | grep -q "^nfs"; then
        local ckpt_name
        local parent_name
        local src_hash
        ckpt_name=$(basename "$src")
        parent_name=$(basename "$(dirname "$src")")
        if command -v sha1sum >/dev/null 2>&1; then
            src_hash=$(printf '%s' "$src" | sha1sum | awk '{print substr($1, 1, 12)}')
        else
            src_hash=$(printf '%s' "$src" | cksum | awk '{print $1}')
        fi
        local local_ckpt="${LOCAL_CACHE_DIR}/${parent_name}_${ckpt_name}_${src_hash}"

        # Check if already cached (use sentinel file to mark complete cache)
        if [ -f "${local_ckpt}/.cache_complete" ]; then
            print_info "Using existing local checkpoint cache: ${local_ckpt}" >&2
            echo "$local_ckpt"
            return
        fi

        # Important: this function is used inside command substitution, so only
        # the resolved checkpoint path may go to stdout. All logs must go to stderr.
        print_warning "Checkpoint is on NFS (${mount_type}). Caching model weights to local disk to avoid I/O D-state..." >&2
        print_info "Source      : ${src}" >&2
        print_info "Destination : ${local_ckpt}" >&2
        mkdir -p "$local_ckpt"

        # Rsync model files only — exclude DeepSpeed optimizer states (global_step*)
        # which can be 80-100G and are not needed for inference
        rsync -ah --progress \
            --exclude="global_step*" \
            "${src}/" "${local_ckpt}/" >&2

        touch "${local_ckpt}/.cache_complete"
        local cached_size
        cached_size=$(du -sh "$local_ckpt" | cut -f1)
        print_success "Checkpoint cached locally (${cached_size}): ${local_ckpt}" >&2
        echo "$local_ckpt"
    else
        # Not NFS, use as-is
        echo "$src"
    fi
}

if [ "$DRY_RUN" != "true" ]; then
    CHECKPOINT_DIR=$(maybe_cache_checkpoint "$CHECKPOINT_DIR")
fi

# ---- PYTHONPATH ----
export PYTHONPATH="${BASELINE_DIR}/src:${BASELINE_DIR}:${PYTHONPATH:-}"

# ---- Launch ----
if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD=(torchrun)
else
    print_warning "torchrun not found, fallback to: python -m torch.distributed.run"
    TORCHRUN_CMD=(python -m torch.distributed.run)
fi

resolve_visible_gpu_list() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "${CUDA_VISIBLE_DEVICES}"
        return
    fi

    python3 - <<'PY'
import subprocess
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader",
        ],
        text=True,
    )
except Exception:
    print("")
else:
    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    print(",".join(gpus))
PY
}

count_visible_gpus() {
    local gpu_list="$1"
    if [ -z "$gpu_list" ]; then
        echo 0
        return
    fi
    awk -F',' '{print NF}' <<< "$gpu_list"
}

run_single_split() {
    local split="$1"
    local satnav_episodes="${SATNAV_DATA_ROOT}/${SATNAV_VERSION}/episodes/eval/${split}/all_episodes.json"
    local satnav_config="${BASELINE_DIR}/configs/.satnav_task_eval_${split}_$$.yaml"
    local output_dir="${OUTPUT_BASE_DIR}/${split}"
    local visible_gpu_list
    local available_gpu_count

    if [ ! -f "$satnav_episodes" ]; then
        print_error "Episodes file not found: ${satnav_episodes}"
        return 1
    fi

    if [ "$DRY_RUN" = "true" ]; then
        visible_gpu_list="${CUDA_VISIBLE_DEVICES:-<dry-run>}"
    else
        visible_gpu_list="$(resolve_visible_gpu_list)"
        available_gpu_count="$(count_visible_gpus "$visible_gpu_list")"

        if [ "$NUM_GPUS" -gt "$available_gpu_count" ]; then
            print_error "Requested ${NUM_GPUS} GPUs, but only ${available_gpu_count} are visible on this host."
            print_error "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
            print_error "Detected visible GPUs: ${visible_gpu_list:-<none>}"
            return 1
        fi
    fi

    cp "$SATNAV_CONFIG_TEMPLATE" "$satnav_config"
    sed -i "s|DATA_PATH:.*|DATA_PATH: ${satnav_episodes}|" "$satnav_config"
    sed -i "s|SCENES_DIR:.*|SCENES_DIR: ${SATNAV_SCENES}|" "$satnav_config"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo "Uni-NaVid Baseline Evaluation"
    echo "=========================================="
    echo "  Eval mode  : ${EVAL_MODE}"
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Config     : ${satnav_config}"
    echo "  Data ver   : ${SATNAV_VERSION}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    echo "  VisibleGPU : ${visible_gpu_list:-<none>}"
    [ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ -n "$MODEL_BASE"   ] && echo "  Model Base  : ${MODEL_BASE}"
    [ "$DRY_RUN" = "true" ] && echo "  Dry Run    : true"
    echo "=========================================="

    if [ "$DRY_RUN" = "true" ]; then
        rm -f "${satnav_config}"
        print_success "Dry run completed for split=${split}."
        return 0
    fi

    COMMON_ARGS=(
        --model_path "${CHECKPOINT_DIR}"
        --satnav_config_path "${satnav_config}"
        --eval_split "${split}"
        --output_path "${output_dir}"
    )

    if [ -n "$MAX_EPISODES" ]; then
        COMMON_ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi

    if [ -n "$MODEL_BASE" ]; then
        COMMON_ARGS+=(--model_base "${MODEL_BASE}")
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        (
            unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT MASTER_PORTS
            export CUDA_VISIBLE_DEVICES="${visible_gpu_list}"
            "${TORCHRUN_CMD[@]}" \
                --nproc_per_node="${NUM_GPUS}" \
                --master_port=$((RANDOM % 10000 + 20000)) \
                "${EVAL_SCRIPT}" \
                "${COMMON_ARGS[@]}" \
                --world_size "${NUM_GPUS}"
        ) 2>&1 | tee "${output_dir}/eval.log"
    else
        (
            unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT MASTER_PORTS
            export CUDA_VISIBLE_DEVICES="${visible_gpu_list}"
            python "${EVAL_SCRIPT}" \
                "${COMMON_ARGS[@]}" \
                --world_size 1 \
                --rank 0
        ) 2>&1 | tee "${output_dir}/eval.log"
    fi

    rm -f "${satnav_config}"
    print_success "Evaluation completed for split=${split}!"
    echo "  Results: ${output_dir}"
}

for SPLIT in ${SPLITS_LIST}; do
    run_single_split "${SPLIT}"
done
