#!/bin/bash
# ==============================================================================
# Evaluate UniNaVid Baseline on SatNav task.
#
# Eval data and split are controlled by:
#   baseline/uninavid/configs/satnav_task.yaml
#
# Required:
#   --model_name   Model directory name under --model_dir
#
# Optional:
#   --model_dir    Model root directory (default: output/uninavid-baseline)
#   --gpus         Number of GPUs (default: 8)
#   --max_episodes Limit episodes for debugging
#   --model_base   Base model path for adapter-only checkpoints
#   --dry_run      Resolve paths and print launch config without running eval
#
# Output:
#   results/uninavid-baseline/<model_name>/<split>/
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
    echo "  bash baseline/uninavid/scripts/eval_satnav.sh \\"
    echo "    --model_dir /path/to/model_root \\"
    echo "    --model_name uninavid-satnav-continue-1ep-lr1e-5 \\"
    echo "    --gpus 8"
    echo ""
    echo "Notes:"
    echo "  - Eval data and split come from baseline/uninavid/configs/satnav_task.yaml."
    echo "  - DATASET.DATA_PATH must be the eval split parent dir, e.g. .../episodes/eval."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
EVAL_SCRIPT="${BASELINE_DIR}/src/eval_satnav.py"
SATNAV_CONFIG_TEMPLATE="${BASELINE_DIR}/configs/satnav_task.yaml"
DEFAULT_MODEL_DIR="${REPO_ROOT}/output/uninavid-baseline"

read_config_value() {
    local key="$1"
    awk -v key="$key" '
        $1 == key ":" {
            sub(/^[^:]+:[[:space:]]*/, "")
            gsub(/^["'\''"]|["'\''"]$/, "")
            print
            exit
        }
    ' "$SATNAV_CONFIG_TEMPLATE"
}

has_model_weights() {
    local path="$1"
    find "$path" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .
}

resolve_model_load_dir() {
    local model_dir="$1"
    local checkpoint_dir

    if [ -f "${model_dir}/config.json" ] && has_model_weights "$model_dir"; then
        echo "$model_dir"
        return 0
    fi

    checkpoint_dir=$(ls -d "${model_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "$checkpoint_dir" ] && [ -f "${checkpoint_dir}/config.json" ] && has_model_weights "$checkpoint_dir"; then
        echo "$checkpoint_dir"
        return 0
    fi

    return 1
}

maybe_cache_checkpoint() {
    local src="$1"
    local cache_root="${LOCAL_CACHE_DIR:-/mnt/data4/jiangjiajun/uninavid_ckpt_cache}"

    if [ -z "$cache_root" ]; then
        echo "$src"
        return
    fi

    local mount_type
    mount_type=$(stat -f -c "%T" "$src" 2>/dev/null || echo "unknown")
    if [ "$mount_type" != "nfs" ] && ! findmnt -n -o FSTYPE --target "$src" 2>/dev/null | grep -q "^nfs"; then
        echo "$src"
        return
    fi

    if ! command -v rsync >/dev/null 2>&1; then
        print_warning "Checkpoint is on NFS (${mount_type}), but rsync is not available. Skipping local cache." >&2
        echo "$src"
        return
    fi

    local parent_name ckpt_name src_hash local_ckpt
    parent_name="$(basename "$(dirname "$src")")"
    ckpt_name="$(basename "$src")"
    if command -v sha1sum >/dev/null 2>&1; then
        src_hash="$(printf '%s' "$src" | sha1sum | awk '{print substr($1, 1, 12)}')"
    else
        src_hash="$(printf '%s' "$src" | cksum | awk '{print $1}')"
    fi
    local_ckpt="${cache_root}/${parent_name}_${ckpt_name}_${src_hash}"

    if [ -f "${local_ckpt}/.cache_complete" ]; then
        print_info "Using existing local checkpoint cache: ${local_ckpt}" >&2
        echo "$local_ckpt"
        return
    fi

    print_warning "Checkpoint is on NFS (${mount_type}). Caching model weights to local disk..." >&2
    print_info "Source      : ${src}" >&2
    print_info "Destination : ${local_ckpt}" >&2
    mkdir -p "$local_ckpt"

    rsync -ah --progress \
        --exclude="global_step*" \
        --exclude="rng_state_*.pth" \
        --exclude="scheduler.pt" \
        --exclude="optimizer.pt" \
        "${src}/" "${local_ckpt}/" >&2

    touch "${local_ckpt}/.cache_complete"
    print_success "Checkpoint cached locally: ${local_ckpt}" >&2
    echo "$local_ckpt"
}

resolve_visible_gpu_list() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES"
        return
    fi

    python3 - <<'PY'
import subprocess
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
except Exception:
    print("")
else:
    print(",".join(line.strip() for line in out.splitlines() if line.strip()))
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

MODEL_DIR_INPUT=""
MODEL_NAME_ARG=""
NUM_GPUS="8"
MAX_EPISODES=""
DRY_RUN="false"
MODEL_BASE="${MODEL_BASE:-}"

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
        --gpus|--num_gpus|--num-gpus)
            NUM_GPUS="${2:-8}"
            shift 2
            ;;
        --max_episodes|--max-episodes)
            MAX_EPISODES="${2:-}"
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
        --checkpoint_path|--checkpoint-path)
            print_error "--checkpoint_path is no longer supported. Use --model_dir + --model_name."
            exit 1
            ;;
        --split|--satnav_version|--satnav-version)
            print_error "$1 is no longer supported. Set DATASET.SPLIT/DATA_PATH in ${SATNAV_CONFIG_TEMPLATE}."
            exit 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            print_error "Positional arguments are not supported: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME_ARG" ]; then
    print_error "Missing required --model_name"
    usage
    exit 1
fi

if [ -z "$MODEL_DIR_INPUT" ]; then
    MODEL_ROOT_DIR="$DEFAULT_MODEL_DIR"
elif [[ "$MODEL_DIR_INPUT" = /* ]]; then
    MODEL_ROOT_DIR="$MODEL_DIR_INPUT"
else
    MODEL_ROOT_DIR="${REPO_ROOT}/${MODEL_DIR_INPUT}"
fi

EXP_NAME="$MODEL_NAME_ARG"
MODEL_DIR="${MODEL_ROOT_DIR}/${EXP_NAME}"
OUTPUT_BASE_DIR="${REPO_ROOT}/results/uninavid-baseline/${EXP_NAME}"

if [ ! -d "$MODEL_DIR" ]; then
    print_error "Model directory not found: ${MODEL_DIR}"
    exit 1
fi

if ! CHECKPOINT_DIR="$(resolve_model_load_dir "$MODEL_DIR")"; then
    print_error "No loadable UniNaVid model root or checkpoint found in: ${MODEL_DIR}"
    exit 1
fi

CONFIG_SPLIT="$(read_config_value SPLIT)"
CONFIG_DATA_PATH="$(read_config_value DATA_PATH)"
CONFIG_SCENES_DIR="$(read_config_value SCENES_DIR)"

if [ -z "$CONFIG_SPLIT" ]; then
    print_error "DATASET.SPLIT not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
elif [ "$CONFIG_SPLIT" = "all" ]; then
    SPLITS_LIST="val_seen val_unseen"
else
    SPLITS_LIST="$CONFIG_SPLIT"
fi

if [ -z "$CONFIG_DATA_PATH" ]; then
    print_error "DATASET.DATA_PATH not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
fi
if [ -z "$CONFIG_SCENES_DIR" ]; then
    print_error "DATASET.SCENES_DIR not found in config: ${SATNAV_CONFIG_TEMPLATE}"
    exit 1
fi
if [ ! -d "$CONFIG_DATA_PATH" ]; then
    print_error "DATASET.DATA_PATH must be an eval split parent directory: ${CONFIG_DATA_PATH}"
    exit 1
fi

SATNAV_SCENES="$CONFIG_SCENES_DIR"
print_info "Using SatNav eval data root from config: ${CONFIG_DATA_PATH}"
print_info "Using UniNaVid model load dir: ${CHECKPOINT_DIR}"

if [ "$DRY_RUN" != "true" ]; then
    CHECKPOINT_DIR="$(maybe_cache_checkpoint "$CHECKPOINT_DIR")"
    source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
    conda activate uninavid-baseline
fi

export PYTHONPATH="${BASELINE_DIR}/src:${BASELINE_DIR}:${PYTHONPATH:-}"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD=(torchrun)
else
    print_warning "torchrun not found, fallback to: python -m torch.distributed.run"
    TORCHRUN_CMD=(python -m torch.distributed.run)
fi

run_single_split() {
    local split="$1"
    local satnav_episodes="${CONFIG_DATA_PATH%/}/${split}/all_episodes.json"
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
    sed -i -E "s|^([[:space:]]*)SPLIT:.*|\\1SPLIT: ${split}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)DATA_PATH:.*|\\1DATA_PATH: ${satnav_episodes}|" "$satnav_config"
    sed -i -E "s|^([[:space:]]*)SCENES_DIR:.*|\\1SCENES_DIR: ${SATNAV_SCENES}|" "$satnav_config"

    mkdir -p "$output_dir"

    echo ""
    echo "=========================================="
    echo "UniNaVid Baseline Evaluation"
    echo "=========================================="
    echo "  EXP_NAME   : ${EXP_NAME}"
    echo "  ModelRoot  : ${MODEL_DIR}"
    echo "  ModelLoad  : ${CHECKPOINT_DIR}"
    echo "  Config     : ${satnav_config}"
    echo "  EvalRoot   : ${CONFIG_DATA_PATH}"
    echo "  Split      : ${split}"
    echo "  Output     : ${output_dir}"
    echo "  GPUs       : ${NUM_GPUS}"
    echo "  VisibleGPU : ${visible_gpu_list:-<none>}"
    [ -n "$MAX_EPISODES" ] && echo "  Max Episodes: ${MAX_EPISODES}"
    [ -n "$MODEL_BASE" ] && echo "  Model Base  : ${MODEL_BASE}"
    [ "$DRY_RUN" = "true" ] && echo "  Dry Run    : true"
    echo "=========================================="

    if [ "$DRY_RUN" = "true" ]; then
        rm -f "$satnav_config"
        print_success "Dry run completed for split=${split}."
        return 0
    fi

    COMMON_ARGS=(
        --model_path "$CHECKPOINT_DIR"
        --satnav_config_path "$satnav_config"
        --eval_split "$split"
        --output_path "$output_dir"
    )

    if [ -n "$MAX_EPISODES" ]; then
        COMMON_ARGS+=(--max_episodes "$MAX_EPISODES")
    fi

    if [ -n "$MODEL_BASE" ]; then
        COMMON_ARGS+=(--model_base "$MODEL_BASE")
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        (
            unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT MASTER_PORTS
            export CUDA_VISIBLE_DEVICES="$visible_gpu_list"
            "${TORCHRUN_CMD[@]}" \
                --nproc_per_node="$NUM_GPUS" \
                --master_port=$((RANDOM % 10000 + 20000)) \
                "$EVAL_SCRIPT" \
                "${COMMON_ARGS[@]}" \
                --world_size "$NUM_GPUS"
        ) 2>&1 | tee "${output_dir}/eval.log"
    else
        (
            unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_NAME MASTER_ADDR MASTER_PORT MASTER_PORTS
            export CUDA_VISIBLE_DEVICES="$visible_gpu_list"
            python "$EVAL_SCRIPT" \
                "${COMMON_ARGS[@]}" \
                --world_size 1 \
                --rank 0
        ) 2>&1 | tee "${output_dir}/eval.log"
    fi

    rm -f "$satnav_config"
    print_success "Evaluation completed for split=${split}!"
    echo "  Results: ${output_dir}"
}

for SPLIT in ${SPLITS_LIST}; do
    run_single_split "$SPLIT"
done
