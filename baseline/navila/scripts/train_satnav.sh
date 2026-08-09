#!/bin/bash
# Train NaVILA baseline on SatNav trajectory data.
# Usage:
#   bash baseline/navila/scripts/train_satnav.sh [scratch|continue] [EXP_NAME]
#   bash baseline/navila/scripts/train_satnav.sh [EXP_NAME]   # backward-compatible, defaults to scratch
#
# Optional env overrides:
#   SATNAV_DATASET=SatNav-v0.1
#   SATNAV_TRAIN_DATA_DIR=...
#   DATA_PATH=...
#   IMAGE_FOLDER=...
#   MODEL_PATH=...
#   SCRATCH_MODEL=...
#   CONTINUE_MODEL=...
#   NUM_GPUS=8
#   TRAIN_BSZ=4
#   GRAD_ACCUM=1
#   NUM_EPOCHS=1
#   LEARNING_RATE=3e-5
#   MAX_STEPS=60000
#   SAVE_STEPS=20000
#   SAVE_COUNT_TARGET=4
#   SAVE_TOTAL_LIMIT=1
#   MODEL_MAX_LENGTH=4096
#   DATALOADER_WORKERS=16
#   MASTER_PORT=29500
#   SATNAV_MAX_EPISODES=...
#   SATNAV_MAX_SAMPLES=...
#   SATNAV_SAMPLE_RATIO=...
#   SATNAV_SAMPLE_STRIDE=...      (forward stride in middle when HEAD_KEEP is set; turns always kept)
#   SATNAV_HEAD_KEEP=...          (enable head+stop+turn-protect mode; default 7)
#   SATNAV_STOP_REPEAT=...        (repeat stop samples this many times; default 1)
#   SATNAV_ACTION_FORMAT=...      (sentence|compact; default compact)
#   ENABLE_GPU_MONITOR=true
#   GPU_MONITOR_INTERVAL=60
#   USE_SWANLAB=false
#   SWANLAB_PROJECT=baseline
#   SWANLAB_MODE=cloud
#   REPORT_TO=...
set -euo pipefail

SWIFTVLN_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASELINE_DIR="${SWIFTVLN_ROOT}/baseline/navila"
# shellcheck source=../../../src/swiftvln/scripts/lib/local_env.sh
source "${SWIFTVLN_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env "${BASELINE_DIR}"

SCRATCH_MODEL="${SCRATCH_MODEL:-${BASELINE_DIR}/model/navila-siglip-llama3-8b-v1.5-pretrain}"
CONTINUE_MODEL="${CONTINUE_MODEL:-${BASELINE_DIR}/model/navila-llama3-8b-8f}"
DS_CONFIG="${DS_CONFIG:-${BASELINE_DIR}/configs/zero2.json}"

SATNAV_DATA_ROOT="${SWIFTVLN_SATNAV_DATA_ROOT:-data/satnav}"
SATNAV_DATASET="${SATNAV_DATASET:-${SATNAV_VERSION:-${SWIFTVLN_SATNAV_DATASET:-SatNav-v0.1}}}"
SATNAV_TRAIN_DATA_DIR="${SATNAV_TRAIN_DATA_DIR:-${SWIFTVLN_SATNAV_TRAIN_DATA_PATH:-${SATNAV_DATA_ROOT}/${SATNAV_DATASET}/trajectory_data}}"
DATA_PATH="${DATA_PATH:-${SATNAV_TRAIN_DATA_DIR}/annotations.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${SATNAV_TRAIN_DATA_DIR}}"

# Sample reduction strategy: head + stop + turn-protected forward stride.
#   - Head  steps 1..HEAD_KEEP : always kept (unique <8-frame input distribution)
#   - Stop  (last step)        : always kept (rare but critical)
#   - Middle turns             : always kept (decision-critical minority class)
#   - Middle forward runs      : keep every SAMPLE_STRIDE-th consecutive forward step
# Default (head=7, fwd_stride=7, stop_repeat=4): ~2.87M samples, full turn coverage,
# and stronger stop supervision without fully balancing the classes.
# Pass empty string to disable: SATNAV_HEAD_KEEP= SATNAV_SAMPLE_STRIDE= (full raw data).
SATNAV_HEAD_KEEP="${SATNAV_HEAD_KEEP-7}"
SATNAV_SAMPLE_STRIDE="${SATNAV_SAMPLE_STRIDE-7}"
SATNAV_STOP_REPEAT="${SATNAV_STOP_REPEAT-4}"
SATNAV_ACTION_FORMAT="${SATNAV_ACTION_FORMAT:-compact}"
# Smoke / debug limits (unset by default for full training).
SATNAV_MAX_EPISODES="${SATNAV_MAX_EPISODES-}"
SATNAV_MAX_SAMPLES="${SATNAV_MAX_SAMPLES-}"
SATNAV_SAMPLE_RATIO="${SATNAV_SAMPLE_RATIO-}"

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BSZ="${TRAIN_BSZ:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
# Empty MAX_STEPS means "do not pass --max_steps", which enables full-data training.
MAX_STEPS="${MAX_STEPS-60000}"
# Default checkpoint cadence is every 20k optimizer steps.
# Explicit empty SAVE_STEPS= switches back to auto scheduling from SAVE_COUNT_TARGET.
SAVE_STEPS="${SAVE_STEPS-20000}"
SAVE_COUNT_TARGET="${SAVE_COUNT_TARGET:-4}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
REPORT_TO="${REPORT_TO:-}"
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-true}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
TRAIN_MODE="${NAVILA_INIT_MODE:-scratch}"
CUSTOM_EXP_NAME=""

if [[ $# -ge 1 ]]; then
    case "${1}" in
        scratch|continue)
            TRAIN_MODE="${1}"
            CUSTOM_EXP_NAME="${2:-}"
            ;;
        *)
            CUSTOM_EXP_NAME="${1}"
            ;;
    esac
fi

case "${TRAIN_MODE}" in
    scratch)
        DEFAULT_MODEL_PATH="${SCRATCH_MODEL}"
        ;;
    continue)
        DEFAULT_MODEL_PATH="${CONTINUE_MODEL}"
        ;;
    *)
        echo "Unsupported training mode: ${TRAIN_MODE}" >&2
        echo "Expected one of: scratch, continue" >&2
        exit 2
        ;;
esac

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
VISION_TOWER="${VISION_TOWER:-${MODEL_PATH}/vision_tower}"

VERSION_TAG="$(echo "${SATNAV_DATASET}" | sed -E 's/^ver_//; s/[^A-Za-z0-9._-]+/-/g')"
if [[ -z "${VERSION_TAG}" ]]; then
    VERSION_TAG="unknown"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
EFFECTIVE_BATCH_SIZE=$((TRAIN_BSZ * GRAD_ACCUM * NUM_GPUS))

if [[ -n "${SATNAV_HEAD_KEEP}" ]]; then
    SAMPLE_TAG="sample-hk${SATNAV_HEAD_KEEP}-fs${SATNAV_SAMPLE_STRIDE:-1}-stopx${SATNAV_STOP_REPEAT:-1}"
else
    SAMPLE_TAG="sample-legacy-stride${SATNAV_SAMPLE_STRIDE:-off}-ratio${SATNAV_SAMPLE_RATIO:-1.0}-stopx${SATNAV_STOP_REPEAT:-1}"
fi

ACTION_TAG=""
case "${SATNAV_ACTION_FORMAT,,}" in
    compact|token|word|default|"")
        SATNAV_ACTION_FORMAT="compact"
        ACTION_TAG=""
        ;;
    sentence|natural|legacy)
        SATNAV_ACTION_FORMAT="sentence"
        ACTION_TAG="-actsentence"
        ;;
    *)
        echo "Unsupported SATNAV_ACTION_FORMAT: ${SATNAV_ACTION_FORMAT}" >&2
        echo "Expected one of: compact, sentence" >&2
        exit 2
        ;;
esac

append_sample_tag() {
    local base_name="$1"
    if [[ "${base_name}" == *"${SAMPLE_TAG}"* ]]; then
        printf '%s\n' "${base_name}"
    else
        printf '%s-%s\n' "${base_name}" "${SAMPLE_TAG}"
    fi
}

append_action_tag() {
    local base_name="$1"
    if [[ -z "${ACTION_TAG}" ]] || [[ "${base_name}" == *"${ACTION_TAG}"* ]]; then
        printf '%s\n' "${base_name}"
    else
        printf '%s%s\n' "${base_name}" "${ACTION_TAG}"
    fi
}

if [[ -n "${CUSTOM_EXP_NAME}" ]]; then
    EXP_NAME="$(append_action_tag "$(append_sample_tag "${CUSTOM_EXP_NAME}")")"
else
    EXP_NAME="navila-baseline-${TRAIN_MODE}-${NUM_EPOCHS}ep-8f-data${VERSION_TAG}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${SAMPLE_TAG}${ACTION_TAG}-${TIMESTAMP}"
fi

OUTPUT_DIR="${SWIFTVLN_ROOT}/output/navila-baseline/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"
GPU_LOG="${OUTPUT_DIR}/gpu_metrics.log"
MAX_STEPS_ARG=()
REPORT_TO_ARG=()
GPU_MONITOR_PID=""

if [[ -t 1 ]]; then
    exec > >(tee -a "${TRAIN_LOG}") 2>&1
else
    exec >> "${TRAIN_LOG}" 2>&1
fi

cleanup_gpu_monitor() {
    local rc=$?
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    return "${rc}"
}

start_gpu_monitor() {
    if [[ "${ENABLE_GPU_MONITOR}" != "true" ]]; then
        echo "[INFO] GPU monitor disabled (ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR})"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] nvidia-smi not found; skip GPU temperature logging."
        return
    fi

    if ! [[ "${GPU_MONITOR_INTERVAL}" =~ ^[0-9]+$ ]] || [[ "${GPU_MONITOR_INTERVAL}" -lt 1 ]]; then
        echo "[WARN] Invalid GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}; skip GPU temperature logging."
        return
    fi

    : > "${GPU_LOG}"
    {
        echo "# ts,index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
        while true; do
            local ts
            ts="$(date '+%Y-%m-%d %H:%M:%S')"
            if ! nvidia-smi \
                --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits 2>/dev/null | sed "s/^/${ts},/"; then
                echo "${ts},ERROR,nvidia-smi_failed"
            fi
            sleep "${GPU_MONITOR_INTERVAL}"
        done
    } >> "${GPU_LOG}" &
    GPU_MONITOR_PID=$!
    echo "[INFO] GPU monitor enabled: interval=${GPU_MONITOR_INTERVAL}s"
    echo "[INFO] GPU metrics log: ${GPU_LOG}"
}

trap cleanup_gpu_monitor EXIT

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Base model directory not found: ${MODEL_PATH}" >&2
    exit 2
fi

if [[ ! -d "${VISION_TOWER}" ]]; then
    echo "Vision tower directory not found: ${VISION_TOWER}" >&2
    exit 2
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Data path not found: ${DATA_PATH}" >&2
    exit 2
fi

# Ensure python is available for sample counting and launcher scripts.
swiftvln_activate_conda navila-baseline

PYTHON_BIN="${NAVILA_PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python interpreter not found after activating navila-baseline" >&2
    exit 2
fi

compute_effective_samples() {
    DATA_PATH="${DATA_PATH}" \
    SATNAV_MAX_EPISODES="${SATNAV_MAX_EPISODES}" \
    SATNAV_MAX_SAMPLES="${SATNAV_MAX_SAMPLES}" \
    SATNAV_SAMPLE_RATIO="${SATNAV_SAMPLE_RATIO}" \
    SATNAV_SAMPLE_STRIDE="${SATNAV_SAMPLE_STRIDE}" \
    SATNAV_HEAD_KEEP="${SATNAV_HEAD_KEEP}" \
    SATNAV_STOP_REPEAT="${SATNAV_STOP_REPEAT}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import zlib

VALID_ACTIONS = {0, 1, 2, 3}

data_path = os.environ["DATA_PATH"]
max_episodes_env = os.getenv("SATNAV_MAX_EPISODES", "").strip()
max_samples_env = os.getenv("SATNAV_MAX_SAMPLES", "").strip()
sample_ratio_env = os.getenv("SATNAV_SAMPLE_RATIO", "").strip()
sample_stride_env = os.getenv("SATNAV_SAMPLE_STRIDE", "").strip()
head_keep_env = os.getenv("SATNAV_HEAD_KEEP", "").strip()
stop_repeat_env = os.getenv("SATNAV_STOP_REPEAT", "").strip()

max_episodes = int(max_episodes_env) if max_episodes_env else None
max_samples = int(max_samples_env) if max_samples_env else None
sample_ratio = float(sample_ratio_env) if sample_ratio_env else None
sample_stride = int(sample_stride_env) if sample_stride_env else None
head_keep = int(head_keep_env) if head_keep_env else None
stop_repeat = int(stop_repeat_env) if stop_repeat_env else 1

with open(data_path) as f:
    episodes = json.load(f)

if max_episodes is not None:
    episodes = episodes[:max_episodes]

def select_head_stop_stride(actions, head_keep, fwd_stride):
    last_step = len(actions) - 1
    if last_step <= 0:
        return []
    kept = set()
    for i in range(1, min(head_keep + 1, last_step + 1)):
        if actions[i] in VALID_ACTIONS:
            kept.add(i)
    if actions[last_step] in VALID_ACTIONS:
        kept.add(last_step)
    mid_start = head_keep + 1
    mid_end = last_step - 1
    if fwd_stride > 0 and mid_start <= mid_end:
        consecutive_fwd = 0
        for i in range(mid_start, mid_end + 1):
            a = actions[i]
            if a not in VALID_ACTIONS:
                consecutive_fwd = 0
                continue
            if a == 1:
                if consecutive_fwd % fwd_stride == 0:
                    kept.add(i)
                consecutive_fwd += 1
            else:
                kept.add(i)
                consecutive_fwd = 0
    return sorted(kept)

count = 0
for episode in episodes:
    episode_id = str(episode.get("id", ""))
    trajectory_id = str(episode.get("trajectory_id", ""))
    actions = episode["actions"]

    if head_keep is not None:
        fwd_stride = sample_stride if (sample_stride is not None and sample_stride > 1) else 1
        step_indices = select_head_stop_stride(actions, head_keep, fwd_stride)
    else:
        step_indices = []
        for i in range(1, len(actions)):
            action = actions[i]
            if action not in VALID_ACTIONS:
                continue
            if sample_stride is not None and sample_stride > 1 and i % sample_stride != 0:
                continue
            if sample_ratio is not None and sample_ratio < 1.0:
                sample_key = f"{episode_id}|{trajectory_id}|{i}"
                sample_hash = zlib.crc32(sample_key.encode("utf-8")) & 0xFFFFFFFF
                if (sample_hash / 0xFFFFFFFF) >= sample_ratio:
                    continue
            step_indices.append(i)

    for i in step_indices:
        count += stop_repeat if actions[i] == 0 else 1
    if max_samples is not None and count >= max_samples:
        print(min(count, max_samples))
        raise SystemExit(0)

print(count)
PY
}

if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

TOTAL_STEPS=""
if [[ -n "${MAX_STEPS}" ]]; then
    TOTAL_STEPS="${MAX_STEPS}"
else
    TOTAL_SAMPLES="$(compute_effective_samples)"
    STEPS_PER_EPOCH=$(((TOTAL_SAMPLES + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE))
    TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
fi

if [[ -z "${SAVE_STEPS}" ]]; then
    SAVE_STEPS=$(((TOTAL_STEPS + SAVE_COUNT_TARGET - 1) / SAVE_COUNT_TARGET))
fi

if [[ "${SAVE_STEPS}" -lt 1 ]]; then
    SAVE_STEPS=1
fi

if [[ -n "${REPORT_TO}" ]]; then
    REPORT_TO_ARG=(--report_to "${REPORT_TO}")
elif [[ "${USE_SWANLAB}" == "true" ]]; then
    export SWANLAB_PROJECT
    export SWANLAB_NAME="${EXP_NAME}"
    export SWANLAB_MODE
    REPORT_TO_ARG=(--report_to swanlab)
else
    REPORT_TO_ARG=(--report_to none)
fi

export SATNAV_HEAD_KEEP SATNAV_SAMPLE_STRIDE SATNAV_STOP_REPEAT
export SATNAV_MAX_EPISODES SATNAV_MAX_SAMPLES SATNAV_SAMPLE_RATIO
export SATNAV_ACTION_FORMAT

echo "=========================================="
echo "NaVILA Baseline Training"
echo "=========================================="
echo "  Init mode  : ${TRAIN_MODE}"
echo "  Model      : ${MODEL_PATH}"
echo "  Dataset    : ${SATNAV_DATASET}"
echo "  Data path  : ${DATA_PATH}"
echo "  Image root : ${IMAGE_FOLDER}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  GPUs       : ${NUM_GPUS}"
echo "  Master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Batch      : ${TRAIN_BSZ} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${EFFECTIVE_BATCH_SIZE}"
echo "  LR         : ${LEARNING_RATE}"
echo "  Total step : ${TOTAL_STEPS}"
echo "  Save every : ${SAVE_STEPS} steps"
echo "  Sampling   : head_keep=${SATNAV_HEAD_KEEP:-off}, stride=${SATNAV_SAMPLE_STRIDE:-off}, stop_repeat=${SATNAV_STOP_REPEAT:-1}, max_ep=${SATNAV_MAX_EPISODES:-off}, max_samples=${SATNAV_MAX_SAMPLES:-off}"
echo "  Action fmt : ${SATNAV_ACTION_FORMAT}"
echo "  Train log  : ${TRAIN_LOG}"
echo "  GPU log    : ${GPU_LOG} (enabled=${ENABLE_GPU_MONITOR}, interval=${GPU_MONITOR_INTERVAL}s)"
echo "  SwanLab    : ${USE_SWANLAB}"
echo "  Report To  : ${REPORT_TO_ARG[*]}"
echo "=========================================="

start_gpu_monitor

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${BASELINE_DIR}/src/train_satnav.py" \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version llama_3 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --data_mixture satnav \
    --longvila_sampler False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    "${MAX_STEPS_ARG[@]}" \
    --per_device_train_batch_size "${TRAIN_BSZ}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --fps 0.0 \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --lazy_preprocess True \
    "${REPORT_TO_ARG[@]}"

echo "=========================================="
echo "Training completed!"
echo "  EXP_NAME : ${EXP_NAME}"
echo "  Output   : ${OUTPUT_DIR}"
echo "=========================================="
