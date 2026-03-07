#!/usr/bin/env bash
# Download Uni-NaVid model weights via hf-mirror.com (HF models) and direct URL (EVA-CLIP).
# Models:
#   1. EVA-CLIP (eva_vit_g.pth)      -- Google Storage direct download
#   2. Vicuna-7B (lmsys/vicuna-7b-v1.5) -- HuggingFace via hf-mirror.com  [optional]
#   3. Uni-NaVid weights (Jzzhang/Uni-NaVid) -- HuggingFace via hf-mirror.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../model"

MONITOR_INTERVAL=20

# ---------- model definitions ----------
EVA_CLIP_URL="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
EVA_CLIP_FILENAME="eva_vit_g.pth"

VICUNA_REPO="lmsys/vicuna-7b-v1.5"
VICUNA_DIR_NAME="vicuna-7b-v1.5"

UNINAVID_REPO="Jzzhang/Uni-NaVid"
UNINAVID_SUBFOLDER="uninavid-7b-full-224-video-fps-1-grid-2"
UNINAVID_DIR_NAME="uninavid-7b-full-224-video-fps-1-grid-2"

# ---------- flags ----------
DOWNLOAD_EVA=true
DOWNLOAD_VICUNA=false     # off by default: large base model, skip unless explicitly requested
DOWNLOAD_UNINAVID=true

usage() {
  cat <<'USAGE'
Usage:
  bash baseline/uninavid/scripts/download_uninavid_models.sh [options]

Options:
  --model-dir <path>        Directory to store models (default: baseline/uninavid/model)
  --monitor-interval <sec>  Progress print interval in seconds (default: 20)
  --all                     Download all models including Vicuna-7B base model
  --skip-eva                Skip EVA-CLIP download
  --skip-uninavid           Skip Uni-NaVid weights download
  --vicuna                  Also download Vicuna-7B base model
  -h, --help                Show this message

Models downloaded by default:
  1. EVA-CLIP encoder  (eva_vit_g.pth)              -- ~3.9 GB  direct URL
  2. Uni-NaVid weights (Jzzhang/Uni-NaVid subdir)  -- HuggingFace via hf-mirror.com

Optional:
  3. Vicuna-7B base    (lmsys/vicuna-7b-v1.5)       -- ~13 GB   HuggingFace via hf-mirror.com
USAGE
}

human_bytes() {
  local bytes="${1:-0}"
  awk -v b="$bytes" 'BEGIN {
    split("B KiB MiB GiB TiB", u, " ");
    i=1;
    while (b >= 1024 && i < 5) { b /= 1024; i++; }
    printf "%.2f %s", b, u[i];
  }'
}

calc_dir_size() {
  local path="$1"
  if [[ -f "$path" ]]; then
    stat -c%s "$path" 2>/dev/null || echo 0
  elif [[ -d "$path" ]]; then
    du -sb "$path" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

monitor_download() {
  local pid="$1"
  local dest="$2"   # file or directory
  local total_size="$3"
  local base_size="$4"

  local last_ts now_ts dt delta speed overall pct
  last_ts="$(date +%s)"
  last_size="$base_size"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    now_ts="$(date +%s)"
    now_size="$(calc_dir_size "$dest")"
    dt=$(( now_ts - last_ts ))
    [[ "$dt" -le 0 ]] && dt=1

    delta=$(( now_size - last_size ))
    [[ "$delta" -lt 0 ]] && delta=0
    speed=$(( delta / dt ))
    overall=$(( now_size - base_size ))
    [[ "$overall" -lt 0 ]] && overall=0

    if [[ "$total_size" -gt 0 ]]; then
      pct=$(( overall * 100 / total_size ))
      [[ "$pct" -gt 100 ]] && pct=100
      echo "[PROGRESS] ${pct}% | $(human_bytes "$overall") / $(human_bytes "$total_size") | speed=$(human_bytes "$speed")/s"
    else
      echo "[PROGRESS] $(human_bytes "$overall") / unknown | speed=$(human_bytes "$speed")/s"
    fi

    last_ts="$now_ts"
    last_size="$now_size"
  done
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="${2:?--model-dir requires a path}"
      shift 2 ;;
    --monitor-interval)
      MONITOR_INTERVAL="${2:?--monitor-interval requires a number}"
      shift 2 ;;
    --all)
      DOWNLOAD_EVA=true; DOWNLOAD_VICUNA=true; DOWNLOAD_UNINAVID=true
      shift ;;
    --skip-eva)
      DOWNLOAD_EVA=false; shift ;;
    --skip-uninavid)
      DOWNLOAD_UNINAVID=false; shift ;;
    --vicuna)
      DOWNLOAD_VICUNA=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$MONITOR_INTERVAL" -le 0 ]]; then
  echo "[ERROR] --monitor-interval must be a positive integer." >&2
  exit 1
fi

# ---------- setup mirror (for HF models) ----------
export HF_ENDPOINT="https://hf-mirror.com"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"

mkdir -p "$MODEL_DIR"

# ============================================================
# 1. EVA-CLIP  (direct wget from Google Storage)
# ============================================================
download_eva_clip() {
  local dest_file="${MODEL_DIR}/${EVA_CLIP_FILENAME}"

  if [[ -f "$dest_file" ]]; then
    local size; size="$(stat -c%s "$dest_file")"
    echo "[INFO] EVA-CLIP already exists ($(human_bytes "$size")): ${dest_file}"
    echo "[INFO] Skipping EVA-CLIP download."
    return 0
  fi

  echo ""
  echo "[INFO] ===== Downloading EVA-CLIP ====="
  echo "[INFO] URL : ${EVA_CLIP_URL}"
  echo "[INFO] Dest: ${dest_file}"

  local base_size=0
  wget --continue --show-progress \
    -O "$dest_file" \
    "$EVA_CLIP_URL" &
  local dl_pid=$!

  monitor_download "$dl_pid" "$dest_file" 0 "$base_size"

  if wait "$dl_pid"; then
    echo "[SUCCESS] EVA-CLIP downloaded: ${dest_file}"
  else
    echo "[ERROR] EVA-CLIP download failed." >&2
    exit 1
  fi
}

# ============================================================
# 2. Vicuna-7B  (HuggingFace via hf-mirror)
# ============================================================
download_vicuna() {
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[ERROR] huggingface-cli not found. Install: pip install huggingface_hub" >&2
    exit 1
  fi

  local dest_dir="${MODEL_DIR}/${VICUNA_DIR_NAME}"
  mkdir -p "$dest_dir"
  local before_size; before_size="$(calc_dir_size "$dest_dir")"

  local total_size
  total_size="$(python3 - "$VICUNA_REPO" <<'PY' 2>/dev/null || echo 0
import sys
repo_id = sys.argv[1]
try:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id)
    print(sum(getattr(s,"size",0) or 0 for s in (getattr(info,"siblings",[]) or [])))
except Exception:
    print(0)
PY
)"
  [[ -z "$total_size" ]] && total_size=0

  echo ""
  echo "[INFO] ===== Downloading Vicuna-7B ====="
  echo "[INFO] Repo  : ${VICUNA_REPO}"
  echo "[INFO] Target: ${dest_dir}"
  [[ "$total_size" -gt 0 ]] && echo "[INFO] Total : $(human_bytes "$total_size")"

  huggingface-cli download "$VICUNA_REPO" \
    --local-dir "$dest_dir" \
    --local-dir-use-symlinks False \
    --resume-download &
  local dl_pid=$!

  monitor_download "$dl_pid" "$dest_dir" "$total_size" "$before_size"

  if wait "$dl_pid"; then
    echo "[SUCCESS] Vicuna-7B downloaded: ${dest_dir}"
  else
    echo "[ERROR] Vicuna-7B download failed." >&2
    exit 1
  fi
}

# ============================================================
# 3. Uni-NaVid weights  (HuggingFace subfolder via hf-mirror)
# ============================================================
download_uninavid() {
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[ERROR] huggingface-cli not found. Install: pip install huggingface_hub" >&2
    exit 1
  fi

  local dest_dir="${MODEL_DIR}/${UNINAVID_DIR_NAME}"
  mkdir -p "$dest_dir"
  local before_size; before_size="$(calc_dir_size "$dest_dir")"

  local total_size
  total_size="$(python3 - "$UNINAVID_REPO" "$UNINAVID_SUBFOLDER" <<'PY' 2>/dev/null || echo 0
import sys
repo_id, subfolder = sys.argv[1], sys.argv[2]
try:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id)
    prefix = subfolder.rstrip("/") + "/"
    total = sum(
        getattr(s, "size", 0) or 0
        for s in (getattr(info, "siblings", []) or [])
        if s.rfilename.startswith(prefix)
    )
    print(total)
except Exception:
    print(0)
PY
)"
  [[ -z "$total_size" ]] && total_size=0

  echo ""
  echo "[INFO] ===== Downloading Uni-NaVid weights ====="
  echo "[INFO] Repo     : ${UNINAVID_REPO}"
  echo "[INFO] Subfolder: ${UNINAVID_SUBFOLDER}"
  echo "[INFO] Target   : ${dest_dir}"
  [[ "$total_size" -gt 0 ]] && echo "[INFO] Total    : $(human_bytes "$total_size")"

  huggingface-cli download "$UNINAVID_REPO" \
    --include "${UNINAVID_SUBFOLDER}/*" \
    --local-dir "$dest_dir" \
    --local-dir-use-symlinks False \
    --resume-download &
  local dl_pid=$!

  monitor_download "$dl_pid" "$dest_dir" "$total_size" "$before_size"

  if wait "$dl_pid"; then
    echo "[SUCCESS] Uni-NaVid weights downloaded: ${dest_dir}"
  else
    echo "[ERROR] Uni-NaVid download failed." >&2
    exit 1
  fi
}

# ============================================================
# Main
# ============================================================
echo "[INFO] Model dir: ${MODEL_DIR}"
echo "[INFO] Downloads: EVA-CLIP=${DOWNLOAD_EVA}  Vicuna=${DOWNLOAD_VICUNA}  Uni-NaVid=${DOWNLOAD_UNINAVID}"

[[ "$DOWNLOAD_EVA"      == true ]] && download_eva_clip
[[ "$DOWNLOAD_VICUNA"   == true ]] && download_vicuna
[[ "$DOWNLOAD_UNINAVID" == true ]] && download_uninavid

echo ""
echo "[INFO] All requested downloads finished."
echo "[INFO] Model dir: ${MODEL_DIR}"
ls -lh "$MODEL_DIR" 2>/dev/null || true
