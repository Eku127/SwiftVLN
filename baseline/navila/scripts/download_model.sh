#!/usr/bin/env bash
# Download NaVILA model checkpoints from HuggingFace (hf-mirror).
# NOTE: NaVILA models are only available on HuggingFace, NOT on ModelScope.
#
# Two official models:
#   1. a8cheng/navila-siglip-llama3-8b-v1.5-pretrain  (pretrain, training start point)
#   2. a8cheng/navila-llama3-8b-8f                      (SFT trained, for evaluation)
#
# ModelScope availability checked via ModelScope API (2026-06-30):
#   - a8cheng/navila-siglip-llama3-8b-v1.5-pretrain: not found
#   - a8cheng/navila-llama3-8b-8f: not found
# Keep these downloads on HF/hf-mirror unless a ModelScope mirror is added later.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../model"

MONITOR_INTERVAL=20

MODELS=(
  "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain"
  "a8cheng/navila-llama3-8b-8f"
)

usage() {
  cat <<'USAGE'
Usage:
  bash baseline/navila/scripts/download_model.sh [options]

Options:
  --repo <owner/repo>        Download a single model instead of all defaults
  --name <dir-name>          Local directory name under model/ (default: repo name after '/')
  --model-dir <path>         Parent directory to store models (default: baseline/navila/model)
  --all                      Download all NaVILA models (default behavior)
  --monitor-interval <sec>   Progress print interval in seconds (default: 20)
  -h, --help                 Show this message

Examples:
  # Download all NaVILA models (pretrain + SFT)
  bash baseline/navila/scripts/download_model.sh

  # Download only the pretrain model
  bash baseline/navila/scripts/download_model.sh \
    --repo a8cheng/navila-siglip-llama3-8b-v1.5-pretrain

  # Download only the SFT-trained evaluation model
  bash baseline/navila/scripts/download_model.sh \
    --repo a8cheng/navila-llama3-8b-8f

NOTE: NaVILA models are ONLY available on HuggingFace (no ModelScope mirror).
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
  local dir="$1"
  [[ ! -d "$dir" ]] && { echo 0; return; }
  du -sb "$dir" 2>/dev/null | awk '{print $1}'
}

monitor_download() {
  local pid="$1"
  local dest_dir="$2"
  local total_size="$3"
  local base_size="$4"

  local last_ts now_ts dt
  local last_size now_size delta speed overall pct

  last_ts="$(date +%s)"
  last_size="$base_size"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    now_ts="$(date +%s)"
    now_size="$(calc_dir_size "$dest_dir")"
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

download_one() {
  local repo="$1"
  local dir_name="${2:-${repo##*/}}"
  local dest="${MODEL_DIR}/${dir_name}"

  mkdir -p "$dest"
  local before_size
  before_size="$(calc_dir_size "$dest")"

  echo ""
  echo "================================================================"
  echo "[INFO] Model  : ${repo}"
  echo "[INFO] Target : ${dest}"

  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[ERROR] huggingface-cli not found. Install first:" >&2
    echo "  pip install huggingface_hub" >&2
    exit 1
  fi

  export HF_ENDPOINT="https://hf-mirror.com"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
  echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"

  local total_size
  total_size="$(python3 - "$repo" <<'PY' 2>/dev/null || echo 0
import sys
repo_id = sys.argv[1]
try:
    from huggingface_hub import HfApi
    import os; os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    info = HfApi().model_info(repo_id)
    total = sum(
        getattr(s, "size", 0) or 0
        for s in (getattr(info, "siblings", []) or [])
    )
    print(total)
except Exception:
    print(0)
PY
)"
  [[ -z "$total_size" ]] && total_size=0

  if [[ "$total_size" -gt 0 ]]; then
    echo "[INFO] Total  : $(human_bytes "$total_size")"
  else
    echo "[INFO] Total  : unknown"
  fi

  huggingface-cli download "$repo" \
    --local-dir "$dest" \
    --local-dir-use-symlinks False \
    --resume-download &
  local dl_pid=$!

  monitor_download "$dl_pid" "$dest" "$total_size" "$before_size"

  if wait "$dl_pid"; then
    echo "[SUCCESS] Download completed: ${repo}"
    echo "[SUCCESS] Local path: ${dest}"
  else
    echo "[ERROR] Download failed: ${repo}" >&2
    return 1
  fi
}

# ─── parse args ───────────────────────────────────────────────────────────────
SINGLE_REPO=""
DIR_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      SINGLE_REPO="${2:?--repo requires a value}"
      shift 2 ;;
    --name)
      DIR_NAME="${2:?--name requires a value}"
      shift 2 ;;
    --model-dir)
      MODEL_DIR="${2:?--model-dir requires a path}"
      shift 2 ;;
    --all)
      shift ;;
    --monitor-interval)
      MONITOR_INTERVAL="${2:?--monitor-interval requires a number}"
      shift 2 ;;
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

# ─── download ─────────────────────────────────────────────────────────────────
if [[ -n "$SINGLE_REPO" ]]; then
  download_one "$SINGLE_REPO" "$DIR_NAME"
else
  echo "[INFO] Downloading all NaVILA models (${#MODELS[@]} total)..."
  for repo in "${MODELS[@]}"; do
    download_one "$repo" || exit 1
  done
  echo ""
  echo "[SUCCESS] All NaVILA models downloaded to: ${MODEL_DIR}"
fi
