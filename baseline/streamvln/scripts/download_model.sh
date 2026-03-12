#!/usr/bin/env bash
# Download any model checkpoint from HuggingFace (hf-mirror) or ModelScope.
# Default: official StreamVLN benchmark checkpoint via hf-mirror.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../model"

MONITOR_INTERVAL=20
SOURCE="hf"   # hf | modelscope

# Default: official StreamVLN benchmark checkpoint (non-realworld)
DEFAULT_MODEL_REPO="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
MODEL_REPO=""
MODEL_DIR_NAME=""

usage() {
  cat <<'USAGE'
Usage:
  bash baseline/streamvln/scripts/download_model.sh [options]

Options:
  --repo <owner/repo>       Model repo ID to download (default: StreamVLN official checkpoint)
  --name <dir-name>         Local directory name under --model-dir (default: repo name after '/')
  --model-dir <path>        Parent directory to store the model (default: baseline/streamvln/model)
  --source <hf|modelscope>  Download source: hf (hf-mirror.com) or modelscope (default: hf)
  --monitor-interval <sec>  Progress print interval in seconds (default: 20)
  -h, --help                Show this message

Examples:
  # Download default StreamVLN checkpoint via hf-mirror
  bash baseline/streamvln/scripts/download_model.sh

  # Download LLaVA-Video via ModelScope (faster in China)
  bash baseline/streamvln/scripts/download_model.sh \
    --repo lmms-lab/LLaVA-Video-7B-Qwen2 --source modelscope

  # Download any HF model with custom local name
  bash baseline/streamvln/scripts/download_model.sh \
    --repo Qwen/Qwen2.5-7B-Instruct --name qwen2_5_7b

Default repo: mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3
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

# ─── parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      MODEL_REPO="${2:?--repo requires a value}"
      shift 2 ;;
    --name)
      MODEL_DIR_NAME="${2:?--name requires a value}"
      shift 2 ;;
    --model-dir)
      MODEL_DIR="${2:?--model-dir requires a path}"
      shift 2 ;;
    --source)
      SOURCE="${2:?--source requires hf or modelscope}"
      shift 2 ;;
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

# Apply defaults
[[ -z "$MODEL_REPO" ]] && MODEL_REPO="$DEFAULT_MODEL_REPO"
[[ -z "$MODEL_DIR_NAME" ]] && MODEL_DIR_NAME="${MODEL_REPO##*/}"

if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$MONITOR_INTERVAL" -le 0 ]]; then
  echo "[ERROR] --monitor-interval must be a positive integer." >&2
  exit 1
fi

if [[ "$SOURCE" != "hf" && "$SOURCE" != "modelscope" ]]; then
  echo "[ERROR] --source must be 'hf' or 'modelscope'" >&2
  exit 1
fi

DEST_DIR="${MODEL_DIR}/${MODEL_DIR_NAME}"
mkdir -p "$DEST_DIR"
before_size="$(calc_dir_size "$DEST_DIR")"

echo "[INFO] Source : ${SOURCE}"
echo "[INFO] Model  : ${MODEL_REPO}"
echo "[INFO] Target : ${DEST_DIR}"

# ─── HuggingFace (hf-mirror) ──────────────────────────────────────────────────
if [[ "$SOURCE" == "hf" ]]; then
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[ERROR] huggingface-cli not found. Install huggingface_hub first:" >&2
    echo "  pip install huggingface_hub" >&2
    exit 1
  fi

  export HF_ENDPOINT="https://hf-mirror.com"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
  echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"

  # query total size from hub for progress display
  total_size="$(python3 - "$MODEL_REPO" <<'PY' 2>/dev/null || echo 0
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
    echo "[INFO] Total  : unknown (progress based on local size growth)"
  fi

  huggingface-cli download "$MODEL_REPO" \
    --local-dir "$DEST_DIR" \
    --local-dir-use-symlinks False \
    --resume-download &
  dl_pid=$!

# ─── ModelScope ───────────────────────────────────────────────────────────────
else
  if ! python3 -c "import modelscope" 2>/dev/null; then
    echo "[ERROR] modelscope not found. Install first:" >&2
    echo "  pip install modelscope" >&2
    exit 1
  fi

  # query total size from modelscope for progress display
  total_size="$(python3 - "$MODEL_REPO" <<'PY' 2>/dev/null || echo 0
import sys
repo_id = sys.argv[1]
try:
    from modelscope.hub.api import HubApi
    info = HubApi().get_model(repo_id)
    size = info.get("StorageSize", 0) or 0
    print(size)
except Exception:
    print(0)
PY
)"
  [[ -z "$total_size" ]] && total_size=0

  if [[ "$total_size" -gt 0 ]]; then
    echo "[INFO] Total  : $(human_bytes "$total_size")"
  else
    echo "[INFO] Total  : unknown (progress based on local size growth)"
  fi

  python3 - "$MODEL_REPO" "$DEST_DIR" <<'PY' &
import sys
from modelscope import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=local_dir)
PY
  dl_pid=$!
fi

# ─── monitor & wait ───────────────────────────────────────────────────────────
monitor_download "$dl_pid" "$DEST_DIR" "$total_size" "$before_size"

if wait "$dl_pid"; then
  echo "[SUCCESS] Download completed: ${MODEL_REPO}"
  echo "[SUCCESS] Local path: ${DEST_DIR}"
else
  echo "[ERROR] Download failed: ${MODEL_REPO}" >&2
  exit 1
fi
