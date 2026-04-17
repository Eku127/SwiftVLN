#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_CACHE_ROOT="${MODELSCOPE_CACHE:-${HOME}/.cache/modelscope}"
CACHE_ROOT=""
MONITOR_INTERVAL=30
BOOTSTRAP_DEPS=true
DRY_RUN=false
MAX_WORKERS=""
DIRECT_NETWORK=true
MAX_ATTEMPTS=0
RETRY_SLEEP_SEC=60
MODELS=(
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
  "Qwen/Qwen3-VL-30B-A3B-Instruct"
)

usage() {
  cat <<'USAGE'
Usage:
  bash src/swiftvln/scripts/download_modelscope_qwen_vl_serial.sh [options]

Options:
  --cache-root <path>        Override ModelScope cache root
  --monitor-interval <sec>   Progress print interval in seconds (default: 30)
  --max-workers <n>          Per-model ModelScope download workers
  --models <csv>             Override model list with comma-separated model IDs
  --keep-proxy               Preserve proxy env vars instead of clearing them
  --max-attempts <n>         Retry attempts per model; 0 means unlimited (default: 0)
  --retry-sleep-sec <sec>    Sleep between failed attempts (default: 60)
  --list                     Print the model list and exit
  --dry-run                  Print plan only; do not download
  --no-bootstrap-deps        Do not auto-install missing Python deps
  -h, --help                 Show this message

Default models:
  Qwen/Qwen2.5-VL-7B-Instruct
  Qwen/Qwen2.5-VL-32B-Instruct
  Qwen/Qwen3-VL-8B-Instruct
  Qwen/Qwen3-VL-30B-A3B-Instruct

Examples:
  bash src/swiftvln/scripts/download_modelscope_qwen_vl_serial.sh

  bash src/swiftvln/scripts/download_modelscope_qwen_vl_serial.sh \
    --monitor-interval 60 --max-workers 8

  bash src/swiftvln/scripts/download_modelscope_qwen_vl_serial.sh \
    --cache-root /mnt/data1/home/jiangjiajun/.cache/modelscope \
    --models Qwen/Qwen2.5-VL-7B-Instruct,Qwen/Qwen3-VL-8B-Instruct

  bash src/swiftvln/scripts/download_modelscope_qwen_vl_serial.sh \
    --max-attempts 20 --retry-sleep-sec 120
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

join_by() {
  local delim="$1"
  shift
  local first=1
  for item in "$@"; do
    if [[ "$first" -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf '%s%s' "$delim" "$item"
    fi
  done
}

print_models() {
  local idx=1
  for repo_id in "${MODELS[@]}"; do
    printf '%d. %s\n' "$idx" "$repo_id"
    idx=$(( idx + 1 ))
  done
}

ensure_python_deps() {
  local missing
  missing="$(python3 - <<'PY'
import importlib.util
mods = {
    'modelscope': 'modelscope',
    'requests': 'requests',
    'urllib3': 'urllib3',
    'certifi': 'certifi',
    'filelock': 'filelock',
    'tqdm': 'tqdm',
}
missing = [pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None]
print(' '.join(missing))
PY
)"

  [[ -z "$missing" ]] && return 0

  if [[ "$BOOTSTRAP_DEPS" != "true" ]]; then
    echo "[ERROR] Missing Python dependencies: ${missing}" >&2
    echo "[ERROR] Re-run without --no-bootstrap-deps or install them manually." >&2
    exit 1
  fi

  echo "[INFO] Installing missing Python dependencies: ${missing}"
  python3 -m pip install --user ${missing}
}

clear_proxy_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
}

get_model_size() {
  local repo_id="$1"
  local size
  size="$(curl -L -s --fail "https://www.modelscope.cn/api/v1/models/${repo_id}" \
    | python3 -c 'import json,sys; obj=json.load(sys.stdin); print((obj.get("Data") or {}).get("StorageSize", 0) or 0)' \
    2>/dev/null || echo 0)"
  [[ -z "$size" ]] && size=0
  echo "$size"
}

get_model_name() {
  local repo_id="$1"
  local name
  name="$(curl -L -s --fail "https://www.modelscope.cn/api/v1/models/${repo_id}" \
    | python3 -c 'import json,sys; obj=json.load(sys.stdin); print((obj.get("Data") or {}).get("Name", ""))' \
    2>/dev/null || true)"
  [[ -z "$name" ]] && name="${repo_id##*/}"
  echo "$name"
}

monitor_download() {
  local pid="$1"
  local watch_dir="$2"
  local total_size="$3"
  local base_size="$4"

  local last_ts now_ts dt
  local last_size now_size delta speed overall pct eta

  last_ts="$(date +%s)"
  last_size="$base_size"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi

    now_ts="$(date +%s)"
    now_size="$(calc_dir_size "$watch_dir")"
    dt=$(( now_ts - last_ts ))
    [[ "$dt" -le 0 ]] && dt=1

    delta=$(( now_size - last_size ))
    [[ "$delta" -lt 0 ]] && delta=0
    speed=$(( delta / dt ))

    overall=$(( now_size - base_size ))
    [[ "$overall" -lt 0 ]] && overall=0

    eta="unknown"
    if [[ "$speed" -gt 0 && "$total_size" -gt "$overall" ]]; then
      eta="$(( (total_size - overall) / speed ))s"
    fi

    if [[ "$total_size" -gt 0 ]]; then
      pct=$(( overall * 100 / total_size ))
      [[ "$pct" -gt 100 ]] && pct=100
      echo "[PROGRESS] ${pct}% | $(human_bytes "$overall") / $(human_bytes "$total_size") | speed=$(human_bytes "$speed")/s | eta=${eta}"
    else
      echo "[PROGRESS] $(human_bytes "$overall") / unknown | speed=$(human_bytes "$speed")/s | eta=${eta}"
    fi

    last_ts="$now_ts"
    last_size="$now_size"
  done
}

download_one() {
  local repo_id="$1"
  local total_size="$2"
  local before_size="$3"
  local watch_root="$4"

  echo "[INFO] Downloading: ${repo_id}"
  [[ "$total_size" -gt 0 ]] && echo "[INFO] Expected size: $(human_bytes "$total_size")"

  PYTHONUNBUFFERED=1 python3 - "$repo_id" "$CACHE_ROOT" "$MAX_WORKERS" <<'PY' &
import sys
from modelscope import snapshot_download

repo_id = sys.argv[1]
cache_root = sys.argv[2]
max_workers = sys.argv[3].strip()
max_workers = int(max_workers) if max_workers else None

kwargs = {
    "model_id": repo_id,
    "max_workers": max_workers,
}
if cache_root:
    kwargs["cache_dir"] = cache_root

local_path = snapshot_download(**kwargs)
print(f"[SUCCESS] Local path: {local_path}", flush=True)
PY
  local dl_pid=$!

  monitor_download "$dl_pid" "$watch_root" "$total_size" "$before_size"

  if wait "$dl_pid"; then
    echo "[SUCCESS] Download finished: ${repo_id}"
  else
    echo "[ERROR] Download failed: ${repo_id}" >&2
    return 1
  fi
}

download_with_retry() {
  local repo_id="$1"
  local total_size="$2"
  local watch_root="$3"
  local attempt=1

  while true; do
    local before_size
    before_size="$(calc_dir_size "$watch_root")"

    if [[ "$MAX_ATTEMPTS" -eq 0 ]]; then
      echo "[INFO] Attempt ${attempt} for ${repo_id} (unlimited retry mode)"
    else
      echo "[INFO] Attempt ${attempt}/${MAX_ATTEMPTS} for ${repo_id}"
    fi

    if download_one "$repo_id" "$total_size" "$before_size" "$watch_root"; then
      return 0
    fi

    if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
      echo "[ERROR] Reached max attempts for ${repo_id}: ${MAX_ATTEMPTS}" >&2
      return 1
    fi

    echo "[WARN] Will retry ${repo_id} after ${RETRY_SLEEP_SEC}s; partial cache will be reused."
    sleep "$RETRY_SLEEP_SEC"
    attempt=$(( attempt + 1 ))
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-root)
      CACHE_ROOT="${2:?--cache-root requires a value}"
      shift 2
      ;;
    --monitor-interval)
      MONITOR_INTERVAL="${2:?--monitor-interval requires a value}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:?--max-workers requires a value}"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="${2:?--max-attempts requires a value}"
      shift 2
      ;;
    --retry-sleep-sec)
      RETRY_SLEEP_SEC="${2:?--retry-sleep-sec requires a value}"
      shift 2
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "${2:?--models requires a value}"
      shift 2
      ;;
    --keep-proxy)
      DIRECT_NETWORK=false
      shift
      ;;
    --list)
      print_models
      exit 0
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-bootstrap-deps)
      BOOTSTRAP_DEPS=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$MONITOR_INTERVAL" -le 0 ]]; then
  echo "[ERROR] --monitor-interval must be a positive integer." >&2
  exit 1
fi

if [[ -n "$MAX_WORKERS" ]] && { ! [[ "$MAX_WORKERS" =~ ^[0-9]+$ ]] || [[ "$MAX_WORKERS" -le 0 ]]; }; then
  echo "[ERROR] --max-workers must be a positive integer." >&2
  exit 1
fi

if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --max-attempts must be a non-negative integer." >&2
  exit 1
fi

if ! [[ "$RETRY_SLEEP_SEC" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --retry-sleep-sec must be a non-negative integer." >&2
  exit 1
fi

WATCH_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
mkdir -p "$WATCH_ROOT"
ensure_python_deps

if [[ "$DIRECT_NETWORK" == "true" ]]; then
  clear_proxy_env
fi

echo "[INFO] Repo root   : ${REPO_ROOT}"
if [[ -n "$CACHE_ROOT" ]]; then
  echo "[INFO] Cache root  : ${CACHE_ROOT} (override)"
else
  echo "[INFO] Cache root  : ${DEFAULT_CACHE_ROOT} (ModelScope default)"
fi
if [[ "$DIRECT_NETWORK" == "true" ]]; then
  echo "[INFO] Network     : direct (proxy env cleared)"
else
  echo "[INFO] Network     : inherited proxy settings"
fi
if [[ "$MAX_ATTEMPTS" -eq 0 ]]; then
  echo "[INFO] Retry mode   : unlimited"
else
  echo "[INFO] Retry mode   : max ${MAX_ATTEMPTS} attempts"
fi
echo "[INFO] Retry sleep  : ${RETRY_SLEEP_SEC}s"
echo "[INFO] Model count : ${#MODELS[@]}"
echo "[INFO] Model list  : $(join_by ', ' "${MODELS[@]}")"

grand_total=0
for repo_id in "${MODELS[@]}"; do
  size="$(get_model_size "$repo_id")"
  grand_total=$(( grand_total + size ))
  echo "[INFO] Planned model: $(get_model_name "$repo_id") | size=$(human_bytes "$size")"
done
echo "[INFO] Planned total size: $(human_bytes "$grand_total")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] Dry run only, exiting."
  exit 0
fi

for repo_id in "${MODELS[@]}"; do
  size="$(get_model_size "$repo_id")"
  download_with_retry "$repo_id" "$size" "$WATCH_ROOT"
done

echo "[SUCCESS] All downloads completed."
