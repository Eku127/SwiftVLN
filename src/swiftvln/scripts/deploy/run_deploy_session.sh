#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "Usage: bash $0 <requests_jsonl> [model_name] [session_root]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${DEPLOY_PYTHON:-python3}"
REQUESTS_FILE="$(realpath "$1")"

if [ ! -f "$REQUESTS_FILE" ]; then
  echo "Requests file does not exist: $REQUESTS_FILE" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEFAULT_MODEL_NAME="$($PYTHON_BIN - <<'PY'
from swiftvln.deployment import DEFAULT_OVERLAPVLN_DEPLOY_MODEL_NAME

print(DEFAULT_OVERLAPVLN_DEPLOY_MODEL_NAME)
PY
)"
MODEL_NAME="${2:-$DEFAULT_MODEL_NAME}"
SESSION_ROOT="${3:-$REPO_ROOT/runtime/deploy/sessions}"

DEPLOY_ARGS=(
  -m swiftvln.cli deploy
  --model overlapvln
  --session-root "$SESSION_ROOT"
)

if [ "$MODEL_NAME" != "$DEFAULT_MODEL_NAME" ]; then
  DEPLOY_ARGS+=(--model-name "$MODEL_NAME")
fi

exec "$PYTHON_BIN" "${DEPLOY_ARGS[@]}" < "$REQUESTS_FILE"
