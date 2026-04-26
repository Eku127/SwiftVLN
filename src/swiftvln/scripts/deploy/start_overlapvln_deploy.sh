#!/bin/bash
set -euo pipefail

if [ $# -gt 2 ]; then
  echo "Usage: bash $0 [model_name] [session_root]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${DEPLOY_PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEFAULT_MODEL_NAME="$($PYTHON_BIN - <<'PY'
from swiftvln.deployment import DEFAULT_OVERLAPVLN_DEPLOY_MODEL_NAME

print(DEFAULT_OVERLAPVLN_DEPLOY_MODEL_NAME)
PY
)"
MODEL_NAME="${1:-$DEFAULT_MODEL_NAME}"
SESSION_ROOT="${2:-$REPO_ROOT/runtime/deploy/sessions}"

DEPLOY_ARGS=(
  -m swiftvln.cli deploy
  --model overlapvln
  --session-root "$SESSION_ROOT"
)

if [ "$MODEL_NAME" != "$DEFAULT_MODEL_NAME" ]; then
  DEPLOY_ARGS+=(--model-name "$MODEL_NAME")
fi

exec "$PYTHON_BIN" "${DEPLOY_ARGS[@]}"
