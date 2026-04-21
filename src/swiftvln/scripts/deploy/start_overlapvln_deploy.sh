#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash $0 <model_name> [session_root]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${DEPLOY_PYTHON:-python3}"
MODEL_NAME="$1"
SESSION_ROOT="${2:-$REPO_ROOT/runtime/deploy/sessions}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m swiftvln.cli deploy \
  --model overlapvln \
  --model-name "$MODEL_NAME" \
  --session-root "$SESSION_ROOT"
