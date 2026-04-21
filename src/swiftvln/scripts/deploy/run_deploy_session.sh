#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash $0 <model_name> <requests_jsonl> [session_root]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${DEPLOY_PYTHON:-python3}"
MODEL_NAME="$1"
REQUESTS_FILE="$(realpath "$2")"
SESSION_ROOT="${3:-$REPO_ROOT/runtime/deploy/sessions}"

if [ ! -f "$REQUESTS_FILE" ]; then
  echo "Requests file does not exist: $REQUESTS_FILE" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m swiftvln.cli deploy \
  --model overlapvln \
  --model-name "$MODEL_NAME" \
  --session-root "$SESSION_ROOT" \
  < "$REQUESTS_FILE"
