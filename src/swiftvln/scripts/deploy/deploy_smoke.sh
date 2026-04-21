#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash $0 <model_name> <image_path> [instruction]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODEL_NAME="$1"
IMAGE_PATH="$(realpath "$2")"
INSTRUCTION="${3:-Drive to the destination.}"
TMP_ROOT="$(mktemp -d "$REPO_ROOT/runtime/tmp/deploy_smoke.XXXXXX")"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Image path does not exist: $IMAGE_PATH" >&2
  exit 1
fi

REQUESTS_FILE="$TMP_ROOT/requests.jsonl"
{
  printf '{"type":"start","instruction":"%s","session_id":"smoke"}\n' "$INSTRUCTION"
  printf '{"type":"image","image_path":"%s"}\n' "$IMAGE_PATH"
  printf '{"type":"end","reason":"smoke"}\n'
} > "$REQUESTS_FILE"

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
python3 -m swiftvln.cli deploy \
  --model overlapvln \
  --model-name "$MODEL_NAME" \
  --session-root "$TMP_ROOT/sessions" \
  < "$REQUESTS_FILE" \
  > "$TMP_ROOT/stdout.jsonl"

cat "$TMP_ROOT/stdout.jsonl"
