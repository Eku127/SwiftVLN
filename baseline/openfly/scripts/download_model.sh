#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
MODEL_DIR="${MODEL_DIR:-${BASELINE_DIR}/model}"

BACKEND="${BACKEND:-${OPENFLY_BACKEND:-continue}}"
CONTINUE_MODEL_ID="${CONTINUE_MODEL_ID:-IPEC-COMMUNITY/openfly-agent-7b}"
SCRATCH_MODEL_ID="${SCRATCH_MODEL_ID:-openvla/openvla-7b-prismatic}"
# ModelScope availability checked via ModelScope API (2026-06-30):
#   - IPEC-COMMUNITY/openfly-agent-7b: not found
#   - openvla/openvla-7b-prismatic: not found
# Keep OpenFly default downloads on HF/hf-mirror unless mirrors are added later.
MODEL_ID="${MODEL_ID:-}"
TARGET_DIR="${TARGET_DIR:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

usage() {
    cat <<'USAGE'
Usage:
  bash baseline/openfly/scripts/download_model.sh [options]

Options:
  --backend <continue|scratch|all>
      Which model set to download. Default: continue.

  --model-id <repo>
      Override the Hugging Face repo for a single backend download.
      Environment alias: MODEL_ID.

  --target-dir <path>
      Override the target directory for a single backend download.
      Environment alias: TARGET_DIR.

  --model-dir <path>
      Parent model directory. Default: baseline/openfly/model.

  --continue-model-id <repo>
      Repo for the continue backend. Default: IPEC-COMMUNITY/openfly-agent-7b.

  --scratch-model-id <repo>
      Repo for the scratch backend native checkpoint. Default: openvla/openvla-7b-prismatic.

  -h, --help
      Show this message.

Examples:
  # Download default continue checkpoint.
  bash baseline/openfly/scripts/download_model.sh

  # Download scratch native OpenVLA/Prismatic checkpoint.
  bash baseline/openfly/scripts/download_model.sh --backend scratch

  # Download both continue and scratch assets.
  bash baseline/openfly/scripts/download_model.sh --backend all
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="${2:?--backend requires continue, scratch, or all}"
            shift 2
            ;;
        --model-id)
            MODEL_ID="${2:?--model-id requires a Hugging Face repo id}"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="${2:?--target-dir requires a path}"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="${2:?--model-dir requires a path}"
            shift 2
            ;;
        --continue-model-id)
            CONTINUE_MODEL_ID="${2:?--continue-model-id requires a Hugging Face repo id}"
            shift 2
            ;;
        --scratch-model-id)
            SCRATCH_MODEL_ID="${2:?--scratch-model-id requires a Hugging Face repo id}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${BACKEND}" in
    continue|scratch|all) ;;
    *)
        echo "[ERROR] --backend must be continue, scratch, or all: ${BACKEND}" >&2
        exit 1
        ;;
esac

if [[ "${BACKEND}" == "all" && ( -n "${MODEL_ID}" || -n "${TARGET_DIR}" ) ]]; then
    echo "[ERROR] --model-id / --target-dir are only valid for a single backend download." >&2
    exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[ERROR] huggingface-cli not found. Install huggingface_hub in the target env first:" >&2
    echo "  pip install huggingface_hub" >&2
    exit 2
fi

export HF_ENDPOINT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

download_one() {
    local backend="$1"
    local repo="$2"
    local target="$3"

    mkdir -p "${target}"
    echo "=========================================="
    echo "OpenFly model download"
    echo "  Backend     : ${backend}"
    echo "  Repo        : ${repo}"
    echo "  Target      : ${target}"
    echo "  HF_ENDPOINT : ${HF_ENDPOINT}"
    echo "=========================================="

    huggingface-cli download "${repo}" \
        --local-dir "${target}" \
        --local-dir-use-symlinks False \
        --resume-download
}

validate_continue() {
    local target="$1"
    if [[ ! -f "${target}/config.json" ]]; then
        echo "[ERROR] Continue model is missing config.json: ${target}" >&2
        exit 3
    fi
}

validate_scratch() {
    local target="$1"
    if [[ ! -d "${target}/checkpoints" ]]; then
        echo "[ERROR] Scratch native checkpoint directory is missing checkpoints/: ${target}" >&2
        exit 3
    fi

    shopt -s nullglob
    local checkpoints=("${target}"/checkpoints/*.pt)
    shopt -u nullglob
    if (( ${#checkpoints[@]} == 0 )); then
        echo "[ERROR] Scratch native checkpoint requires at least one checkpoints/*.pt file: ${target}" >&2
        exit 3
    fi
}

download_continue() {
    local repo="${MODEL_ID:-${CONTINUE_MODEL_ID}}"
    local target="${TARGET_DIR:-${MODEL_DIR}/openfly-agent-7b}"
    download_one "continue" "${repo}" "${target}"
    validate_continue "${target}"
    echo "[SUCCESS] Continue checkpoint ready: ${target}"
}

download_scratch() {
    local repo="${MODEL_ID:-${SCRATCH_MODEL_ID}}"
    local target="${TARGET_DIR:-${MODEL_DIR}/openvlaopenvla-7b-prismatic}"
    download_one "scratch" "${repo}" "${target}"
    validate_scratch "${target}"
    echo "[SUCCESS] Scratch native checkpoint ready: ${target}"
}

case "${BACKEND}" in
    continue)
        download_continue
        ;;
    scratch)
        download_scratch
        ;;
    all)
        download_continue
        MODEL_ID=""
        TARGET_DIR=""
        download_scratch
        ;;
esac
