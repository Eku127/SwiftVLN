#!/usr/bin/env bash

# Shared helpers for SwiftVLN eval shell entrypoints.

swiftvln_eval_python() {
    local executable="${PYTHON_EXECUTABLE:-python}"
    if ! command -v "$executable" >/dev/null 2>&1; then
        executable=python3
    fi
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "[ERROR] Python is unavailable (tried python and python3)." >&2
        return 127
    fi
    "$executable" "$@"
}

parse_env_type_from_model() {
    swiftvln_eval_python -m swiftvln.experiment parse-name "$1" \
        --format value --field env_type
}

parse_embed_slot_from_model() {
    local embedding
    embedding=$(
        swiftvln_eval_python -m swiftvln.experiment parse-name "$1" \
            --format value --field embedding
    ) || return 1
    if [[ "$embedding" == "none" ]]; then
        echo "noembed"
    else
        echo "$embedding"
    fi
}

infer_eval_splits() {
    local env_type="$1"
    local explicit_split="${2:-}"

    if [[ -n "$explicit_split" ]]; then
        echo "$explicit_split"
    elif [[ "$env_type" == "satnav" ]]; then
        echo "val_seen val_unseen"
    else
        echo "val_unseen"
    fi
}
