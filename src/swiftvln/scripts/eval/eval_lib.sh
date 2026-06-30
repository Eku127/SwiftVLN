#!/usr/bin/env bash

# Shared helpers for SwiftVLN eval shell entrypoints.

parse_env_type_from_model() {
    local name="$1"
    local second_field

    second_field=$(echo "$name" | cut -d'-' -f2)
    if [[ "$second_field" == "habitat" ]] || [[ "$second_field" == "satnav" ]]; then
        echo "$second_field"
    else
        echo "habitat"
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
