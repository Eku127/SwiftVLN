#!/usr/bin/env bash
# Optional machine-local environment loader shared by SwiftVLN entrypoints.
#
# Public scripts remain usable without a local file: every caller owns its
# editable fallback values.  To keep machine paths out of Git, copy
# local.env.example to .local/env.sh and edit it instead.

if [[ -z "${SWIFTVLN_ROOT:-}" ]]; then
    _swiftvln_env_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SWIFTVLN_ROOT="$(cd "${_swiftvln_env_lib_dir}/../.." && pwd)"
fi
export SWIFTVLN_ROOT

swiftvln_load_local_env() {
    local component_dir="${1:-}"
    local shared_env_file="${SWIFTVLN_LOCAL_ENV_FILE:-${SWIFTVLN_ROOT}/.local/env.sh}"
    local component_env_file=""

    if [[ -f "${shared_env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${shared_env_file}"
    fi

    if [[ -n "${component_dir}" ]]; then
        component_env_file="${component_dir}/.local/env.sh"
        if [[ -f "${component_env_file}" && "${component_env_file}" != "${shared_env_file}" ]]; then
            # shellcheck disable=SC1090
            source "${component_env_file}"
        fi
    fi
}

swiftvln_init_conda() {
    if [[ -n "${SWIFTVLN_CONDA_SH:-}" ]]; then
        if [[ ! -f "${SWIFTVLN_CONDA_SH}" ]]; then
            echo "[ERROR] SWIFTVLN_CONDA_SH does not exist: ${SWIFTVLN_CONDA_SH}" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "${SWIFTVLN_CONDA_SH}"
    elif command -v conda >/dev/null 2>&1; then
        if [[ "$(type -t conda)" != "function" ]]; then
            eval "$(conda shell.bash hook)"
        fi
    else
        echo "[ERROR] conda is unavailable. Set SWIFTVLN_CONDA_SH or edit this script's environment setup." >&2
        return 1
    fi
}

swiftvln_activate_conda() {
    local env_name="$1"

    swiftvln_init_conda
    conda activate "${env_name}"
}

unset _swiftvln_env_lib_dir
