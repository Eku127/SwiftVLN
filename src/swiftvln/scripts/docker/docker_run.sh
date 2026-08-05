#!/usr/bin/env bash
# Bootstrap the persistent SwiftVLN runtime container on server 17.

set -Eeuo pipefail

# This is deliberately non-interactive. Existing containers are reused by
# default; replacement only happens when the caller explicitly sets
# RECREATE=true.
DOCKER_IMAGE="${DOCKER_IMAGE:-ubuntu:22.04}"
CONTAINER_NAME="${CONTAINER_NAME:-streamvln-container}"
RECREATE="${RECREATE:-false}"
ATTACH="${ATTACH:-false}"
DRY_RUN="${DRY_RUN:-false}"

HOST_WORKSPACE="${HOST_WORKSPACE:-/mnt/data1/home/jiangjiajun/workspace}"
HOST_CONDA="${HOST_CONDA:-/mnt/data1/home/jiangjiajun/miniconda3}"
HOST_DATA="${HOST_DATA:-/mnt/data3/jiangjiajun/dataset}"
HOST_CACHE="${HOST_CACHE:-/mnt/data1/home/jiangjiajun/.cache}"
HOST_CUDA="${HOST_CUDA:-/usr/local/cuda-13.0}"

CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-${HOST_WORKSPACE}}"
CONTAINER_CONDA="${CONTAINER_CONDA:-${HOST_CONDA}}"
CONTAINER_DATA="${CONTAINER_DATA:-${HOST_DATA}}"
CONTAINER_CACHE="${CONTAINER_CACHE:-${HOST_CACHE}}"
CONTAINER_CUDA="${CONTAINER_CUDA:-/usr/local/cuda-13.0}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${CONTAINER_WORKSPACE}/SwiftVLN}"

USE_GPU="${USE_GPU:-true}"
CUDA_DEVICES="${CUDA_DEVICES:-all}"
NETWORK_MODE="${NETWORK_MODE:-host}"

require_bool() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "true" && "${value}" != "false" ]]; then
        echo "[ERROR] ${name} must be true or false, got: ${value}" >&2
        exit 2
    fi
}

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

attach_if_requested() {
    if [[ "${ATTACH}" == "true" ]]; then
        exec docker exec -it "${CONTAINER_NAME}" /bin/bash
    fi
    echo "[INFO] Enter with: docker exec -it ${CONTAINER_NAME} /bin/bash"
}

require_bool RECREATE "${RECREATE}"
require_bool ATTACH "${ATTACH}"
require_bool DRY_RUN "${DRY_RUN}"
require_bool USE_GPU "${USE_GPU}"

gpu_request="${CUDA_DEVICES}"
if [[ "${CUDA_DEVICES}" != "all" ]]; then
    gpu_request="device=${CUDA_DEVICES}"
fi

docker_args=(
    run --detach
    --name "${CONTAINER_NAME}"
    --hostname "${CONTAINER_NAME}"
    --restart unless-stopped
    --network "${NETWORK_MODE}"
    --workdir "${CONTAINER_WORKDIR}"
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --label swiftvln.role=server17-runtime
    --volume "${HOST_WORKSPACE}:${CONTAINER_WORKSPACE}"
    --volume "${HOST_CONDA}:${CONTAINER_CONDA}"
    --volume "${HOST_DATA}:${CONTAINER_DATA}"
    --volume "${HOST_CACHE}:${CONTAINER_CACHE}"
    --volume "${HOST_CUDA}:${CONTAINER_CUDA}:ro"
    --env "CUDA_HOME=${CONTAINER_CUDA}"
    --env "LD_LIBRARY_PATH=${CONTAINER_CUDA}/lib64:${CONTAINER_CUDA}/lib:/usr/lib64:/usr/lib"
    --env "PATH=${CONTAINER_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --env PYTORCH_ALLOC_CONF=expandable_segments:True
    --env NCCL_DEBUG=ERROR
    --env NCCL_TIMEOUT=1800
    --env 'NCCL_SOCKET_IFNAME=^docker0,lo'
    --env NCCL_BUFFSIZE=2097152
    --env NCCL_MAX_NCHANNELS=4
    --env "MODELSCOPE_CACHE=${CONTAINER_CACHE}/modelscope"
    --env PYTHONUNBUFFERED=1
)

if [[ "${USE_GPU}" == "true" ]]; then
    docker_args+=(--gpus "${gpu_request}" --env "NVIDIA_VISIBLE_DEVICES=${CUDA_DEVICES}")
fi
if [[ "${CUDA_DEVICES}" != "all" ]]; then
    docker_args+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}")
fi

conda_source="source ${CONTAINER_CONDA}/etc/profile.d/conda.sh"
printf -v container_cuda_q '%q' "${CONTAINER_CUDA}"
printf -v conda_source_q '%q' "${conda_source}"
init_command="ln -sfn ${container_cuda_q} /usr/local/cuda; grep -Fqx ${conda_source_q} /root/.bashrc 2>/dev/null || printf '%s\\n' ${conda_source_q} >> /root/.bashrc; exec sleep infinity"
docker_args+=("${DOCKER_IMAGE}" bash -lc "${init_command}")

echo "=========================================="
echo "SwiftVLN server-17 container bootstrap"
echo "=========================================="
echo "Image:      ${DOCKER_IMAGE}"
echo "Container:  ${CONTAINER_NAME}"
echo "Workdir:    ${CONTAINER_WORKDIR}"
echo "GPU:        ${USE_GPU} (${CUDA_DEVICES})"
echo "Network:    ${NETWORK_MODE}"
echo "Recreate:   ${RECREATE}"
echo "Dry run:    ${DRY_RUN}"

if [[ "${DRY_RUN}" == "true" ]]; then
    print_command docker "${docker_args[@]}"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker is not available; run this script on server 17." >&2
    exit 1
fi

for host_path in \
    "${HOST_WORKSPACE}" \
    "${HOST_CONDA}" \
    "${HOST_DATA}" \
    "${HOST_CACHE}" \
    "${HOST_CUDA}"; do
    if [[ ! -e "${host_path}" ]]; then
        echo "[ERROR] Required server-17 mount does not exist: ${host_path}" >&2
        exit 1
    fi
done

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "${RECREATE}" == "true" ]]; then
        echo "[INFO] Explicitly recreating ${CONTAINER_NAME}"
        docker container rm --force "${CONTAINER_NAME}"
    else
        if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
            echo "[INFO] Starting existing container ${CONTAINER_NAME}"
            docker container start "${CONTAINER_NAME}" >/dev/null
        else
            echo "[INFO] Reusing running container ${CONTAINER_NAME}"
        fi
        attach_if_requested
        exit 0
    fi
fi

echo "[INFO] Creating persistent container ${CONTAINER_NAME}"
docker "${docker_args[@]}" >/dev/null
echo "[INFO] Container ${CONTAINER_NAME} is running"
attach_if_requested
