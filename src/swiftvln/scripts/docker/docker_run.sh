#!/bin/bash
# StreamVLN Docker Container Startup Script
# 
# Usage:
#   bash src/swiftvln/scripts/docker/docker_run.sh
#
# This script starts a Docker container with Ubuntu 22.04 and mounts necessary
# directories including NFS-mounted conda environment.
# You can manually run training/evaluation scripts inside the container.

set -e

# ============================================================================
# Configuration
# ============================================================================
# Docker image (Ubuntu 22.04)
# Found local image: docker.internal.silassz.com/public/ubuntu:22.04
# Note: CUDA support will be provided by nvidia-docker runtime and host CUDA drivers
DOCKER_IMAGE="${DOCKER_IMAGE:-docker.internal.silassz.com/public/ubuntu:22.04}"

# Container name
CONTAINER_NAME="${CONTAINER_NAME:-streamvln-container}"

# Host paths (NFS mounted from 98 server)
# Keep all paths identical between host and container for compatibility
HOST_WORKSPACE="/mnt/data1/home/jiangjiajun/workspace"
HOST_CONDA="/mnt/data1/home/jiangjiajun/miniconda3"
HOST_DATA="/mnt/data3/jiangjiajun/dataset"
HOST_CACHE="/mnt/data1/home/jiangjiajun/.cache"
HOST_CUDA="/usr/local/cuda-13.0"  # CUDA installation on host

# Container paths (SAME as host paths for NFS compatibility)
# This ensures all hardcoded paths in scripts work correctly
CONTAINER_WORKSPACE="/mnt/data1/home/jiangjiajun/workspace"
CONTAINER_CONDA="/mnt/data1/home/jiangjiajun/miniconda3"
CONTAINER_DATA="/mnt/data3/jiangjiajun/dataset"
CONTAINER_CACHE="/mnt/data1/home/jiangjiajun/.cache"
CONTAINER_CUDA="/usr/local/cuda-13.0"

# Working directory inside container
CONTAINER_WORKDIR="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor"

# GPU configuration
# Note: Use specific GPU IDs (e.g., "0,1,2,3,4,5,6,7") or "all" for --gpus flag
CUDA_DEVICES="${CUDA_DEVICES:-all}"
USE_GPU="${USE_GPU:-true}"

# Network mode (host mode for NFS access)
NETWORK_MODE="${NETWORK_MODE:-host}"

# ============================================================================
# Check if container already exists
# ============================================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Container '${CONTAINER_NAME}' already exists"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Stopping and removing existing container..."
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    else
        echo "[INFO] Starting existing container..."
        docker start "${CONTAINER_NAME}" 2>/dev/null || true
        docker exec -it "${CONTAINER_NAME}" /bin/bash
        exit 0
    fi
fi

# ============================================================================
# Build docker run command
# ============================================================================
DOCKER_RUN_ARGS=(
    --name "${CONTAINER_NAME}"
    --rm
    --network "${NETWORK_MODE}"
    --hostname "${CONTAINER_NAME}"
    --workdir "${CONTAINER_WORKDIR}"
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
)

# GPU support
if [ "$USE_GPU" = "true" ]; then
    DOCKER_RUN_ARGS+=(
        --gpus "${CUDA_DEVICES}"
        -e NVIDIA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    )
fi

# Mount volumes
DOCKER_RUN_ARGS+=(
    -v "${HOST_WORKSPACE}:${CONTAINER_WORKSPACE}"
    -v "${HOST_CONDA}:${CONTAINER_CONDA}"
    -v "${HOST_DATA}:${CONTAINER_DATA}"
    -v "${HOST_CACHE}:${CONTAINER_CACHE}"
    -v "${HOST_CUDA}:${CONTAINER_CUDA}:ro"  # Mount CUDA libraries (read-only)
)

# Environment variables
# Note: Don't set CUDA_VISIBLE_DEVICES when using --gpus all (let nvidia-docker handle it)
DOCKER_RUN_ARGS+=(
    -e CUDA_HOME="${CONTAINER_CUDA}"
    -e LD_LIBRARY_PATH="${CONTAINER_CUDA}/lib64:${CONTAINER_CUDA}/lib:/usr/lib64:/usr/lib"
    -e PATH="${CONTAINER_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e PYTORCH_ALLOC_CONF=expandable_segments:True
    -e NCCL_DEBUG=ERROR
    -e NCCL_TIMEOUT=1800
    -e NCCL_SOCKET_IFNAME=^docker0,lo
    -e NCCL_BUFFSIZE=2097152
    -e NCCL_MAX_NCHANNELS=4
    -e MODELSCOPE_CACHE="${CONTAINER_CACHE}/modelscope"
    -e PYTHONUNBUFFERED=1
)

# Only set CUDA_VISIBLE_DEVICES if specific GPUs are requested (not "all")
if [ "${CUDA_DEVICES}" != "all" ]; then
    DOCKER_RUN_ARGS+=(-e CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}")
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "=========================================="
echo "StreamVLN Docker Container Startup"
echo "=========================================="
echo "Image:           ${DOCKER_IMAGE}"
echo "Container:      ${CONTAINER_NAME}"
echo "Network:        ${NETWORK_MODE}"
echo "GPU:            ${USE_GPU} (${CUDA_DEVICES})"
echo "Workdir:        ${CONTAINER_WORKDIR}"
echo "------------------------------------------"
echo "Mounts (same path mapping for NFS compatibility):"
echo "  Workspace:    ${HOST_WORKSPACE} -> ${CONTAINER_WORKSPACE}"
echo "  Conda:        ${HOST_CONDA} -> ${CONTAINER_CONDA}"
echo "  Data:         ${HOST_DATA} -> ${CONTAINER_DATA}"
echo "  Cache:        ${HOST_CACHE} -> ${CONTAINER_CACHE}"
echo "  CUDA:         ${HOST_CUDA} -> ${CONTAINER_CUDA} (read-only)"
echo "------------------------------------------"
echo "Note: All paths are identical to host for script compatibility"
echo "=========================================="

# ============================================================================
# Run container
# ============================================================================
echo "[INFO] Starting container..."
# Create CUDA symlink, setup conda in .bashrc, and start bash
docker run -it "${DOCKER_RUN_ARGS[@]}" "${DOCKER_IMAGE}" \
    bash -c "if [ ! -L /usr/local/cuda ]; then ln -sf /usr/local/cuda-13.0 /usr/local/cuda; fi && \
             if ! grep -q 'source.*conda.sh' ~/.bashrc 2>/dev/null; then \
                 echo 'source ${CONTAINER_CONDA}/etc/profile.d/conda.sh' >> ~/.bashrc; \
             fi && \
             exec /bin/bash"

echo "[INFO] Container execution completed."

