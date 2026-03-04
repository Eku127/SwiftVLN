#!/bin/bash

# =============================================================================
# Docker 交互式启动脚本 - 进入容器手动操作
# =============================================================================

set -e

# Docker 镜像
DOCKER_IMAGE="modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3"

# GPU 配置
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 挂载路径配置
HOME_DIR="/mnt/data1/home/jiangjiajun"
WORKSPACE_DIR="${HOME_DIR}/workspace"
MODEL_CACHE_DIR="${HOME_DIR}/.cache"
HF_HOME="${HOME_DIR}/.cache/huggingface"
MODELSCOPE_CACHE="${HOME_DIR}/.cache/modelscope"

# 容器名称
CONTAINER_NAME="qwen3_vl_interactive"

# 创建必要的目录
mkdir -p "${MODEL_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${MODELSCOPE_CACHE}"

# 停止已存在的同名容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

echo "=========================================="
echo "启动交互式容器..."
echo "=========================================="

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODELSCOPE_CACHE=/root/.cache/modelscope \
    -e HF_ENDPOINT=https://hf-mirror.com \
    \
    -v "${WORKSPACE_DIR}:/workspace:rw" \
    -v "${MODEL_CACHE_DIR}:/root/.cache:rw" \
    -v "${MODELSCOPE_CACHE}/hub/models:/root/.cache/modelscope/models:rw" \
    -v "/mnt/data1:/mnt/data1:rw" \
    -v "/mnt/data2:/mnt/data2:rw" \
    \
    --ipc=host \
    --network=host \
    \
    "${DOCKER_IMAGE}" \
    bash

# 进入容器后可以手动运行：
# swift deploy \
#     --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
#     --infer_backend vllm \
#     --served_model_name Qwen3-VL-235B \
#     --vllm_tensor_parallel_size 8 \
#     --vllm_enable_expert_parallel \
#     --max_model_len 32768 \
#     --vllm_gpu_memory_utilization 0.95 \
#     --host 0.0.0.0 \
#     --port 8000
