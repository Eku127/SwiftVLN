#!/bin/bash

# =============================================================================
# Docker 启动脚本 - Qwen3-VL-235B-A22B-FP8 推理
# 使用 swift/vllm 后端进行多模态大模型推理
# 用法: ./run_qwen3_vl_235b.sh [instruct|thinking]
#       默认使用 instruct 模式
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 解析命令行参数
# -----------------------------------------------------------------------------

MODEL_TYPE="${1:-instruct}"  # 默认使用 instruct

# 验证参数有效性
if [[ "${MODEL_TYPE}" != "instruct" && "${MODEL_TYPE}" != "thinking" ]]; then
    echo "错误: 无效的参数 '${MODEL_TYPE}'"
    echo "用法: $0 [instruct|thinking]"
    echo "      默认使用 instruct 模式"
    exit 1
fi

# 将首字母大写（兼容性更好的方法）
MODEL_TYPE_CAPITALIZED=$(echo "${MODEL_TYPE}" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')

# -----------------------------------------------------------------------------
# 配置参数（可根据需要修改）
# -----------------------------------------------------------------------------

# Docker 镜像
DOCKER_IMAGE="modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3"

# 模型配置（根据参数动态设置）
MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-${MODEL_TYPE_CAPITALIZED}-FP8"
SERVED_MODEL_NAME="Qwen3-VL-235B"

# GPU 配置（8 张 H100 80GB）
# 可以修改为 "0,1,2,3" 等指定部分 GPU
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPU_COUNT=8

# 端口配置
HOST_PORT=8000
CONTAINER_PORT=8000

# 挂载路径配置
HOME_DIR="/mnt/data1/home/jiangjiajun"
WORKSPACE_DIR="${HOME_DIR}/workspace"
MODEL_CACHE_DIR="${HOME_DIR}/.cache"           # 模型缓存目录（已下载的模型在这里）
HF_HOME="${HOME_DIR}/.cache/huggingface"       # HuggingFace 缓存
MODELSCOPE_CACHE="${HOME_DIR}/.cache/modelscope"  # ModelScope 缓存（模型已下载在此）

# 已下载的本地模型路径（根据参数动态设置）
LOCAL_MODEL_PATH="${MODELSCOPE_CACHE}/hub/models/Qwen/Qwen3-VL-235B-A22B-${MODEL_TYPE_CAPITALIZED}-FP8"

# 容器名称（包含模型类型）
CONTAINER_NAME="qwen3_vl_235b_${MODEL_TYPE}_inference"

# vLLM 推理参数
TENSOR_PARALLEL_SIZE=${GPU_COUNT}              # 张量并行数量，通常等于 GPU 数量
MAX_MODEL_LEN=32768                            # 最大序列长度
GPU_MEMORY_UTILIZATION=0.95                    # GPU 显存利用率

# -----------------------------------------------------------------------------
# 创建必要的目录
# -----------------------------------------------------------------------------

echo "=========================================="
echo "创建缓存目录..."
echo "=========================================="

mkdir -p "${MODEL_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${MODELSCOPE_CACHE}"

# -----------------------------------------------------------------------------
# 停止并删除已存在的同名容器
# -----------------------------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "停止并删除已存在的容器: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# 启动 Docker 容器
# -----------------------------------------------------------------------------

echo "=========================================="
echo "启动 Docker 容器..."
echo "模型类型: ${MODEL_TYPE}"
echo "模型: ${MODEL_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES} (共 ${GPU_COUNT} 张)"
echo "端口: ${HOST_PORT}"
echo "=========================================="

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODELSCOPE_CACHE=/root/.cache/modelscope \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
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
    \
    swift deploy \
        --model "${MODEL_NAME}" \
        --infer_backend vllm \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --vllm_tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --vllm_enable_expert_parallel \
        --max_model_len ${MAX_MODEL_LEN} \
        --vllm_gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --host 0.0.0.0 \
        --port ${CONTAINER_PORT}

echo ""
echo "=========================================="
echo "容器已启动！"
echo "=========================================="
echo ""
echo "查看日志:        docker logs -f ${CONTAINER_NAME}"
echo "进入容器:        docker exec -it ${CONTAINER_NAME} bash"
echo "停止容器:        docker stop ${CONTAINER_NAME}"
echo "删除容器:        docker rm ${CONTAINER_NAME}"
echo ""
echo "API 测试命令:"
echo "=========================================="
cat << 'EOF'
# 文本对话测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-235B",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# 图像理解测试（需要提供图片 URL 或 base64）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-235B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "请描述这张图片"}
        ]
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
EOF
echo "=========================================="
