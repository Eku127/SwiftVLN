#!/bin/bash

# =============================================================================
# 停止并删除 Qwen3-VL-235B 推理容器（支持 instruct 和 thinking 模式）
# =============================================================================

CONTAINER_PATTERN="qwen3_vl_235b_*_inference"
CONTAINERS=$(docker ps -a --format '{{.Names}}' | grep -E "^qwen3_vl_235b_(instruct|thinking)_inference$" || true)

echo "=========================================="
echo "停止并删除 Qwen3-VL-235B 推理容器"
echo "=========================================="

if [ -z "${CONTAINERS}" ]; then
    echo "⚠️  未找到相关容器"
else
    # 遍历所有匹配的容器
    for CONTAINER_NAME in ${CONTAINERS}; do
        echo ""
        echo "处理容器: ${CONTAINER_NAME}"
        
        # 检查容器是否正在运行
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "  正在停止容器..."
            docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        fi
        
        echo "  正在删除容器..."
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
        
        echo "  ✅ 容器 ${CONTAINER_NAME} 已成功停止并删除"
    done
    
    echo ""
    echo "✅ 所有相关容器已处理完成"
fi

echo ""
echo "=========================================="
echo "当前运行中的 Qwen3-VL-235B 容器:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAMES|qwen3_vl_235b" || echo "无相关容器运行"
echo "=========================================="
