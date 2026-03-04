#!/bin/bash
# sync_and_verify.sh - SatNav 数据同步与验证脚本
# Step 3: 将处理后的数据同步到 73 和 17 服务器，并验证一致性

# ================= 配置区 =================
SRC_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
DEST_USER="jiangjiajun"
# 目标服务器IP映射
DEST_IP_73="10.246.152.73"
DEST_IP_17="10.246.132.17"
DEST_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================= 参数处理 =================
VERSION="$1"

if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: 请指定数据集版本${NC}"
    echo "Usage: sync_and_verify.sh <version>"
    echo "Example: sync_and_verify.sh ver_260211"
    exit 1
fi

VERSION_PATH="${SRC_ROOT}/${VERSION}"

# 检查版本目录是否存在
if [ ! -d "${VERSION_PATH}" ]; then
    echo -e "${RED}Error: 版本目录不存在: ${VERSION_PATH}${NC}"
    exit 1
fi

echo "========================================================================"
echo "SatNav 数据同步与验证 (Step 3)"
echo "========================================================================"
echo "版本: ${VERSION}"
echo "源目录: ${VERSION_PATH}"
echo "目标服务器: 73, 17"
echo "========================================================================"

# ========================================================================
# 同步到 73 服务器
# ========================================================================

echo ""
echo "========================================================================"
echo "步骤 1: 同步到 73 服务器"
echo "========================================================================"
echo "目标: ${DEST_IP_73}:${DEST_ROOT}/${VERSION}/"
echo ""

# 使用 rsync 同步到 73 服务器
rsync -avP --update "${VERSION_PATH}/" ${DEST_USER}@${DEST_IP_73}:${DEST_ROOT}/${VERSION}/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 同步到 73 服务器完成！${NC}"
else
    echo -e "${RED}❌ 同步到 73 服务器失败！${NC}"
    read -p "是否继续同步到 17 服务器? [y/N]: " CONTINUE_17
    if [[ ! "$CONTINUE_17" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ========================================================================
# 同步到 17 服务器
# ========================================================================

echo ""
echo "========================================================================"
echo "步骤 2: 同步到 17 服务器"
echo "========================================================================"
echo "目标: ${DEST_IP_17}:${DEST_ROOT}/${VERSION}/"
echo ""

# 使用 rsync 同步到 17 服务器
rsync -avP --update "${VERSION_PATH}/" ${DEST_USER}@${DEST_IP_17}:${DEST_ROOT}/${VERSION}/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 同步到 17 服务器完成！${NC}"
else
    echo -e "${RED}❌ 同步到 17 服务器失败！${NC}"
    echo -e "${YELLOW}注意: 73 服务器可能已同步成功${NC}"
fi

# ========================================================================
# 验证数据一致性
# ========================================================================

echo ""
echo "========================================================================"
echo "步骤 3: 验证三台服务器数据一致性"
echo "========================================================================"

# 创建临时目录存放验证结果
TEMP_DIR=$(mktemp -d)
echo "临时目录: ${TEMP_DIR}"

# 统计本地数据
echo ""
echo -e "${BLUE}正在统计 98 服务器（本地）数据...${NC}"
find "${VERSION_PATH}" -type f > "${TEMP_DIR}/local_files.txt"
echo "98 服务器文件数: $(wc -l < ${TEMP_DIR}/local_files.txt)"

# 统计 73 服务器数据
echo ""
echo -e "${BLUE}正在统计 73 服务器数据...${NC}"
ssh ${DEST_USER}@${DEST_IP_73} "find ${DEST_ROOT}/${VERSION} -type f" > "${TEMP_DIR}/server73_files.txt"
echo "73 服务器文件数: $(wc -l < ${TEMP_DIR}/server73_files.txt)"

# 统计 17 服务器数据
echo ""
echo -e "${BLUE}正在统计 17 服务器数据...${NC}"
ssh ${DEST_USER}@${DEST_IP_17} "find ${DEST_ROOT}/${VERSION} -type f" > "${TEMP_DIR}/server17_files.txt"
echo "17 服务器文件数: $(wc -l < ${TEMP_DIR}/server17_files.txt)"

# 比较差异
echo ""
echo "========================================================================"
echo "数据一致性检查结果"
echo "========================================================================"

# 比较 98 和 73
diff "${TEMP_DIR}/local_files.txt" "${TEMP_DIR}/server73_files.txt" > "${TEMP_DIR}/diff_73.txt" 2>&1
if [ $? -eq 0 ] && [ ! -s "${TEMP_DIR}/diff_73.txt" ]; then
    echo -e "${GREEN}✅ 98 与 73 服务器: 数据一致${NC}"
else
    echo -e "${YELLOW}⚠️  98 与 73 服务器: 存在差异${NC}"
    DIFF_COUNT_73=$(wc -l < ${TEMP_DIR}/diff_73.txt)
    echo "   差异行数: ${DIFF_COUNT_73}"
    if [ "$DIFF_COUNT_73" -lt 50 ]; then
        echo "   差异详情:"
        head -20 "${TEMP_DIR}/diff_73.txt" | sed 's/^/   /'
    fi
fi

# 比较 98 和 17
diff "${TEMP_DIR}/local_files.txt" "${TEMP_DIR}/server17_files.txt" > "${TEMP_DIR}/diff_17.txt" 2>&1
if [ $? -eq 0 ] && [ ! -s "${TEMP_DIR}/diff_17.txt" ]; then
    echo -e "${GREEN}✅ 98 与 17 服务器: 数据一致${NC}"
else
    echo -e "${YELLOW}⚠️  98 与 17 服务器: 存在差异${NC}"
    DIFF_COUNT_17=$(wc -l < ${TEMP_DIR}/diff_17.txt)
    echo "   差异行数: ${DIFF_COUNT_17}"
    if [ "$DIFF_COUNT_17" -lt 50 ]; then
        echo "   差异详情:"
        head -20 "${TEMP_DIR}/diff_17.txt" | sed 's/^/   /'
    fi
fi

# 比较 73 和 17
diff "${TEMP_DIR}/server73_files.txt" "${TEMP_DIR}/server17_files.txt" > "${TEMP_DIR}/diff_73_17.txt" 2>&1
if [ $? -eq 0 ] && [ ! -s "${TEMP_DIR}/diff_73_17.txt" ]; then
    echo -e "${GREEN}✅ 73 与 17 服务器: 数据一致${NC}"
else
    echo -e "${YELLOW}⚠️  73 与 17 服务器: 存在差异${NC}"
    DIFF_COUNT_73_17=$(wc -l < ${TEMP_DIR}/diff_73_17.txt)
    echo "   差异行数: ${DIFF_COUNT_73_17}"
fi

# 关键数据文件验证
echo ""
echo "========================================================================"
echo "关键数据文件验证"
echo "========================================================================"

# 检查 episodes
echo ""
echo "检查 episodes/train/all_episodes.json..."
LOCAL_SIZE=$(stat -f%z "${VERSION_PATH}/episodes/train/all_episodes.json" 2>/dev/null || stat -c%s "${VERSION_PATH}/episodes/train/all_episodes.json" 2>/dev/null)
SERVER73_SIZE=$(ssh ${DEST_USER}@${DEST_IP_73} "stat -f%z ${DEST_ROOT}/${VERSION}/episodes/train/all_episodes.json 2>/dev/null || stat -c%s ${DEST_ROOT}/${VERSION}/episodes/train/all_episodes.json 2>/dev/null")
SERVER17_SIZE=$(ssh ${DEST_USER}@${DEST_IP_17} "stat -f%z ${DEST_ROOT}/${VERSION}/episodes/train/all_episodes.json 2>/dev/null || stat -c%s ${DEST_ROOT}/${VERSION}/episodes/train/all_episodes.json 2>/dev/null")

if [ "$LOCAL_SIZE" = "$SERVER73_SIZE" ] && [ "$LOCAL_SIZE" = "$SERVER17_SIZE" ]; then
    echo -e "${GREEN}✅ episodes/train/all_episodes.json 大小一致: ${LOCAL_SIZE} bytes${NC}"
else
    echo -e "${YELLOW}⚠️  episodes/train/all_episodes.json 大小不一致:${NC}"
    echo "   98: ${LOCAL_SIZE} bytes"
    echo "   73: ${SERVER73_SIZE} bytes"
    echo "   17: ${SERVER17_SIZE} bytes"
fi

# 检查 trajectory_data
echo ""
echo "检查 trajectory_data/annotations.json..."
LOCAL_SIZE=$(stat -f%z "${VERSION_PATH}/trajectory_data/annotations.json" 2>/dev/null || stat -c%s "${VERSION_PATH}/trajectory_data/annotations.json" 2>/dev/null)
SERVER73_SIZE=$(ssh ${DEST_USER}@${DEST_IP_73} "stat -f%z ${DEST_ROOT}/${VERSION}/trajectory_data/annotations.json 2>/dev/null || stat -c%s ${DEST_ROOT}/${VERSION}/trajectory_data/annotations.json 2>/dev/null")
SERVER17_SIZE=$(ssh ${DEST_USER}@${DEST_IP_17} "stat -f%z ${DEST_ROOT}/${VERSION}/trajectory_data/annotations.json 2>/dev/null || stat -c%s ${DEST_ROOT}/${VERSION}/trajectory_data/annotations.json 2>/dev/null")

if [ "$LOCAL_SIZE" = "$SERVER73_SIZE" ] && [ "$LOCAL_SIZE" = "$SERVER17_SIZE" ]; then
    echo -e "${GREEN}✅ trajectory_data/annotations.json 大小一致: ${LOCAL_SIZE} bytes${NC}"
else
    echo -e "${YELLOW}⚠️  trajectory_data/annotations.json 大小不一致:${NC}"
    echo "   98: ${LOCAL_SIZE} bytes"
    echo "   73: ${SERVER73_SIZE} bytes"
    echo "   17: ${SERVER17_SIZE} bytes"
fi

# 统计 images 目录数量
echo ""
echo "检查 trajectory_data/images/ 目录数量..."
LOCAL_COUNT=$(find "${VERSION_PATH}/trajectory_data/images" -maxdepth 1 -type d | wc -l)
SERVER73_COUNT=$(ssh ${DEST_USER}@${DEST_IP_73} "find ${DEST_ROOT}/${VERSION}/trajectory_data/images -maxdepth 1 -type d 2>/dev/null | wc -l")
SERVER17_COUNT=$(ssh ${DEST_USER}@${DEST_IP_17} "find ${DEST_ROOT}/${VERSION}/trajectory_data/images -maxdepth 1 -type d 2>/dev/null | wc -l")

if [ "$LOCAL_COUNT" = "$SERVER73_COUNT" ] && [ "$LOCAL_COUNT" = "$SERVER17_COUNT" ]; then
    echo -e "${GREEN}✅ trajectory_data/images/ 目录数一致: ${LOCAL_COUNT}${NC}"
else
    echo -e "${YELLOW}⚠️  trajectory_data/images/ 目录数不一致:${NC}"
    echo "   98: ${LOCAL_COUNT} dirs"
    echo "   73: ${SERVER73_COUNT} dirs"
    echo "   17: ${SERVER17_COUNT} dirs"
fi

# 清理临时文件
rm -rf "${TEMP_DIR}"

# 最终总结
echo ""
echo "========================================================================"
echo "Step 3 完成！"
echo "========================================================================"
echo -e "${GREEN}数据已同步到 73 和 17 服务器${NC}"
echo -e "运行以下命令查看详细状态：${NC}"
echo "  73 服务器: ssh ${DEST_USER}@${DEST_IP_73}"
echo "  17 服务器: ssh ${DEST_USER}@${DEST_IP_17}"
echo "========================================================================"
