#!/bin/bash
# sync_data.sh - 放在 98 的 workspace 下
# 作用：数据同步工具 (支持全量检查 或 指定子目录快速同步)

# ================= 配置区 =================
# 注意：路径末尾不要带 "/"，我们在逻辑中动态处理
SRC_ROOT="/mnt/data3/jiangjiajun/dataset"
DEST_USER="jiangjiajun"
# 目标服务器IP映射
DEST_IP_73="10.246.152.73"  # 当前服务器（73服务器）
DEST_IP_17="10.246.132.17"  # 第三台服务器（go17）
DEST_ROOT="/mnt/data3/jiangjiajun/dataset"
# ==========================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 数据同步工具启动...${NC}"

# 1. 选择目标服务器IP (第二个参数优先，无参数则询问)
SERVER_CHOICE="$2"
if [ -z "$SERVER_CHOICE" ]; then
    echo -e "${YELLOW}请选择目标服务器：${NC}"
    echo "  [73] : 73服务器 (当前服务器) - $DEST_IP_73"
    echo "  [17] : 17服务器 (go17) - $DEST_IP_17"
    read -p "👉 请输入选择 [73/17，默认73]: " SERVER_INPUT
    
    if [ -z "$SERVER_INPUT" ] || [ "$SERVER_INPUT" == "73" ]; then
        SERVER_CHOICE="73"
        DEST_IP="$DEST_IP_73"
    elif [ "$SERVER_INPUT" == "17" ]; then
        SERVER_CHOICE="17"
        DEST_IP="$DEST_IP_17"
    else
        echo -e "${RED}❌ 无效选择，使用默认73服务器${NC}"
        SERVER_CHOICE="73"
        DEST_IP="$DEST_IP_73"
    fi
else
    if [ "$SERVER_CHOICE" == "73" ]; then
        DEST_IP="$DEST_IP_73"
    elif [ "$SERVER_CHOICE" == "17" ]; then
        DEST_IP="$DEST_IP_17"
    else
        echo -e "${RED}❌ 无效的服务器选择 '$SERVER_CHOICE'，使用默认73服务器${NC}"
        DEST_IP="$DEST_IP_73"
    fi
fi

echo -e "${GREEN}目标服务器: $DEST_IP${NC}"

# 2. 获取用户输入 (参数优先，无参数则询问)
TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo -e "${YELLOW}请选择同步模式：${NC}"
    echo "  数据目录: $SRC_ROOT"
    echo "  [直接回车] : 同步 satnav_datasets 目录 (默认)"
    echo "  [all]      : 同步整个 dataset 目录"
    echo "  [输入目录] : 仅同步指定子目录 (例如 'mp3d')"
    read -p "👉 请输入: " USER_INPUT

    if [ -z "$USER_INPUT" ]; then
        TARGET_DIR="satnav_datasets"
    else
        TARGET_DIR="$USER_INPUT"
    fi
fi

# 3. 构建 Rsync 命令
if [ "$TARGET_DIR" == "ALL" ] || [ "$TARGET_DIR" == "all" ]; then
    # === 模式一：全量同步 ===
    echo -e "${YELLOW}正在进行 [全量同步] 检查...${NC}"
    # 注意：源路径末尾加 /，表示同步该目录下的内容，不包含目录本身
    CMD="rsync -avP --update $SRC_ROOT/ $DEST_USER@$DEST_IP:$DEST_ROOT/"
else
    # === 模式二：子目录同步 (策略A) ===
    FULL_SRC_PATH="$SRC_ROOT/$TARGET_DIR"
    
    # 检查本地目录是否存在
    if [ ! -d "$FULL_SRC_PATH" ]; then
        echo -e "${RED}❌ 错误：在源服务器上找不到目录：$FULL_SRC_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}正在进行 [子目录同步]：$TARGET_DIR${NC}"
    # 注意：这里源路径末尾 不加 /，表示把这个文件夹本身传过去
    CMD="rsync -avP --update $FULL_SRC_PATH $DEST_USER@$DEST_IP:$DEST_ROOT/"
fi

# 4. 显示并执行命令
echo "------------------------------------------------"
echo "执行命令: $CMD"
echo "------------------------------------------------"

# 使用 time 统计耗时
time $CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 同步完成！${NC}"
else
    echo -e "${RED}❌ 同步过程中发生错误。${NC}"
fi