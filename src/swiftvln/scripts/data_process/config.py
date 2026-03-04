#!/usr/bin/env python3
"""
数据处理配置文件
"""

from pathlib import Path

# 数据集根目录
DATASET_ROOT = Path("/mnt/data3/jiangjiajun/dataset/satnav_datasets")

# 城市分类配置
TRAIN_CITIES = ["Geneva", "LA", "LON", "MN", "MN2", "PAR", "ROM", "SF", "BOS", "NY"]
EVAL_CITIES = ["BER", "LA2"]
ALL_CITIES = TRAIN_CITIES + EVAL_CITIES

# Episode 类型
EPISODE_TYPES = {
    "boundary": "Boundary",
    "landmark": "LandmarkSet",
    "road": "Road"
}

# 输出文件名
EPISODE_FILES = {
    "all": "all_episodes.json",
    "boundary": "boundary_episodes.json",
    "landmark": "landmark_episodes.json",
    "road": "road_episodes.json"
}

QA_OUTPUT_FILE = "qa_swift.jsonl"


def get_dataset_path(version: str) -> Path:
    """获取数据集版本路径"""
    return DATASET_ROOT / version


def get_data_dir(version: str) -> Path:
    """获取数据目录路径"""
    return get_dataset_path(version) / "data"


def get_episodes_dir(version: str) -> Path:
    """获取episodes输出目录"""
    return get_dataset_path(version) / "episodes"


def get_qa_output_path(version: str, filename: str = None) -> Path:
    """获取QA输出文件路径"""
    if filename is None:
        filename = QA_OUTPUT_FILE
    return get_data_dir(version) / filename
