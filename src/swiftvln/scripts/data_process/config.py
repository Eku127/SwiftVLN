#!/usr/bin/env python3
"""
数据处理配置文件
"""

from pathlib import Path

# 数据集根目录
DATASET_ROOT = Path("/mnt/data3/jiangjiajun/dataset/satnav_datasets")

# 城市分类配置（0316 默认划分）
# eval: Amsterdam-1, Rome-1, NewYork-1
# train: 其余全部城市
TRAIN_CITIES = [
    "Amsterdam-2",
    "Berlin-1",
    "Berlin-2",
    "Berlin-3",
    "Berlin-4",
    "Berlin-5",
    "Boston-1",
    "Boston-2",
    "Boston-3",
    "Boston-4",
    "Boston-5",
    "Brugge-1",
    "Geneva-1",
    "Geneva-2",
    "Geneva-3",
    "Geneva-4",
    "Geneva-5",
    "London-1",
    "London-2",
    "London-3",
    "London-4",
    "LosAngeles-1",
    "LosAngeles-2",
    "LosAngeles-3",
    "LosAngeles-4",
    "Minneapolis-1",
    "Minneapolis-2",
    "Minneapolis-3",
    "Minneapolis-4",
    "Minneapolis-5",
    "Minneapolis-6",
    "NewYork-2",
    "NewYork-3",
    "NewYork-4",
    "NewYork-5",
    "Paris-1",
    "Paris-2",
    "Paris-3",
    "Paris-4",
    "Paris-5",
    "RiodeJaneiro-1",
    "Rome-2",
    "Rome-3",
    "Rome-4",
    "Rome-5",
    "Sydney-1",
    "TheBayArea-1",
    "TheBayArea-2",
    "TheBayArea-3",
    "TheBayArea-4",
    "TheBayArea-5",
    "TheBayArea-6",
]
EVAL_CITIES = ["Amsterdam-1", "Rome-1", "NewYork-1"]
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
