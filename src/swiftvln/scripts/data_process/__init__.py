"""
SatNav VLN 数据处理工具包

提供以下功能：
1. process_episodes - 处理VLN episodes数据，按城市和类型整理成train/eval数据集
2. run_all - 一键运行所有数据处理步骤

Usage:
    # 处理所有数据
    python run_all.py ver_260202
    
    # 只处理episodes
    python process_episodes.py ver_260202
    
"""

from .config import (
    DATASET_ROOT,
    TRAIN_CITIES,
    EVAL_CITIES,
    ALL_CITIES,
    get_dataset_path,
    get_data_dir,
    get_episodes_dir,
)

from .process_episodes import process_episodes
from .run_all import run_all

__all__ = [
    "DATASET_ROOT",
    "TRAIN_CITIES",
    "EVAL_CITIES",
    "ALL_CITIES",
    "get_dataset_path",
    "get_data_dir",
    "get_episodes_dir",
    "process_episodes",
    "run_all"
]
