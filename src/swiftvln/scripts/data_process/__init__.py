"""
SatNav VLN 数据处理工具包

提供以下功能：
1. process_episodes - 处理VLN episodes数据，按城市和类型整理成train/eval数据集
2. convert_qa_to_swift - 将qa.json转换为ms-swift JSONL格式
3. run_all - 一键运行所有数据处理步骤

Usage:
    # 处理所有数据
    python run_all.py ver_260202
    
    # 只处理episodes
    python process_episodes.py ver_260202
    
    # 只转换QA数据
    python convert_qa_to_swift.py ver_260202
"""

from .config import (
    DATASET_ROOT,
    TRAIN_CITIES,
    EVAL_CITIES,
    ALL_CITIES,
    get_dataset_path,
    get_data_dir,
    get_episodes_dir,
    get_qa_output_path
)

from .process_episodes import process_episodes
from .convert_qa_to_swift import convert_qa_to_swift
from .run_all import run_all

__all__ = [
    "DATASET_ROOT",
    "TRAIN_CITIES",
    "EVAL_CITIES",
    "ALL_CITIES",
    "get_dataset_path",
    "get_data_dir",
    "get_episodes_dir",
    "get_qa_output_path",
    "process_episodes",
    "convert_qa_to_swift",
    "run_all"
]
