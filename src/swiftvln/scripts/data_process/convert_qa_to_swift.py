#!/usr/bin/env python3
"""
将 qa.json 转换为 ms-swift 支持的 JSONL 格式
用于 Qwen2.5-VL 等多模态模型微调

默认只处理训练集城市（自动排除eval城市如BER）

Usage:
    python convert_qa_to_swift.py ver_260202
    python convert_qa_to_swift.py ver_260202 --include-eval  # 包含eval城市
    python convert_qa_to_swift.py ver_260202 --relative-path
    python convert_qa_to_swift.py ver_260202 --output qa_custom.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from .config import (
        TRAIN_CITIES,
        EVAL_CITIES,
        QA_OUTPUT_FILE,
        get_data_dir,
        get_qa_output_path,
    )
except ImportError:
    from config import (
        TRAIN_CITIES,
        EVAL_CITIES,
        QA_OUTPUT_FILE,
        get_data_dir,
        get_qa_output_path,
    )


def convert_message_content(content: List[Dict]) -> Tuple[str, List[str]]:
    """
    将 content 数组格式转换为 ms-swift 格式
    
    Args:
        content: 原始content数组，包含image和text类型
        
    Returns:
        (text_content, images_list): 转换后的文本内容和图片路径列表
    """
    text_parts = []
    images = []
    
    for item in content:
        if item["type"] == "image":
            text_parts.append("<image>")
            images.append(item["image"])
        elif item["type"] == "text":
            text_parts.append(item["text"])
    
    return "".join(text_parts), images


def convert_single_qa_item(
    item: Dict[str, Any], 
    data_dir: Path, 
    use_absolute_path: bool = True
) -> Dict[str, Any]:
    """
    转换单条QA数据为ms-swift格式
    
    Args:
        item: 原始QA数据项
        data_dir: 数据根目录
        use_absolute_path: 是否使用绝对路径
        
    Returns:
        转换后的数据项
    """
    new_item = {
        "messages": [],
        "images": []
    }
    
    all_images = []
    
    for msg in item["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, list):
            # 转换 content 数组格式
            text_content, images = convert_message_content(content)
            all_images.extend(images)
        else:
            # 已经是字符串格式
            text_content = content
        
        new_item["messages"].append({
            "role": role,
            "content": text_content
        })
    
    # 处理图片路径
    if use_absolute_path:
        new_item["images"] = [
            str(data_dir / img) for img in all_images
        ]
    else:
        new_item["images"] = all_images
    
    # 保留有用的元数据
    if "id" in item:
        new_item["id"] = item["id"]
    if "task" in item:
        new_item["task"] = item["task"]
    
    return new_item


def convert_qa_to_swift(
    version: str,
    output_filename: str = None,
    include_eval: bool = False,
    use_absolute_path: bool = True,
    cities: List[str] = None
) -> str:
    """
    转换指定版本的QA数据为ms-swift格式
    
    默认只处理训练集城市，自动过滤eval城市的数据
    
    Args:
        version: 数据集版本名称 (e.g., ver_260202)
        output_filename: 输出文件名，默认为qa_swift.jsonl
        include_eval: 是否包含eval城市数据（默认False，即自动过滤）
        use_absolute_path: 是否使用绝对路径
        cities: 指定要处理的城市列表，None表示使用默认城市
        
    Returns:
        输出文件路径
    """
    data_dir = get_data_dir(version)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 确定要处理的城市（默认只处理训练集城市）
    if cities is not None:
        target_cities = cities
    elif include_eval:
        target_cities = TRAIN_CITIES + EVAL_CITIES
    else:
        target_cities = TRAIN_CITIES
    
    # 确定输出文件名
    if output_filename is None:
        output_filename = QA_OUTPUT_FILE
    
    output_path = get_qa_output_path(version, output_filename)
    
    print(f"Converting QA data for version: {version}")
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_path}")
    print(f"Cities to process: {', '.join(target_cities)}")
    print(f"Use absolute path: {use_absolute_path}")
    print("-" * 60)
    
    all_data = []
    processed_cities = []
    
    for city in sorted(target_cities):
        city_dir = data_dir / city
        qa_file = city_dir / "qa.json"
        
        if not city_dir.exists():
            print(f"  Warning: City directory {city_dir} not found, skipping...")
            continue
            
        if not qa_file.exists():
            print(f"  Warning: {qa_file} not found, skipping...")
            continue
        
        print(f"Processing city: {city}")
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            city_data = json.load(f)
        
        city_count = 0
        for item in city_data:
            converted_item = convert_single_qa_item(
                item, data_dir, use_absolute_path
            )
            all_data.append(converted_item)
            city_count += 1
        
        processed_cities.append(city)
        print(f"  - {city}: {city_count} items")
    
    # 写入 jsonl 格式
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("-" * 60)
    print(f"Conversion complete!")
    print(f"  Processed cities: {', '.join(processed_cities)}")
    print(f"  Total items: {len(all_data)}")
    print(f"  Output file: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="将 qa.json 转换为 ms-swift 支持的 JSONL 格式（默认只处理训练集城市）"
    )
    parser.add_argument(
        "version",
        type=str,
        help="数据集版本名称 (e.g., ver_260202)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件名 (默认: qa_swift.jsonl)"
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="包含eval城市数据（默认自动过滤eval城市）"
    )
    parser.add_argument(
        "--relative-path",
        action="store_true",
        help="使用相对路径（默认使用绝对路径）"
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的城市列表 (e.g., --cities Geneva LA SF)"
    )
    
    args = parser.parse_args()
    
    convert_qa_to_swift(
        version=args.version,
        output_filename=args.output,
        include_eval=args.include_eval,
        use_absolute_path=not args.relative_path,
        cities=args.cities
    )


if __name__ == "__main__":
    main()
