#!/usr/bin/env python3
"""
检查 SatNav 数据集的情况
列出所有城市及其 episodes 统计信息

Usage:
    python inspect_data.py ver_260211
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


def inspect_city(city_path: Path) -> Dict[str, Any]:
    """检查单个城市的数据情况"""
    result = {
        "city": city_path.name,
        "has_qa": False,
        "has_episodes": False,
        "qa_items": 0,
        "episodes": 0,
        "episodes_by_type": {},
        "traj_dirs": 0,
        "img_dirs": 0
    }

    # 检查 qa.json
    qa_file = city_path / "qa.json"
    if qa_file.exists():
        result["has_qa"] = True
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
            result["qa_items"] = len(qa_data)

    # 检查 VLN_episodes.json
    episodes_file = city_path / "VLN_episodes.json"
    if episodes_file.exists():
        result["has_episodes"] = True
        with open(episodes_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            episodes = data.get("episodes", [])
            result["episodes"] = len(episodes)

            # 统计各类型 episodes
            type_counter = Counter(ep.get("trajectory_type", "Unknown") for ep in episodes)
            result["episodes_by_type"] = dict(type_counter)

    # 检查子目录
    for item in city_path.iterdir():
        if item.is_dir():
            if "traj" in item.name.lower():
                result["traj_dirs"] += 1
            elif "img" in item.name.lower():
                result["img_dirs"] += 1

    return result


def inspect_all(data_dir: Path):
    """检查所有城市的数据"""
    print("=" * 80)
    print("SatNav Data Inspection")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("=" * 80)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    # 获取所有城市目录
    city_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not city_dirs:
        print("No city directories found!")
        return

    print(f"\nFound {len(city_dirs)} city directories:\n")

    all_results = []

    for city_dir in city_dirs:
        result = inspect_city(city_dir)
        all_results.append(result)

        # 打印城市信息
        print(f"📍 {result['city']}")
        if result['has_episodes']:
            print(f"   Episodes: {result['episodes']}")
            for ep_type, count in result['episodes_by_type'].items():
                print(f"     - {ep_type}: {count}")
        else:
            print(f"   Episodes: ❌ No VLN_episodes.json")

        if result['has_qa']:
            print(f"   QA items: {result['qa_items']}")
        else:
            print(f"   QA items: ❌ No qa.json")

        # 显示子目录统计
        dirs_info = []
        if result['traj_dirs'] > 0:
            dirs_info.append(f"{result['traj_dirs']} traj dirs")
        if result['img_dirs'] > 0:
            dirs_info.append(f"{result['img_dirs']} img dirs")
        if dirs_info:
            print(f"   Subdirs: {', '.join(dirs_info)}")

        print()

    # 汇总统计
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    total_episodes = sum(r['episodes'] for r in all_results)
    total_qa = sum(r['qa_items'] for r in all_results)
    cities_with_episodes = sum(1 for r in all_results if r['has_episodes'])
    cities_with_qa = sum(1 for r in all_results if r['has_qa'])

    print(f"Total cities: {len(all_results)}")
    print(f"Cities with episodes: {cities_with_episodes}")
    print(f"Cities with QA: {cities_with_qa}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total QA items: {total_qa}")

    # Episodes 类型汇总
    all_types = Counter()
    for r in all_results:
        all_types.update(r['episodes_by_type'])

    if all_types:
        print("\nEpisodes by type:")
        for ep_type, count in sorted(all_types.items()):
            print(f"  - {ep_type}: {count}")

    print("\n" + "=" * 80)
    print("Cities list for config.py:")
    print("=" * 80)
    all_city_names = [r['city'] for r in all_results if r['has_episodes'] or r['has_qa']]
    print(f"ALL_CITIES = {all_city_names}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="检查 SatNav 数据集情况"
    )
    parser.add_argument(
        "version",
        type=str,
        help="数据集版本名称 (e.g., ver_260211)"
    )

    args = parser.parse_args()

    from config import get_data_dir
    data_dir = get_data_dir(args.version)
    inspect_all(data_dir)


if __name__ == "__main__":
    main()
