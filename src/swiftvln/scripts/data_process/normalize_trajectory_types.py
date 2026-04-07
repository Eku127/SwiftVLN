#!/usr/bin/env python3
"""
标准化 SatNav episode 的 trajectory 类型。

- 删除不完整城市目录（目前仅移除 Venezia）
- 将 Highway / Multway / Waterway 统一映射为:
  - trajectory_type = Road
  - trajectory_subtype = 原始类型
- 若原始类型已是 Road 且缺少 subtype，则补一个同名 subtype
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .config import get_data_dir
except ImportError:
    from config import get_data_dir

ROAD_SUBTYPE_MAP = {
    "highway": "Highway",
    "multway": "Multiway",
    "multiway": "Multiway",
    "waterway": "Waterway",
}
INCOMPLETE_CITIES = {"Venezia"}


def normalize_episode(episode: Dict) -> bool:
    """原地规范化单条 episode，返回是否发生修改。"""
    original_type = episode.get("trajectory_type")
    original_subtype = episode.get("trajectory_subtype")
    changed = False

    normalized_key = (
        original_type.strip().lower()
        if isinstance(original_type, str)
        else None
    )

    if normalized_key in ROAD_SUBTYPE_MAP:
        episode["trajectory_type"] = "Road"
        episode["trajectory_subtype"] = ROAD_SUBTYPE_MAP[normalized_key]
        changed = True
    elif original_type == "Road" and "trajectory_subtype" not in episode:
        episode["trajectory_subtype"] = "Road"
        changed = True

    normalized_subtype_key = (
        original_subtype.strip().lower()
        if isinstance(original_subtype, str)
        else None
    )
    if normalized_subtype_key in ROAD_SUBTYPE_MAP:
        canonical_subtype = ROAD_SUBTYPE_MAP[normalized_subtype_key]
        if original_subtype != canonical_subtype:
            episode["trajectory_subtype"] = canonical_subtype
            changed = True

    return changed


def normalize_city(city_dir: Path) -> Tuple[int, int]:
    """规范化单个城市的 episodes 文件。"""
    episodes_path = city_dir / "VLN_episodes.json"
    if not episodes_path.exists():
        return 0, 0

    with open(episodes_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes: List[Dict] = payload.get("episodes", [])
    changed = 0
    for episode in episodes:
        if normalize_episode(episode):
            changed += 1

    if changed:
        with open(episodes_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

    return len(episodes), changed


def normalize_dataset(version: str, remove_incomplete: bool = True) -> Dict[str, object]:
    """原地规范化整个数据集版本。"""
    data_dir = get_data_dir(version)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    removed_cities = []
    if remove_incomplete:
        for city in sorted(INCOMPLETE_CITIES):
            city_dir = data_dir / city
            if city_dir.exists():
                shutil.rmtree(city_dir)
                removed_cities.append(city)

    city_stats = {}
    total_episodes = 0
    total_changed = 0

    for city_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        episode_count, changed_count = normalize_city(city_dir)
        city_stats[city_dir.name] = {
            "episodes": episode_count,
            "normalized": changed_count,
        }
        total_episodes += episode_count
        total_changed += changed_count

    return {
        "version": version,
        "removed_cities": removed_cities,
        "cities": city_stats,
        "total_episodes": total_episodes,
        "normalized_episodes": total_changed,
    }


def main():
    parser = argparse.ArgumentParser(description="标准化 SatNav trajectory 类型")
    parser.add_argument("version", type=str, help="数据集版本名称 (e.g., ver_260312)")
    parser.add_argument(
        "--keep-incomplete-cities",
        action="store_true",
        help="保留不完整的城市目录（默认会移除 Venezia）",
    )
    args = parser.parse_args()

    result = normalize_dataset(
        version=args.version,
        remove_incomplete=not args.keep_incomplete_cities,
    )

    print(f"Version: {result['version']}")
    print(f"Removed cities: {result['removed_cities']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Normalized episodes: {result['normalized_episodes']}")


if __name__ == "__main__":
    main()
