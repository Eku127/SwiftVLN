#!/usr/bin/env python3
"""
一键处理数据集：生成episodes和转换QA数据

QA数据默认只包含训练集城市（自动过滤eval城市如BER）

Usage:
    python run_all.py ver_260202
    python run_all.py ver_260202 --episodes-only
    python run_all.py ver_260202 --qa-only
    python run_all.py ver_260202 --include-eval  # 包含eval城市的QA数据
"""

import argparse
import sys
from pathlib import Path

from config import get_data_dir, get_dataset_path, TRAIN_CITIES, EVAL_CITIES
from process_episodes import process_episodes
from convert_qa_to_swift import convert_qa_to_swift
from normalize_trajectory_types import normalize_dataset


def run_all(
    version: str,
    episodes_only: bool = False,
    qa_only: bool = False,
    include_eval: bool = False,
    use_absolute_path: bool = True
):
    """
    运行所有数据处理步骤
    
    Args:
        version: 数据集版本名称
        episodes_only: 只处理episodes
        qa_only: 只转换QA数据
        include_eval: QA数据是否包含eval城市（默认False，自动过滤）
        use_absolute_path: 是否使用绝对路径
    """
    dataset_path = get_dataset_path(version)
    data_dir = get_data_dir(version)
    
    print("=" * 70)
    print(f"SatNav Data Processing Pipeline")
    print("=" * 70)
    print(f"Version: {version}")
    print(f"Dataset path: {dataset_path}")
    print(f"Data directory: {data_dir}")
    print(f"Train cities: {', '.join(TRAIN_CITIES)}")
    print(f"Eval cities: {', '.join(EVAL_CITIES)}")
    print("=" * 70)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    results = {}

    print("\n" + "=" * 70)
    print("STEP 0: Normalizing Trajectory Types")
    print("=" * 70)
    try:
        results["normalize"] = normalize_dataset(version)
        print(f"  Removed cities: {results['normalize']['removed_cities']}")
        print(f"  Normalized episodes: {results['normalize']['normalized_episodes']}")
    except Exception as e:
        print(f"Error normalizing trajectory types: {e}")
        results["normalize"] = {"error": str(e)}

    # Step 1: Process episodes
    if not qa_only:
        print("\n" + "=" * 70)
        print("STEP 1: Processing Episodes")
        print("=" * 70)
        try:
            results["episodes"] = process_episodes(version, normalize_first=False)
        except Exception as e:
            print(f"Error processing episodes: {e}")
            results["episodes"] = {"error": str(e)}
    
    # Step 2: Convert QA data (默认只包含训练集城市)
    if not episodes_only:
        print("\n" + "=" * 70)
        print("STEP 2: Converting QA Data (Training Cities Only)")
        if include_eval:
            print("  Note: Including eval cities as requested")
        print("=" * 70)
        try:
            output_file = convert_qa_to_swift(
                version=version,
                include_eval=include_eval,
                use_absolute_path=use_absolute_path
            )
            results["qa"] = output_file
                
        except Exception as e:
            print(f"Error converting QA data: {e}")
            results["qa"] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    if "episodes" in results and isinstance(results["episodes"], dict):
        if "error" not in results["episodes"]:
            print("\nEpisodes:")
            for split, stats in results["episodes"].items():
                if stats:
                    print(f"  {split}: {stats['total']} total "
                          f"({stats['boundary']} boundary, {stats['landmark']} landmark, {stats['road']} road)")
    
    if "qa" in results and not isinstance(results["qa"], dict):
        print(f"\nQA Data: {results['qa']}")
    
    print("\n" + "=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="一键处理数据集：生成episodes和转换QA数据（QA默认只包含训练集城市）"
    )
    parser.add_argument(
        "version",
        type=str,
        help="数据集版本名称 (e.g., ver_260202)"
    )
    parser.add_argument(
        "--episodes-only",
        action="store_true",
        help="只处理episodes"
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="只转换QA数据"
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="QA数据包含eval城市（默认自动过滤eval城市）"
    )
    parser.add_argument(
        "--relative-path",
        action="store_true",
        help="QA数据使用相对路径（默认使用绝对路径）"
    )
    
    args = parser.parse_args()
    
    if args.episodes_only and args.qa_only:
        parser.error("Cannot specify both --episodes-only and --qa-only")
    
    run_all(
        version=args.version,
        episodes_only=args.episodes_only,
        qa_only=args.qa_only,
        include_eval=args.include_eval,
        use_absolute_path=not args.relative_path
    )


if __name__ == "__main__":
    main()
