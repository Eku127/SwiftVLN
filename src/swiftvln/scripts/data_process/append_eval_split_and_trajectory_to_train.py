#!/usr/bin/env python3
"""
Append an eval split and its standalone trajectory data into train.

Typical workflow:
1. Generate trajectories for an eval split into a standalone output dir.
2. Run this script to merge those trajectories into ``trajectory_data``.
3. Append the corresponding episodes into ``episodes/train``.

By default the script requires every source episode to have a generated
trajectory record. If generation is incomplete, use ``--successful-only`` to
append only the subset with successful trajectories.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

try:
    from .config import DATASET_ROOT, EPISODE_FILES, EPISODE_TYPES
except ImportError:
    from config import DATASET_ROOT, EPISODE_FILES, EPISODE_TYPES


TASK_ORDER = [
    EPISODE_TYPES["boundary"],
    EPISODE_TYPES["landmark"],
    EPISODE_TYPES["road"],
]


def read_episode_records(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("episodes", []))


def write_episode_records(file_path: Path, episodes: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump({"episodes": episodes}, handle, indent=2, ensure_ascii=False)


def read_annotations(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def write_annotations(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def read_summary(file_path: Path) -> list[dict[str, Any]]:
    records = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_scene_name(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    if "/" in text:
        text = text.split("/")[-1]
    return text


def normalize_episode_id(value: Any) -> str:
    return str(value)


def episode_key(episode: dict[str, Any]) -> tuple[str, str]:
    return normalize_scene_name(episode["scene_id"]), normalize_episode_id(episode["episode_id"])


def summary_key(record: dict[str, Any]) -> tuple[str, str]:
    return normalize_scene_name(record.get("scene_id")), normalize_episode_id(record.get("episode_id"))


def compute_type_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(episode["trajectory_type"] for episode in episodes)
    return {task: counter.get(task, 0) for task in TASK_ORDER}


def filter_by_type(episodes: list[dict[str, Any]], trajectory_type: str) -> list[dict[str, Any]]:
    return [episode for episode in episodes if episode["trajectory_type"] == trajectory_type]


def write_train_split(train_dir: Path, episodes: list[dict[str, Any]]) -> None:
    write_episode_records(train_dir / EPISODE_FILES["all"], episodes)
    write_episode_records(
        train_dir / EPISODE_FILES["boundary"],
        filter_by_type(episodes, EPISODE_TYPES["boundary"]),
    )
    write_episode_records(
        train_dir / EPISODE_FILES["landmark"],
        filter_by_type(episodes, EPISODE_TYPES["landmark"]),
    )
    write_episode_records(
        train_dir / EPISODE_FILES["road"],
        filter_by_type(episodes, EPISODE_TYPES["road"]),
    )


def ensure_unique_episode_keys(episodes: list[dict[str, Any]], label: str) -> None:
    seen: set[tuple[str, str]] = set()
    for episode in episodes:
        key = episode_key(episode)
        if key in seen:
            raise RuntimeError(f"Duplicate scene_id + episode_id found in {label}: {key}")
        seen.add(key)


def index_summary(records: list[dict[str, Any]], label: str) -> OrderedDict[str, dict[str, Any]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        video = str(record["video"])
        if video in indexed:
            raise RuntimeError(f"Duplicate summary video in {label}: {video}")
        indexed[video] = record
    return indexed


def index_annotations(records: list[dict[str, Any]], label: str) -> OrderedDict[str, dict[str, Any]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        video = str(record["video"])
        if video in indexed:
            raise RuntimeError(f"Duplicate annotation video in {label}: {video}")
        indexed[video] = record
    return indexed


def files_identical(path_a: Path, path_b: Path) -> bool:
    if not path_a.exists() or not path_b.exists():
        return False
    if path_a.stat().st_size != path_b.stat().st_size:
        return False
    with path_a.open("rb") as handle_a, path_b.open("rb") as handle_b:
        while True:
            chunk_a = handle_a.read(1024 * 1024)
            chunk_b = handle_b.read(1024 * 1024)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def link_or_copy_file(source: Path, dest: Path, copy_mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if files_identical(source, dest):
            return
        raise RuntimeError(f"Destination exists with different content: {dest}")
    if copy_mode == "hardlink":
        try:
            os.link(source, dest)
            return
        except OSError:
            pass
    shutil.copy2(source, dest)


def copy_tree(source: Path, dest: Path, copy_mode: str) -> None:
    if source.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            copy_tree(child, dest / child.name, copy_mode)
        return
    link_or_copy_file(source, dest, copy_mode)


def reindex_records_in_order(
    ordered_videos: list[str],
    records_by_video: OrderedDict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for idx, video in enumerate(ordered_videos):
        item = copy.deepcopy(records_by_video[video])
        item["id"] = idx
        output.append(item)
    return output


def append_eval_split_and_trajectory_to_train(
    version: str,
    source_split: str,
    source_trajectory_dirname: str,
    target_trajectory_dirname: str,
    successful_only: bool,
    copy_mode: str,
) -> dict[str, Any]:
    dataset_root = DATASET_ROOT / version
    train_dir = dataset_root / "episodes" / "train"
    source_split_dir = dataset_root / "episodes" / "eval" / source_split
    source_trajectory_root = dataset_root / source_trajectory_dirname
    target_trajectory_root = dataset_root / target_trajectory_dirname

    train_episodes = read_episode_records(train_dir / EPISODE_FILES["all"])
    source_episodes = read_episode_records(source_split_dir / EPISODE_FILES["all"])
    target_summary = read_summary(target_trajectory_root / "summary.json")
    target_annotations = read_annotations(target_trajectory_root / "annotations.json")
    source_summary = read_summary(source_trajectory_root / "summary.json")
    source_annotations = read_annotations(source_trajectory_root / "annotations.json")

    ensure_unique_episode_keys(train_episodes, "train split")
    ensure_unique_episode_keys(source_episodes, f"eval split {source_split}")

    train_keys = {episode_key(episode) for episode in train_episodes}
    source_episode_by_key = OrderedDict((episode_key(episode), episode) for episode in source_episodes)
    source_summary_by_key = OrderedDict((summary_key(record), record) for record in source_summary)

    missing_trajectory_keys = [
        key for key in source_episode_by_key if key not in source_summary_by_key
    ]
    if missing_trajectory_keys and not successful_only:
        raise RuntimeError(
            f"{source_split} has {len(missing_trajectory_keys)} episodes without standalone trajectory output; "
            "rerun with --successful-only if you want to merge only the successful subset"
        )

    selected_source_keys = [
        key for key in source_episode_by_key if successful_only is False or key in source_summary_by_key
    ]
    selected_source_episodes = [source_episode_by_key[key] for key in selected_source_keys if key in source_summary_by_key or not successful_only]
    if successful_only:
        selected_source_episodes = [source_episode_by_key[key] for key in selected_source_keys]

    duplicate_train_keys = [key for key in selected_source_keys if key in train_keys]
    if duplicate_train_keys:
        raise RuntimeError(f"Source split already overlaps with train: {duplicate_train_keys[:5]}")

    target_summary_by_video = index_summary(target_summary, "target trajectory")
    source_summary_by_video = index_summary(source_summary, "source trajectory")
    target_annotations_by_video = index_annotations(target_annotations, "target trajectory")
    source_annotations_by_video = index_annotations(source_annotations, "source trajectory")

    if set(target_summary_by_video) != set(target_annotations_by_video):
        raise RuntimeError("Target summary / annotations video sets differ")
    if set(source_summary_by_video) != set(source_annotations_by_video):
        raise RuntimeError("Source summary / annotations video sets differ")

    selected_source_videos = []
    for key in selected_source_keys:
        if key not in source_summary_by_key:
            continue
        selected_source_videos.append(str(source_summary_by_key[key]["video"]))

    overlap_videos = set(target_summary_by_video) & set(selected_source_videos)
    if overlap_videos:
        raise RuntimeError(f"Source trajectory overlaps with target trajectory videos: {sorted(overlap_videos)[:5]}")

    merged_train = list(train_episodes) + selected_source_episodes
    write_train_split(train_dir, merged_train)

    merged_order = list(target_summary_by_video) + selected_source_videos
    merged_summary_by_video = OrderedDict(target_summary_by_video)
    merged_annotations_by_video = OrderedDict(target_annotations_by_video)
    for video in selected_source_videos:
        merged_summary_by_video[video] = source_summary_by_video[video]
        merged_annotations_by_video[video] = source_annotations_by_video[video]
        source_dir = source_trajectory_root / video
        dest_dir = target_trajectory_root / video
        copy_tree(source_dir, dest_dir, copy_mode=copy_mode)

    merged_summary = reindex_records_in_order(merged_order, merged_summary_by_video)
    merged_annotations = reindex_records_in_order(merged_order, merged_annotations_by_video)
    write_summary(target_trajectory_root / "summary.json", merged_summary)
    write_annotations(target_trajectory_root / "annotations.json", merged_annotations)

    image_dir_names = {
        path.name for path in (target_trajectory_root / "images").iterdir() if path.is_dir()
    }
    referenced_image_dirs = {Path(video).name for video in merged_order}
    if image_dir_names != referenced_image_dirs:
        raise RuntimeError("Merged image directory set does not match merged summary videos")

    manifest = {
        "version": version,
        "source_split": source_split,
        "source_trajectory_dirname": source_trajectory_dirname,
        "target_trajectory_dirname": target_trajectory_dirname,
        "successful_only": successful_only,
        "copy_mode": copy_mode,
        "train_total_before": len(train_episodes),
        "train_total_after": len(merged_train),
        "source_split_total": len(source_episodes),
        "source_selected_total": len(selected_source_episodes),
        "source_missing_trajectory_count": len(missing_trajectory_keys),
        "target_trajectory_before": len(target_summary),
        "target_trajectory_after": len(merged_summary),
        "added_type_counts": compute_type_counts(selected_source_episodes),
        "train_type_counts_after": compute_type_counts(merged_train),
    }
    manifest_path = source_split_dir / f"{source_split}_append_to_train_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append an eval split and its standalone trajectory output into train"
    )
    parser.add_argument("version", help="Dataset version, e.g. ver_260418")
    parser.add_argument("--source-split", default="val_seen")
    parser.add_argument("--source-trajectory-dirname", default="trajectory_data_val_seen")
    parser.add_argument("--target-trajectory-dirname", default="trajectory_data")
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Append only the episodes that exist in the standalone trajectory output",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize source trajectory files into the target trajectory root",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = append_eval_split_and_trajectory_to_train(
        version=args.version,
        source_split=args.source_split,
        source_trajectory_dirname=args.source_trajectory_dirname,
        target_trajectory_dirname=args.target_trajectory_dirname,
        successful_only=args.successful_only,
        copy_mode=args.copy_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
