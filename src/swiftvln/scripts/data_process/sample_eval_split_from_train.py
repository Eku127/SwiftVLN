#!/usr/bin/env python3
"""
Sample a new eval split from train episodes and prune matching trajectory data.

The workflow is:
1. Read train episodes from ``episodes/train/all_episodes.json``.
2. Read a reference eval split to obtain exact per-task quotas.
3. Sample eligible train episodes per task with scene-aware round-robin randomization.
4. Write sampled episodes to ``episodes/eval/<output_split>/``.
5. Remove sampled episodes from ``episodes/train/*.json``.
6. Remove matching trajectory summary / annotations / image directories.

Example:
    python -m swiftvln.scripts.data_process.sample_eval_split_from_train \
        ver_260418 --output-split val_seen_update --reference-split val_seen
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter, OrderedDict, defaultdict, deque
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
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_summary(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_episode_id(value: Any) -> str:
    return str(value)


def is_numeric_id(value: str) -> bool:
    return value.isdigit()


def normalize_scene_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.rstrip("/")
    if "/" in text:
        text = text.split("/")[-1]
    return text


def build_episode_key(scene_name: str, episode_id: Any) -> tuple[str, str]:
    return normalize_scene_name(scene_name), normalize_episode_id(episode_id)


def episode_key(episode: dict[str, Any]) -> tuple[str, str]:
    return build_episode_key(episode["scene_id"], episode["episode_id"])


def build_video_rel(scene_name: str, episode_id: Any) -> str:
    episode_text = normalize_episode_id(episode_id)
    if is_numeric_id(episode_text):
        suffix = f"{int(episode_text):06d}"
    else:
        suffix = episode_text
    return f"images/{scene_name}_satnav_{suffix}"


def trajectory_key_from_summary(record: dict[str, Any]) -> tuple[str, str]:
    return build_episode_key(record.get("scene_id"), record.get("episode_id"))


def ensure_unique_episode_keys(episodes: list[dict[str, Any]], label: str) -> None:
    seen: set[tuple[str, str]] = set()
    duplicates = []
    for episode in episodes:
        key = episode_key(episode)
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
    if duplicates:
        raise RuntimeError(f"Duplicate scene_id + episode_id found in {label}: {duplicates[:5]}")


def ensure_unique_summary_keys(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = trajectory_key_from_summary(record)
        if key in indexed:
            raise RuntimeError(f"Duplicate trajectory summary key found: {key}")
        indexed[key] = record
    return indexed


def index_annotations_by_video(records: list[dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        video = str(record["video"])
        if video in indexed:
            raise RuntimeError(f"Duplicate trajectory annotation video found: {video}")
        indexed[video] = record
    return indexed


def compute_type_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(episode["trajectory_type"] for episode in episodes)
    return {task: counter.get(task, 0) for task in TASK_ORDER}


def filter_by_type(episodes: list[dict[str, Any]], trajectory_type: str) -> list[dict[str, Any]]:
    return [episode for episode in episodes if episode["trajectory_type"] == trajectory_type]


def sample_category_scene_round_robin(
    episodes: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(episodes) < target_count:
        raise RuntimeError(
            f"Not enough eligible episodes for sampling: need {target_count}, have {len(episodes)}"
        )

    scene_to_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        scene_to_episodes[episode["scene_id"]].append(episode)

    scenes = list(scene_to_episodes)
    rng.shuffle(scenes)
    for scene_name in scenes:
        rng.shuffle(scene_to_episodes[scene_name])

    queue = deque(scenes)
    selected = []
    while len(selected) < target_count:
        if not queue:
            raise RuntimeError("Sampling queue exhausted before reaching target count")
        scene_name = queue.popleft()
        picked = scene_to_episodes[scene_name].pop()
        selected.append(picked)
        if scene_to_episodes[scene_name]:
            queue.append(scene_name)
    return selected


def reindex_records_in_order(
    ordered_videos: list[str],
    records_by_video: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for new_id, video in enumerate(ordered_videos):
        record = copy.deepcopy(records_by_video[video])
        record["id"] = new_id
        output.append(record)
    return output


def write_split_files(split_dir: Path, episodes: list[dict[str, Any]]) -> None:
    write_episode_records(split_dir / EPISODE_FILES["all"], episodes)
    write_episode_records(
        split_dir / EPISODE_FILES["boundary"],
        filter_by_type(episodes, EPISODE_TYPES["boundary"]),
    )
    write_episode_records(
        split_dir / EPISODE_FILES["landmark"],
        filter_by_type(episodes, EPISODE_TYPES["landmark"]),
    )
    write_episode_records(
        split_dir / EPISODE_FILES["road"],
        filter_by_type(episodes, EPISODE_TYPES["road"]),
    )


def ensure_output_split_dir(split_dir: Path, overwrite: bool) -> None:
    if not split_dir.exists():
        split_dir.mkdir(parents=True, exist_ok=True)
        return
    existing = list(split_dir.iterdir())
    if existing and not overwrite:
        raise RuntimeError(f"Output split already exists and is not empty: {split_dir}")
    if overwrite:
        for child in existing:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def build_selection_manifest(
    version: str,
    output_split: str,
    reference_split: str,
    seed: int,
    reference_counts: dict[str, int],
    selected_episodes: list[dict[str, Any]],
    selected_videos: dict[tuple[str, str], str],
) -> dict[str, Any]:
    selected_counter = compute_type_counts(selected_episodes)
    scene_counter = Counter(episode["scene_id"] for episode in selected_episodes)
    return {
        "version": version,
        "output_split": output_split,
        "reference_split": reference_split,
        "seed": seed,
        "reference_counts": reference_counts,
        "selected_counts": selected_counter,
        "selected_total": len(selected_episodes),
        "selected_scene_count": len(scene_counter),
        "selected_scene_counts": dict(sorted(scene_counter.items())),
        "selected_episode_keys": [
            {
                "scene_id": episode["scene_id"],
                "episode_id": normalize_episode_id(episode["episode_id"]),
                "trajectory_type": episode["trajectory_type"],
                "video": selected_videos[episode_key(episode)],
            }
            for episode in selected_episodes
        ],
    }


def sample_eval_split_from_train(
    version: str,
    output_split: str,
    reference_split: str,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_root = DATASET_ROOT / version
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset version not found: {dataset_root}")

    train_split_dir = dataset_root / "episodes" / "train"
    reference_split_dir = dataset_root / "episodes" / "eval" / reference_split
    output_split_dir = dataset_root / "episodes" / "eval" / output_split
    trajectory_root = dataset_root / "trajectory_data"
    images_root = trajectory_root / "images"

    ensure_output_split_dir(output_split_dir, overwrite=overwrite)

    train_episodes = read_episode_records(train_split_dir / EPISODE_FILES["all"])
    reference_episodes = read_episode_records(reference_split_dir / EPISODE_FILES["all"])
    summary_records = read_summary(trajectory_root / "summary.json")
    annotation_records = read_annotations(trajectory_root / "annotations.json")

    ensure_unique_episode_keys(train_episodes, "train split")
    ensure_unique_episode_keys(reference_episodes, f"reference split {reference_split}")

    summary_by_key = ensure_unique_summary_keys(summary_records)
    annotations_by_video = index_annotations_by_video(annotation_records)
    summary_videos = {str(record["video"]) for record in summary_records}
    annotation_videos = set(annotations_by_video)
    if summary_videos != annotation_videos:
        raise RuntimeError("trajectory summary / annotations do not cover the same video set")

    eligible_train = []
    missing_trajectory = []
    for episode in train_episodes:
        key = episode_key(episode)
        summary_record = summary_by_key.get(key)
        if summary_record is None:
            missing_trajectory.append(key)
            continue
        video = str(summary_record["video"])
        if not (trajectory_root / video).is_dir():
            raise RuntimeError(f"Missing trajectory image directory for {key}: {trajectory_root / video}")
        eligible_train.append(episode)

    reference_counts = compute_type_counts(reference_episodes)
    eligible_by_type: dict[str, list[dict[str, Any]]] = {
        task: filter_by_type(eligible_train, task) for task in TASK_ORDER
    }

    rng = random.Random(seed)
    selected_keys: set[tuple[str, str]] = set()
    selected_episodes_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for task in TASK_ORDER:
        sampled = sample_category_scene_round_robin(
            eligible_by_type[task],
            reference_counts[task],
            rng,
        )
        for episode in sampled:
            key = episode_key(episode)
            if key in selected_keys:
                raise RuntimeError(f"Episode selected more than once: {key}")
            selected_keys.add(key)
            selected_episodes_by_key[key] = episode

    selected_episodes = [episode for episode in train_episodes if episode_key(episode) in selected_keys]
    remaining_train_episodes = [
        episode for episode in train_episodes if episode_key(episode) not in selected_keys
    ]

    selected_videos = {
        key: str(summary_by_key[key]["video"])
        for key in selected_keys
    }
    selected_video_set = set(selected_videos.values())

    write_split_files(output_split_dir, selected_episodes)
    write_split_files(train_split_dir, remaining_train_episodes)

    remaining_summary_order = [
        str(record["video"])
        for record in summary_records
        if trajectory_key_from_summary(record) not in selected_keys
    ]
    remaining_summary_by_video = OrderedDict()
    for record in summary_records:
        key = trajectory_key_from_summary(record)
        if key in selected_keys:
            continue
        remaining_summary_by_video[str(record["video"])] = record

    remaining_annotations_by_video = OrderedDict()
    for video, record in annotations_by_video.items():
        if video in selected_video_set:
            continue
        remaining_annotations_by_video[video] = record

    if set(remaining_summary_by_video) != set(remaining_annotations_by_video):
        raise RuntimeError("Filtered trajectory summary / annotations video sets differ")

    remaining_summary_records = reindex_records_in_order(
        remaining_summary_order,
        remaining_summary_by_video,
    )
    remaining_annotation_records = reindex_records_in_order(
        remaining_summary_order,
        remaining_annotations_by_video,
    )

    write_summary(trajectory_root / "summary.json", remaining_summary_records)
    write_annotations(trajectory_root / "annotations.json", remaining_annotation_records)

    removed_image_dirs = 0
    for video in selected_video_set:
        image_dir = trajectory_root / video
        if image_dir.exists():
            shutil.rmtree(image_dir)
            removed_image_dirs += 1

    manifest = build_selection_manifest(
        version=version,
        output_split=output_split,
        reference_split=reference_split,
        seed=seed,
        reference_counts=reference_counts,
        selected_episodes=selected_episodes,
        selected_videos=selected_videos,
    )
    manifest["train_total_before"] = len(train_episodes)
    manifest["train_total_after"] = len(remaining_train_episodes)
    manifest["trajectory_summary_before"] = len(summary_records)
    manifest["trajectory_summary_after"] = len(remaining_summary_records)
    manifest["missing_train_trajectory_count_before"] = len(missing_trajectory)
    manifest["removed_image_dir_count"] = removed_image_dirs

    with (output_split_dir / "selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    output_counts = compute_type_counts(selected_episodes)
    if output_counts != reference_counts:
        raise RuntimeError(
            f"Output split counts do not match reference counts: {output_counts} vs {reference_counts}"
        )

    train_counts_after = compute_type_counts(remaining_train_episodes)
    if len(selected_episodes) != sum(reference_counts.values()):
        raise RuntimeError("Selected episode count does not match reference total")
    if len(remaining_summary_records) != len(summary_records) - len(selected_episodes):
        raise RuntimeError("Trajectory summary count did not shrink by selected episode count")
    if len(remaining_annotation_records) != len(annotation_records) - len(selected_episodes):
        raise RuntimeError("Trajectory annotation count did not shrink by selected episode count")

    referenced_image_dirs = {
        Path(record["video"]).name for record in remaining_summary_records
    }
    actual_image_dirs = {
        path.name for path in images_root.iterdir() if path.is_dir()
    }
    if referenced_image_dirs != actual_image_dirs:
        raise RuntimeError("Trajectory image directories are not exactly aligned with summary videos")

    return {
        "version": version,
        "output_split": output_split,
        "reference_split": reference_split,
        "seed": seed,
        "selected_total": len(selected_episodes),
        "reference_counts": reference_counts,
        "output_counts": output_counts,
        "train_total_before": len(train_episodes),
        "train_total_after": len(remaining_train_episodes),
        "train_counts_after": train_counts_after,
        "trajectory_summary_before": len(summary_records),
        "trajectory_summary_after": len(remaining_summary_records),
        "trajectory_missing_before": len(missing_trajectory),
        "removed_image_dir_count": removed_image_dirs,
        "selected_scene_count": len({episode["scene_id"] for episode in selected_episodes}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a new eval split from train and prune matching trajectory data"
    )
    parser.add_argument("version", help="Dataset version, e.g. ver_260418")
    parser.add_argument(
        "--output-split",
        default="val_seen_update",
        help="Name of the new eval split under episodes/eval/",
    )
    parser.add_argument(
        "--reference-split",
        default="val_seen",
        help="Existing eval split whose exact type counts will be matched",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260418,
        help="Random seed for scene-aware sampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting a non-empty output split directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = sample_eval_split_from_train(
        version=args.version,
        output_split=args.output_split,
        reference_split=args.reference_split,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
