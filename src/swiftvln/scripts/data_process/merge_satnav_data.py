#!/usr/bin/env python3
"""
Merge two SatNav dataset versions into a new version with duplicate-safe remapping.

The merge strategy is:
1. Analyze per-city `VLN_episodes.json` overlap.
2. Keep identical duplicates once.
3. If `scene_id + episode_id` collides but payload differs, remap the secondary
   episode to a fresh `episode_id` instead of dropping it.
4. Regenerate merged `episodes/` via `process_episodes.py`.
5. Merge `trajectory_data` and rewrite remapped secondary `episode_id` / `video`.
6. Regenerate `data/qa_swift.jsonl` from merged `qa.json` when possible, or fall
   back to merging source `qa_swift.jsonl`.

Examples:
    python -m swiftvln.scripts.data_process.merge_satnav_data \
        ver_260327 ver_260403 ver_260404 --analyze-only

    python -m swiftvln.scripts.data_process.merge_satnav_data \
        ver_260327 ver_260403 ver_260404
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from .config import (
        DATASET_ROOT,
        EVAL_CITIES,
        TRAIN_CITIES,
        classify_eval_cities,
    )
    from .convert_qa_to_swift import convert_qa_to_swift
    from .process_episodes import process_episodes
except ImportError:
    from config import DATASET_ROOT, EVAL_CITIES, TRAIN_CITIES, classify_eval_cities
    from convert_qa_to_swift import convert_qa_to_swift
    from process_episodes import process_episodes


PRIMARY = "primary"
SECONDARY = "secondary"
TRAIN_CITIES_SET = set(TRAIN_CITIES)
VAL_SEEN_CITIES, VAL_UNSEEN_CITIES = classify_eval_cities()
VAL_SEEN_CITIES_SET = set(VAL_SEEN_CITIES)
VAL_UNSEEN_CITIES_SET = set(VAL_UNSEEN_CITIES)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_scene_name(value: Any, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    if not text:
        return fallback
    text = text.rstrip("/")
    if "/" in text:
        text = text.split("/")[-1]
    return text


def normalize_episode_id(value: Any) -> str:
    return str(value)


def is_numeric_id(value: str) -> bool:
    return value.isdigit()


def cast_like(original: Any, new_value: str) -> Any:
    if isinstance(original, int) and is_numeric_id(new_value):
        return int(new_value)
    return new_value


def build_episode_key(scene_name: str, episode_id: str) -> tuple[str, str]:
    return normalize_scene_name(scene_name), normalize_episode_id(episode_id)


def episode_key(episode: dict[str, Any], fallback_scene: str) -> tuple[str, str]:
    scene_name = normalize_scene_name(episode.get("scene_id"), fallback_scene)
    if "episode_id" in episode:
        episode_id = normalize_episode_id(episode["episode_id"])
    else:
        episode_id = normalize_episode_id(episode.get("id", ""))
    return build_episode_key(scene_name, episode_id)


def canonical_episode(episode: dict[str, Any], fallback_scene: str) -> str:
    normalized = copy.deepcopy(episode)
    scene_name = normalize_scene_name(normalized.get("scene_id"), fallback_scene)
    if scene_name:
        normalized["scene_id"] = scene_name
    if "episode_id" in normalized:
        normalized["episode_id"] = normalize_episode_id(normalized["episode_id"])
    elif "id" in normalized:
        normalized["id"] = normalize_episode_id(normalized["id"])
    if "trajectory_id" in normalized:
        normalized["trajectory_id"] = normalize_episode_id(normalized["trajectory_id"])
    return canonical_json(normalized)


def set_episode_id(episode: dict[str, Any], new_episode_id: str) -> dict[str, Any]:
    if "episode_id" in episode:
        episode["episode_id"] = cast_like(episode["episode_id"], new_episode_id)
    elif "id" in episode:
        episode["id"] = cast_like(episode["id"], new_episode_id)
    else:
        episode["episode_id"] = new_episode_id
    return episode


def ensure_scene_id(episode: dict[str, Any], scene_name: str) -> dict[str, Any]:
    episode["scene_id"] = normalize_scene_name(episode.get("scene_id"), scene_name)
    return episode


def source_root(version: str) -> Path:
    return DATASET_ROOT / version


def version_exists(version: str) -> bool:
    return source_root(version).exists()


def discover_city_episode_sources(version: str) -> dict[str, Path]:
    root = source_root(version)
    result: dict[str, Path] = {}
    for subdir in ("data", "raw_data"):
        base = root / subdir
        if not base.exists():
            continue
        for city_dir in sorted(base.iterdir()):
            if not city_dir.is_dir():
                continue
            if city_dir.name in result:
                continue
            if (city_dir / "VLN_episodes.json").exists():
                result[city_dir.name] = city_dir
    return result


def discover_city_asset_dirs(version: str) -> dict[str, Path]:
    base = source_root(version) / "data"
    result: dict[str, Path] = {}
    if not base.exists():
        return result
    for city_dir in sorted(base.iterdir()):
        if city_dir.is_dir():
            result[city_dir.name] = city_dir
    return result


def load_episode_records(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("episodes", []))


def write_episode_records(file_path: Path, episodes: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump({"episodes": episodes}, handle, indent=2, ensure_ascii=False)


def read_json_file(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_jsonl_records(file_path: Path) -> list[dict[str, Any]]:
    records = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rel_file_digest(file_path: Path) -> tuple[int, str]:
    hasher = hashlib.sha1()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return file_path.stat().st_size, hasher.hexdigest()


def files_identical(path_a: Path, path_b: Path) -> bool:
    if not path_a.exists() or not path_b.exists():
        return False
    if path_a.stat().st_size != path_b.stat().st_size:
        return False
    return rel_file_digest(path_a) == rel_file_digest(path_b)


def link_or_copy_file(source: Path, dest: Path, copy_mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if files_identical(source, dest):
            return
        raise RuntimeError(f"File conflict with different content: {source} -> {dest}")
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


def list_relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    items = []
    for path in root.rglob("*"):
        if path.is_file():
            items.append(str(path.relative_to(root)))
    return sorted(items)


class EpisodeIdAllocator:
    def __init__(self, existing_ids: set[str]):
        self.taken = {normalize_episode_id(item) for item in existing_ids}
        numeric_ids = [int(item) for item in self.taken if is_numeric_id(item)]
        self.next_numeric = max(numeric_ids, default=-1) + 1

    def allocate(self) -> str:
        while True:
            candidate = str(self.next_numeric)
            self.next_numeric += 1
            if candidate not in self.taken:
                self.taken.add(candidate)
                return candidate


def merge_episode_lists(
    primary_episodes: list[dict[str, Any]],
    secondary_episodes: list[dict[str, Any]],
    city_name: str,
    secondary_version: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    existing: dict[tuple[str, str], str] = {}
    existing_ids: set[str] = set()
    remap: dict[tuple[str, str], str] = {}
    stats = {
        "primary_count": len(primary_episodes),
        "secondary_count": len(secondary_episodes),
        "identical_overlap": 0,
        "conflicting_overlap": 0,
        "unique_secondary_additions": 0,
        "remapped_secondary_additions": 0,
        "sample_conflicts": [],
    }

    for episode in primary_episodes:
        item = ensure_scene_id(copy.deepcopy(episode), city_name)
        key = episode_key(item, city_name)
        merged.append(item)
        existing[key] = canonical_episode(item, city_name)
        existing_ids.add(key[1])

    secondary_original_ids = {
        episode_key(ensure_scene_id(copy.deepcopy(episode), city_name), city_name)[1]
        for episode in secondary_episodes
    }
    allocator = EpisodeIdAllocator(existing_ids | secondary_original_ids)

    for episode in secondary_episodes:
        item = ensure_scene_id(copy.deepcopy(episode), city_name)
        key = episode_key(item, city_name)
        canonical = canonical_episode(item, city_name)
        if key not in existing:
            merged.append(item)
            existing[key] = canonical
            existing_ids.add(key[1])
            stats["unique_secondary_additions"] += 1
            continue
        if existing[key] == canonical:
            stats["identical_overlap"] += 1
            continue

        stats["conflicting_overlap"] += 1
        new_episode_id = allocator.allocate()
        remap[key] = new_episode_id
        item = set_episode_id(item, new_episode_id)
        new_key = build_episode_key(city_name, new_episode_id)
        merged.append(item)
        existing[new_key] = canonical_episode(item, city_name)
        stats["remapped_secondary_additions"] += 1
        if len(stats["sample_conflicts"]) < 10:
            stats["sample_conflicts"].append(
                {
                    "source_version": secondary_version,
                    "scene_id": city_name,
                    "old_episode_id": key[1],
                    "new_episode_id": new_episode_id,
                }
            )

    if len({episode_key(item, city_name) for item in merged}) != len(merged):
        raise RuntimeError(f"Merged city {city_name} still contains duplicate episode keys")

    stats["merged_count"] = len(merged)
    return merged, remap, stats


def qa_key(item: dict[str, Any]) -> str:
    if item.get("id") is not None:
        return f"id:{item['id']}"
    return f"hash:{sha1_text(canonical_json(item))}"


def rename_qa_id(item: dict[str, Any], version: str, used_ids: set[str]) -> dict[str, Any]:
    updated = copy.deepcopy(item)
    base_id = str(updated.get("id", f"qa_{sha1_text(canonical_json(updated))[:12]}"))
    candidate = f"{base_id}__{version}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base_id}__{version}_{suffix}"
        suffix += 1
    updated["id"] = candidate
    used_ids.add(candidate)
    return updated


def merge_qa_items(qa_sources: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[str, str] = {}
    used_ids: set[str] = set()
    stats = {
        "sources": [],
        "merged_count": 0,
        "identical_overlap": 0,
        "conflicting_overlap": 0,
    }

    for version, qa_path in qa_sources:
        if not qa_path.exists():
            continue
        items = read_json_file(qa_path)
        stats["sources"].append({"version": version, "path": str(qa_path), "count": len(items)})
        for item in items:
            candidate = copy.deepcopy(item)
            key = qa_key(candidate)
            canonical = canonical_json(candidate)
            if key not in by_key:
                merged.append(candidate)
                by_key[key] = canonical
                if candidate.get("id") is not None:
                    used_ids.add(str(candidate["id"]))
                continue
            if by_key[key] == canonical:
                stats["identical_overlap"] += 1
                continue
            stats["conflicting_overlap"] += 1
            updated = rename_qa_id(candidate, version, used_ids)
            merged.append(updated)
            by_key[qa_key(updated)] = canonical_json(updated)

    stats["merged_count"] = len(merged)
    return merged, stats


def rewrite_swift_record_images(record: dict[str, Any], source_version: str, output_version: str) -> dict[str, Any]:
    updated = copy.deepcopy(record)
    source_token = f"/satnav_datasets/{source_version}/"
    output_token = f"/satnav_datasets/{output_version}/"
    images = []
    for image_path in updated.get("images", []):
        image_text = str(image_path)
        if source_token in image_text:
            image_text = image_text.replace(source_token, output_token)
        images.append(image_text)
    updated["images"] = images
    return updated


def swift_record_key(item: dict[str, Any]) -> str:
    if item.get("id") is not None:
        return f"id:{item['id']}"
    payload = {
        "messages": item.get("messages", []),
        "images": item.get("images", []),
        "task": item.get("task"),
    }
    return f"hash:{sha1_text(canonical_json(payload))}"


def merge_qa_swift_files(
    sources: list[tuple[str, Path]],
    output_path: Path,
    output_version: str,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    by_key: dict[str, str] = {}
    used_ids: set[str] = set()
    stats = {
        "sources": [],
        "identical_overlap": 0,
        "conflicting_overlap": 0,
        "merged_count": 0,
    }

    for version, file_path in sources:
        if not file_path.exists():
            continue
        items = read_jsonl_records(file_path)
        stats["sources"].append({"version": version, "path": str(file_path), "count": len(items)})
        for item in items:
            candidate = rewrite_swift_record_images(item, version, output_version)
            key = swift_record_key(candidate)
            canonical = canonical_json(candidate)
            if key not in by_key:
                merged.append(candidate)
                by_key[key] = canonical
                if candidate.get("id") is not None:
                    used_ids.add(str(candidate["id"]))
                continue
            if by_key[key] == canonical:
                stats["identical_overlap"] += 1
                continue
            stats["conflicting_overlap"] += 1
            updated = rename_qa_id(candidate, version, used_ids)
            merged.append(updated)
            by_key[swift_record_key(updated)] = canonical_json(updated)

    write_jsonl_records(output_path, merged)
    stats["merged_count"] = len(merged)
    return stats


def split_name_for_city(city_name: str) -> str:
    if city_name in TRAIN_CITIES_SET:
        return "train"
    if city_name in VAL_SEEN_CITIES_SET:
        return "val_seen"
    if city_name in VAL_UNSEEN_CITIES_SET:
        return "val_unseen"
    if city_name in EVAL_CITIES:
        return "eval_unclassified"
    return "unknown"


def summarize_split_counts(city_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {
        "train": {"primary": 0, "secondary": 0, "merged": 0},
        "val_seen": {"primary": 0, "secondary": 0, "merged": 0},
        "val_unseen": {"primary": 0, "secondary": 0, "merged": 0},
        "unknown": {"primary": 0, "secondary": 0, "merged": 0},
    }
    for city_name, stats in city_stats.items():
        split = split_name_for_city(city_name)
        if split not in result:
            split = "unknown"
        result[split]["primary"] += stats["primary_count"]
        result[split]["secondary"] += stats["secondary_count"]
        result[split]["merged"] += stats["merged_count"]
    for split_stats in result.values():
        split_stats["growth_vs_primary"] = split_stats["merged"] - split_stats["primary"]
    return result


def planned_episode_analysis(
    primary_version: str,
    secondary_version: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, int]],
]:
    primary_sources = discover_city_episode_sources(primary_version)
    secondary_sources = discover_city_episode_sources(secondary_version)
    all_cities = sorted(set(primary_sources) | set(secondary_sources))
    merged_by_city: dict[str, list[dict[str, Any]]] = {}
    remap: dict[tuple[str, str], str] = {}
    city_stats: dict[str, dict[str, Any]] = {}

    for city_name in all_cities:
        primary_episodes = []
        if city_name in primary_sources:
            primary_episodes = load_episode_records(primary_sources[city_name] / "VLN_episodes.json")
        secondary_episodes = []
        if city_name in secondary_sources:
            secondary_episodes = load_episode_records(secondary_sources[city_name] / "VLN_episodes.json")
        merged, city_remap, stats = merge_episode_lists(
            primary_episodes,
            secondary_episodes,
            city_name,
            secondary_version,
        )
        merged_by_city[city_name] = merged
        remap.update(city_remap)
        city_stats[city_name] = stats

    split_stats = summarize_split_counts(city_stats)
    return merged_by_city, remap, city_stats, split_stats


def normalize_trajectory_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(record)
    normalized.pop("id", None)
    if "episode_id" in normalized:
        normalized["episode_id"] = normalize_episode_id(normalized["episode_id"])
    if "trajectory_id" in normalized:
        normalized["trajectory_id"] = normalize_episode_id(normalized["trajectory_id"])
    if "scene_id" in normalized:
        normalized["scene_id"] = normalize_scene_name(normalized["scene_id"])
    if "video" in normalized:
        normalized["video"] = str(normalized["video"])
    return normalized


def canonical_trajectory_record(record: dict[str, Any]) -> str:
    return canonical_json(normalize_trajectory_record(record))


def build_video_rel(scene_name: str, episode_id: str) -> str:
    if is_numeric_id(episode_id):
        suffix = f"{int(episode_id):06d}"
    else:
        suffix = episode_id
    return f"images/{scene_name}_satnav_{suffix}"


def parse_scene_and_episode_from_video(video_rel: str) -> tuple[str, str]:
    name = Path(video_rel).name
    if "_satnav_" not in name:
        raise ValueError(f"Unexpected trajectory video path: {video_rel}")
    scene_name, suffix = name.rsplit("_satnav_", 1)
    return scene_name, suffix


def read_annotations(file_path: Path) -> list[dict[str, Any]]:
    return list(read_json_file(file_path))


def write_annotations(file_path: Path, records: list[dict[str, Any]]) -> None:
    write_json_file(file_path, records)


def read_summary(file_path: Path) -> list[dict[str, Any]]:
    return read_jsonl_records(file_path)


def write_summary(file_path: Path, records: list[dict[str, Any]]) -> None:
    write_jsonl_records(file_path, records)


def build_secondary_summary_transform(
    summary_records: list[dict[str, Any]],
    remap: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    transformed_records = []
    video_map: dict[str, str] = {}
    for record in summary_records:
        updated = copy.deepcopy(record)
        old_video = str(updated["video"])
        scene_name = normalize_scene_name(updated.get("scene_id"), parse_scene_and_episode_from_video(old_video)[0])
        old_episode_id = normalize_episode_id(updated.get("episode_id", parse_scene_and_episode_from_video(old_video)[1]))
        new_episode_id = remap.get((scene_name, old_episode_id), old_episode_id)
        updated["episode_id"] = cast_like(updated.get("episode_id", old_episode_id), new_episode_id)
        updated["scene_id"] = updated.get("scene_id", scene_name)
        new_video = build_video_rel(scene_name, new_episode_id)
        updated["video"] = new_video
        transformed_records.append(updated)
        video_map[old_video] = new_video
    return transformed_records, video_map


def transform_annotations(
    annotations: list[dict[str, Any]],
    video_map: dict[str, str],
) -> list[dict[str, Any]]:
    transformed = []
    for record in annotations:
        updated = copy.deepcopy(record)
        old_video = str(updated["video"])
        updated["video"] = video_map.get(old_video, old_video)
        transformed.append(updated)
    return transformed


def index_by_video(records: list[dict[str, Any]], kind: str) -> tuple[OrderedDict[str, dict[str, Any]], list[str]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    order: list[str] = []
    for record in records:
        video = str(record["video"])
        if video in indexed:
            prev = indexed[video]
            if canonical_trajectory_record(prev) != canonical_trajectory_record(record):
                raise RuntimeError(f"Conflicting {kind} records share the same video path: {video}")
            continue
        indexed[video] = record
        order.append(video)
    return indexed, order


def merge_video_index(
    base_index: OrderedDict[str, dict[str, Any]],
    base_order: list[str],
    new_records: list[dict[str, Any]],
    kind: str,
) -> tuple[OrderedDict[str, dict[str, Any]], list[str], int]:
    indexed = OrderedDict(base_index)
    order = list(base_order)
    identical_overlap = 0
    for record in new_records:
        video = str(record["video"])
        if video not in indexed:
            indexed[video] = record
            order.append(video)
            continue
        if canonical_trajectory_record(indexed[video]) == canonical_trajectory_record(record):
            identical_overlap += 1
            continue
        raise RuntimeError(f"Conflicting {kind} records share video path: {video}")
    return indexed, order, identical_overlap


def reindex_records_in_order(
    order: list[str],
    records_by_video: OrderedDict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for idx, video in enumerate(order):
        item = copy.deepcopy(records_by_video[video])
        item["id"] = idx
        result.append(item)
    return result


def merge_trajectory_data(
    primary_version: str,
    secondary_version: str,
    output_root: Path,
    remap: dict[tuple[str, str], str],
    copy_mode: str,
) -> dict[str, Any]:
    primary_root = source_root(primary_version) / "trajectory_data"
    secondary_root = source_root(secondary_version) / "trajectory_data"
    output_trajectory_root = output_root / "trajectory_data"
    output_images_root = output_trajectory_root / "images"

    primary_summary = read_summary(primary_root / "summary.json")
    secondary_summary = read_summary(secondary_root / "summary.json")
    primary_annotations = read_annotations(primary_root / "annotations.json")
    secondary_annotations = read_annotations(secondary_root / "annotations.json")

    primary_summary_by_video, primary_summary_order = index_by_video(primary_summary, "summary")
    primary_annotations_by_video, _ = index_by_video(primary_annotations, "annotation")
    secondary_summary_by_video, _ = index_by_video(secondary_summary, "summary")
    secondary_annotations_by_video, _ = index_by_video(secondary_annotations, "annotation")

    if set(primary_summary_by_video) != set(primary_annotations_by_video):
        raise RuntimeError(f"{primary_version} summary/annotations video sets differ")
    if set(secondary_summary_by_video) != set(secondary_annotations_by_video):
        raise RuntimeError(f"{secondary_version} summary/annotations video sets differ")

    transformed_secondary_summary, secondary_video_map = build_secondary_summary_transform(secondary_summary, remap)
    transformed_secondary_annotations = transform_annotations(secondary_annotations, secondary_video_map)

    merged_summary_by_video, merged_order, summary_identical_overlap = merge_video_index(
        primary_summary_by_video,
        primary_summary_order,
        transformed_secondary_summary,
        "summary",
    )
    merged_annotations_by_video, _, annotations_identical_overlap = merge_video_index(
        primary_annotations_by_video,
        primary_summary_order,
        transformed_secondary_annotations,
        "annotation",
    )

    if set(merged_summary_by_video) != set(merged_annotations_by_video):
        raise RuntimeError("Merged trajectory summary/annotations video sets differ")

    output_images_root.mkdir(parents=True, exist_ok=True)
    copy_tree(primary_root / "images", output_images_root, copy_mode)
    for old_video, new_video in secondary_video_map.items():
        source_dir = secondary_root / old_video
        dest_dir = output_trajectory_root / new_video
        if dest_dir.exists():
            continue
        copy_tree(source_dir, dest_dir, copy_mode)

    merged_summary = reindex_records_in_order(merged_order, merged_summary_by_video)
    merged_annotations = reindex_records_in_order(merged_order, merged_annotations_by_video)
    write_summary(output_trajectory_root / "summary.json", merged_summary)
    write_annotations(output_trajectory_root / "annotations.json", merged_annotations)

    referenced_image_dirs = {Path(item["video"]).name for item in merged_summary}
    removed_unused_image_dirs = 0
    for path in list(output_images_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in referenced_image_dirs:
            continue
        shutil.rmtree(path)
        removed_unused_image_dirs += 1

    image_dirs = [path for path in output_images_root.iterdir() if path.is_dir()]
    return {
        "primary_summary": len(primary_summary),
        "secondary_summary": len(secondary_summary),
        "primary_annotations": len(primary_annotations),
        "secondary_annotations": len(secondary_annotations),
        "merged_summary": len(merged_summary),
        "merged_annotations": len(merged_annotations),
        "summary_identical_overlap": summary_identical_overlap,
        "annotations_identical_overlap": annotations_identical_overlap,
        "image_dir_count": len(image_dirs),
        "removed_unused_image_dirs": removed_unused_image_dirs,
    }


def merge_city_assets(
    city_name: str,
    primary_asset_dir: Path | None,
    secondary_asset_dir: Path | None,
    output_city_dir: Path,
    primary_version: str,
    secondary_version: str,
    copy_mode: str,
) -> dict[str, Any]:
    output_city_dir.mkdir(parents=True, exist_ok=True)
    qa_sources: list[tuple[str, Path]] = []
    copied_assets = []

    for version, source_dir in (
        (primary_version, primary_asset_dir),
        (secondary_version, secondary_asset_dir),
    ):
        if source_dir is None or not source_dir.exists():
            continue
        for child in sorted(source_dir.iterdir()):
            if child.name == "VLN_episodes.json":
                continue
            if child.name == "qa.json":
                qa_sources.append((version, child))
                continue
            copy_tree(child, output_city_dir / child.name, copy_mode)
            copied_assets.append(str((output_city_dir / child.name).relative_to(output_city_dir)))

    qa_stats = None
    if qa_sources:
        merged_qa, qa_stats = merge_qa_items(qa_sources)
        write_json_file(output_city_dir / "qa.json", merged_qa)

    return {
        "copied_assets": sorted(set(copied_assets)),
        "qa": qa_stats,
    }


def ensure_empty_output(output_root: Path) -> None:
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing:
            raise RuntimeError(f"Output version already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def output_has_any_qa_json(output_root: Path) -> bool:
    data_root = output_root / "data"
    if not data_root.exists():
        return False
    return any((city_dir / "qa.json").exists() for city_dir in data_root.iterdir() if city_dir.is_dir())


def count_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def validate_merged_outputs(
    output_root: Path,
    split_stats: dict[str, dict[str, int]],
    trajectory_stats: dict[str, Any],
    require_growth: bool,
) -> dict[str, Any]:
    output_version = output_root.name
    results = process_episodes(output_version)
    expected_train = split_stats["train"]["merged"]
    expected_seen = split_stats["val_seen"]["merged"]
    expected_unseen = split_stats["val_unseen"]["merged"]

    actual_train = results["train"]["total"] if results.get("train") else 0
    actual_seen = results["val_seen"]["total"] if results.get("val_seen") else 0
    actual_unseen = results["val_unseen"]["total"] if results.get("val_unseen") else 0

    if actual_train != expected_train:
        raise RuntimeError(f"Merged train count mismatch: expected {expected_train}, got {actual_train}")
    if actual_seen != expected_seen:
        raise RuntimeError(f"Merged val_seen count mismatch: expected {expected_seen}, got {actual_seen}")
    if actual_unseen != expected_unseen:
        raise RuntimeError(f"Merged val_unseen count mismatch: expected {expected_unseen}, got {actual_unseen}")

    def check_no_duplicates(file_path: Path) -> int:
        episodes = load_episode_records(file_path)
        keys = {episode_key(item, "") for item in episodes}
        if len(keys) != len(episodes):
            raise RuntimeError(f"Duplicate scene_id + episode_id found in {file_path}")
        return len(episodes)

    check_no_duplicates(output_root / "episodes" / "train" / "all_episodes.json")
    check_no_duplicates(output_root / "episodes" / "eval" / "val_seen" / "all_episodes.json")
    check_no_duplicates(output_root / "episodes" / "eval" / "val_unseen" / "all_episodes.json")

    summary_records = read_summary(output_root / "trajectory_data" / "summary.json")
    annotation_records = read_annotations(output_root / "trajectory_data" / "annotations.json")
    if len(summary_records) != trajectory_stats["merged_summary"]:
        raise RuntimeError("Merged trajectory summary count changed after write")
    if len(annotation_records) != trajectory_stats["merged_annotations"]:
        raise RuntimeError("Merged trajectory annotation count changed after write")
    summary_videos = {str(item["video"]) for item in summary_records}
    annotation_videos = {str(item["video"]) for item in annotation_records}
    if summary_videos != annotation_videos:
        raise RuntimeError("Merged summary / annotations do not cover the same video set")
    for video in summary_videos:
        if not (output_root / "trajectory_data" / video).exists():
            raise RuntimeError(f"Missing trajectory image directory for video {video}")
    image_dir_names = {
        path.name
        for path in (output_root / "trajectory_data" / "images").iterdir()
        if path.is_dir()
    }
    if image_dir_names != {Path(video).name for video in summary_videos}:
        raise RuntimeError("Trajectory image directories are not exactly aligned with summary videos")

    if require_growth:
        for split_name in ("train", "val_seen", "val_unseen"):
            if split_stats[split_name]["growth_vs_primary"] <= 0:
                raise RuntimeError(f"{split_name} did not grow after merge")

    return {
        "train": actual_train,
        "val_seen": actual_seen,
        "val_unseen": actual_unseen,
        "trajectory_summary": len(summary_records),
        "trajectory_annotations": len(annotation_records),
    }


def write_remap_log(output_root: Path, secondary_version: str, remap: dict[tuple[str, str], str]) -> Path:
    output_path = output_root / "episode_id_remap.jsonl"
    records = [
        {
            "source_version": secondary_version,
            "scene_id": scene_name,
            "old_episode_id": old_episode_id,
            "new_episode_id": new_episode_id,
        }
        for (scene_name, old_episode_id), new_episode_id in sorted(remap.items())
    ]
    write_jsonl_records(output_path, records)
    return output_path


def build_analysis_payload(
    primary_version: str,
    secondary_version: str,
    output_version: str,
    city_stats: dict[str, dict[str, Any]],
    split_stats: dict[str, dict[str, int]],
    remap: dict[tuple[str, str], str],
) -> dict[str, Any]:
    return {
        "primary_version": primary_version,
        "secondary_version": secondary_version,
        "output_version": output_version,
        "city_count": len(city_stats),
        "remapped_episode_count": len(remap),
        "sample_remaps": [
            {
                "scene_id": scene_name,
                "old_episode_id": old_episode_id,
                "new_episode_id": new_episode_id,
            }
            for (scene_name, old_episode_id), new_episode_id in list(sorted(remap.items()))[:20]
        ],
        "split_stats": split_stats,
        "city_stats": city_stats,
    }


def execute_merge(
    primary_version: str,
    secondary_version: str,
    output_version: str,
    copy_mode: str,
    require_growth: bool,
    analyze_only: bool,
) -> dict[str, Any]:
    if primary_version == secondary_version:
        raise RuntimeError("Primary and secondary version must differ")
    if not version_exists(primary_version):
        raise FileNotFoundError(f"Primary version not found: {source_root(primary_version)}")
    if not version_exists(secondary_version):
        raise FileNotFoundError(f"Secondary version not found: {source_root(secondary_version)}")

    merged_by_city, remap, city_stats, split_stats = planned_episode_analysis(
        primary_version,
        secondary_version,
    )
    analysis = build_analysis_payload(
        primary_version,
        secondary_version,
        output_version,
        city_stats,
        split_stats,
        remap,
    )
    if analyze_only:
        return {"analysis": analysis}

    output_root = source_root(output_version)
    ensure_empty_output(output_root)

    primary_assets = discover_city_asset_dirs(primary_version)
    secondary_assets = discover_city_asset_dirs(secondary_version)
    asset_stats = {}
    for city_name, merged_episodes in merged_by_city.items():
        output_data_city = output_root / "data" / city_name
        output_raw_city = output_root / "raw_data" / city_name
        asset_stats[city_name] = merge_city_assets(
            city_name,
            primary_assets.get(city_name),
            secondary_assets.get(city_name),
            output_data_city,
            primary_version,
            secondary_version,
            copy_mode,
        )
        write_episode_records(output_data_city / "VLN_episodes.json", merged_episodes)
        write_episode_records(output_raw_city / "VLN_episodes.json", merged_episodes)

    trajectory_stats = merge_trajectory_data(
        primary_version,
        secondary_version,
        output_root,
        remap,
        copy_mode,
    )

    qa_stats = {}
    if output_has_any_qa_json(output_root):
        convert_qa_to_swift(output_version)
        qa_output = output_root / "data" / "qa_swift.jsonl"
        qa_stats = {
            "mode": "regen-from-qa-json",
            "line_count": count_lines(qa_output),
            "path": str(qa_output),
        }
    else:
        qa_output = output_root / "data" / "qa_swift.jsonl"
        qa_stats = merge_qa_swift_files(
            [
                (primary_version, source_root(primary_version) / "data" / "qa_swift.jsonl"),
                (secondary_version, source_root(secondary_version) / "data" / "qa_swift.jsonl"),
            ],
            qa_output,
            output_version,
        )
        qa_stats["mode"] = "merged-source-qa-swift"
        qa_stats["path"] = str(qa_output)

    validation = validate_merged_outputs(output_root, split_stats, trajectory_stats, require_growth)
    remap_log_path = write_remap_log(output_root, secondary_version, remap)

    manifest = {
        "analysis": analysis,
        "copy_mode": copy_mode,
        "asset_stats": asset_stats,
        "trajectory": trajectory_stats,
        "qa": qa_stats,
        "validation": validation,
        "remap_log": str(remap_log_path),
    }
    write_json_file(output_root / "merge_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two SatNav dataset versions into a new version with conflict-safe remapping."
    )
    parser.add_argument("primary_version", help="Primary/base version, e.g. ver_260327")
    parser.add_argument("secondary_version", help="Secondary version to merge in, e.g. ver_260403")
    parser.add_argument("output_version", help="Output merged version, e.g. ver_260404")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize copied assets and trajectory images in the merged version.",
    )
    parser.add_argument(
        "--allow-no-growth",
        action="store_true",
        help="Do not fail when train / eval splits fail to grow.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the merge plan and print JSON without writing output files.",
    )
    args = parser.parse_args()

    result = execute_merge(
        primary_version=args.primary_version,
        secondary_version=args.secondary_version,
        output_version=args.output_version,
        copy_mode=args.copy_mode,
        require_growth=not args.allow_no_growth,
        analyze_only=args.analyze_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
