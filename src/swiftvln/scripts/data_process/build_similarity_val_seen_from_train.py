#!/usr/bin/env python3
"""
Build a similarity-biased, route-disjoint val_seen split from train using
trajectory groups.

The selection unit is a complete trajectory group identified by
``(scene_id, trajectory_id)``. Once a group is selected, all episodes in that
group move into the new eval split and all corresponding trajectory outputs are
removed from the main train trajectory_data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import DATASET_ROOT, EPISODE_FILES, EPISODE_TYPES
except ImportError:
    from config import DATASET_ROOT, EPISODE_FILES, EPISODE_TYPES


TASK_ORDER = [
    EPISODE_TYPES["boundary"],
    EPISODE_TYPES["landmark"],
    EPISODE_TYPES["road"],
]

EPS = 1e-9
FULL_PATH_POINTS = 32
PREFIX_POINTS = 16


@dataclass(frozen=True)
class SupportCandidate:
    neighbor_key: tuple[str, str]
    score: float
    details: dict[str, Any]


@dataclass
class TrajectoryGroup:
    key: tuple[str, str]
    scene_id: str
    trajectory_id: str
    trajectory_type: str
    trajectory_subtype: str | None
    episodes: list[dict[str, Any]]
    representative_path: np.ndarray
    full_flat: np.ndarray
    prefix_flat: np.ndarray
    path_length_m: float
    boundary_id: str | None = None
    direction: str | None = None
    label: str | None = None
    landmarks: frozenset[str] = field(default_factory=frozenset)
    turn_count: int | None = None
    task_num: int | None = None
    videos: list[str] = field(default_factory=list)
    support_candidates: list[SupportCandidate] = field(default_factory=list)
    best_support_score: float = float("-inf")

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def episode_keys(self) -> list[tuple[str, str]]:
        return [episode_key(ep) for ep in self.episodes]

    @property
    def episode_ids(self) -> list[str]:
        return [normalize_episode_id(ep["episode_id"]) for ep in self.episodes]


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


def group_key_from_episode(episode: dict[str, Any]) -> tuple[str, str]:
    return normalize_scene_name(episode["scene_id"]), str(episode["trajectory_id"])


def trajectory_type_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(ep["trajectory_type"] for ep in episodes)
    return {task: counts.get(task, 0) for task in TASK_ORDER}


def ratio_map(counts: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {task: 0.0 for task in TASK_ORDER}
    return {task: counts.get(task, 0) / total for task in TASK_ORDER}


def max_ratio_error(counts: Counter[str], total: int, target_ratios: dict[str, float]) -> float:
    ratios = ratio_map(counts, total)
    return max(abs(ratios[task] - target_ratios[task]) for task in TASK_ORDER)


def type_deficit(task: str, counts: Counter[str], total: int, target_ratios: dict[str, float]) -> float:
    ratios = ratio_map(counts, total)
    return target_ratios[task] - ratios[task]


def ensure_unique_episode_keys(episodes: list[dict[str, Any]], label: str) -> None:
    seen: set[tuple[str, str]] = set()
    duplicates = []
    for episode in episodes:
        key = episode_key(episode)
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    if duplicates:
        raise RuntimeError(f"Duplicate scene_id + episode_id in {label}: {duplicates[:5]}")


def ensure_unique_summary_keys(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = summary_key(record)
        if key in indexed:
            raise RuntimeError(f"Duplicate summary key found: {key}")
        indexed[key] = record
    return indexed


def index_annotations_by_video(records: list[dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        video = str(record["video"])
        if video in indexed:
            raise RuntimeError(f"Duplicate annotation video found: {video}")
        indexed[video] = record
    return indexed


def resolve_video_dir(images_root: Path, video: str) -> Path:
    video_path = Path(video)
    if video_path.parts and video_path.parts[0] == 'images':
        return images_root / Path(*video_path.parts[1:])
    return images_root / video_path.name


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


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise RuntimeError(f"Output directory already exists and is not empty: {path}")
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def pairwise_flat_l2(flattened: np.ndarray, point_count: int) -> np.ndarray:
    norms = np.sum(flattened * flattened, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (flattened @ flattened.T)
    np.maximum(d2, 0.0, out=d2)
    distances = np.sqrt(d2)
    return distances / math.sqrt(max(point_count, 1))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    idx = q * (len(ordered) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return ordered[low]
    alpha = idx - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def meters_projection(points: list[list[float]]) -> np.ndarray:
    coords = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    if coords.size == 0:
        return np.zeros((1, 2), dtype=np.float64)
    lat_ref = float(np.mean(coords[:, 1]))
    lon_scale = 111320.0 * math.cos(math.radians(lat_ref))
    lat_scale = 110540.0
    projected = np.empty_like(coords)
    projected[:, 0] = coords[:, 0] * lon_scale
    projected[:, 1] = coords[:, 1] * lat_scale
    return projected


def cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) <= 1:
        return np.zeros((len(path_xy),), dtype=np.float64)
    deltas = path_xy[1:] - path_xy[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    return cum


def resample_path(path_xy: np.ndarray, point_count: int) -> tuple[np.ndarray, float]:
    if len(path_xy) == 0:
        return np.zeros((point_count, 2), dtype=np.float64), 0.0
    if len(path_xy) == 1:
        repeated = np.repeat(path_xy, point_count, axis=0)
        return repeated.astype(np.float64), 0.0

    cum = cumulative_lengths(path_xy)
    total_length = float(cum[-1])
    if total_length <= EPS:
        repeated = np.repeat(path_xy[:1], point_count, axis=0)
        return repeated.astype(np.float64), 0.0

    targets = np.linspace(0.0, total_length, point_count)
    resampled = np.empty((point_count, 2), dtype=np.float64)
    segment_idx = 0
    for idx, target in enumerate(targets):
        while segment_idx + 1 < len(cum) and cum[segment_idx + 1] < target:
            segment_idx += 1
        if segment_idx + 1 >= len(cum):
            resampled[idx] = path_xy[-1]
            continue
        start_len = cum[segment_idx]
        end_len = cum[segment_idx + 1]
        if end_len - start_len <= EPS:
            resampled[idx] = path_xy[segment_idx]
            continue
        alpha = (target - start_len) / (end_len - start_len)
        resampled[idx] = path_xy[segment_idx] * (1.0 - alpha) + path_xy[segment_idx + 1] * alpha
    return resampled, total_length


def build_group_features(episodes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    reference_path = episodes[0]["reference_path"]
    projected = meters_projection(reference_path)
    resampled, path_length_m = resample_path(projected, FULL_PATH_POINTS)
    prefix = resampled[:PREFIX_POINTS]
    return projected, resampled.reshape(-1), prefix.reshape(-1), path_length_m


def extract_group_metadata(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode = episodes[0]
    aux_info = episode.get("aux_info") or {}
    trajectory_type = episode["trajectory_type"]
    metadata: dict[str, Any] = {
        "trajectory_subtype": episode.get("trajectory_subtype"),
        "boundary_id": None,
        "direction": None,
        "label": None,
        "landmarks": frozenset(),
        "turn_count": None,
        "task_num": None,
    }
    if trajectory_type == EPISODE_TYPES["boundary"]:
        metadata["boundary_id"] = str(aux_info.get("boundary_id")) if aux_info.get("boundary_id") is not None else None
        metadata["direction"] = str(aux_info.get("direction")) if aux_info.get("direction") else None
        metadata["label"] = str(aux_info.get("label")) if aux_info.get("label") else None
    elif trajectory_type == EPISODE_TYPES["landmark"]:
        landmarks = aux_info.get("landmarks") or []
        metadata["landmarks"] = frozenset(str(item).strip().lower() for item in landmarks if str(item).strip())
        if aux_info.get("turn_count") is not None:
            metadata["turn_count"] = int(aux_info["turn_count"])
    elif trajectory_type == EPISODE_TYPES["road"]:
        if aux_info.get("task_num") is not None:
            metadata["task_num"] = int(aux_info["task_num"])
    return metadata


def build_complete_groups(
    train_episodes: list[dict[str, Any]],
    summary_by_key: dict[tuple[str, str], dict[str, Any]],
    images_root: Path,
) -> tuple[dict[tuple[str, str], TrajectoryGroup], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in train_episodes:
        grouped[group_key_from_episode(episode)].append(episode)

    complete_groups: dict[tuple[str, str], TrajectoryGroup] = {}
    incomplete_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for key, episodes in grouped.items():
        missing = []
        videos = []
        for episode in episodes:
            e_key = episode_key(episode)
            summary_record = summary_by_key.get(e_key)
            if summary_record is None:
                missing.append({"episode_key": e_key, "reason": "missing_summary"})
                continue
            video = str(summary_record["video"])
            if not resolve_video_dir(images_root, video).is_dir():
                missing.append({"episode_key": e_key, "reason": "missing_image_dir", "video": video})
                continue
            videos.append(video)

        if missing:
            incomplete_groups[key] = {
                "group_episode_count": len(episodes),
                "present_episode_count": len(videos),
                "trajectory_type": episodes[0]["trajectory_type"],
                "scene_id": normalize_scene_name(episodes[0]["scene_id"]),
                "trajectory_id": str(episodes[0]["trajectory_id"]),
                "missing": missing[:10],
            }
            continue

        representative_path, full_flat, prefix_flat, path_length_m = build_group_features(episodes)
        metadata = extract_group_metadata(episodes)
        complete_groups[key] = TrajectoryGroup(
            key=key,
            scene_id=normalize_scene_name(episodes[0]["scene_id"]),
            trajectory_id=str(episodes[0]["trajectory_id"]),
            trajectory_type=episodes[0]["trajectory_type"],
            trajectory_subtype=metadata["trajectory_subtype"],
            episodes=list(episodes),
            representative_path=representative_path,
            full_flat=full_flat,
            prefix_flat=prefix_flat,
            path_length_m=path_length_m,
            boundary_id=metadata["boundary_id"],
            direction=metadata["direction"],
            label=metadata["label"],
            landmarks=metadata["landmarks"],
            turn_count=metadata["turn_count"],
            task_num=metadata["task_num"],
            videos=videos,
        )

    stats = {
        "all_group_count": len(grouped),
        "complete_group_count": len(complete_groups),
        "incomplete_group_count": len(incomplete_groups),
        "complete_episode_count": sum(group.episode_count for group in complete_groups.values()),
        "incomplete_episode_count": sum(info["group_episode_count"] for info in incomplete_groups.values()),
        "incomplete_present_episode_count": sum(info["present_episode_count"] for info in incomplete_groups.values()),
        "incomplete_by_type": dict(sorted(Counter(info["trajectory_type"] for info in incomplete_groups.values()).items())),
    }
    return complete_groups, {"stats": stats, "groups": incomplete_groups}


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / max(union, 1)


def build_supports(
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    top_k: int,
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    reverse_supports: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    buckets: dict[tuple[str, str], list[TrajectoryGroup]] = defaultdict(list)
    for group in groups_by_key.values():
        buckets[(group.scene_id, group.trajectory_type)].append(group)

    for (_, trajectory_type), bucket in buckets.items():
        if len(bucket) <= 1:
            continue

        full_matrix = np.stack([group.full_flat for group in bucket], axis=0)
        prefix_matrix = np.stack([group.prefix_flat for group in bucket], axis=0)
        full_dist = pairwise_flat_l2(full_matrix, FULL_PATH_POINTS)
        prefix_dist = pairwise_flat_l2(prefix_matrix, PREFIX_POINTS)
        geom_full = np.exp(-full_dist / 80.0)
        geom_prefix = np.exp(-prefix_dist / 50.0)

        n = len(bucket)
        score = np.zeros((n, n), dtype=np.float64)
        detail_cache: dict[tuple[int, int], dict[str, Any]] = {}

        if trajectory_type == EPISODE_TYPES["boundary"]:
            boundary_ids = np.array([group.boundary_id or "" for group in bucket], dtype=object)
            directions = np.array([group.direction or "" for group in bucket], dtype=object)
            labels = np.array([group.label or "" for group in bucket], dtype=object)
            same_boundary = (boundary_ids[:, None] == boundary_ids[None, :]).astype(np.float64)
            same_direction = (directions[:, None] == directions[None, :]).astype(np.float64)
            same_label = (labels[:, None] == labels[None, :]).astype(np.float64)
            score = (
                0.45 * same_boundary
                + 0.15 * same_direction
                + 0.10 * same_label
                + 0.30 * geom_full
            )
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    detail_cache[(i, j)] = {
                        "same_boundary_id": bool(same_boundary[i, j]),
                        "same_direction": bool(same_direction[i, j]),
                        "same_label": bool(same_label[i, j]),
                        "geom_full": float(geom_full[i, j]),
                        "full_distance": float(full_dist[i, j]),
                    }

        elif trajectory_type == EPISODE_TYPES["landmark"]:
            landmark_jaccard = np.zeros((n, n), dtype=np.float64)
            turn_count_sim = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    landmark_jaccard[i, j] = jaccard_similarity(bucket[i].landmarks, bucket[j].landmarks)
                    if bucket[i].turn_count is not None and bucket[j].turn_count is not None:
                        turn_count_sim[i, j] = max(
                            0.0,
                            1.0 - abs(float(bucket[i].turn_count) - float(bucket[j].turn_count)) / 3.0,
                        )
            score = (
                0.35 * landmark_jaccard
                + 0.20 * turn_count_sim
                + 0.20 * geom_prefix
                + 0.25 * geom_full
            )
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    detail_cache[(i, j)] = {
                        "landmark_jaccard": float(landmark_jaccard[i, j]),
                        "turn_count_sim": float(turn_count_sim[i, j]),
                        "geom_prefix": float(geom_prefix[i, j]),
                        "geom_full": float(geom_full[i, j]),
                        "prefix_distance": float(prefix_dist[i, j]),
                        "full_distance": float(full_dist[i, j]),
                    }

        elif trajectory_type == EPISODE_TYPES["road"]:
            subtypes = np.array([group.trajectory_subtype or "" for group in bucket], dtype=object)
            same_subtype = (subtypes[:, None] == subtypes[None, :]).astype(np.float64)
            task_num_sim = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if bucket[i].task_num is not None and bucket[j].task_num is not None:
                        task_num_sim[i, j] = max(
                            0.0,
                            1.0 - abs(float(bucket[i].task_num) - float(bucket[j].task_num)) / 3.0,
                        )
            score = (
                0.25 * same_subtype
                + 0.20 * task_num_sim
                + 0.30 * geom_prefix
                + 0.25 * geom_full
            )
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    detail_cache[(i, j)] = {
                        "same_subtype": bool(same_subtype[i, j]),
                        "task_num_sim": float(task_num_sim[i, j]),
                        "geom_prefix": float(geom_prefix[i, j]),
                        "geom_full": float(geom_full[i, j]),
                        "prefix_distance": float(prefix_dist[i, j]),
                        "full_distance": float(full_dist[i, j]),
                    }
        else:
            raise RuntimeError(f"Unsupported trajectory type: {trajectory_type}")

        np.fill_diagonal(score, -np.inf)

        for i, group in enumerate(bucket):
            k = min(top_k, len(bucket) - 1)
            top_indices = np.argpartition(score[i], -k)[-k:]
            top_indices = top_indices[np.argsort(score[i][top_indices])[::-1]]
            supports = []
            for j in top_indices.tolist():
                if not np.isfinite(score[i, j]):
                    continue
                neighbor = bucket[j]
                support = SupportCandidate(
                    neighbor_key=neighbor.key,
                    score=float(score[i, j]),
                    details=detail_cache[(i, j)],
                )
                supports.append(support)
                reverse_supports[neighbor.key].add(group.key)
            group.support_candidates = supports
            group.best_support_score = supports[0].score if supports else float("-inf")

    return reverse_supports


def find_available_support(
    group_key: tuple[str, str],
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    selected_groups: set[tuple[str, str]],
) -> SupportCandidate | None:
    for support in groups_by_key[group_key].support_candidates:
        if support.neighbor_key not in selected_groups:
            return support
    return None


def would_preserve_supports(
    candidate_key: tuple[str, str],
    selected_groups: set[tuple[str, str]],
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    reverse_supports: dict[tuple[str, str], set[tuple[str, str]]],
    current_total_eps: int,
    max_total_eps: int,
) -> SupportCandidate | None:
    candidate = groups_by_key[candidate_key]
    if candidate_key in selected_groups:
        return None
    if current_total_eps + candidate.episode_count > max_total_eps:
        return None

    selected_plus = set(selected_groups)
    selected_plus.add(candidate_key)
    candidate_support = find_available_support(candidate_key, groups_by_key, selected_plus)
    if candidate_support is None:
        return None

    for impacted_key in reverse_supports.get(candidate_key, set()):
        if impacted_key not in selected_groups:
            continue
        if find_available_support(impacted_key, groups_by_key, selected_plus) is None:
            return None
    return candidate_support


def build_scene_candidates(
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    tie_breakers: dict[tuple[str, str], float],
) -> dict[str, list[tuple[str, str]]]:
    per_scene: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group in groups_by_key.values():
        per_scene[group.scene_id].append(group.key)
    for scene_id, keys in per_scene.items():
        keys.sort(
            key=lambda group_key: (
                -groups_by_key[group_key].best_support_score,
                groups_by_key[group_key].episode_count,
                tie_breakers[group_key],
                group_key[1],
            )
        )
    return per_scene


def build_type_candidates(
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    tie_breakers: dict[tuple[str, str], float],
) -> dict[str, list[tuple[str, str]]]:
    per_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group in groups_by_key.values():
        per_type[group.trajectory_type].append(group.key)
    for trajectory_type, keys in per_type.items():
        keys.sort(
            key=lambda group_key: (
                -groups_by_key[group_key].best_support_score,
                groups_by_key[group_key].episode_count,
                tie_breakers[group_key],
                groups_by_key[group_key].scene_id,
                groups_by_key[group_key].trajectory_id,
            )
        )
    return per_type


def select_trajectory_groups(
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    reverse_supports: dict[tuple[str, str], set[tuple[str, str]]],
    target_total_eps: int,
    target_ratios: dict[str, float],
    scene_coverage_min: int,
    size_tolerance: float,
    ratio_tolerance_pp: float,
    seed: int,
) -> dict[str, Any]:
    min_total_eps = math.floor(target_total_eps * (1.0 - size_tolerance))
    max_total_eps = math.ceil(target_total_eps * (1.0 + size_tolerance))

    rng = random.Random(seed)
    tie_breakers = {group_key: rng.random() for group_key in groups_by_key}

    scene_candidates = build_scene_candidates(groups_by_key, tie_breakers)
    type_candidates = build_type_candidates(groups_by_key, tie_breakers)
    scene_candidate_indices = {scene_id: 0 for scene_id in scene_candidates}
    type_candidate_indices = {trajectory_type: 0 for trajectory_type in type_candidates}

    selected_group_order: list[tuple[str, str]] = []
    selected_groups: set[tuple[str, str]] = set()
    selected_counts: Counter[str] = Counter()
    selected_total_eps = 0
    covered_scenes: set[str] = set()

    def candidate_priority(group_key: tuple[str, str], current_counts: Counter[str], current_total: int) -> float:
        group = groups_by_key[group_key]
        deficit = type_deficit(group.trajectory_type, current_counts, current_total, target_ratios)
        return group.best_support_score + 0.25 * deficit - 0.01 * max(group.episode_count - 3, 0)

    def peek_best_scene_candidate(scene_id: str) -> tuple[tuple[str, str], SupportCandidate] | None:
        keys = scene_candidates[scene_id]
        idx = scene_candidate_indices[scene_id]
        while idx < len(keys):
            group_key = keys[idx]
            support = would_preserve_supports(
                candidate_key=group_key,
                selected_groups=selected_groups,
                groups_by_key=groups_by_key,
                reverse_supports=reverse_supports,
                current_total_eps=selected_total_eps,
                max_total_eps=max_total_eps,
            )
            if support is not None:
                scene_candidate_indices[scene_id] = idx
                return group_key, support
            idx += 1
        scene_candidate_indices[scene_id] = idx
        return None

    def peek_best_type_candidate(trajectory_type: str) -> tuple[tuple[str, str], SupportCandidate] | None:
        keys = type_candidates.get(trajectory_type, [])
        idx = type_candidate_indices.get(trajectory_type, 0)
        while idx < len(keys):
            group_key = keys[idx]
            support = would_preserve_supports(
                candidate_key=group_key,
                selected_groups=selected_groups,
                groups_by_key=groups_by_key,
                reverse_supports=reverse_supports,
                current_total_eps=selected_total_eps,
                max_total_eps=max_total_eps,
            )
            if support is not None:
                type_candidate_indices[trajectory_type] = idx
                return group_key, support
            idx += 1
        type_candidate_indices[trajectory_type] = idx
        return None

    def select_group(group_key: tuple[str, str]) -> None:
        nonlocal selected_total_eps
        group = groups_by_key[group_key]
        selected_groups.add(group_key)
        selected_group_order.append(group_key)
        covered_scenes.add(group.scene_id)
        selected_counts[group.trajectory_type] += group.episode_count
        selected_total_eps += group.episode_count

    while len(covered_scenes) < scene_coverage_min:
        best_choice: tuple[float, tuple[str, str], SupportCandidate] | None = None
        for scene_id in sorted(set(scene_candidates) - covered_scenes):
            scene_candidate = peek_best_scene_candidate(scene_id)
            if scene_candidate is None:
                continue
            group_key, support = scene_candidate
            priority = candidate_priority(group_key, selected_counts, selected_total_eps)
            if best_choice is None or priority > best_choice[0]:
                best_choice = (priority, group_key, support)
        if best_choice is None:
            raise RuntimeError(
                f"Unable to reach required scene coverage {scene_coverage_min}; "
                f"only covered {len(covered_scenes)} scenes"
            )
        select_group(best_choice[1])

    while True:
        ratio_error = max_ratio_error(selected_counts, selected_total_eps, target_ratios)
        if selected_total_eps >= min_total_eps and ratio_error <= ratio_tolerance_pp:
            break
        if selected_total_eps >= max_total_eps:
            break

        preferred_types = sorted(
            TASK_ORDER,
            key=lambda task: (
                type_deficit(task, selected_counts, selected_total_eps, target_ratios),
                -selected_counts.get(task, 0),
            ),
            reverse=True,
        )

        chosen: tuple[tuple[str, str], SupportCandidate] | None = None
        for trajectory_type in preferred_types:
            chosen = peek_best_type_candidate(trajectory_type)
            if chosen is not None:
                break
        if chosen is None:
            break

        select_group(chosen[0])

    final_ratio_error = max_ratio_error(selected_counts, selected_total_eps, target_ratios)
    if selected_total_eps < min_total_eps or selected_total_eps > max_total_eps:
        raise RuntimeError(
            "Selected episode total is outside tolerance: "
            f"{selected_total_eps} not in [{min_total_eps}, {max_total_eps}]"
        )
    if final_ratio_error > ratio_tolerance_pp:
        raise RuntimeError(
            "Selected episode ratios are outside tolerance: "
            f"max ratio error {final_ratio_error:.4f} > {ratio_tolerance_pp:.4f}"
        )

    selected_supports: dict[tuple[str, str], SupportCandidate] = {}
    missing_final_supports = []
    for group_key in selected_group_order:
        support = find_available_support(group_key, groups_by_key, selected_groups)
        if support is None:
            missing_final_supports.append(group_key)
        else:
            selected_supports[group_key] = support
    if missing_final_supports:
        raise RuntimeError(f"Selected groups lost all supports: {missing_final_supports[:5]}")

    return {
        "selected_group_order": selected_group_order,
        "selected_group_set": selected_groups,
        "selected_counts": selected_counts,
        "selected_total_eps": selected_total_eps,
        "selected_scenes": covered_scenes,
        "selected_supports": selected_supports,
        "target_total_eps": target_total_eps,
        "min_total_eps": min_total_eps,
        "max_total_eps": max_total_eps,
        "target_ratios": target_ratios,
        "ratio_tolerance_pp": ratio_tolerance_pp,
        "size_tolerance": size_tolerance,
    }


def selection_manifest(
    version: str,
    output_split: str,
    reference_split: str,
    selection: dict[str, Any],
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    complete_group_stats: dict[str, Any],
) -> dict[str, Any]:
    selected_group_order = selection["selected_group_order"]
    selected_counts = selection["selected_counts"]
    selected_total_eps = selection["selected_total_eps"]
    target_ratios = selection["target_ratios"]
    selected_ratios = ratio_map(selected_counts, selected_total_eps)

    entries = []
    for group_key in selected_group_order:
        group = groups_by_key[group_key]
        support = selection["selected_supports"][group_key]
        entries.append(
            {
                "scene_id": group.scene_id,
                "trajectory_id": group.trajectory_id,
                "trajectory_type": group.trajectory_type,
                "trajectory_subtype": group.trajectory_subtype,
                "episode_count": group.episode_count,
                "episode_ids": group.episode_ids,
                "videos": list(group.videos),
                "support_group": {
                    "scene_id": groups_by_key[support.neighbor_key].scene_id,
                    "trajectory_id": groups_by_key[support.neighbor_key].trajectory_id,
                    "score": support.score,
                    "details": support.details,
                },
                "top_support_candidates": [
                    {
                        "scene_id": groups_by_key[candidate.neighbor_key].scene_id,
                        "trajectory_id": groups_by_key[candidate.neighbor_key].trajectory_id,
                        "score": candidate.score,
                        "details": candidate.details,
                    }
                    for candidate in group.support_candidates[:3]
                ],
            }
        )

    return {
        "version": version,
        "output_split": output_split,
        "reference_split": reference_split,
        "selection_unit": "trajectory_group",
        "target_total_eps": selection["target_total_eps"],
        "selected_total_eps": selected_total_eps,
        "min_total_eps": selection["min_total_eps"],
        "max_total_eps": selection["max_total_eps"],
        "ratio_tolerance_pp": selection["ratio_tolerance_pp"],
        "size_tolerance": selection["size_tolerance"],
        "selected_group_count": len(selected_group_order),
        "selected_scene_count": len(selection["selected_scenes"]),
        "selected_scene_ids": sorted(selection["selected_scenes"]),
        "selected_episode_counts_by_type": {task: int(selected_counts.get(task, 0)) for task in TASK_ORDER},
        "selected_episode_ratios_by_type": {task: round(selected_ratios[task], 6) for task in TASK_ORDER},
        "target_episode_ratios_by_type": {task: round(target_ratios[task], 6) for task in TASK_ORDER},
        "complete_group_stats": complete_group_stats,
        "selected_groups": entries,
    }


def quality_manifest(
    selection: dict[str, Any],
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
) -> dict[str, Any]:
    per_type_scores: dict[str, list[float]] = defaultdict(list)
    boundary_same_id = []
    boundary_same_direction = []
    boundary_same_label = []
    landmark_jaccard = []
    landmark_turn_count = []
    road_same_subtype = []
    road_task_num = []

    for group_key, support in selection["selected_supports"].items():
        group = groups_by_key[group_key]
        per_type_scores[group.trajectory_type].append(float(support.score))
        if group.trajectory_type == EPISODE_TYPES["boundary"]:
            boundary_same_id.append(float(support.details.get("same_boundary_id", 0.0)))
            boundary_same_direction.append(float(support.details.get("same_direction", 0.0)))
            boundary_same_label.append(float(support.details.get("same_label", 0.0)))
        elif group.trajectory_type == EPISODE_TYPES["landmark"]:
            landmark_jaccard.append(float(support.details.get("landmark_jaccard", 0.0)))
            landmark_turn_count.append(float(support.details.get("turn_count_sim", 0.0)))
        elif group.trajectory_type == EPISODE_TYPES["road"]:
            road_same_subtype.append(float(support.details.get("same_subtype", 0.0)))
            road_task_num.append(float(support.details.get("task_num_sim", 0.0)))

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "p10": None, "median": None, "p90": None, "max": None}
        return {
            "count": len(values),
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "p10": quantile(values, 0.10),
            "median": quantile(values, 0.50),
            "p90": quantile(values, 0.90),
            "max": float(max(values)),
        }

    return {
        "support_score_by_type": {task: summary(per_type_scores.get(task, [])) for task in TASK_ORDER},
        "boundary_quality": {
            "same_boundary_id": summary(boundary_same_id),
            "same_direction": summary(boundary_same_direction),
            "same_label": summary(boundary_same_label),
        },
        "landmark_quality": {
            "landmark_jaccard": summary(landmark_jaccard),
            "turn_count_similarity": summary(landmark_turn_count),
        },
        "road_quality": {
            "same_subtype": summary(road_same_subtype),
            "task_num_similarity": summary(road_task_num),
        },
    }


def reindex_records_in_order(
    ordered_videos: list[str],
    records_by_video: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for idx, video in enumerate(ordered_videos):
        item = copy.deepcopy(records_by_video[video])
        item["id"] = idx
        output.append(item)
    return output


def write_split_files(split_dir: Path, episodes: list[dict[str, Any]]) -> None:
    write_episode_records(split_dir / EPISODE_FILES["all"], episodes)
    write_episode_records(
        split_dir / EPISODE_FILES["boundary"],
        [episode for episode in episodes if episode["trajectory_type"] == EPISODE_TYPES["boundary"]],
    )
    write_episode_records(
        split_dir / EPISODE_FILES["landmark"],
        [episode for episode in episodes if episode["trajectory_type"] == EPISODE_TYPES["landmark"]],
    )
    write_episode_records(
        split_dir / EPISODE_FILES["road"],
        [episode for episode in episodes if episode["trajectory_type"] == EPISODE_TYPES["road"]],
    )


def materialize_selection(
    version: str,
    output_split: str,
    output_trajectory_dirname: str,
    overwrite: bool,
    copy_mode: str,
    train_episodes: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
    groups_by_key: dict[tuple[str, str], TrajectoryGroup],
    selection: dict[str, Any],
    selection_manifest_data: dict[str, Any],
    quality_manifest_data: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = DATASET_ROOT / version
    train_dir = dataset_root / "episodes" / "train"
    output_split_dir = dataset_root / "episodes" / "eval" / output_split
    trajectory_root = dataset_root / "trajectory_data"
    output_traj_dir = dataset_root / output_trajectory_dirname

    prepare_output_dir(output_split_dir, overwrite=overwrite)
    prepare_output_dir(output_traj_dir, overwrite=overwrite)

    selected_group_set: set[tuple[str, str]] = selection["selected_group_set"]
    selected_episode_keys = {
        episode_key(episode)
        for group_key in selected_group_set
        for episode in groups_by_key[group_key].episodes
    }

    selected_episodes = [episode for episode in train_episodes if episode_key(episode) in selected_episode_keys]
    remaining_train_episodes = [episode for episode in train_episodes if episode_key(episode) not in selected_episode_keys]

    selected_total = len(selected_episodes)
    expected_total = selection["selected_total_eps"]
    if selected_total != expected_total:
        raise RuntimeError(f"Selected episode total mismatch: {selected_total} vs expected {expected_total}")

    write_split_files(output_split_dir, selected_episodes)
    with (output_split_dir / "selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_manifest_data, handle, indent=2, ensure_ascii=False)
    with (output_split_dir / "quality_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(quality_manifest_data, handle, indent=2, ensure_ascii=False)

    ensure_unique_summary_keys(summary_records)
    annotations_by_video = index_annotations_by_video(annotation_records)
    summary_by_video = OrderedDict((str(record["video"]), record) for record in summary_records)
    if set(summary_by_video) != set(annotations_by_video):
        raise RuntimeError("Main summary / annotations video sets differ before materialization")

    selected_video_order = [
        str(record["video"])
        for record in summary_records
        if summary_key(record) in selected_episode_keys
    ]
    remaining_video_order = [
        str(record["video"])
        for record in summary_records
        if summary_key(record) not in selected_episode_keys
    ]

    selected_summary_by_video = OrderedDict((video, summary_by_video[video]) for video in selected_video_order)
    selected_annotations_by_video = OrderedDict((video, annotations_by_video[video]) for video in selected_video_order)
    remaining_summary_by_video = OrderedDict((video, summary_by_video[video]) for video in remaining_video_order)
    remaining_annotations_by_video = OrderedDict((video, annotations_by_video[video]) for video in remaining_video_order)

    output_images_root = output_traj_dir / "images"
    output_images_root.mkdir(parents=True, exist_ok=True)
    for video in selected_video_order:
        source_dir = trajectory_root / video
        if not source_dir.is_dir():
            raise RuntimeError(f"Missing trajectory image directory in main trajectory_data: {source_dir}")
        copy_tree(source_dir, output_traj_dir / video, copy_mode=copy_mode)

    selected_summary_records = reindex_records_in_order(selected_video_order, selected_summary_by_video)
    selected_annotation_records = reindex_records_in_order(selected_video_order, selected_annotations_by_video)
    write_summary(output_traj_dir / "summary.json", selected_summary_records)
    write_annotations(output_traj_dir / "annotations.json", selected_annotation_records)
    subset_manifest = {
        "version": version,
        "source_trajectory_dirname": "trajectory_data",
        "selection_unit": "trajectory_group",
        "selected_episode_total": selected_total,
        "selected_video_total": len(selected_video_order),
        "output_dir": str(output_traj_dir),
        "copy_mode": copy_mode,
    }
    with (output_traj_dir / "subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(subset_manifest, handle, indent=2, ensure_ascii=False)

    remaining_summary_records = reindex_records_in_order(remaining_video_order, remaining_summary_by_video)
    remaining_annotation_records = reindex_records_in_order(remaining_video_order, remaining_annotations_by_video)
    write_split_files(train_dir, remaining_train_episodes)
    write_summary(trajectory_root / "summary.json", remaining_summary_records)
    write_annotations(trajectory_root / "annotations.json", remaining_annotation_records)

    removed_image_dirs = 0
    for video in selected_video_order:
        image_dir = trajectory_root / video
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
            removed_image_dirs += 1

    actual_output_dirs = {path.name for path in output_images_root.iterdir() if path.is_dir()}
    expected_output_dirs = {Path(video).name for video in selected_video_order}
    if actual_output_dirs != expected_output_dirs:
        raise RuntimeError("Output val_seen trajectory images do not match selected videos")

    remaining_actual_image_dirs = {
        path.name for path in (trajectory_root / "images").iterdir() if path.is_dir()
    }
    remaining_expected_image_dirs = {Path(video).name for video in remaining_video_order}
    if remaining_actual_image_dirs != remaining_expected_image_dirs:
        raise RuntimeError("Remaining main trajectory_data images do not match remaining summary videos")

    return {
        "output_split": output_split,
        "output_trajectory_dirname": output_trajectory_dirname,
        "selected_episode_total": selected_total,
        "selected_group_total": len(selected_group_set),
        "selected_counts": trajectory_type_counts(selected_episodes),
        "train_total_after": len(remaining_train_episodes),
        "trajectory_total_after": len(remaining_summary_records),
        "removed_image_dir_count": removed_image_dirs,
    }


def build_similarity_val_seen_from_train(
    version: str,
    reference_split: str,
    output_split: str,
    output_trajectory_dirname: str,
    scene_coverage_min: int,
    size_tolerance: float,
    ratio_tolerance_pp: float,
    top_k: int,
    seed: int,
    dry_run: bool,
    overwrite: bool,
    copy_mode: str,
) -> dict[str, Any]:
    dataset_root = DATASET_ROOT / version
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset version not found: {dataset_root}")

    train_path = dataset_root / "episodes" / "train" / EPISODE_FILES["all"]
    reference_split_path = dataset_root / "episodes" / "eval" / reference_split / EPISODE_FILES["all"]
    trajectory_root = dataset_root / "trajectory_data"
    summary_path = trajectory_root / "summary.json"
    annotations_path = trajectory_root / "annotations.json"
    images_root = trajectory_root / "images"

    train_episodes = read_episode_records(train_path)
    reference_episodes = read_episode_records(reference_split_path)
    summary_records = read_summary(summary_path)
    annotation_records = read_annotations(annotations_path)

    ensure_unique_episode_keys(train_episodes, "train split")
    ensure_unique_episode_keys(reference_episodes, f"reference split {reference_split}")

    summary_by_key = ensure_unique_summary_keys(summary_records)
    annotations_by_video = index_annotations_by_video(annotation_records)
    summary_videos = {str(record["video"]) for record in summary_records}
    if summary_videos != set(annotations_by_video):
        raise RuntimeError("trajectory summary / annotations do not cover the same video set")

    groups_by_key, completeness = build_complete_groups(train_episodes, summary_by_key, images_root)
    reverse_supports = build_supports(groups_by_key, top_k=top_k)

    target_total_eps = len(reference_episodes)
    target_counts = trajectory_type_counts(reference_episodes)
    target_ratios = ratio_map(Counter(target_counts), target_total_eps)

    selection = select_trajectory_groups(
        groups_by_key=groups_by_key,
        reverse_supports=reverse_supports,
        target_total_eps=target_total_eps,
        target_ratios=target_ratios,
        scene_coverage_min=scene_coverage_min,
        size_tolerance=size_tolerance,
        ratio_tolerance_pp=ratio_tolerance_pp,
        seed=seed,
    )

    selection_manifest_data = selection_manifest(
        version=version,
        output_split=output_split,
        reference_split=reference_split,
        selection=selection,
        groups_by_key=groups_by_key,
        complete_group_stats=completeness["stats"],
    )
    quality_manifest_data = quality_manifest(selection, groups_by_key)

    result = {
        "version": version,
        "reference_split": reference_split,
        "output_split": output_split,
        "output_trajectory_dirname": output_trajectory_dirname,
        "scene_coverage_min": scene_coverage_min,
        "size_tolerance": size_tolerance,
        "ratio_tolerance_pp": ratio_tolerance_pp,
        "top_k": top_k,
        "seed": seed,
        "dry_run": dry_run,
        "target_total_eps": target_total_eps,
        "target_counts": target_counts,
        "target_ratios": target_ratios,
        "complete_group_stats": completeness["stats"],
        "selected_total_eps": selection["selected_total_eps"],
        "selected_total_groups": len(selection["selected_group_order"]),
        "selected_scene_count": len(selection["selected_scenes"]),
        "selected_counts": {task: int(selection["selected_counts"].get(task, 0)) for task in TASK_ORDER},
        "selected_ratios": ratio_map(selection["selected_counts"], selection["selected_total_eps"]),
        "selection_manifest": selection_manifest_data,
        "quality_manifest": quality_manifest_data,
    }

    if dry_run:
        return result

    materialized = materialize_selection(
        version=version,
        output_split=output_split,
        output_trajectory_dirname=output_trajectory_dirname,
        overwrite=overwrite,
        copy_mode=copy_mode,
        train_episodes=train_episodes,
        summary_records=summary_records,
        annotation_records=annotation_records,
        groups_by_key=groups_by_key,
        selection=selection,
        selection_manifest_data=selection_manifest_data,
        quality_manifest_data=quality_manifest_data,
    )
    result["materialized"] = materialized
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a similarity-biased route-disjoint val_seen split from train using trajectory groups"
    )
    parser.add_argument("version", help="Dataset version, e.g. ver_260418")
    parser.add_argument("--reference-split", default="val_seen_update")
    parser.add_argument("--output-split", default="val_seen")
    parser.add_argument("--output-trajectory-dirname", default="trajectory_data_val_seen")
    parser.add_argument("--scene-coverage-min", type=int, default=45)
    parser.add_argument("--size-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--ratio-tolerance-pp",
        type=float,
        default=0.02,
        help="Absolute ratio tolerance in fractional points, e.g. 0.02 means 2 percentage points",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = build_similarity_val_seen_from_train(
        version=args.version,
        reference_split=args.reference_split,
        output_split=args.output_split,
        output_trajectory_dirname=args.output_trajectory_dirname,
        scene_coverage_min=args.scene_coverage_min,
        size_tolerance=args.size_tolerance,
        ratio_tolerance_pp=args.ratio_tolerance_pp,
        top_k=args.top_k,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        copy_mode=args.copy_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
