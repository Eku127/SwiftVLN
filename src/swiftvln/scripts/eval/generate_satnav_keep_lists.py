#!/usr/bin/env python3
"""Generate deterministic SatNav keep lists and deletion reports.

This script builds a balanced eval subset for a SatNav dataset version by:

- enforcing explicit per-split / per-trajectory-type keep counts
- preserving scene coverage via proportional per-scene quotas
- prioritizing deletion of high-confidence dirty episodes
- deprioritizing deletion of universally hard episodes
- using overlapvln per-episode outcome separation to keep more discriminative cases

Important: ``episode_id`` is not globally unique within a split for SatNav 0404.
All downstream filtering must therefore use the composite key
``(scene_id, episode_id)`` instead of bare ``episode_id``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCENE_EPISODE_KEY_SEP = "::"
TRAJECTORY_TYPES = ("Boundary", "LandmarkSet", "Road")
BASELINE_RESULT_FAMILIES = (
    "streamvln-baseline",
    "navila-baseline",
    "uninavid-baseline",
    "openfly-baseline",
)

PRESETS: Dict[str, Dict[str, Any]] = {
    "data260404_road345": {
        "version": "260404",
        "keep_counts": {
            "val_seen": {"Boundary": 1584, "LandmarkSet": 1589, "Road": 1671},
            "val_unseen": {"Boundary": 2863, "LandmarkSet": 2872, "Road": 3021},
        },
        "min_compare_all_sr": 0.05,
        "never_success_delete_cap_ratio": 0.05,
        "never_success_delete_cap_min": 8,
    }
}


def metric_from_summary(summary: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def stable_hash(parts: Sequence[str]) -> str:
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def scene_episode_key(scene_id: str, episode_id: str) -> str:
    return f"{scene_id}{SCENE_EPISODE_KEY_SEP}{episode_id}"


def split_scene_episode_key(key: str) -> Tuple[str, str]:
    scene_id, episode_id = key.split(SCENE_EPISODE_KEY_SEP, 1)
    return scene_id, episode_id


def mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def euclidean_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def largest_remainder_allocation(
    counts_by_group: Mapping[str, int],
    total_keep: int,
    min_keep_if_present: int = 1,
) -> Dict[str, int]:
    present_groups = {group: count for group, count in counts_by_group.items() if count > 0}
    if not present_groups:
        return {group: 0 for group in counts_by_group}

    if total_keep < len(present_groups) * min_keep_if_present:
        raise ValueError(
            f"total_keep={total_keep} is too small for {len(present_groups)} present groups"
        )

    keep = {group: 0 for group in counts_by_group}
    base = {group: min_keep_if_present for group in present_groups}
    remaining = total_keep - sum(base.values())

    if remaining == 0:
        keep.update(base)
        return keep

    adjustable_total = sum(count - min_keep_if_present for count in present_groups.values())
    if adjustable_total <= 0:
        keep.update(base)
        return keep

    raw_extras: Dict[str, float] = {}
    extra_floor: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    for group, count in present_groups.items():
        raw = (count - min_keep_if_present) * remaining / adjustable_total
        floored = int(math.floor(raw))
        extra_floor[group] = floored
        raw_extras[group] = raw
        remainders.append((raw - floored, group))

    for group, floored in extra_floor.items():
        keep[group] = base[group] + floored

    leftover = total_keep - sum(keep.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, group in remainders[:leftover]:
        keep[group] += 1

    for group, count in present_groups.items():
        keep[group] = min(keep[group], count)

    gap = total_keep - sum(keep.values())
    if gap > 0:
        spare_groups = sorted(
            (
                (count - keep[group], group)
                for group, count in present_groups.items()
                if keep[group] < count
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _, group in spare_groups:
            if gap == 0:
                break
            keep[group] += 1
            gap -= 1

    if sum(keep.values()) != total_keep:
        raise ValueError(f"allocation failed: expected {total_keep}, got {sum(keep.values())}")
    return keep


@dataclass(frozen=True)
class EpisodeRecord:
    split: str
    scene_id: str
    episode_id: str
    trajectory_type: str
    instruction_type: str
    instruction_text: str
    key: str
    index_in_split: int
    payload: Mapping[str, Any]


@dataclass
class ModelRun:
    model_name: str
    family: str
    split: str
    result_path: Path
    summary_path: Path
    all_sr: float
    success_by_key: Dict[str, float]


def load_dataset_split(version: str, split: str) -> List[EpisodeRecord]:
    path = Path(f"/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_{version}/episodes/eval/{split}/all_episodes.json")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        episodes = payload.get("episodes", [])
    else:
        episodes = payload

    records: List[EpisodeRecord] = []
    for index, episode in enumerate(episodes):
        scene_id = str(episode["scene_id"])
        episode_id = str(episode["episode_id"])
        records.append(
            EpisodeRecord(
                split=split,
                scene_id=scene_id,
                episode_id=episode_id,
                trajectory_type=str(episode["trajectory_type"]),
                instruction_type=str(episode.get("instruction", {}).get("instruction_type", "")),
                instruction_text=str(episode.get("instruction", {}).get("instruction_text", "")),
                key=scene_episode_key(scene_id, episode_id),
                index_in_split=index,
                payload=episode,
            )
        )
    return records


def load_jsonl_success_by_key(result_path: Path) -> Dict[str, float]:
    success_by_key: Dict[str, float] = {}
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scene_id = row.get("scene_id")
            episode_id = row.get("episode_id")
            if scene_id is None or episode_id is None:
                continue
            key = scene_episode_key(str(scene_id), str(episode_id))
            success_by_key[key] = float(row.get("success", 0.0))
    return success_by_key


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_overlap_runs(version: str, split: str) -> List[ModelRun]:
    root = Path("results/eval/overlapvln")
    runs_by_model: Dict[str, Path] = {}
    for model_dir in root.glob(f"*data{version}*"):
        split_dir = model_dir / split
        if not split_dir.is_dir():
            continue
        for run_dir in sorted(split_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "evaluation_summary.json"
            result_path = run_dir / "all_results.jsonl"
            if not summary_path.is_file() or not result_path.is_file():
                continue
            current = runs_by_model.get(model_dir.name)
            if current is None or run_dir.name > current.name:
                runs_by_model[model_dir.name] = run_dir

    model_runs: List[ModelRun] = []
    for model_name, run_dir in sorted(runs_by_model.items()):
        summary_path = run_dir / "evaluation_summary.json"
        result_path = run_dir / "all_results.jsonl"
        summary = load_summary(summary_path)
        model_runs.append(
            ModelRun(
                model_name=model_name,
                family="overlapvln",
                split=split,
                result_path=result_path,
                summary_path=summary_path,
                all_sr=metric_from_summary(summary, "success_rate", default=0.0),
                success_by_key=load_jsonl_success_by_key(result_path),
            )
        )
    return model_runs


def find_baseline_runs(version: str, split: str) -> List[ModelRun]:
    model_runs: List[ModelRun] = []
    for family in BASELINE_RESULT_FAMILIES:
        family_root = Path("results") / family
        if not family_root.is_dir():
            continue
        for model_dir in sorted(family_root.glob(f"*data{version}*")):
            split_dir = model_dir / split
            summary_path = split_dir / "evaluation_summary.json"
            result_path = split_dir / "result.jsonl"
            if not summary_path.is_file() or not result_path.is_file():
                continue
            summary = load_summary(summary_path)
            model_runs.append(
                ModelRun(
                    model_name=model_dir.name,
                    family=family,
                    split=split,
                    result_path=result_path,
                    summary_path=summary_path,
                    all_sr=metric_from_summary(summary, "success_rate", default=0.0),
                    success_by_key=load_jsonl_success_by_key(result_path),
                )
            )
    return model_runs


def is_overlap_anchor_model(model_name: str, version: str) -> bool:
    token = f"-overlap0-pf-h8-b1.0-pool-s2-noembed-data{version}-"
    if token not in model_name:
        return False
    forbidden_tokens = (
        "-qa",
        "-initial-",
        "-pose-",
        "-posefilm-",
        "-gtc-",
        "-sgtc-",
        "-h0-nomem-",
        "-overlap16-",
        "-map-",
    )
    return not any(token in model_name for token in forbidden_tokens)


def find_anchor_and_compare_runs(
    overlap_runs: Sequence[ModelRun],
    version: str,
    min_compare_all_sr: float,
) -> Tuple[List[ModelRun], List[ModelRun], List[str]]:
    anchor_runs = [run for run in overlap_runs if is_overlap_anchor_model(run.model_name, version)]
    if not anchor_runs:
        raise SystemExit(f"[ERROR] no anchor overlap baseline found for data{version}")

    excluded_for_compare: List[str] = []
    compare_runs: List[ModelRun] = []
    anchor_names = {run.model_name for run in anchor_runs}
    for run in overlap_runs:
        if run.model_name in anchor_names:
            continue
        if run.all_sr < min_compare_all_sr:
            excluded_for_compare.append(run.model_name)
            continue
        compare_runs.append(run)

    if not compare_runs:
        raise SystemExit("[ERROR] compare run pool is empty after filtering")
    return anchor_runs, compare_runs, excluded_for_compare


def dirty_reasons_for_episode(record: EpisodeRecord) -> List[str]:
    episode = record.payload
    reasons: List[str] = []
    instruction = record.instruction_text.strip()
    reference_path = episode.get("reference_path", [])
    goals = episode.get("goals", [])
    waypoints = episode.get("waypoints", [])

    if not instruction:
        reasons.append("empty_instruction")
    if len(reference_path) < 2:
        reasons.append("short_reference_path")
    if not goals:
        reasons.append("missing_goals")
    if not waypoints:
        reasons.append("missing_waypoints")

    if record.trajectory_type == "Boundary":
        direction = str(episode.get("aux_info", {}).get("direction", "")).lower()
        text = instruction.lower()
        has_ccw = "counter-clockwise" in text or "counterclockwise" in text
        has_cw = "clockwise" in text and not has_ccw
        if direction == "ccw" and has_cw:
            reasons.append("boundary_direction_conflict")
        if direction == "cw" and has_ccw:
            reasons.append("boundary_direction_conflict")

    if reference_path:
        start_gap = euclidean_xy(episode["start_position"], reference_path[0])
        if start_gap > 1e-3:
            reasons.append("start_reference_mismatch")

    if reference_path and goals:
        goal_gap = euclidean_xy(goals[0]["position"], reference_path[-1])
        if goal_gap > 1e-3:
            reasons.append("goal_reference_mismatch")

        text = instruction.lower()
        if any(token in text for token in ("return to", "back to", "return toward")) and goal_gap > 1e-3:
            reasons.append("return_goal_mismatch")

    return sorted(set(reasons))


def weak_suspect_reasons_for_episode(record: EpisodeRecord) -> List[str]:
    episode = record.payload
    reasons: List[str] = []
    if record.trajectory_type == "LandmarkSet":
        raw_landmarks = [str(item).strip() for item in episode.get("aux_info", {}).get("landmarks", [])]
        normalized = [normalize_text(item) for item in raw_landmarks if item]
        if normalized and len(normalized) != len(set(normalized)):
            reasons.append("duplicate_landmark_names")

    reference_path = episode.get("reference_path", [])
    if reference_path:
        start_gap = euclidean_xy(episode["start_position"], reference_path[0])
        if 1e-4 < start_gap <= 1e-3:
            reasons.append("minor_start_reference_mismatch")

    return sorted(set(reasons))


@dataclass
class EpisodeScore:
    anchor_success: float
    compare_mean_success: float
    compare_separation: float
    overlap_mean_success: float
    all_family_mean_success: float
    consensus_easy: bool
    no_model_success_all_families: bool

    @property
    def delete_score(self) -> float:
        score = (1.0 - self.compare_separation) * 10.0
        score += self.all_family_mean_success * 3.0
        if self.consensus_easy:
            score += 2.0
        if self.no_model_success_all_families:
            score -= 1.5
        return score


def build_episode_scores(
    dataset_records: Sequence[EpisodeRecord],
    anchor_runs: Sequence[ModelRun],
    compare_runs: Sequence[ModelRun],
    auxiliary_runs: Sequence[ModelRun],
) -> Dict[str, EpisodeScore]:
    scores: Dict[str, EpisodeScore] = {}
    for record in dataset_records:
        anchor_successes = [
            run.success_by_key.get(record.key, 0.0)
            for run in anchor_runs
        ]
        compare_successes = [
            run.success_by_key.get(record.key, 0.0)
            for run in compare_runs
        ]
        auxiliary_successes = [
            run.success_by_key.get(record.key, 0.0)
            for run in auxiliary_runs
        ]

        anchor_success = mean(anchor_successes, default=0.0)
        compare_mean_success = mean(compare_successes, default=0.0)
        compare_separation = mean(
            [abs(anchor_success - value) for value in compare_successes],
            default=0.0,
        )
        overlap_values = list(anchor_successes) + list(compare_successes)
        all_family_values = list(auxiliary_successes)
        consensus_easy = bool(all_family_values) and all(value >= 1.0 for value in all_family_values)
        no_model_success = bool(all_family_values) and all(value <= 0.0 for value in all_family_values)

        scores[record.key] = EpisodeScore(
            anchor_success=anchor_success,
            compare_mean_success=compare_mean_success,
            compare_separation=compare_separation,
            overlap_mean_success=mean(overlap_values, default=0.0),
            all_family_mean_success=mean(all_family_values, default=0.0),
            consensus_easy=consensus_easy,
            no_model_success_all_families=no_model_success,
        )
    return scores


def make_delete_reason(record: EpisodeRecord, dirty_reasons: Sequence[str], score: EpisodeScore) -> str:
    if dirty_reasons:
        return "dirty"
    if score.consensus_easy and score.compare_separation <= 0.05:
        return "low_info_easy"
    if score.no_model_success_all_families:
        return "no_model_success"
    if score.compare_separation <= 0.15:
        return "low_separation"
    return "fill_quota"


def candidate_sort_key(
    record: EpisodeRecord,
    dirty_reasons: Sequence[str],
    score: EpisodeScore,
) -> Tuple[float, float, float, str]:
    dirty_rank = 1.0 if dirty_reasons else 0.0
    consensus_easy_rank = 1.0 if score.consensus_easy else 0.0
    return (
        dirty_rank,
        consensus_easy_rank,
        score.delete_score,
        stable_hash((record.split, record.scene_id, record.episode_id)),
    )


def summarize_counts(records: Iterable[EpisodeRecord]) -> Dict[str, Dict[str, int]]:
    counter: Dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counter[record.split][record.trajectory_type] += 1
    return {split: dict(type_counter) for split, type_counter in counter.items()}


def build_scene_keep_quotas(
    records: Sequence[EpisodeRecord],
    keep_counts: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    scene_counts: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for record in records:
        scene_counts[record.split][record.trajectory_type][record.scene_id] += 1

    quotas: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for split, type_counts in scene_counts.items():
        for trajectory_type, counts_by_scene in type_counts.items():
            total_keep = int(keep_counts[split][trajectory_type])
            quotas[split][trajectory_type] = largest_remainder_allocation(counts_by_scene, total_keep)
    return {split: dict(type_quota) for split, type_quota in quotas.items()}


def select_deletions(
    records: Sequence[EpisodeRecord],
    keep_counts: Mapping[str, Mapping[str, int]],
    scene_keep_quotas: Mapping[str, Mapping[str, Mapping[str, int]]],
    dirty_reason_map: Mapping[str, Sequence[str]],
    weak_suspect_reason_map: Mapping[str, Sequence[str]],
    score_map: Mapping[str, EpisodeScore],
    never_success_delete_cap_ratio: float,
    never_success_delete_cap_min: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[EpisodeRecord]]]:
    records_by_bucket: Dict[Tuple[str, str, str], List[EpisodeRecord]] = defaultdict(list)
    for record in records:
        records_by_bucket[(record.split, record.trajectory_type, record.scene_id)].append(record)

    total_delete_by_split: Dict[str, int] = {}
    for split in keep_counts:
        total_records = sum(1 for record in records if record.split == split)
        total_keep = sum(int(value) for value in keep_counts[split].values())
        total_delete_by_split[split] = total_records - total_keep

    never_success_cap_by_split = {
        split: max(never_success_delete_cap_min, int(total_delete * never_success_delete_cap_ratio))
        for split, total_delete in total_delete_by_split.items()
    }
    never_success_deleted = Counter()

    deleted_rows: List[Dict[str, Any]] = []
    deleted_by_split: Dict[str, List[EpisodeRecord]] = defaultdict(list)

    for split in sorted(scene_keep_quotas):
        for trajectory_type in TRAJECTORY_TYPES:
            if trajectory_type not in scene_keep_quotas[split]:
                continue
            for scene_id in sorted(scene_keep_quotas[split][trajectory_type]):
                bucket = (split, trajectory_type, scene_id)
                bucket_records = records_by_bucket[bucket]
                keep_quota = int(scene_keep_quotas[split][trajectory_type][scene_id])
                delete_quota = len(bucket_records) - keep_quota
                if delete_quota <= 0:
                    continue

                ranked = sorted(
                    bucket_records,
                    key=lambda record: candidate_sort_key(
                        record,
                        dirty_reason_map.get(record.key, ()),
                        score_map[record.key],
                    ),
                    reverse=True,
                )

                chosen: List[EpisodeRecord] = []
                chosen_keys: set[str] = set()
                for record in ranked:
                    if len(chosen) >= delete_quota:
                        break
                    score = score_map[record.key]
                    if score.no_model_success_all_families and (
                        never_success_deleted[split] >= never_success_cap_by_split[split]
                    ):
                        continue
                    chosen.append(record)
                    chosen_keys.add(record.key)
                    if score.no_model_success_all_families:
                        never_success_deleted[split] += 1

                if len(chosen) < delete_quota:
                    for record in ranked:
                        if len(chosen) >= delete_quota:
                            break
                        if record.key in chosen_keys:
                            continue
                        chosen.append(record)
                        chosen_keys.add(record.key)

                if len(chosen) != delete_quota:
                    raise RuntimeError(
                        f"failed to fill delete quota for {split}/{trajectory_type}/{scene_id}: "
                        f"expected {delete_quota}, got {len(chosen)}"
                    )

                for record in chosen:
                    score = score_map[record.key]
                    deleted_by_split[split].append(record)
                    deleted_rows.append(
                        {
                            "split": split,
                            "scene_id": record.scene_id,
                            "episode_id": record.episode_id,
                            "scene_episode_key": record.key,
                            "trajectory_type": record.trajectory_type,
                            "instruction_type": record.instruction_type,
                            "index_in_split": record.index_in_split,
                            "delete_reason": make_delete_reason(
                                record,
                                dirty_reason_map.get(record.key, ()),
                                score,
                            ),
                            "dirty_reasons": "|".join(dirty_reason_map.get(record.key, ())),
                            "weak_suspect_reasons": "|".join(weak_suspect_reason_map.get(record.key, ())),
                            "anchor_success": f"{score.anchor_success:.4f}",
                            "compare_mean_success": f"{score.compare_mean_success:.4f}",
                            "compare_separation": f"{score.compare_separation:.4f}",
                            "overlap_mean_success": f"{score.overlap_mean_success:.4f}",
                            "all_family_mean_success": f"{score.all_family_mean_success:.4f}",
                            "consensus_easy": str(score.consensus_easy).lower(),
                            "no_model_success_all_families": str(score.no_model_success_all_families).lower(),
                            "delete_score": f"{score.delete_score:.4f}",
                        }
                    )

    return deleted_rows, deleted_by_split


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_keep_list(path: Path, records: Sequence[EpisodeRecord]) -> None:
    rows = [
        {
            "scene_id": record.scene_id,
            "episode_id": record.episode_id,
            "scene_episode_key": record.key,
            "trajectory_type": record.trajectory_type,
            "instruction_type": record.instruction_type,
            "index_in_split": record.index_in_split,
        }
        for record in sorted(
            records,
            key=lambda item: (item.scene_id, item.trajectory_type, item.index_in_split, item.episode_id),
        )
    ]
    write_csv(
        path,
        ["scene_id", "episode_id", "scene_episode_key", "trajectory_type", "instruction_type", "index_in_split"],
        rows,
    )


def build_suspect_rows(
    records: Sequence[EpisodeRecord],
    dirty_reason_map: Mapping[str, Sequence[str]],
    weak_suspect_reason_map: Mapping[str, Sequence[str]],
    score_map: Mapping[str, EpisodeScore],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        dirty_reasons = dirty_reason_map.get(record.key, ())
        weak_reasons = weak_suspect_reason_map.get(record.key, ())
        if not dirty_reasons and not weak_reasons:
            continue
        score = score_map[record.key]
        rows.append(
            {
                "split": record.split,
                "scene_id": record.scene_id,
                "episode_id": record.episode_id,
                "scene_episode_key": record.key,
                "trajectory_type": record.trajectory_type,
                "instruction_type": record.instruction_type,
                "dirty_reasons": "|".join(dirty_reasons),
                "weak_suspect_reasons": "|".join(weak_reasons),
                "anchor_success": f"{score.anchor_success:.4f}",
                "compare_mean_success": f"{score.compare_mean_success:.4f}",
                "compare_separation": f"{score.compare_separation:.4f}",
                "all_family_mean_success": f"{score.all_family_mean_success:.4f}",
            }
        )
    rows.sort(key=lambda row: (row["split"], row["scene_id"], row["trajectory_type"], int(row["episode_id"])))
    return rows


def validate_keep_counts(
    keep_records: Mapping[str, Sequence[EpisodeRecord]],
    target_keep_counts: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, int]]:
    actual: Dict[str, Counter[str]] = defaultdict(Counter)
    for split, records in keep_records.items():
        for record in records:
            actual[split][record.trajectory_type] += 1

    for split, type_counts in target_keep_counts.items():
        for trajectory_type, expected_count in type_counts.items():
            actual_count = actual[split][trajectory_type]
            if actual_count != expected_count:
                raise RuntimeError(
                    f"keep count mismatch for {split}/{trajectory_type}: "
                    f"expected {expected_count}, got {actual_count}"
                )
    return {split: dict(counter) for split, counter in actual.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default="data260404_road345",
        choices=sorted(PRESETS),
        help="Keep-list preset to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/analysis/satnav_keep_lists",
        help="Base output directory. A preset-named subdir will be created inside it.",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    version = preset["version"]
    keep_counts = preset["keep_counts"]
    out_dir = Path(args.output_dir) / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[EpisodeRecord] = []
    split_records: Dict[str, List[EpisodeRecord]] = {}
    for split in ("val_seen", "val_unseen"):
        records = load_dataset_split(version, split)
        split_records[split] = records
        all_records.extend(records)

    overlap_runs_by_split = {
        split: find_latest_overlap_runs(version, split)
        for split in ("val_seen", "val_unseen")
    }
    baseline_runs_by_split = {
        split: find_baseline_runs(version, split)
        for split in ("val_seen", "val_unseen")
    }

    dirty_reason_map = {record.key: dirty_reasons_for_episode(record) for record in all_records}
    weak_suspect_reason_map = {record.key: weak_suspect_reasons_for_episode(record) for record in all_records}

    scene_keep_quotas = build_scene_keep_quotas(all_records, keep_counts)

    split_deleted_rows: List[Dict[str, Any]] = []
    keep_records: Dict[str, List[EpisodeRecord]] = {}
    summary: Dict[str, Any] = {
        "preset": args.preset,
        "version": version,
        "keep_counts": keep_counts,
        "scene_episode_key_format": f"scene_id{SCENE_EPISODE_KEY_SEP}episode_id",
        "splits": {},
    }

    for split in ("val_seen", "val_unseen"):
        overlap_runs = overlap_runs_by_split[split]
        anchor_runs, compare_runs, excluded_for_compare = find_anchor_and_compare_runs(
            overlap_runs=overlap_runs,
            version=version,
            min_compare_all_sr=float(preset["min_compare_all_sr"]),
        )

        valid_baseline_runs = [
            run
            for run in baseline_runs_by_split[split]
            if run.all_sr >= float(preset["min_compare_all_sr"])
        ]
        auxiliary_runs = list(anchor_runs) + list(compare_runs) + valid_baseline_runs
        score_map = build_episode_scores(
            dataset_records=split_records[split],
            anchor_runs=anchor_runs,
            compare_runs=compare_runs,
            auxiliary_runs=auxiliary_runs,
        )

        deleted_rows, _ = select_deletions(
            records=split_records[split],
            keep_counts={split: keep_counts[split]},
            scene_keep_quotas={split: scene_keep_quotas[split]},
            dirty_reason_map=dirty_reason_map,
            weak_suspect_reason_map=weak_suspect_reason_map,
            score_map=score_map,
            never_success_delete_cap_ratio=float(preset["never_success_delete_cap_ratio"]),
            never_success_delete_cap_min=int(preset["never_success_delete_cap_min"]),
        )
        split_deleted_rows.extend(deleted_rows)
        deleted_keys = {row["scene_episode_key"] for row in deleted_rows}
        keep_records[split] = [record for record in split_records[split] if record.key not in deleted_keys]

        suspect_rows = build_suspect_rows(
            records=split_records[split],
            dirty_reason_map=dirty_reason_map,
            weak_suspect_reason_map=weak_suspect_reason_map,
            score_map=score_map,
        )
        summary["splits"][split] = {
            "anchor_models": [run.model_name for run in anchor_runs],
            "compare_models": [run.model_name for run in compare_runs],
            "excluded_compare_models": excluded_for_compare,
            "baseline_family_models": [run.model_name for run in baseline_runs_by_split[split]],
            "never_success_delete_cap": max(
                int(preset["never_success_delete_cap_min"]),
                int(
                    (len(split_records[split]) - sum(keep_counts[split].values()))
                    * float(preset["never_success_delete_cap_ratio"])
                ),
            ),
            "scene_keep_quotas": scene_keep_quotas[split],
            "deleted_reason_counts": dict(Counter(row["delete_reason"] for row in deleted_rows)),
            "dirty_reason_counts": dict(
                Counter(
                    reason
                    for row in deleted_rows
                    for reason in str(row["dirty_reasons"]).split("|")
                    if reason
                )
            ),
            "suspect_rows": len(suspect_rows),
        }
        write_keep_list(out_dir / f"{split}_keep.csv", keep_records[split])
        write_csv(
            out_dir / f"{split}_suspect_episode_report.csv",
            [
                "split",
                "scene_id",
                "episode_id",
                "scene_episode_key",
                "trajectory_type",
                "instruction_type",
                "dirty_reasons",
                "weak_suspect_reasons",
                "anchor_success",
                "compare_mean_success",
                "compare_separation",
                "all_family_mean_success",
            ],
            suspect_rows,
        )

    actual_keep_counts = validate_keep_counts(keep_records, keep_counts)

    deleted_rows_sorted = sorted(
        split_deleted_rows,
        key=lambda row: (row["split"], row["scene_id"], row["trajectory_type"], int(row["episode_id"])),
    )
    write_csv(
        out_dir / "deletion_report.csv",
        [
            "split",
            "scene_id",
            "episode_id",
            "scene_episode_key",
            "trajectory_type",
            "instruction_type",
            "index_in_split",
            "delete_reason",
            "dirty_reasons",
            "weak_suspect_reasons",
            "anchor_success",
            "compare_mean_success",
            "compare_separation",
            "overlap_mean_success",
            "all_family_mean_success",
            "consensus_easy",
            "no_model_success_all_families",
            "delete_score",
        ],
        deleted_rows_sorted,
    )

    summary["actual_keep_counts"] = actual_keep_counts
    summary["deleted_counts"] = {
        split: {
            trajectory_type: len(
                [
                    row
                    for row in deleted_rows_sorted
                    if row["split"] == split and row["trajectory_type"] == trajectory_type
                ]
            )
            for trajectory_type in TRAJECTORY_TYPES
        }
        for split in ("val_seen", "val_unseen")
    }
    summary["original_counts"] = summarize_counts(all_records)
    summary["keep_list_files"] = {
        split: str(out_dir / f"{split}_keep.csv")
        for split in ("val_seen", "val_unseen")
    }

    with (out_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[INFO] preset: {args.preset}")
    print(f"[INFO] output_dir: {out_dir}")
    print(f"[INFO] keep counts: {json.dumps(actual_keep_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
