#!/usr/bin/env python3
"""Apply a SatNav keep subset to baseline eval results.

This script updates the baseline result artifacts referenced by
``reports/baseline_reports/baseline_0404.md``:

- Seq2Seq scratch
- CMA scratch
- StreamVLN scratch / continue
- NaVILA scratch / continue
- UniNaVid scratch / continue

For SwiftVLN baselines, the active ``result.jsonl`` and ``evaluation_summary.json``
are rewritten from a keep-list-filtered backup.

For traditional SatNav baselines, the active
``eval_ckpt_0_<split>.json`` and ``eval_ckpt_0_<split>_diagnostics.json``
are rewritten from a keep-list-filtered backup.

Top-level ``ALL`` metrics are direct averages over the matched keep episodes
for each model. For Seq2Seq / CMA, some keep episodes are unavailable because
they were skipped during the original eval; this script reports the matched
count explicitly and averages over the available keep-episode intersection.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


KEEP_KEY_SEP = "::"


@dataclass(frozen=True)
class BaselineTarget:
    name: str
    family: str
    root: Path
    kind: str  # swift_jsonl | satsim_diagnostics


TARGETS: Sequence[BaselineTarget] = (
    BaselineTarget(
        name="seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab",
        family="seq2seq",
        root=Path("/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab"),
        kind="satsim_diagnostics",
    ),
    BaselineTarget(
        name="cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab",
        family="cma",
        root=Path("/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/results/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab"),
        kind="satsim_diagnostics",
    ),
    BaselineTarget(
        name="streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630",
        family="streamvln",
        root=Path("results/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630"),
        kind="swift_jsonl",
    ),
    BaselineTarget(
        name="streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737",
        family="streamvln",
        root=Path("results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737"),
        kind="swift_jsonl",
    ),
    BaselineTarget(
        name="navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4",
        family="navila",
        root=Path("results/navila-baseline/navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4"),
        kind="swift_jsonl",
    ),
    BaselineTarget(
        name="navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4",
        family="navila",
        root=Path("results/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4"),
        kind="swift_jsonl",
    ),
    BaselineTarget(
        name="uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606",
        family="uninavid",
        root=Path("results/uninavid-baseline/uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606"),
        kind="swift_jsonl",
    ),
    BaselineTarget(
        name="uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606",
        family="uninavid",
        root=Path("results/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606"),
        kind="swift_jsonl",
    ),
)


def keep_key(scene_id: str, episode_id: str) -> str:
    return f"{Path(scene_id).name}{KEEP_KEY_SEP}{episode_id}"


def read_keep_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["scene_episode_key"] for row in csv.DictReader(f)}


def write_backup_if_missing(active_path: Path, backup_path: Path) -> Path:
    if backup_path.exists():
        return backup_path
    backup_path.write_bytes(active_path.read_bytes())
    return backup_path


def mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def build_swift_summary_from_rows(
    source_summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    split: str,
    keep_keys: set[str],
    keep_list_path: Path,
) -> Dict[str, Any]:
    by_type_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type_rows[str(row.get("trajectory_type", "unknown"))].append(dict(row))

    by_trajectory_type: Dict[str, Dict[str, Any]] = {}
    for trajectory_type, type_rows in sorted(by_type_rows.items()):
        n = len(type_rows)
        by_trajectory_type[trajectory_type] = {
            "SR": mean([float(r.get("success", 0.0)) for r in type_rows]),
            "SPL": mean([float(r.get("spl", 0.0)) for r in type_rows]),
            "OS": mean([float(r.get("oracle_success", 0.0)) for r in type_rows]),
            "NE": mean([float(r.get("distance_to_goal", 0.0)) for r in type_rows]),
            "avg_steps": mean([float(r.get("steps", 0.0)) for r in type_rows]),
            "count": n,
        }

    summary: Dict[str, Any] = {
        "eval_split": split,
        "SR": mean([float(r.get("success", 0.0)) for r in rows]),
        "SPL": mean([float(r.get("spl", 0.0)) for r in rows]),
        "OS": mean([float(r.get("oracle_success", 0.0)) for r in rows]),
        "NE": mean([float(r.get("distance_to_goal", 0.0)) for r in rows]),
        "avg_steps": mean([float(r.get("steps", 0.0)) for r in rows]),
        "total_episodes": len(rows),
        "model_path": source_summary.get("model_path", ""),
        "by_trajectory_type": by_trajectory_type,
        "subset_mode": "keep_list_direct_average",
        "subset_keep_list_path": str(keep_list_path.resolve()),
        "subset_key_format": f"scene_id{KEEP_KEY_SEP}episode_id",
        "subset_target_keep_episodes": len(keep_keys),
        "subset_available_keep_episodes": len(rows),
        "subset_missing_keep_episodes": len(keep_keys) - len(rows),
        "subset_total_episodes_before_filter": int(source_summary.get("total_episodes", 0)),
    }
    for passthrough_key in ("action_format", "num_frames", "num_history", "num_future_steps", "swanlab_url"):
        if passthrough_key in source_summary:
            summary[passthrough_key] = source_summary[passthrough_key]
    return summary


def compute_swift_overall_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "SR": mean([float(r.get("success", 0.0)) for r in rows]),
        "SPL": mean([float(r.get("spl", 0.0)) for r in rows]),
        "OS": mean([float(r.get("oracle_success", 0.0)) for r in rows]),
        "NE": mean([float(r.get("distance_to_goal", 0.0)) for r in rows]),
        "avg_steps": mean([float(r.get("steps", 0.0)) for r in rows]),
        "total_episodes": len(rows),
    }


def normalize_swift_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "trajectory_type": row.get("trajectory_type"),
        "success": float(row.get("success", 0.0)),
        "spl": float(row.get("spl", 0.0)),
        "oracle_success": float(row.get("oracle_success", 0.0)),
        "distance_to_goal": float(row.get("distance_to_goal", 0.0)),
        "steps": float(row.get("steps", 0.0)),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_swift_target(
    target: BaselineTarget,
    keep_keys_by_split: Mapping[str, set[str]],
    backup_suffix: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "family": target.family,
        "name": target.name,
        "root": str(target.root.resolve()),
        "splits": {},
    }
    overall_rows: List[Dict[str, Any]] = []

    for split in ("val_seen", "val_unseen"):
        split_dir = target.root / split
        active_result = split_dir / "result.jsonl"
        active_summary = split_dir / "evaluation_summary.json"
        backup_result = split_dir / f"result_full_before_{backup_suffix}.jsonl"
        backup_summary = split_dir / f"evaluation_summary_full_before_{backup_suffix}.json"
        keep_list_path = Path(f"runtime/analysis/satnav_keep_lists/{backup_suffix}/{split}_keep.csv")

        source_result = write_backup_if_missing(active_result, backup_result)
        source_summary = write_backup_if_missing(active_summary, backup_summary)

        with source_result.open("r", encoding="utf-8") as f:
            source_rows = [json.loads(line) for line in f if line.strip()]
        with source_summary.open("r", encoding="utf-8") as f:
            source_summary_data = json.load(f)

        keep_keys = keep_keys_by_split[split]
        filtered_rows = [
            row
            for row in source_rows
            if keep_key(str(row.get("scene_id", "")), str(row.get("episode_id", ""))) in keep_keys
        ]

        write_jsonl(active_result, filtered_rows)
        summary = build_swift_summary_from_rows(
            source_summary=source_summary_data,
            rows=filtered_rows,
            split=split,
            keep_keys=keep_keys,
            keep_list_path=keep_list_path,
        )
        write_json(active_summary, summary)

        normalized_rows = [normalize_swift_row(row) for row in filtered_rows]
        overall_rows.extend(normalized_rows)
        result["splits"][split] = {
            "result_path": str(active_result.resolve()),
            "summary_path": str(active_summary.resolve()),
            "source_result_path": str(source_result.resolve()),
            "source_summary_path": str(source_summary.resolve()),
            "target_keep_episodes": len(keep_keys),
            "matched_keep_episodes": len(filtered_rows),
            "missing_keep_episodes": len(keep_keys) - len(filtered_rows),
            "metrics": {
                "SR": summary["SR"],
                "SPL": summary["SPL"],
                "OS": summary["OS"],
                "NE": summary["NE"],
                "avg_steps": summary["avg_steps"],
                "total_episodes": summary["total_episodes"],
            },
        }

    result["overall"] = compute_swift_overall_metrics(overall_rows)
    return result


def recompute_action_counts(episodes: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        action_counts = episode.get("action_counts", {}) or {}
        for action, count in action_counts.items():
            counts[str(action)] += int(count)
    total = sum(counts.values())
    if total == 0:
        return dict(counts), {}
    distribution = {action: counts[action] / total for action in sorted(counts)}
    return dict(counts), distribution


def recompute_failure_mode_counts(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        counts[str(episode.get("failure_mode", "unknown"))] += 1
    return dict(counts)


def build_satsim_summary_from_episodes(
    source_summary: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    split: str,
    keep_keys: set[str],
    keep_list_path: Path,
) -> Dict[str, Any]:
    summary = {
        "spl": mean([float(ep.get("spl", 0.0)) for ep in episodes]),
        "success": mean([float(ep.get("success", 0.0)) for ep in episodes]),
        "distance_to_goal": mean([float(ep.get("final_distance_to_goal", ep.get("distance_to_goal", 0.0))) for ep in episodes]),
        "path_length": mean([float(ep.get("path_length", 0.0)) for ep in episodes]),
        "steps_taken": mean([float(ep.get("steps_taken", 0.0)) for ep in episodes]),
        "num_episodes": len(episodes),
        "split": split,
        "checkpoint_index": int(source_summary.get("checkpoint_index", 0)),
        "subset_mode": "keep_list_direct_average",
        "subset_keep_list_path": str(keep_list_path.resolve()),
        "subset_key_format": f"scene_id{KEEP_KEY_SEP}episode_id",
        "subset_target_keep_episodes": len(keep_keys),
        "subset_available_keep_episodes": len(episodes),
        "subset_missing_keep_episodes": len(keep_keys) - len(episodes),
        "subset_total_episodes_before_filter": int(source_summary.get("num_episodes", 0)),
    }
    return summary


def process_satsim_target(
    target: BaselineTarget,
    keep_keys_by_split: Mapping[str, set[str]],
    backup_suffix: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "family": target.family,
        "name": target.name,
        "root": str(target.root.resolve()),
        "splits": {},
    }
    overall_episodes: List[Dict[str, Any]] = []

    for split in ("val_seen", "val_unseen"):
        split_dir = target.root / split
        active_summary = split_dir / f"eval_ckpt_0_{split}.json"
        active_diag = split_dir / f"eval_ckpt_0_{split}_diagnostics.json"
        backup_summary = split_dir / f"eval_ckpt_0_{split}_full_before_{backup_suffix}.json"
        backup_diag = split_dir / f"eval_ckpt_0_{split}_diagnostics_full_before_{backup_suffix}.json"
        keep_list_path = Path(f"runtime/analysis/satnav_keep_lists/{backup_suffix}/{split}_keep.csv")

        source_summary_path = write_backup_if_missing(active_summary, backup_summary)
        source_diag_path = write_backup_if_missing(active_diag, backup_diag)

        with source_summary_path.open("r", encoding="utf-8") as f:
            source_summary = json.load(f)
        with source_diag_path.open("r", encoding="utf-8") as f:
            source_diag = json.load(f)

        keep_keys = keep_keys_by_split[split]
        source_episodes = list(source_diag.get("episodes", []))
        filtered_episodes = [
            episode
            for episode in source_episodes
            if keep_key(str(episode.get("scene_id", "")), str(episode.get("episode_id", ""))) in keep_keys
        ]

        summary = build_satsim_summary_from_episodes(
            source_summary=source_summary,
            episodes=filtered_episodes,
            split=split,
            keep_keys=keep_keys,
            keep_list_path=keep_list_path,
        )
        action_counts, action_distribution = recompute_action_counts(filtered_episodes)
        diagnostics = {
            "checkpoint_index": int(source_diag.get("checkpoint_index", source_summary.get("checkpoint_index", 0))),
            "split": split,
            "num_episodes": len(filtered_episodes),
            "global_action_counts": action_counts,
            "global_action_distribution": action_distribution,
            "failure_mode_counts": recompute_failure_mode_counts(filtered_episodes),
            "episodes": filtered_episodes,
            "subset_mode": "keep_list_direct_average",
            "subset_keep_list_path": str(keep_list_path.resolve()),
            "subset_key_format": f"scene_id{KEEP_KEY_SEP}episode_id",
            "subset_target_keep_episodes": len(keep_keys),
            "subset_available_keep_episodes": len(filtered_episodes),
            "subset_missing_keep_episodes": len(keep_keys) - len(filtered_episodes),
            "subset_total_episodes_before_filter": int(source_diag.get("num_episodes", 0)),
        }

        write_json(active_summary, summary)
        write_json(active_diag, diagnostics)

        overall_episodes.extend(filtered_episodes)
        result["splits"][split] = {
            "summary_path": str(active_summary.resolve()),
            "diagnostics_path": str(active_diag.resolve()),
            "source_summary_path": str(source_summary_path.resolve()),
            "source_diagnostics_path": str(source_diag_path.resolve()),
            "target_keep_episodes": len(keep_keys),
            "matched_keep_episodes": len(filtered_episodes),
            "missing_keep_episodes": len(keep_keys) - len(filtered_episodes),
            "metrics": {
                "SR": summary["success"],
                "SPL": summary["spl"],
                "NE": summary["distance_to_goal"],
                "path_length": summary["path_length"],
                "avg_steps": summary["steps_taken"],
                "total_episodes": summary["num_episodes"],
            },
        }

    result["overall"] = {
        "SR": mean([float(ep.get("success", 0.0)) for ep in overall_episodes]),
        "SPL": mean([float(ep.get("spl", 0.0)) for ep in overall_episodes]),
        "NE": mean([float(ep.get("final_distance_to_goal", ep.get("distance_to_goal", 0.0))) for ep in overall_episodes]),
        "path_length": mean([float(ep.get("path_length", 0.0)) for ep in overall_episodes]),
        "avg_steps": mean([float(ep.get("steps_taken", 0.0)) for ep in overall_episodes]),
        "total_episodes": len(overall_episodes),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-dir",
        default="runtime/analysis/satnav_keep_lists/data260404_road345",
        help="Directory containing val_seen_keep.csv and val_unseen_keep.csv.",
    )
    parser.add_argument(
        "--report-path",
        default="runtime/analysis/baseline_keep_subset_data260404_road345.json",
        help="Output verification report path.",
    )
    args = parser.parse_args()

    keep_dir = Path(args.keep_dir)
    keep_keys_by_split = {
        split: read_keep_keys(keep_dir / f"{split}_keep.csv")
        for split in ("val_seen", "val_unseen")
    }
    backup_suffix = keep_dir.name

    report: Dict[str, Any] = {
        "keep_dir": str(keep_dir.resolve()),
        "keep_key_format": f"scene_id{KEEP_KEY_SEP}episode_id",
        "targets": [],
    }

    for target in TARGETS:
        if target.kind == "swift_jsonl":
            result = process_swift_target(target, keep_keys_by_split, backup_suffix)
        elif target.kind == "satsim_diagnostics":
            result = process_satsim_target(target, keep_keys_by_split, backup_suffix)
        else:
            raise SystemExit(f"unknown target kind: {target.kind}")
        report["targets"].append(result)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"[INFO] updated targets: {len(report['targets'])}")
    print(f"[INFO] report: {report_path}")


if __name__ == "__main__":
    main()
