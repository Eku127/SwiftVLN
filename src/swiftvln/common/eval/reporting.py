# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Reusable result/timing reporting helpers for evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def compute_trajectory_type_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute statistics grouped by trajectory_type for SatNav results."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        trajectory_type = result.get("trajectory_type", "unknown")
        grouped.setdefault(trajectory_type, []).append(result)

    stats: Dict[str, Dict[str, Any]] = {}
    for trajectory_type, type_results in grouped.items():
        n = len(type_results)
        if n == 0:
            continue

        successes = sum(r.get("success", 0.0) for r in type_results)
        spls = sum(r.get("spl", 0.0) for r in type_results)
        oracle_successes = sum(r.get("oracle_success", 0.0) for r in type_results)

        navigation_errors = [r.get("distance_to_goal", float("inf")) for r in type_results]
        valid_errors = [err for err in navigation_errors if err < 1000]
        mean_navigation_error = sum(valid_errors) / len(valid_errors) if valid_errors else 0.0

        steps = [r.get("steps", 0) for r in type_results]
        avg_steps = sum(steps) / len(steps) if steps else 0.0

        stats[trajectory_type] = {
            "success_rate": successes / n,
            "mean_spl": spls / n,
            "oracle_success": oracle_successes / n,
            "navigation_error": mean_navigation_error,
            "avg_steps": round(avg_steps, 2),
            "total_episodes": n,
        }

    return stats


def clean_results_for_output(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove internal timing fields before persisting user-facing outputs."""
    timing_keys = {"_timing_stats", "_total_time", "_step_count"}
    return [{k: v for k, v in result.items() if k not in timing_keys} for result in results]


def get_swanlab_url() -> Optional[str]:
    """Try to retrieve SwanLab run URL if SwanLab is active."""
    try:
        import swanlab

        run = swanlab.get_run()
        if run is None:
            return None

        if hasattr(run, "url"):
            return run.url
        if hasattr(run, "get_url"):
            return run.get_url()
        if hasattr(run, "public_url"):
            return run.public_url

        if hasattr(run, "project") and hasattr(run, "name"):
            username = getattr(run, "username", None) or getattr(run, "user", None)
            if username:
                return f"https://swanlab.cn/@{username}/{run.project}/runs/{run.name}"
    except Exception:
        return None

    return None


def get_swanlab_url_from_train_metadata(model_path: str) -> Optional[str]:
    """Read SwanLab URL from train_metadata.json located in the training output dir.

    The model_path typically points to a checkpoint like:
      output/<arch>/<exp_name>/v0-<ts>/checkpoint-N
    We walk up to find train_metadata.json at the <exp_name> level.
    """
    from pathlib import Path

    p = Path(model_path)
    for ancestor in [p.parent.parent, p.parent, p]:
        candidate = ancestor / "train_metadata.json"
        if candidate.is_file():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
                url = meta.get("swanlab_url", "")
                if url:
                    return url
            except Exception:
                pass
    return None


def save_timing_stats(output_dir: str, timing_stats_list: List[Dict[str, Any]]) -> str:
    """Save timing statistics summary and return output file path."""
    num_episodes = len(timing_stats_list)
    avg_total_time = sum(item.get("_total_time", 0.0) for item in timing_stats_list) / num_episodes
    avg_step_count = sum(item.get("_step_count", 0) for item in timing_stats_list) / num_episodes

    component_totals: Dict[str, List[float]] = {}
    for episode_stats in timing_stats_list:
        for component, elapsed in episode_stats.get("_timing_stats", {}).items():
            component_totals.setdefault(component, []).append(elapsed)

    avg_components = {
        component: (sum(times) / len(times))
        for component, times in component_totals.items()
    }

    timing_summary: Dict[str, Any] = {
        "num_episodes": num_episodes,
        "avg_episode_time": round(avg_total_time, 3),
        "avg_steps": round(avg_step_count, 1),
        "breakdown_by_component": {},
    }

    for component, avg_time in sorted(avg_components.items(), key=lambda x: x[1], reverse=True):
        percentage = (avg_time / avg_total_time * 100) if avg_total_time > 0 else 0.0
        timing_summary["breakdown_by_component"][component] = {
            "avg_time": round(avg_time, 3),
            "percentage": round(percentage, 1),
        }

    output_path = os.path.join(output_dir, "timing_summary.json")
    with open(output_path, "w") as file:
        json.dump(timing_summary, file, indent=2)
    return output_path
