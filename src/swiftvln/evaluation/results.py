# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable evaluation results, resume state, and summary reporting."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .reporting import (
    clean_results_for_output,
    compute_trajectory_type_stats,
    get_swanlab_url,
    get_swanlab_url_from_train_metadata,
    save_timing_stats,
)
from swiftvln.backends.specs import get_environment_spec


class ResultRecorder:
    """Persist episode results and coordinate distributed completion.

    ``result.jsonl`` is the append-only recovery log. The final
    ``all_results.jsonl`` and ``evaluation_summary.json`` files are derived from
    that log after every rank has finished.
    """

    def __init__(self, output_dir: str, *, rank: int = 0, world_size: int = 1):
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size

    @property
    def result_file(self) -> str:
        return os.path.join(self.output_dir, "result.jsonl")

    @property
    def rank_sync_dir(self) -> str:
        return os.path.join(self.output_dir, ".dist_sync")

    @staticmethod
    def episode_key(episode_id: Any, scene_id: Any) -> str:
        """Build the stable resume/dedup key for one episode."""
        ep_id = str(episode_id) if episode_id is not None else ""
        scene = str(scene_id) if scene_id is not None else ""
        if not ep_id:
            return ""
        return f"{scene}::{ep_id}" if scene else ep_id

    def load_results(self) -> list[dict[str, Any]]:
        """Load the recovery log, keeping the latest valid row per episode."""
        if not os.path.exists(self.result_file):
            return []

        results_by_episode: dict[str, dict[str, Any]] = {}
        with open(self.result_file, encoding="utf-8") as handle:
            for line in handle:
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_key = self.episode_key(
                    result.get("episode_id", ""),
                    result.get("scene_id", ""),
                )
                if episode_key:
                    results_by_episode[episode_key] = result
        return list(results_by_episode.values())

    def append(self, result: dict[str, Any]) -> None:
        """Append one episode result as a durable JSONL row."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.result_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    def clear_rank_markers(self) -> None:
        """Remove stale distributed completion markers before a new run."""
        os.makedirs(self.rank_sync_dir, exist_ok=True)
        for name in os.listdir(self.rank_sync_dir):
            if name.startswith("rank_") and name.endswith(".done.json"):
                try:
                    os.remove(os.path.join(self.rank_sync_dir, name))
                except FileNotFoundError:
                    pass

    def mark_rank_complete(
        self,
        *,
        processed_count: int,
        resumed_count: int,
        local_total: int,
    ) -> None:
        """Atomically publish this rank's completion state."""
        os.makedirs(self.rank_sync_dir, exist_ok=True)
        final_path = os.path.join(self.rank_sync_dir, f"rank_{self.rank}.done.json")
        temporary_path = f"{final_path}.tmp"
        payload = {
            "rank": self.rank,
            "processed_count": processed_count,
            "resumed_count": resumed_count,
            "local_total": local_total,
            "completed_at": time.time(),
        }
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, final_path)

    def wait_for_ranks(
        self,
        expected_ranks: list[int],
        *,
        timeout_seconds: int = 1800,
    ) -> None:
        """Wait until all expected ranks have published completion markers."""
        os.makedirs(self.rank_sync_dir, exist_ok=True)
        deadline = time.time() + timeout_seconds
        last_missing: list[int] | None = None
        expected_ranks = sorted(set(int(rank) for rank in expected_ranks))

        while True:
            missing = [
                rank
                for rank in expected_ranks
                if not os.path.exists(
                    os.path.join(self.rank_sync_dir, f"rank_{rank}.done.json")
                )
            ]
            if not missing:
                return

            if time.time() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for rank completion markers. "
                    f"Missing ranks: {missing}, dir={self.rank_sync_dir}"
                )

            if missing != last_missing:
                print(
                    f"[Sync] rank markers: {len(expected_ranks) - len(missing)}/"
                    f"{len(expected_ranks)} ready, missing={missing}"
                )
                last_missing = missing
            time.sleep(5)

    def save_summary(
        self,
        results: list[dict[str, Any]],
        timing_stats: list[dict[str, Any]],
        *,
        args: Any,
        model_description: str,
        summary_extras: dict[str, Any],
        uses_compression: bool,
    ) -> None:
        """Write final metrics and compatibility result files."""
        total_episodes = len(results)
        successes = [float(result.get("success", 0.0)) for result in results]
        spls = [float(result.get("spl", 0.0)) for result in results]
        oracle_successes = [
            float(result.get("oracle_success", 0.0)) for result in results
        ]
        navigation_errors = [
            float(result.get("distance_to_goal", 0.0)) for result in results
        ]
        valid_navigation_errors = [error for error in navigation_errors if error < 1000]

        success_rate = sum(successes) / total_episodes if total_episodes else 0
        mean_spl = sum(spls) / total_episodes if total_episodes else 0
        mean_oracle_success = (
            sum(oracle_successes) / total_episodes if total_episodes else 0
        )
        mean_navigation_error = (
            sum(valid_navigation_errors) / len(valid_navigation_errors)
            if valid_navigation_errors
            else 0
        )
        steps = [result.get("steps", 0) for result in results]
        average_steps = sum(steps) / len(steps) if steps else 0

        summary = {
            "eval_split": args.eval_split,
            "success_rate": success_rate,
            "mean_spl": mean_spl,
            "oracle_success": mean_oracle_success,
            "navigation_error": mean_navigation_error,
            "avg_steps": round(average_steps, 2),
            "total_episodes": total_episodes,
            "world_size": self.world_size,
            "model_path": args.model_path,
            "num_history": args.num_history,
        }
        summary.update(summary_extras)

        swanlab_url = get_swanlab_url()
        if not swanlab_url:
            swanlab_url = get_swanlab_url_from_train_metadata(args.model_path)
        if swanlab_url:
            summary["swanlab_url"] = swanlab_url

        environment_spec = get_environment_spec(args.env_type)
        if environment_spec.reports_trajectory_types:
            trajectory_type_stats = compute_trajectory_type_stats(results)
            if trajectory_type_stats:
                summary["by_trajectory_type"] = trajectory_type_stats

        print("\n" + "=" * 60)
        print(f"{model_description} Evaluation Summary ({args.eval_split})")
        print("=" * 60)
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Mean SPL: {summary['mean_spl']:.4f}")
        print(f"Oracle Success: {summary['oracle_success']:.2%}")
        print(f"Navigation Error: {summary['navigation_error']:.2f}m")
        print(f"Average Steps: {summary['avg_steps']:.2f}")

        if environment_spec.reports_trajectory_types and "by_trajectory_type" in summary:
            print("\n--- By Trajectory Type ---")
            for trajectory_type, type_stats in summary["by_trajectory_type"].items():
                print(
                    f"  [{trajectory_type}] SR: {type_stats['success_rate']:.2%}, "
                    f"SPL: {type_stats['mean_spl']:.4f}, "
                    f"OS: {type_stats['oracle_success']:.2%}, "
                    f"NE: {type_stats['navigation_error']:.2f}m, "
                    f"Steps: {type_stats['avg_steps']:.2f}, "
                    f"N: {type_stats['total_episodes']}"
                )

        if uses_compression and hasattr(args, "compress_stride"):
            print(
                f"\nCompression: stride={args.compress_stride} "
                f"({args.compress_stride**2}x)"
            )

        print(f"Total Episodes: {total_episodes}")
        if self.world_size > 1:
            print(f"Distributed: {self.world_size} GPUs")
        if swanlab_url:
            print(f"SwanLab URL: {swanlab_url}")
        print("=" * 60)

        with open(
            os.path.join(self.output_dir, "evaluation_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2)

        cleaned_results = clean_results_for_output(results)

        def result_sort_key(result: dict[str, Any]):
            episode_id = result.get("episode_id", 0)
            try:
                episode_sort_id = (0, int(episode_id))
            except (TypeError, ValueError):
                episode_sort_id = (1, str(episode_id))
            return str(result.get("scene_id", "")), episode_sort_id

        cleaned_results = sorted(cleaned_results, key=result_sort_key, reverse=True)
        with open(
            os.path.join(self.output_dir, "all_results.jsonl"),
            "w",
            encoding="utf-8",
        ) as handle:
            for result in cleaned_results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

        if timing_stats:
            timing_summary_path = save_timing_stats(self.output_dir, timing_stats)
            print(f"Timing statistics summary saved to {timing_summary_path}")

        print(f"Results saved to {self.output_dir}")

        if args.video_compression and args.save_video:
            try:
                from swiftvln.utils.video import compress_videos

                compress_videos(self.output_dir, cleaned_results, chunk_size=400)
            except ImportError:
                pass
