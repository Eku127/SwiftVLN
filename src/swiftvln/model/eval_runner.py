# Copyright (c) Alibaba, Inc. and its affiliates.
"""Concrete SwiftVLN evaluation orchestration."""

from __future__ import annotations

import importlib
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

from swiftvln.common.env.base import EnvWrapper
from swiftvln.common.eval.results import ResultRecorder
from swiftvln.common.utils.distributed_utils import init_distributed

DEFAULT_EVAL_SEED = 42
MODEL_DESCRIPTION = "SwiftVLN"

_DEBUG_RANK = int(os.environ.get("SATNAV_DEBUG_RANK", "-1"))
_DEBUG_LOG_FILE = os.environ.get("SATNAV_DEBUG_LOG")
_debug_file_handle = None


def _debug_log(rank: int, message: str) -> None:
    """Write optional per-rank evaluation diagnostics."""
    global _debug_file_handle
    if rank != _DEBUG_RANK:
        return

    log_message = f"[Rank {rank}][SwiftVLNEval] {message}"
    if _DEBUG_LOG_FILE:
        if _debug_file_handle is None:
            _debug_file_handle = open(_DEBUG_LOG_FILE, "a", encoding="utf-8")
        _debug_file_handle.write(log_message + "\n")
        _debug_file_handle.flush()
    else:
        print(log_message, file=sys.stderr, flush=True)


def scene_name(episode: Any) -> str:
    """Normalize an episode scene path for result keys."""
    raw_scene = str(getattr(episode, "scene_id", "unknown") or "unknown")
    return os.path.basename(raw_scene).replace(".glb", "").replace(".tif", "")


def distribute_episodes(
    episodes: list[Any],
    *,
    rank: int,
    world_size: int,
) -> tuple[list[Any], dict[str, list[Any]]]:
    """Return a deterministic, globally balanced rank shard.

    Episodes remain grouped by sorted scene for stable ordering, then a single
    global round-robin distributes them. This avoids assigning several scenes'
    first episodes to the same low ranks.
    """
    episodes_by_scene: dict[str, list[Any]] = defaultdict(list)
    for episode in episodes:
        episodes_by_scene[str(getattr(episode, "scene_id", "default_scene"))].append(
            episode
        )
    ordered_episodes = [
        episode
        for current_scene in sorted(episodes_by_scene)
        for episode in episodes_by_scene[current_scene]
    ]
    if world_size > 1:
        rank_episodes = ordered_episodes[rank::world_size]
    else:
        rank_episodes = ordered_episodes
    return rank_episodes, dict(episodes_by_scene)


class SwiftVLNEvaluationRunner:
    """Load SwiftVLN and coordinate evaluation across ranks and environments."""

    def __init__(self, args: Any, summary_extras: dict[str, Any]):
        self.args = args
        self.summary_extras = summary_extras
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self.package_root = Path(__file__).resolve().parents[1]
        self.repo_root = self.package_root.parents[1]

    @staticmethod
    def set_seed(seed: int = DEFAULT_EVAL_SEED) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def initialize_distributed(self) -> None:
        if self.args.distributed:
            self.rank, self.world_size, self.local_rank = init_distributed()
        self.is_main = self.rank == 0

    def prepare_output(self, recorder: ResultRecorder) -> None:
        if self.is_main:
            os.makedirs(self.args.output_dir, exist_ok=True)
            mode = (
                f"[Distributed Mode] {self.world_size} processes"
                if self.world_size > 1
                else "[Single Process Mode]"
            )
            print(mode)
            print(f"[{MODEL_DESCRIPTION}] compress_stride={self.args.compress_stride}")
            if self.world_size > 1:
                recorder.clear_rank_markers()
        if self.world_size > 1:
            torch.distributed.barrier()

    def load_model(self):
        """Load the selected Qwen2.5/Qwen3 SwiftVLN wrapper and processor."""
        importlib.import_module("swiftvln.model")
        from swift.model import get_model_processor

        device_map = {"": self.local_rank} if self.world_size > 1 else "auto"
        return get_model_processor(
            model_id_or_path=self.args.model_path,
            model_type=self.args.model_type,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_impl="flash_attn",
            embedding_mode=self.args.embedding_mode,
            uav_adapter_path=self.args.uav_adapter_path,
            uav_adapter_type=self.args.uav_adapter_type,
            uav_adapter_apply_scope=self.args.uav_adapter_apply_scope,
            pose_norm_scale=self.args.pose_norm_scale,
        )

    def resolve_config_path(self) -> str:
        configured_path = (
            self.args.habitat_config_path
            if self.args.env_type == "habitat"
            else self.args.satnav_config
        )
        config_path = Path(configured_path)
        if config_path.is_absolute():
            return str(config_path)

        package_path = self.package_root / config_path
        repo_path = self.repo_root / config_path
        if package_path.exists() or not repo_path.exists():
            return str(package_path)
        return str(repo_path)

    def create_evaluator(self, model: Any, processor: Any):
        from swiftvln.model.evaluator import SwiftVLNEvaluator

        return SwiftVLNEvaluator(
            config_path=self.resolve_config_path(),
            model=model,
            processor=processor,
            args=self.args,
            env_type=self.args.env_type,
        )

    def load_episodes(self, env_wrapper: EnvWrapper) -> list[Any]:
        episodes = list(env_wrapper.episodes)
        if self.args.max_episodes is not None:
            episodes = episodes[: self.args.max_episodes]
        return episodes

    def evaluate_episode(
        self,
        evaluator: Any,
        env_wrapper: EnvWrapper,
        episode: Any,
    ) -> dict[str, Any]:
        """Convert one evaluator metrics dictionary into the durable row schema."""
        current_scene = scene_name(episode)
        instruction = env_wrapper.get_instruction(episode)
        trajectory_type = getattr(episode, "trajectory_type", None)
        _debug_log(self.rank, "=" * 70)
        _debug_log(self.rank, f"evaluate_episode() called for {episode.episode_id}")
        _debug_log(self.rank, f"  trajectory_type: {trajectory_type}")
        _debug_log(self.rank, f"  scene_id: {current_scene}")
        if hasattr(episode, "start_position"):
            _debug_log(self.rank, f"  start_position: {episode.start_position}")
            _debug_log(self.rank, f"  start_rotation: {episode.start_rotation}")

        try:
            metrics = evaluator.eval_episode(env_wrapper, episode, env_idx=0)
            result = {
                "episode_id": episode.episode_id,
                "scene_id": current_scene,
                "success": float(metrics.get("success", 0)),
                "spl": float(metrics.get("spl", 0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0)),
                "oracle_success": float(metrics.get("oracle_success", 0)),
                "steps": int(metrics.get("_step_count", 0)),
                "instruction": instruction,
            }
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type
            if "_timing_stats" in metrics:
                result.update(
                    {
                        "_timing_stats": metrics.get("_timing_stats", {}),
                        "_total_time": metrics.get("_total_time", 0.0),
                        "_step_count": metrics.get("_step_count", 0),
                    }
                )
            if "_error" in metrics:
                result["error"] = metrics["_error"]
            if self.world_size > 1:
                result["rank"] = self.rank
        except Exception as exc:
            error_traceback = traceback.format_exc()
            print(f"\n[Rank {self.rank}] Error on episode {episode.episode_id}: {exc}")
            _debug_log(self.rank, f"  Error: {exc}\n{error_traceback}")
            result = {
                "episode_id": episode.episode_id,
                "scene_id": current_scene,
                "success": 0.0,
                "spl": 0.0,
                "distance_to_goal": float("inf"),
                "oracle_success": 0.0,
                "steps": 0,
                "error": str(exc),
                "instruction": instruction,
            }
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type

        _debug_log(self.rank, f"evaluate_episode() finished for {episode.episode_id}")
        return result

    def resume_state(
        self,
        recorder: ResultRecorder,
        rank_episodes: list[Any],
    ) -> tuple[set[str], int]:
        existing_results = recorder.load_results()
        done_ids = {
            recorder.episode_key(
                result.get("episode_id", ""),
                result.get("scene_id", ""),
            )
            for result in existing_results
        }
        done_ids.discard("")
        if existing_results and self.is_main:
            print(
                f"[Resume] Loaded {len(done_ids)} completed episodes from "
                f"{recorder.result_file}"
            )
        resumed_count = sum(
            recorder.episode_key(
                getattr(episode, "episode_id", ""), scene_name(episode)
            )
            in done_ids
            for episode in rank_episodes
        )
        return done_ids, resumed_count

    def evaluate_rank(
        self,
        evaluator: Any,
        env_wrapper: EnvWrapper,
        rank_episodes: list[Any],
        recorder: ResultRecorder,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        done_ids, resumed_count = self.resume_state(recorder, rank_episodes)
        results: list[dict[str, Any]] = []
        timing_stats: list[dict[str, Any]] = []
        description = (
            f"Rank 0 ({len(rank_episodes)} eps, {self.world_size} GPUs total)"
            if self.world_size > 1
            else f"Evaluating ({MODEL_DESCRIPTION})"
        )
        progress = tqdm.tqdm(
            rank_episodes,
            desc=description,
            disable=not self.is_main,
        )
        for episode in progress:
            episode_key = recorder.episode_key(
                getattr(episode, "episode_id", ""),
                scene_name(episode),
            )
            if episode_key in done_ids:
                continue

            result = self.evaluate_episode(evaluator, env_wrapper, episode)
            results.append(result)
            done_ids.add(episode_key)
            recorder.append(result)
            if self.rank == 0 and "_timing_stats" in result:
                timing_stats.append(result)
            if self.is_main:
                self.update_progress(progress, results, resumed_count)
        return results, timing_stats, resumed_count

    @staticmethod
    def update_progress(progress: Any, results: list[dict[str, Any]], resumed: int):
        recent = results[-20:]
        success_rate = sum(float(row.get("success", 0.0)) for row in recent) / len(
            recent
        )
        valid_errors = [
            float(row.get("distance_to_goal", 0.0))
            for row in recent
            if float(row.get("distance_to_goal", 0.0)) < 1000
        ]
        navigation_error = (
            sum(valid_errors) / len(valid_errors) if valid_errors else 0.0
        )
        progress.set_postfix(
            SR=f"{success_rate:.2%}",
            NE=f"{navigation_error:.1f}m",
            done=resumed + len(results),
        )

    def finalize(
        self,
        recorder: ResultRecorder,
        results: list[dict[str, Any]],
        timing_stats: list[dict[str, Any]],
        resumed_count: int,
        local_total: int,
    ) -> None:
        recorder.mark_rank_complete(
            processed_count=len(results),
            resumed_count=resumed_count,
            local_total=local_total,
        )
        if not self.is_main:
            return
        if self.world_size > 1:
            recorder.wait_for_ranks(list(range(self.world_size)))
        merged_results = recorder.load_results()
        print(
            f"[Summary] Loaded {len(merged_results)} deduplicated episodes from "
            f"{recorder.result_file}"
        )
        recorder.save_summary(
            merged_results,
            timing_stats,
            args=self.args,
            model_description=MODEL_DESCRIPTION,
            summary_extras=self.summary_extras,
            uses_compression=True,
        )

    def run(self) -> None:
        """Execute model loading, environment setup, rank loop, and reporting."""
        self.set_seed()
        self.initialize_distributed()
        recorder = ResultRecorder(
            self.args.output_dir,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.prepare_output(recorder)
        if self.is_main:
            print(f"Loading model from {self.args.model_path}...")
        model, processor = self.load_model()
        evaluator = self.create_evaluator(model, processor)
        env_wrapper = evaluator.create_environment()
        try:
            episodes = self.load_episodes(env_wrapper)
            rank_episodes, episodes_by_scene = distribute_episodes(
                episodes,
                rank=self.rank,
                world_size=self.world_size,
            )
            if self.is_main:
                print(
                    f"Environment: {self.args.eval_split}, Total: {len(episodes)}, "
                    f"This process: {len(rank_episodes)}, "
                    f"Scenes: {len(episodes_by_scene)}"
                )
            results, timing_stats, resumed_count = self.evaluate_rank(
                evaluator,
                env_wrapper,
                rank_episodes,
                recorder,
            )
            self.finalize(
                recorder,
                results,
                timing_stats,
                resumed_count,
                len(rank_episodes),
            )
        finally:
            env_wrapper.close()
            if self.world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
