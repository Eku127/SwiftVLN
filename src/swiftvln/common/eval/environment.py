# Copyright (c) Alibaba, Inc. and its affiliates.
"""Environment-specific configuration and episode support for evaluation."""

from __future__ import annotations

import os
import random
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from ..constants import DEFAULT_ACTION_MAP
from ..env.base import EnvWrapper
from ..utils.image_utils import append_text_to_image

DEFAULT_EVAL_SEED = 42


class EvaluationEnvironment:
    """Own one evaluation backend's config, wrapper, actions, and visuals."""

    def __init__(self, config_path: str, args: Any, env_type: str):
        self.args = args
        self.env_type = env_type
        self.config_path = config_path
        self.config = self._load_config()

        self.idx2actions = DEFAULT_ACTION_MAP.copy()
        self.actions2idx = {symbol: index for index, symbol in self.idx2actions.items()}
        self.save_video = getattr(args, "save_video", False)
        self.output_path = getattr(args, "output_dir", "./results/eval")

    def _load_config(self):
        if self.env_type == "habitat":
            return self._load_habitat_config()
        if self.env_type == "satnav":
            return self._load_satnav_config()
        raise ValueError(
            f"Unknown env_type: {self.env_type}. Must be 'habitat' or 'satnav'."
        )

    def _load_habitat_config(self):
        """Load Habitat config and attach evaluation visual measurements."""
        import habitat
        from habitat.config.default_structured_configs import (
            CollisionsMeasurementConfig,
            FogOfWarConfig,
            TopDownMapMeasurementConfig,
        )
        from habitat_baselines.config.default import get_config as get_habitat_config

        # Register SwiftVLN's retained custom measures before resolving config.
        from swiftvln import habitat_extensions  # noqa: F401

        config = get_habitat_config(self.config_path)
        with habitat.config.read_write(config):
            config.habitat.dataset.split = self.args.eval_split
            habitat_data_path = os.environ.get(
                "SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH"
            )
            habitat_scenes_dir = os.environ.get("SWIFTVLN_HABITAT_SCENES_DIR")
            if habitat_data_path:
                config.habitat.dataset.data_path = habitat_data_path
            if habitat_scenes_dir:
                config.habitat.dataset.scenes_dir = habitat_scenes_dir
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
            config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        return config

    def _load_satnav_config(self):
        """Load SatNav config and resolve its optional ``{split}`` data path."""
        config = OmegaConf.load(self.config_path)
        OmegaConf.set_struct(config, False)
        config.DATASET.SPLIT = self.args.eval_split

        raw_path = os.environ.get(
            "SWIFTVLN_SATNAV_EVAL_DATA_PATH", config.DATASET.DATA_PATH
        )
        scenes_dir = os.environ.get("SWIFTVLN_SATNAV_SCENES_DIR")
        if scenes_dir:
            config.DATASET.SCENES_DIR = scenes_dir
        if "{split}" in raw_path:
            resolved_path = raw_path.replace("{split}", self.args.eval_split)
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(
                    f"[SatNav] eval_split='{self.args.eval_split}' -> "
                    f"'{resolved_path}' does not exist. Run process_episodes.py "
                    "to generate this split first."
                )
            config.DATASET.DATA_PATH = resolved_path
            print(
                f"[SatNav] eval_split='{self.args.eval_split}' -> "
                f"DATA_PATH: {resolved_path}"
            )
        else:
            config.DATASET.DATA_PATH = raw_path

        OmegaConf.set_struct(config, True)
        return config

    def create_wrapper(self) -> EnvWrapper:
        """Construct the configured Habitat or SatNav environment wrapper."""
        if self.env_type == "habitat":
            from habitat import Env

            from ..env.habitat import HabitatEnvWrapper

            habitat_env = Env(config=self.config)
            max_steps = self.config.habitat.environment.max_episode_steps
            return HabitatEnvWrapper(habitat_env, max_episode_steps=max_steps)

        from satnav.core.env import Env as SatNavEnv
        from satnav.dataset.satnav_dataset import SatNavDataset

        from ..env.satnav import SatNavEnvWrapper

        dataset = SatNavDataset(self.config.DATASET)
        satnav_env = SatNavEnv(self.config, dataset=dataset, cycle=False)
        return SatNavEnvWrapper(satnav_env, config=self.config)

    def set_seed(self, seed: int = DEFAULT_EVAL_SEED) -> None:
        """Seed model generation and environment-side random helpers."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def parse_actions(self, output: str) -> list[int]:
        """Parse action symbols from a model response."""
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        matches = re.findall(action_patterns, output)
        return [self.actions2idx[match] for match in matches]

    def save_habitat_video(
        self,
        episode_id: str,
        frames: list[np.ndarray],
        metrics: dict[str, Any],
    ) -> None:
        """Save one Habitat visualization video."""
        if not frames:
            return

        from habitat.utils.visualizations.utils import images_to_video

        video_dir = os.path.join(self.output_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        success_percentage = int(metrics.get("success", 0.0) * 100)
        video_filename = f"{episode_id}_{success_percentage}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                images_to_video(
                    frames,
                    video_dir,
                    video_filename,
                    fps=6,
                    quality=9,
                )
        except Exception as exc:
            print(
                f"[Warning] Failed to save Habitat video for episode "
                f"{episode_id}: {exc}"
            )

    def save_satnav_video(
        self,
        episode_id: str,
        instruction: str,
        rgb_frames: list[np.ndarray],
        topdown_frames: list[np.ndarray],
        metrics: dict[str, Any],
    ) -> None:
        """Save one SatNav RGB/top-down visualization video."""
        if not rgb_frames:
            print(f"[Warning] No RGB frames for episode {episode_id}, skipping video")
            return
        if not topdown_frames:
            print(
                f"[Warning] No topdown frames for episode {episode_id}, skipping video"
            )
            return

        from satnav.utils.maps import make_video

        video_dir = os.path.join(self.output_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        success_percentage = int(metrics.get("success", 0.0) * 100)
        video_path = Path(video_dir) / f"{episode_id}_{success_percentage}.mp4"
        frame_count = min(len(rgb_frames), len(topdown_frames))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                make_video(
                    rgb_frames=rgb_frames[:frame_count],
                    topdown_frames=topdown_frames[:frame_count],
                    instruction_text=instruction,
                    output_path=video_path,
                    fps=5,
                    frame_width=2048,
                    quality=5,
                )
        except Exception as exc:
            print(
                f"[Warning] Failed to save SatNav video for episode {episode_id}: {exc}"
            )

    @staticmethod
    def collect_habitat_frame(
        observations: dict[str, Any],
        instruction: str,
        env_wrapper: EnvWrapper,
    ) -> np.ndarray | None:
        """Render one Habitat RGB/top-down frame when map metrics are present."""
        from habitat.utils.visualizations.utils import observations_to_image

        info = env_wrapper.get_metrics()
        if info.get("top_down_map") is None:
            return None
        frame = observations_to_image({"rgb": observations["rgb"]}, info)
        return append_text_to_image(
            frame,
            f"Instruction: {instruction}",
            position="top",
        )

    def collect_satnav_topdown(
        self,
        env_wrapper: EnvWrapper,
        episode: Any,
        action: int,
        step_id: int,
        topdown_frames: list[np.ndarray],
        rgb_fallback: np.ndarray,
    ) -> None:
        """Append one SatNav top-down frame, falling back to RGB on errors."""
        try:
            from satnav.core.utils import geodesic_distance
            from satnav.utils.maps import annotate_topdown_map

            info = env_wrapper.get_last_step_info()
            agent_state = env_wrapper.get_agent_state()
            config = env_wrapper.config
            if "top_down_map" not in info.get("metrics", {}):
                topdown_frames.append(rgb_fallback.copy())
                return

            waypoints = getattr(episode, "reference_path", [])
            if not waypoints and getattr(episode, "goals", None):
                waypoints = [goal.position for goal in episode.goals]

            if waypoints:
                current_distance = geodesic_distance(
                    agent_state.position,
                    waypoints[-1],
                )
            else:
                current_distance = 0.0

            success_distance = getattr(config.TASK, "SUCCESS_DISTANCE", 8.0)
            if isinstance(success_distance, (int, float)):
                goal_radius = float(success_distance)
            elif hasattr(success_distance, "DEFAULT"):
                trajectory_type = getattr(episode, "trajectory_type", None)
                if trajectory_type and hasattr(success_distance, trajectory_type):
                    goal_radius = float(getattr(success_distance, trajectory_type))
                else:
                    goal_radius = float(success_distance.DEFAULT)
            else:
                goal_radius = 8.0

            action_symbol = self.idx2actions.get(action, "STOP")
            action_name = {
                "↑": "FORWARD",
                "←": "LEFT",
                "→": "RIGHT",
                "STOP": "STOP",
            }.get(action_symbol, action_symbol)
            annotate_topdown_map(
                info=info,
                agent_state=agent_state,
                waypoints=waypoints,
                current_waypoint_idx=len(waypoints) - 1,
                step_count=step_id,
                action=action_name,
                current_distance=current_distance,
                goal_radius=goal_radius,
                config=config,
                topdown_frames=topdown_frames,
            )
            if len(topdown_frames) < step_id:
                topdown_frames.append(rgb_fallback.copy())
        except Exception:
            if len(topdown_frames) < step_id:
                topdown_frames.append(rgb_fallback.copy())
