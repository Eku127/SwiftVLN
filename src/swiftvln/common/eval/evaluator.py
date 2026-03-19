# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN Evaluator for VLN tasks (Habitat and SatNav).

This base class provides common functionality for VLN evaluation across different
model architectures (StreamVLN, MonoVLN, etc.).

Subclasses should implement:
- eval_episode(): The main evaluation logic for a single episode
"""

import os
import re
import time
import random
import torch
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple
from PIL import Image
from omegaconf import OmegaConf

# ============================================================================
# Default random seed for reproducible evaluation
# ============================================================================
DEFAULT_EVAL_SEED = 42

import habitat
from habitat import Env
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

# Import from common module
from ..constants import DEFAULT_ACTION_MAP, DEFAULT_IMAGE_TOKEN
from ..env.base import EnvWrapper
from ..env.habitat import HabitatEnvWrapper
from ..env.satnav import SatNavEnvWrapper
from ..utils import append_text_to_image, TrajectoryRecorder, ErrorAnalyzer

# Trigger registration of measures
try:
    from swiftvln import habitat_extensions  # noqa: F401
except ImportError:
    pass

class BaseVLNEvaluator(ABC):
    """
    Base VLN Evaluator providing common functionality.
    
    Key features:
    - Environment configuration (Habitat, SatNav)
    - Action parsing
    - History sampling
    - Video generation
    - Error analysis
    
    Subclasses must implement:
    - eval_episode(): Episode evaluation logic
    """
    
    def __init__(
        self,
        config_path: str,
        model: Any,
        processor: Any,
        template: Any,
        args: Any,
        env_type: str = "habitat",
    ):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.processor = processor
        self.template = template
        self.env_type = env_type
        self.config_path = config_path
        
        # Load config based on environment type
        if env_type == "habitat":
            self._init_habitat_config(config_path, args)
        elif env_type == "satnav":
            self._init_satnav_config(config_path, args)
        else:
            raise ValueError(f"Unknown env_type: {env_type}. Must be 'habitat' or 'satnav'.")

        # Action mapping (same across all VLN models)
        self.idx2actions = DEFAULT_ACTION_MAP.copy()
        self.actions2idx = {v: k for k, v in self.idx2actions.items()}
        
        # VLN parameters (from args)
        self.num_history = getattr(args, 'num_history', 8)
        self.num_future_steps = getattr(args, 'num_future_steps', 4)
        
        # Video generation parameters
        self.save_video = getattr(args, 'save_video', False)
        self.output_path = getattr(args, 'output_dir', './results/eval')
        
        # Random seed for reproducibility
        self.eval_seed = DEFAULT_EVAL_SEED
    
    def set_eval_seed(self, seed: int = None):
        """Set random seeds for reproducible evaluation.
        
        Args:
            seed: Random seed value. If None, uses self.eval_seed.
        """
        if seed is None:
            seed = self.eval_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _init_habitat_config(self, config_path: str, args: Any) -> None:
        """Initialize Habitat configuration."""
        from habitat_baselines.config.default import get_config as get_habitat_config
        self.config = get_habitat_config(config_path)
        
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = args.eval_split
            
            # CRITICAL: Set gpu_device_id based on LOCAL_RANK for distributed rendering
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
            
            # Add metrics
            self.config.habitat.task.measurements.update(
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
    
    def _init_satnav_config(self, config_path: str, args: Any) -> None:
        """Initialize SatNav configuration.

        ``DATASET.DATA_PATH`` may contain a ``{split}`` placeholder which is
        expanded with ``args.eval_split`` at runtime, e.g.::

            eval/{split}/all_episodes.json → eval/val_seen/all_episodes.json

        Raises ``FileNotFoundError`` if the expanded path does not exist.
        """
        self.config = OmegaConf.load(config_path)
        OmegaConf.set_struct(self.config, False)
        self.config.DATASET.SPLIT = args.eval_split

        raw_path = self.config.DATASET.DATA_PATH
        if '{split}' in raw_path:
            resolved_path = raw_path.replace('{split}', args.eval_split)
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(
                    f"[SatNav] eval_split='{args.eval_split}' → '{resolved_path}' does not exist. "
                    f"Run process_episodes.py to generate this split first."
                )
            self.config.DATASET.DATA_PATH = resolved_path
            print(f"[SatNav] eval_split='{args.eval_split}' → DATA_PATH: {resolved_path}")

        OmegaConf.set_struct(self.config, True)
        
    def config_env(self) -> EnvWrapper:
        """Create and return an environment wrapper.
        
        Returns:
            EnvWrapper: Unified environment wrapper (HabitatEnvWrapper or SatNavEnvWrapper)
        """
        if self.env_type == "habitat":
            habitat_env = Env(config=self.config)
            max_steps = self.config.habitat.environment.max_episode_steps
            return HabitatEnvWrapper(habitat_env, max_episode_steps=max_steps)
        elif self.env_type == "satnav":
            from satnav.core.env import Env as SatNavEnv
            from satnav.dataset.satnav_dataset import SatNavDataset
            
            dataset = SatNavDataset(self.config.DATASET)
            satnav_env = SatNavEnv(self.config, dataset=dataset, cycle=False)
            return SatNavEnvWrapper(satnav_env, config=self.config)
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

    def parse_actions(self, output: str) -> List[int]:
        """
        Parse action sequence from model output string.
        
        Args:
            output: Model output text containing action symbols (↑←→STOP)
            
        Returns:
            List of action indices
        """
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        return actions

    def sample_history_indices(self, total_frames: int, num_to_sample: int) -> List[int]:
        """
        Uniformly sample indices from historical frames.
        
        Args:
            total_frames: Total number of historical frames available
            num_to_sample: Number of frames to sample
            
        Returns:
            List of sampled frame indices
        """
        if total_frames <= 0:
            return []
        
        if total_frames <= num_to_sample:
            return list(range(total_frames))
        
        # Uniform sampling
        step = max(total_frames // num_to_sample, 1)
        indices = list(range(0, total_frames, step))
        
        # Ensure we don't exceed num_to_sample
        if len(indices) > num_to_sample:
            selected = np.linspace(0, len(indices) - 1, num_to_sample, dtype=int)
            indices = [indices[i] for i in selected]
        
        return indices

    @abstractmethod
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode.
        
        This is the main method that subclasses must implement.
        
        Args:
            env_wrapper: Unified environment wrapper
            episode: Episode to evaluate
            env_idx: Environment index for multi-env support
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def _analyze_trajectory_errors(
        self,
        trajectory_recorder: TrajectoryRecorder,
        episode: Any,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze trajectory for error patterns.
        
        Args:
            trajectory_recorder: Recorder containing agent trajectory
            episode: Episode object with GT path and goal information
            metrics: Evaluation metrics dictionary
            
        Returns:
            Dictionary with error_tags, had_deviation, deviation_recovered
        """
        trajectory = trajectory_recorder.get_trajectory()
        path_lengths = trajectory_recorder.get_path_lengths()
        
        if len(trajectory) == 0:
            return {
                "error_tags": [],
                "had_deviation": False,
                "deviation_recovered": False,
            }
        
        # Extract GT path and goal position from episode
        gt_path = None
        if hasattr(episode, 'reference_path') and episode.reference_path is not None:
            gt_path = [np.array(pos) for pos in episode.reference_path]
        
        # Get goal position
        goal_position = None
        if hasattr(episode, 'goals') and len(episode.goals) > 0:
            goal_position = np.array(episode.goals[0].position)
        elif hasattr(episode, 'goal_position'):
            goal_position = np.array(episode.goal_position)
        
        if goal_position is None and len(trajectory) > 0:
            goal_position = trajectory[-1]
        
        # Get final distance and oracle success from metrics
        final_distance = metrics.get('distance_to_goal', float('inf'))
        oracle_success = metrics.get('oracle_success', 0.0) > 0.5
        
        # Analyze trajectory
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(
            trajectory=trajectory,
            path_lengths=path_lengths,
            gt_path=gt_path,
            goal_position=goal_position,
            final_distance=final_distance,
            oracle_success=oracle_success
        )
        
        return error_analysis
    
    def _save_habitat_video(
        self,
        episode_id: str,
        vis_frames: List[np.ndarray],
        metrics: Dict[str, Any]
    ) -> None:
        """Save Habitat visualization video."""
        if len(vis_frames) == 0:
            return
            
        video_dir = os.path.join(self.output_path, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        success_rate = metrics.get("success", 0.0)
        success_rate_str = f"{int(success_rate * 100)}"
        video_filename = f"{episode_id}_{success_rate_str}"
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                images_to_video(
                    vis_frames, 
                    video_dir, 
                    video_filename,
                    fps=6,
                    quality=9
                )
        except Exception as e:
            print(f"[Warning] Failed to save Habitat video for episode {episode_id}: {e}")
    
    def _save_satnav_video(
        self,
        episode_id: str,
        instruction: str,
        rgb_frames: List[np.ndarray],
        topdown_frames: List[np.ndarray],
        metrics: Dict[str, Any]
    ) -> None:
        """Save SatNav visualization video."""
        if len(rgb_frames) == 0:
            print(f"[Warning] No RGB frames for episode {episode_id}, skipping video")
            return
        if len(topdown_frames) == 0:
            print(f"[Warning] No topdown frames for episode {episode_id}, skipping video")
            return
            
        from satnav.utils.maps import make_video
        from pathlib import Path
        
        video_dir = os.path.join(self.output_path, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        success_rate = metrics.get("success", 0.0)
        success_rate_str = f"{int(success_rate * 100)}"
        video_filename = f"{episode_id}_{success_rate_str}.mp4"
        video_path = Path(video_dir) / video_filename
        
        # Ensure same number of frames
        min_frames = min(len(rgb_frames), len(topdown_frames))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                make_video(
                    rgb_frames=rgb_frames[:min_frames],
                    topdown_frames=topdown_frames[:min_frames],
                    instruction_text=instruction,
                    output_path=video_path,
                    fps=5,
                    frame_width=2048,
                    quality=5
                )
        except Exception as e:
            print(f"[Warning] Failed to save SatNav video for episode {episode_id}: {e}")
    
    def _collect_habitat_frame(
        self,
        observations: Dict[str, Any],
        instruction: str,
        env_wrapper: EnvWrapper
    ) -> Optional[np.ndarray]:
        """Collect a Habitat visualization frame."""
        info = env_wrapper.get_metrics()
        if info.get('top_down_map') is not None:
            frame = observations_to_image({'rgb': observations['rgb']}, info)
            frame = append_text_to_image(frame, f"Instruction: {instruction}", position='top')
            return frame
        return None
    
    def _collect_satnav_topdown(
        self,
        env_wrapper: EnvWrapper,
        episode: Any,
        action: int,
        step_id: int,
        topdown_frames: List[np.ndarray],
        rgb_fallback: np.ndarray
    ) -> None:
        """Collect SatNav topdown frame after step."""
        try:
            from satnav.utils.maps import annotate_topdown_map
            from satnav.core.utils import geodesic_distance
            
            info = env_wrapper.get_last_step_info()
            agent_state = env_wrapper.get_agent_state()
            config = env_wrapper.config
            
            # Debug: check if top_down_map is in metrics
            env_metrics = info.get("metrics", {})
            if "top_down_map" not in env_metrics:
                # Silently use rgb_fallback if no top_down_map
                topdown_frames.append(rgb_fallback.copy())
                return
            
            waypoints = getattr(episode, 'reference_path', [])
            if not waypoints and hasattr(episode, 'goals') and episode.goals:
                waypoints = [g.position for g in episode.goals]
            
            if waypoints:
                goal_pos = waypoints[-1] if isinstance(waypoints[-1], list) else waypoints[-1]
                current_distance = geodesic_distance(agent_state.position, goal_pos)
            else:
                current_distance = 0.0
            
            # Handle SUCCESS_DISTANCE config (supports both old format and new dict format)
            # Old format: SUCCESS_DISTANCE: 10.0
            # New format: SUCCESS_DISTANCE: {DEFAULT: 10.0, Boundary: 10.0, LandmarkSet: 30.0, Road: 10.0}
            success_distance_config = getattr(config.TASK, 'SUCCESS_DISTANCE', 8.0)
            if isinstance(success_distance_config, (int, float)):
                goal_radius = float(success_distance_config)
            elif hasattr(success_distance_config, 'DEFAULT'):
                # New dict format with type-specific thresholds
                # Use trajectory_type if available, otherwise use DEFAULT
                trajectory_type = getattr(episode, 'trajectory_type', None)
                if trajectory_type and hasattr(success_distance_config, trajectory_type):
                    goal_radius = float(getattr(success_distance_config, trajectory_type))
                else:
                    goal_radius = float(success_distance_config.DEFAULT)
            else:
                goal_radius = 8.0  # Fallback default
            
            action_symbol = self.idx2actions.get(action, "STOP")
            action_name_map = {"↑": "FORWARD", "←": "LEFT", "→": "RIGHT", "STOP": "STOP"}
            action_name = action_name_map.get(action_symbol, action_symbol)
            
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
                topdown_frames=topdown_frames
            )
            
            # If annotate_topdown_map didn't add a frame (returned None), use fallback
            if len(topdown_frames) < step_id:
                topdown_frames.append(rgb_fallback.copy())
                
        except Exception as e:
            # Use rgb_fallback on any error
            if len(topdown_frames) < step_id:
                topdown_frames.append(rgb_fallback.copy())
