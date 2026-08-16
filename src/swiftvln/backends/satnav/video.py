"""SatNav-specific map rendering and video persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import warnings
from typing import Any

import numpy as np

from swiftvln.backends.base import EnvWrapper


@dataclass
class SatNavVideoState:
    rgb_frames: list[np.ndarray] = field(default_factory=list)
    topdown_frames: list[np.ndarray] = field(default_factory=list)


def collect_topdown(
    state: SatNavVideoState,
    *,
    env_wrapper: EnvWrapper,
    episode: Any,
    action_symbol: str,
    step_id: int,
    rgb_fallback: np.ndarray,
) -> None:
    try:
        from satnav.core.utils import geodesic_distance
        from satnav.utils.maps import annotate_topdown_map

        info = env_wrapper.get_last_step_info()
        agent_state = env_wrapper.get_agent_state()
        config = env_wrapper.config
        if "top_down_map" not in info.get("metrics", {}):
            state.topdown_frames.append(rgb_fallback.copy())
            return

        waypoints = getattr(episode, "reference_path", [])
        if not waypoints and getattr(episode, "goals", None):
            waypoints = [goal.position for goal in episode.goals]

        current_distance = (
            geodesic_distance(agent_state.position, waypoints[-1])
            if waypoints
            else 0.0
        )
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
            topdown_frames=state.topdown_frames,
        )
        if len(state.topdown_frames) < step_id:
            state.topdown_frames.append(rgb_fallback.copy())
    except Exception:
        if len(state.topdown_frames) < step_id:
            state.topdown_frames.append(rgb_fallback.copy())


def save_video(
    state: SatNavVideoState,
    *,
    output_path: str,
    episode_id: str,
    instruction: str,
    metrics: dict[str, Any],
) -> None:
    if not state.rgb_frames:
        print(f"[Warning] No RGB frames for episode {episode_id}, skipping video")
        return
    if not state.topdown_frames:
        print(f"[Warning] No topdown frames for episode {episode_id}, skipping video")
        return

    from satnav.utils.maps import make_video

    video_dir = os.path.join(output_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    success_percentage = int(metrics.get("success", 0.0) * 100)
    video_path = Path(video_dir) / f"{episode_id}_{success_percentage}.mp4"
    frame_count = min(len(state.rgb_frames), len(state.topdown_frames))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make_video(
                rgb_frames=state.rgb_frames[:frame_count],
                topdown_frames=state.topdown_frames[:frame_count],
                instruction_text=instruction,
                output_path=video_path,
                fps=5,
                frame_width=2048,
                quality=5,
            )
    except Exception as exc:
        print(f"[Warning] Failed to save SatNav video for episode {episode_id}: {exc}")
