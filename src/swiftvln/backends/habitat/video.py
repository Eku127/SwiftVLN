"""Habitat-specific frame rendering and video persistence."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from swiftvln.backends.base import EnvWrapper
from swiftvln.utils.images import append_text_to_image


def collect_frame(
    observations: dict[str, Any],
    instruction: str,
    env_wrapper: EnvWrapper,
) -> np.ndarray | None:
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


def save_video(
    *,
    output_path: str,
    episode_id: str,
    frames: list[np.ndarray],
    metrics: dict[str, Any],
) -> None:
    if not frames:
        return

    from habitat.utils.visualizations.utils import images_to_video

    video_dir = os.path.join(output_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    success_percentage = int(metrics.get("success", 0.0) * 100)
    video_filename = f"{episode_id}_{success_percentage}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images_to_video(frames, video_dir, video_filename, fps=6, quality=9)
    except Exception as exc:
        print(f"[Warning] Failed to save Habitat video for episode {episode_id}: {exc}")
