# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pose utility helpers for VLN embedding enhancement."""

import math
from typing import Iterable, Union

import numpy as np


def _wrap_heading_deg(angle: float) -> float:
    """Wrap heading angle to [-180, 180)."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return wrapped


def reconstruct_pose_from_actions(
    actions: Iterable[Union[int, str]],
    step_size: float = 10.0,
    turn_angle: float = 15.0,
    norm_scale: float = 100.0,
) -> np.ndarray:
    """
    Reconstruct per-step pose sequence from action sequence.

    Output pose format:
        [delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]

    Notes:
    - This function follows action integration logic from SatNav tests.
    - `norm_scale` is accepted for API consistency; normalization is handled
      by PoseEmbedding at runtime.
    """
    _ = norm_scale  # Reserved for compatibility with earlier planning APIs.

    delta_forward = 0.0
    delta_right = 0.0
    delta_heading = 0.0
    poses = []

    for action in actions:
        if isinstance(action, str):
            action_key = action.strip().upper()
            is_left = action_key in ('TURN_LEFT', 'LEFT', 'L', '2')
            is_right = action_key in ('TURN_RIGHT', 'RIGHT', 'R', '3')
            is_forward = action_key in ('MOVE_FORWARD', 'FORWARD', 'F', '1')
        else:
            action_int = int(action)
            is_left = action_int == 2
            is_right = action_int == 3
            is_forward = action_int == 1

        if is_left:
            delta_heading = _wrap_heading_deg(delta_heading - turn_angle)
        elif is_right:
            delta_heading = _wrap_heading_deg(delta_heading + turn_angle)
        elif is_forward:
            heading_rad = math.radians(delta_heading)
            delta_forward += step_size * math.cos(heading_rad)
            delta_right += step_size * math.sin(heading_rad)

        heading_rad = math.radians(delta_heading)
        poses.append([
            delta_forward,
            delta_right,
            math.sin(heading_rad),
            math.cos(heading_rad),
        ])

    if not poses:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(poses, dtype=np.float32)
