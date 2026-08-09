# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sliding-window state and overlap handling for SwiftVLN inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from swiftvln.modeling.embeddings import reconstruct_pose_from_actions


@dataclass
class OverlapContext:
    """Context copied from the trailing turns of the previous window."""

    input_ids: Optional[torch.Tensor] = None
    image_embeds: List[torch.Tensor] = field(default_factory=list)


@dataclass
class TurnContext:
    """Prompt and visual context for one completed turn."""

    user_input_ids: torch.Tensor
    assistant_response: str
    image_embed: torch.Tensor


class WindowStateMixin:
    """Own per-episode window state while the session coordinates execution."""

    def reset(self) -> None:
        """Reset all caches at the start of each episode."""
        self.history_cache: List[
            Tuple[torch.Tensor, Optional[List[float]]]
        ] = []

        # For GTC: cache VIT features by step_id to avoid re-encoding.
        self.vit_feature_cache: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, Optional[List[float]]]
        ] = {}

        self.initial_features: Optional[torch.Tensor] = None
        self.overlap_context: Optional[OverlapContext] = None
        self.window_turns: List[TurnContext] = []
        self.window_start_step = 0
        self.current_window_idx = 0
        self.pose_history: List[List[float]] = []
        self.executed_actions: List[int] = [-1]

    def _get_pose_from_actions(self) -> List[float]:
        """Fallback pose from executed actions (action integral)."""
        poses = reconstruct_pose_from_actions(
            self.executed_actions,
            step_size=self.environment_spec.forward_step_m,
            turn_angle=self.environment_spec.turn_angle_deg,
        )
        if poses.shape[0] == 0:
            return [0.0, 0.0, 0.0, 1.0]
        return poses[-1].tolist()

    def current_pose(self) -> List[float]:
        """Return the action-integrated pose used by the training pipeline."""
        return self._get_pose_from_actions()

    def observe_pose(self) -> List[float]:
        """Record and return the action-integrated pose for the current frame."""
        pose = self.current_pose()
        self.pose_history.append(pose)
        return pose

    def start_first_window(
        self,
        rgb_list: List[Image.Image],
        episode: Any,
    ) -> None:
        """Initialize window zero and synthesize map memory when requested."""
        self.window_start_step = 0
        self.current_window_idx = 0
        if self.memory_method == "map":
            self._compute_history_cache(rgb_list, self.pose_history, 0, episode)

    def record_action(self, action: int) -> None:
        """Append an executed action for pose and map reconstruction."""
        self.executed_actions.append(int(action))

    def _save_turn_to_window(
        self,
        conjunction: str,
        response: str,
        vit_features: torch.Tensor,
    ) -> None:
        """Save a completed user/assistant turn to the current window."""
        token_count = vit_features.shape[0]
        user_ids = self._build_user_turn_ids(conjunction, token_count)
        self.window_turns.append(
            TurnContext(
                user_input_ids=user_ids,
                assistant_response=response,
                image_embed=vit_features,
            )
        )

    def _prepare_overlap_context(self) -> None:
        """Cache the configured trailing turns for the next window."""
        if self.overlap_turns <= 0:
            self.overlap_context = None
            return

        if len(self.window_turns) < self.overlap_turns:
            self.overlap_context = None
            return

        overlap_turns = self.window_turns[-self.overlap_turns :]
        ids_parts = []
        image_embeds = []
        for turn in overlap_turns:
            ids_parts.append(turn.user_input_ids)
            ids_parts.append(
                self._build_assistant_turn_ids(turn.assistant_response)
            )
            image_embeds.append(turn.image_embed)

        self.overlap_context = OverlapContext(
            input_ids=torch.cat(ids_parts, dim=1),
            image_embeds=image_embeds,
        )

    def slide_window(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        new_window_start: int,
        episode: Any,
    ) -> None:
        """Advance the window and rebuild its history cache."""
        self._prepare_overlap_context()
        self._compute_history_cache(
            rgb_list,
            pose_list,
            new_window_start,
            episode,
        )
        self.diagnostics.map_slide_window(self, new_window_start)
        self.window_turns = []
        self.window_start_step = new_window_start
        self.current_window_idx += 1


__all__ = ["OverlapContext", "TurnContext", "WindowStateMixin"]
