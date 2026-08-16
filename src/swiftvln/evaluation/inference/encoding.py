# Copyright (c) Alibaba, Inc. and its affiliates.
"""Visual encoding and history-memory construction for inference."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from PIL import Image

from swiftvln.modeling.history.per_frame import (
    sample_per_frame_history_indices,
)


class VisualEncodingMixin:
    """Encode current/history frames without owning session orchestration."""

    @staticmethod
    def _extract_visual_features(visual_res: Any) -> torch.Tensor:
        """Normalize Qwen-family visual outputs to pooled visual tokens."""
        if hasattr(visual_res, "pooler_output"):
            return visual_res.pooler_output
        if isinstance(visual_res, tuple):
            return visual_res[0]
        return visual_res

    def encode_initial_view(
        self,
        image: Image.Image,
        pose: List[float],
    ) -> torch.Tensor:
        """Cache the uncompressed initial view used by initial prompt mode."""
        self.initial_features, _ = self._encode_frame(image, pose=pose)
        return self.initial_features

    def _encode_frame(
        self,
        image: Image.Image,
        pose: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one frame and return its visual features and grid."""
        media_inputs = self.processor.image_processor(
            images=[image],
            return_tensors="pt",
        )
        pixel_values = (
            media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        )
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )
            if self.has_embed_enhance:
                _, height, width = image_grid_thw[0].tolist()
                height_merged = int(height) // self.merge_size
                width_merged = int(width) // self.merge_size
                vit_features = self.model.embed_enhance(
                    vit_features,
                    height_merged,
                    width_merged,
                    pose=pose,
                )

        return vit_features, image_grid_thw[0]

    def _encode_batch_frames(
        self,
        images: List[Image.Image],
        poses: Optional[List[Optional[List[float]]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Batch-encode frames while preserving per-image token boundaries."""
        if not images:
            return [], []

        media_inputs = self.processor.image_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = (
            media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        )
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            all_vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )

        merge_length = self.merge_size**2
        features_list = []
        grid_thw_list = []
        embed_idx = 0
        for index in range(len(images)):
            num_tokens = int(image_grid_thw[index].prod() // merge_length)
            image_features = all_vit_features[
                embed_idx : embed_idx + num_tokens
            ]
            embed_idx += num_tokens

            if self.has_embed_enhance:
                _, height, width = image_grid_thw[index].tolist()
                height_merged = int(height) // self.merge_size
                width_merged = int(width) // self.merge_size
                pose = None
                if poses is not None and index < len(poses):
                    pose = poses[index]
                with torch.no_grad():
                    image_features = self.model.embed_enhance(
                        image_features,
                        height_merged,
                        width_merged,
                        pose=pose,
                    )

            features_list.append(image_features)
            grid_thw_list.append(image_grid_thw[index])

        return features_list, grid_thw_list

    def _compress_features(
        self,
        features: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Compress visual features with the configured history compressor."""
        grid_after_merge = grid_thw.clone()
        grid_after_merge[1] = grid_after_merge[1] // self.merge_size
        grid_after_merge[2] = grid_after_merge[2] // self.merge_size
        compressed, _ = self.compressor.compress(
            features,
            grid_after_merge,
            stride=self.compress_stride,
        )
        return compressed

    def _compute_history_cache(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
        episode: Any,
    ) -> None:
        """Build map, per-frame, or clustered history for a new window."""
        if self.memory_method == "map":
            self._compute_history_cache_map(episode, window_start)
            return

        if window_start <= 0:
            self.history_cache = []
            return

        if self.history_processor_type in ("gtc", "segment_gtc"):
            self._compute_history_cache_gtc(
                rgb_list,
                pose_list,
                window_start,
            )
        else:
            self._compute_history_cache_per_frame(
                rgb_list,
                pose_list,
                window_start,
            )

    def _compute_history_cache_map(
        self,
        episode: Any,
        window_start: int,
    ) -> None:
        if self.map_builder is None:
            raise RuntimeError(
                "memory_method=map requested but map_builder is not initialized."
            )
        scene_id = getattr(episode, "scene_id", None)
        start_position = getattr(episode, "start_position", None)
        start_rotation = getattr(episode, "start_rotation", None)
        if scene_id is None or start_position is None or start_rotation is None:
            raise ValueError(
                "SatNav episode is missing scene/start metadata required for map memory."
            )

        map_images = self.map_builder.render_from_actions(
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=float(start_rotation),
            actions=self.executed_actions,
            window_start=window_start,
            step_size=self.environment_spec.forward_step_m,
            turn_angle=self.environment_spec.turn_angle_deg,
        )
        features_list, grid_thw_list = self._encode_batch_frames(
            map_images,
            poses=[None] * len(map_images),
        )
        self.history_cache = []
        for features, grid_thw in zip(features_list, grid_thw_list):
            compressed = self._compress_features(features, grid_thw)
            self.history_cache.append((compressed, None))
        self.diagnostics.map_cache(
            self,
            episode,
            window_start,
            features_list,
        )

    def _compute_history_cache_per_frame(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
    ) -> None:
        """Sample and independently compress per-frame history."""
        available_history_frames = min(window_start, len(rgb_list))
        if self.num_history <= 0 or available_history_frames <= 0:
            self.history_cache = []
            return

        history_indices = sample_per_frame_history_indices(
            num_frames=available_history_frames,
            num_samples=self.num_history,
            log_base=self.log_base,
            use_random=self.use_random,
        )
        if not history_indices:
            self.history_cache = []
            return

        history_images = [rgb_list[index] for index in history_indices]
        history_poses = [
            pose_list[index] if index < len(pose_list) else None
            for index in history_indices
        ]
        features_list, grid_thw_list = self._encode_batch_frames(
            history_images,
            poses=history_poses,
        )

        self.history_cache = []
        for features, grid_thw, pose in zip(
            features_list,
            grid_thw_list,
            history_poses,
        ):
            compressed = self._compress_features(features, grid_thw)
            self.history_cache.append((compressed, pose))

    def _compute_history_cache_gtc(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
    ) -> None:
        """Cluster all prediction-time features preceding the new window."""
        del rgb_list, pose_list
        history_features = []
        for step_id in sorted(self.vit_feature_cache):
            if step_id < window_start:
                features, _grid_thw, _pose = self.vit_feature_cache[step_id]
                history_features.append(features)

        if not history_features:
            self.history_cache = []
            return

        clustered = self.history_processor.process(
            frame_embeds_list=history_features,
            frame_grid_thws=[],
        )
        self.history_cache = [(clustered, None)]


__all__ = ["VisualEncodingMixin"]
