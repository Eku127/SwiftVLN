# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SwiftVLN Evaluator with VIT Feature Caching and Overlap Context Reuse

This evaluator implements an efficient inference pipeline that:
1. Caches VIT features to avoid redundant computation
2. Reuses overlap context from previous windows
3. Aligns evaluation with training data format

Prompt Format (aligned with training):
- History: Unified <history_memory> block in system prompt (compressed, no ROPE)
- Current: <|vision_start|><current_image>×N<|vision_end|> per observation (with ROPE)

This design ensures:
- History is treated as semantic memory (compressed features, no spatial encoding)
- Current observations have full ROPE position encoding for spatial understanding

Key Components:
- history_cache: Compressed VIT features for global history frames (unified block)
- overlap_context: Cached input_ids + image embeddings from configured overlap turns

Window Sliding Strategy:
- Window size: num_frames (default 32 actions = 8 turns)
- Overlap: num_overlap (default 0 for current baseline)
- Stride: num_frames - num_overlap
"""

import os
import time
import random
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional, Union
from PIL import Image
from dataclasses import dataclass, field
import json as json_module
import math as math_module

# Import from common module
try:
    from swiftvln.common import (
        BaseVLNEvaluator,
        CURRENT_IMAGE_TOKEN,
        DEFAULT_CONJUNCTIONS,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        HISTORY_MEMORY_TOKEN,
        PROMPT_TEMPLATE_HABITAT,
        PROMPT_TEMPLATE_SATNAV,
        TrajectoryRecorder,
        HistoryTokenCompressor,
    )
    from swiftvln.common.history_processors import (
        create_history_processor,
        HistoryProcessor,
        GlobalTokenClustering,
        PerFrameCompressor,
    )
    from swiftvln.common.history_processors.per_frame import sample_per_frame_history_indices
    from swiftvln.common.embedding_enhancement import reconstruct_pose_from_actions
    from swiftvln.model.map_memory import SatNavMapMemoryBuilder
except ImportError:
    from ..common import (
        BaseVLNEvaluator,
        CURRENT_IMAGE_TOKEN,
        DEFAULT_CONJUNCTIONS,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        HISTORY_MEMORY_TOKEN,
        PROMPT_TEMPLATE_HABITAT,
        PROMPT_TEMPLATE_SATNAV,
        TrajectoryRecorder,
        HistoryTokenCompressor,
    )
    from ..common.history_processors import (
        create_history_processor,
        HistoryProcessor,
        GlobalTokenClustering,
        PerFrameCompressor,
    )
    from ..common.history_processors.per_frame import sample_per_frame_history_indices
    from ..common.embedding_enhancement import reconstruct_pose_from_actions
    from .map_memory import SatNavMapMemoryBuilder


def _debug_enabled() -> bool:
    return os.environ.get('SWIFTVLN_DEBUG', '') != ''


def _debug_rank() -> int:
    raw = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    try:
        return int(raw)
    except ValueError:
        return 0


def _preview_text(text: str, limit: int = 260) -> str:
    text = str(text).replace('\n', '\\n')
    if len(text) <= limit:
        return text
    return text[:limit] + '...'


def _tensor_debug_stats(tensor: Optional[torch.Tensor], sample_limit: int = 4) -> str:
    if tensor is None:
        return "none"
    if not isinstance(tensor, torch.Tensor):
        return f"type={type(tensor).__name__}"
    if tensor.numel() == 0:
        return f"shape={tuple(tensor.shape)} empty"
    with torch.no_grad():
        flat = tensor.detach().float().cpu().reshape(-1)
        mean = float(flat.mean().item())
        std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
        checksum = float(flat.sum().item())
        abs_checksum = float(flat.abs().sum().item())
        l2 = float(torch.linalg.vector_norm(flat).item())
        sample = ", ".join(f"{v:.4f}" for v in flat[:sample_limit].tolist())
    return (
        f"shape={tuple(tensor.shape)} mean={mean:.6f} std={std:.6f} "
        f"sum={checksum:.6f} abs_sum={abs_checksum:.6f} l2={l2:.6f} "
        f"sample=[{sample}]"
    )


def _derive_dataset_cache_dir(data_path_tmpl: str) -> Optional[str]:
    """Infer {dataset_root}/map_cache from a SatNav DATA_PATH template."""
    if not data_path_tmpl:
        return None

    parts = str(data_path_tmpl).split(os.sep)
    if 'episodes' in parts:
        idx = parts.index('episodes')
        if idx > 0:
            dataset_root = os.sep.join(parts[:idx])
            if os.path.isabs(str(data_path_tmpl)) and not dataset_root.startswith(os.sep):
                dataset_root = os.sep + dataset_root
            return os.path.join(os.path.abspath(dataset_root), 'map_cache')

    probe_dir = os.path.dirname(str(data_path_tmpl))
    for _ in range(8):
        base = os.path.basename(probe_dir.rstrip('/'))
        if base.startswith('ver_'):
            return os.path.join(os.path.abspath(probe_dir), 'map_cache')
        parent = os.path.dirname(probe_dir)
        if not parent or parent == probe_dir:
            break
        probe_dir = parent
    return None


@dataclass
class OverlapContext:
    """Context from the last num_overlap/num_future_steps turns of previous window."""
    # Token IDs for the overlap turns (excluding system prompt)
    input_ids: Optional[torch.Tensor] = None
    # Image embeddings (already processed through VIT, may be compressed)
    image_embeds: List[torch.Tensor] = field(default_factory=list)


@dataclass
class TurnContext:
    """Context for a single turn in the current window."""
    user_input_ids: torch.Tensor
    assistant_response: str
    image_embed: torch.Tensor  # VIT features for this turn's image


class SwiftVLNEvaluator(BaseVLNEvaluator):
    """
    SwiftVLN Evaluator with VIT feature caching and overlap context reuse.
    
    Key Features:
    - VIT feature caching: Avoid recomputing features for cached frames
    - Overlap context: Reuse configured overlap turns from previous window
    - Manual embedding construction: Bypass template for efficient inference
    
    Cache Architecture:
    - history_cache: List[Tensor] - Compressed features for NUM_HISTORY global frames
    - overlap_context: OverlapContext - Cached context from last window's overlap turns
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ==========================================================================
        # Window parameters (shared by all history processor types)
        # ==========================================================================
        self.num_frames = getattr(self.args, 'num_frames', 32)
        self.num_overlap = getattr(self.args, 'num_overlap', 0)
        
        # Calculate derived window parameters
        self.stride = self.num_frames - self.num_overlap
        self.turns_per_window = self.num_frames // self.num_future_steps
        self.overlap_turns = self.num_overlap // self.num_future_steps
        self.new_turns_per_window = self.turns_per_window - self.overlap_turns
        
        # ==========================================================================
        # History processor configuration
        # ==========================================================================
        self.history_processor_type = getattr(self.args, 'history_processor_type', 'per_frame')
        
        # Per-frame specific parameters
        self.compress_stride = getattr(self.args, 'compress_stride', 2)
        self.use_tome = getattr(self.args, 'use_tome', False)
        self.log_base = getattr(self.args, 'log_base', 1.0)
        self.use_random = getattr(self.args, 'use_random', False)
        
        # GTC specific parameters
        self.gtc_output_tokens = getattr(self.args, 'gtc_output_tokens', 512)
        self.gtc_temperature = getattr(self.args, 'gtc_temperature', 0.1)
        self.gtc_num_iterations = getattr(self.args, 'gtc_num_iterations', 1)
        
        # Initialize history processor based on type
        if self.history_processor_type == 'gtc':
            self.history_processor = create_history_processor(
                processor_type='gtc',
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None  # Not used for GTC
        elif self.history_processor_type == 'segment_gtc':
            self.history_processor = create_history_processor(
                processor_type='segment_gtc',
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None  # Not used for Segment GTC
        else:
            # Per-frame compression (default)
            # Supports both uniform (log_base=1.0) and logarithmic (log_base>1.0) sampling
            compress_method = 'tome' if self.use_tome else 'pooling'
            self.history_processor = create_history_processor(
                processor_type='per_frame',
                compress_stride=self.compress_stride,
                compress_method=compress_method,
                num_history=self.num_history,
                log_base=self.log_base,
            )
            # Keep compressor for _compress_features method
            self.compressor = HistoryTokenCompressor(
                stride=self.compress_stride, 
                method=compress_method
            )
        
        # ==========================================================================
        # System prompt setting
        # ==========================================================================
        self.system_prompt_setting = getattr(self.args, 'system_prompt_setting', 'vanilla').lower()
        self.memory_method = getattr(self.args, 'memory_method', 'history').lower()
        self.map_global_side_m = float(getattr(self.args, 'map_global_side_m', 1000.0))
        self.map_local_side_m = float(getattr(self.args, 'map_local_side_m', 400.0))
        self.map_render_px = int(getattr(self.args, 'map_render_px', 448))
        self.map_mask_method = str(getattr(self.args, 'map_mask_method', 'dilate20')).lower()
        self.map_builder: Optional[SatNavMapMemoryBuilder] = None
        if self.memory_method not in ('history', 'map'):
            raise ValueError(f"Unsupported memory_method: {self.memory_method}")
        if self.memory_method == 'map':
            if self.env_type != 'satnav':
                raise ValueError("SwiftVLN memory_method=map currently supports only satnav.")
            if self.history_processor_type != 'per_frame':
                raise ValueError("SwiftVLN memory_method=map currently requires history_processor_type=per_frame.")
            if self.use_tome:
                raise ValueError("SwiftVLN memory_method=map currently requires use_tome=false.")
            # Map images are synthesized top-down views, not real camera frames,
            # so pose / uav_adapter embed enhancements are not meaningful
            # and must stay disabled to match the training-time constraint.
            if getattr(self.args, 'use_pose_embed', False):
                raise ValueError("SwiftVLN memory_method=map requires use_pose_embed=false.")
            if getattr(self.args, 'use_uav_adapter', False):
                raise ValueError("SwiftVLN memory_method=map requires use_uav_adapter=false.")
            # Derive cache dir from DATA_PATH so any dataset root name, such as
            # SatNav-v0.1 or a custom abcd directory, maps to {root}/map_cache.
            # SWIFTVLN_MAP_CACHE_DIR overrides this; "off" disables it.
            data_path_tmpl = getattr(self.config.DATASET, 'DATA_PATH', '') or ''
            default_map_cache_dir = _derive_dataset_cache_dir(str(data_path_tmpl))
            self.map_builder = SatNavMapMemoryBuilder(
                scenes_dir=self.config.DATASET.SCENES_DIR,
                global_side_m=self.map_global_side_m,
                local_side_m=self.map_local_side_m,
                render_px=self.map_render_px,
                mask_method=self.map_mask_method,
                hfov=float(self.config.SIMULATOR.RGB_SENSOR.HFOV),
                sensor_width=int(self.config.SIMULATOR.RGB_SENSOR.WIDTH),
                sensor_height=int(self.config.SIMULATOR.RGB_SENSOR.HEIGHT),
                cache_dir=default_map_cache_dir,
            )

        # ==========================================================================
        # Prompt templates (must match dataset.py)
        # ==========================================================================
        self.conjunctions = DEFAULT_CONJUNCTIONS.copy()
        
        # ==========================================================================
        # Token IDs and visual processor config
        # ==========================================================================
        self.history_memory_token_id = self.processor.tokenizer.convert_tokens_to_ids(HISTORY_MEMORY_TOKEN)
        self.current_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        self.merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
        
        # Cache initialization (will be reset per episode)
        self._reset_caches()
        
        # ==========================================================================
        # Embedding enhancement pipeline (auto-detect from model checkpoint)
        # ==========================================================================
        self.has_embed_enhance = (
            hasattr(self.model, 'embed_enhance') and 
            not self.model.embed_enhance.is_empty
        )
        # ==========================================================================
        # Print configuration
        # ==========================================================================
        print(f"[SwiftVLNEvaluator] Initialized:")
        print(f"  Window: num_frames={self.num_frames}, num_overlap={self.num_overlap}, stride={self.stride}")
        print(f"  Turns: per_window={self.turns_per_window}, overlap={self.overlap_turns}")
        print(f"  History Processor: {self.history_processor.name}")
        if self.history_processor_type in ('gtc', 'segment_gtc'):
            print(f"    Sampling: every {self.num_future_steps} frames (same as training)")
        else:
            if self.use_random:
                sampling_type = "random"
            else:
                sampling_type = "uniform" if self.log_base == 1.0 else f"logarithmic (b={self.log_base})"
            print(f"    Sampling: {sampling_type}, {self.num_history} frames")
        print(f"  System Prompt: {self.system_prompt_setting}")
        if self.system_prompt_setting == "initial":
            print(f"    [INITIAL] Initial view ENABLED: first frame (uncompressed) in system prompt")
        print(f"  Memory Method: {self.memory_method}")
        if self.memory_method == 'map':
            print(
                f"    map: global={self.map_global_side_m:.0f}m, "
                f"local={self.map_local_side_m:.0f}m, "
                f"render={self.map_render_px}px, mask={self.map_mask_method}"
            )
        if self.has_embed_enhance:
            print(f"  Embed Enhance: {self.model.embed_enhance} (auto-detected from checkpoint)")
        
        # Debug counter for initial verification
        self._debug_initial_eval_count = 0
        self._debug_map_eval_count = 0
        self._debug_map_prompt_count = 0
        self._debug_map_user_prompt_count = 0
        self._debug_map_embed_count = 0
    
    def _reset_caches(self):
        """Reset all caches at the start of each episode."""
        # Reset random seed for each episode to ensure reproducibility
        self.set_eval_seed()
        
        # VIT feature caches
        self.history_cache: List[Tuple[torch.Tensor, Optional[List[float]]]] = []
        
        # For GTC: cache VIT features by step_id to avoid re-encoding
        # Key: step_id (prediction step), Value: (vit_features, grid_thw, pose)
        self.vit_feature_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, Optional[List[float]]]] = {}
        
        # Initial view features (uncompressed, for initial prompt setting)
        self.initial_features: Optional[torch.Tensor] = None
        
        # Overlap context from previous window
        self.overlap_context: Optional[OverlapContext] = None
        
        # Current window state
        self.window_turns: List[TurnContext] = []  # Turns in current window
        self.window_start_step: int = 0
        self.current_window_idx: int = 0
        self.prediction_step: int = 0  # Track which prediction step we're on (for GTC cache)
        
        # Pose tracking
        self.pose_history: List[List[float]] = []
        self.executed_actions: List[int] = [-1]  # Start with INITIAL action marker
        self._start_state_pos: Optional[List[float]] = None
        self._start_state_heading: Optional[float] = None

    def _wrap_heading_deg(self, angle: float) -> float:
        """Wrap heading angle to [-180, 180)."""
        return (angle + 180.0) % 360.0 - 180.0

    def _extract_numeric_heading(self, rotation: Any) -> Optional[float]:
        """Extract numeric heading in degrees from simulator rotation object."""
        if rotation is None:
            return None
        if isinstance(rotation, (int, float)):
            return float(rotation)
        if hasattr(rotation, 'item'):
            try:
                return float(rotation.item())
            except Exception:
                pass
        return None

    def _get_pose_from_simulator(self, env_wrapper: EnvWrapper, episode: Any) -> Optional[List[float]]:
        """Compute current ego-frame pose from simulator state if available."""
        if not hasattr(env_wrapper, 'get_agent_state'):
            return None

        try:
            state = env_wrapper.get_agent_state()
            pos = getattr(state, 'position', None)
            if pos is None:
                return None
            pos = [float(pos[0]), float(pos[1])]

            heading = self._extract_numeric_heading(getattr(state, 'rotation', None))
            if heading is None:
                return None

            if self._start_state_pos is None:
                if hasattr(episode, 'start_position') and episode.start_position is not None:
                    self._start_state_pos = [float(episode.start_position[0]), float(episode.start_position[1])]
                else:
                    self._start_state_pos = [pos[0], pos[1]]
            if self._start_state_heading is None:
                start_rot = getattr(episode, 'start_rotation', None)
                start_heading = self._extract_numeric_heading(start_rot)
                if start_heading is None:
                    start_heading = heading
                self._start_state_heading = float(start_heading)

            dx = pos[0] - self._start_state_pos[0]
            dy = pos[1] - self._start_state_pos[1]
            h0_rad = math_module.radians(self._start_state_heading)
            delta_forward = dx * math_module.cos(h0_rad) + dy * math_module.sin(h0_rad)
            delta_right = -dx * math_module.sin(h0_rad) + dy * math_module.cos(h0_rad)

            delta_heading = self._wrap_heading_deg(float(heading) - self._start_state_heading)
            heading_rad = math_module.radians(delta_heading)
            return [
                float(delta_forward),
                float(delta_right),
                float(math_module.sin(heading_rad)),
                float(math_module.cos(heading_rad)),
            ]
        except Exception:
            return None

    def _get_pose_from_actions(self) -> List[float]:
        """Fallback pose from executed actions (action integral)."""
        step_size = 10.0 if self.env_type == 'satnav' else 0.25
        turn_angle = 15.0 if self.env_type == 'satnav' else 30.0
        poses = reconstruct_pose_from_actions(
            self.executed_actions,
            step_size=step_size,
            turn_angle=turn_angle,
        )
        if poses.shape[0] == 0:
            return [0.0, 0.0, 0.0, 1.0]
        return poses[-1].tolist()

    def _get_current_pose(self, env_wrapper: EnvWrapper, episode: Any) -> List[float]:
        """Get current pose vector from action integral (consistent with training).

        We always use action integration here so that the pose representation
        matches exactly what ``reconstruct_pose_from_actions`` produces during
        training.  ``_get_pose_from_simulator`` is kept for future use once
        proper coordinate conversion (lonlat_to_ego_displacement) is integrated.
        """
        return self._get_pose_from_actions()
    
    def _resize_image(self, image: Image.Image, resize_stride: float = 1.0) -> Image.Image:
        """Resize image (optional, for efficiency)."""
        if resize_stride <= 1.0:
            return image
        
        orig_w, orig_h = image.size
        new_w = int(orig_w / resize_stride)
        new_h = int(orig_h / resize_stride)
        new_w = max(new_w, 28)
        new_h = max(new_h, 28)
        
        return image.resize((new_w, new_h), Image.BILINEAR)

    @staticmethod
    def _extract_visual_features(visual_res: Any) -> torch.Tensor:
        """Normalize Qwen-family visual outputs to pooled visual tokens."""
        if hasattr(visual_res, 'pooler_output'):
            return visual_res.pooler_output
        if isinstance(visual_res, tuple):
            return visual_res[0]
        return visual_res
    
    def _encode_frame(
        self,
        image: Image.Image,
        pose: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single frame through VIT.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (vit_features [num_tokens, hidden], grid_thw [3])
        """
        media_inputs = self.processor.image_processor(
            images=[image], return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        with torch.no_grad():
            vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )
            
            # Apply embedding enhancement pipeline if enabled (auto-detected from checkpoint)
            if self.has_embed_enhance:
                t, h, w = image_grid_thw[0].tolist()
                h_m, w_m = int(h) // self.merge_size, int(w) // self.merge_size
                vit_features = self.model.embed_enhance(vit_features, h_m, w_m, pose=pose)
        
        return vit_features, image_grid_thw[0]
    
    def _encode_batch_frames(
        self,
        images: List[Image.Image],
        poses: Optional[List[Optional[List[float]]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Batch encode multiple frames through VIT.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tuple of (list of vit_features, list of grid_thw)
        """
        if not images:
            return [], []
        
        media_inputs = self.processor.image_processor(
            images=images, return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        with torch.no_grad():
            all_vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )
        
        # Split features by image
        merge_length = self.merge_size ** 2
        features_list = []
        grid_thw_list = []
        embed_idx = 0
        
        for i in range(len(images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            img_features = all_vit_features[embed_idx:embed_idx + num_tokens]
            embed_idx += num_tokens
            
            # Apply embedding enhancement pipeline if enabled (auto-detected from checkpoint)
            if self.has_embed_enhance:
                t, h, w = image_grid_thw[i].tolist()
                h_m, w_m = int(h) // self.merge_size, int(w) // self.merge_size
                pose_i = None
                if poses is not None and i < len(poses):
                    pose_i = poses[i]
                with torch.no_grad():
                    img_features = self.model.embed_enhance(img_features, h_m, w_m, pose=pose_i)
            
            features_list.append(img_features)
            grid_thw_list.append(image_grid_thw[i])
        
        return features_list, grid_thw_list
    
    def _compress_features(self, features: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compress VIT features using configured method (pooling or ToMe).
        
        Args:
            features: [num_tokens, hidden_size]
            grid_thw: [3] tensor (t, h, w) BEFORE merge
            
        Returns:
            Compressed features
        """
        # Get grid after spatial merge
        grid_after_merge = grid_thw.clone()
        grid_after_merge[1] = grid_after_merge[1] // self.merge_size
        grid_after_merge[2] = grid_after_merge[2] // self.merge_size
        
        # Use the compress method which handles both pooling and ToMe
        compressed, _ = self.compressor.compress(
            features, grid_after_merge, stride=self.compress_stride
        )
        return compressed
    
    def _compute_history_cache(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
        episode: Any,
    ):
        """
        Compute history features based on history_processor_type.
        
        For per_frame:
            - Sample num_history frames using the shared train/eval helper
            - use_random=True: Uniform random sampling without replacement
            - log_base=1.0: Uniform sampling
            - log_base>1.0: Logarithmic sampling (more recent frames)
            - Compress each frame independently
            
        For GTC:
            - Sample with num_future_steps interval from [0, window_start)
            - Use cached VIT features when available (from previous predictions)
            - Cluster all features into fixed output_tokens
        
        Args:
            rgb_list: All collected RGB images so far
            window_start: Start index of current window
        """
        if self.memory_method == 'map':
            self._compute_history_cache_map(episode, window_start)
            return

        if window_start <= 0:
            self.history_cache = []
            return

        if self.history_processor_type in ('gtc', 'segment_gtc'):
            self._compute_history_cache_gtc(rgb_list, pose_list, window_start)
        else:
            self._compute_history_cache_per_frame(rgb_list, pose_list, window_start)

    def _compute_history_cache_map(
        self,
        episode: Any,
        window_start: int,
    ):
        if self.map_builder is None:
            raise RuntimeError("memory_method=map requested but map_builder is not initialized.")
        scene_id = getattr(episode, 'scene_id', None)
        start_position = getattr(episode, 'start_position', None)
        start_rotation = getattr(episode, 'start_rotation', None)
        if scene_id is None or start_position is None or start_rotation is None:
            raise ValueError("SatNav episode is missing scene/start metadata required for map memory.")

        step_size = 10.0 if self.env_type == 'satnav' else 0.25
        turn_angle = 15.0 if self.env_type == 'satnav' else 30.0
        map_images = self.map_builder.render_from_actions(
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=float(start_rotation),
            actions=self.executed_actions,
            window_start=window_start,
            step_size=step_size,
            turn_angle=turn_angle,
        )

        features_list, grid_thw_list = self._encode_batch_frames(
            map_images,
            poses=[None] * len(map_images),
        )
        self.history_cache = []
        for features, grid_thw in zip(features_list, grid_thw_list):
            compressed = self._compress_features(features, grid_thw)
            self.history_cache.append((compressed, None))
        if _debug_enabled() and self._debug_map_eval_count < 6:
            raw_token_counts = [int(features.shape[0]) for features in features_list]
            token_counts = [int(item[0].shape[0]) for item in self.history_cache]
            map_labels = ['global', 'local']
            print(
                f"[MAP DEBUG][eval] cache[{self._debug_map_eval_count}] "
                f"rank={_debug_rank()} "
                f"episode={getattr(episode, 'episode_id', 'unknown')} "
                f"window_start={window_start} executed_actions={len(self.executed_actions)} "
                f"map_images={len(map_images)} raw_tokens={raw_token_counts} "
                f"compressed_tokens={token_counts}"
            )
            for idx, (features, cache_item) in enumerate(zip(features_list, self.history_cache)):
                label = map_labels[idx] if idx < len(map_labels) else f"map{idx}"
                print(
                    f"[MAP DEBUG][eval] cache[{self._debug_map_eval_count}].{label} "
                    f"raw={_tensor_debug_stats(features)} "
                    f"compressed={_tensor_debug_stats(cache_item[0])}"
                )
            self._debug_map_eval_count += 1

    def _compute_history_cache_per_frame(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
    ):
        """
        Compute history cache using per-frame compression.
        
        Sampling strategy (aligned with training in dataset.py):
            Uses the same helper as training:
            - use_random=True: uniform random sampling without replacement
            - log_base=1.0: Uniform/linear sampling
            - log_base>1.0: Logarithmic sampling (more recent frames)

        Each frame is compressed independently using pooling or ToMe.
        """
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
        
        # Get history images
        history_images = [rgb_list[i] for i in history_indices]
        history_poses = [pose_list[i] if i < len(pose_list) else None for i in history_indices]
        
        # Batch encode
        features_list, grid_thw_list = self._encode_batch_frames(history_images, poses=history_poses)
        
        # Compress each frame
        self.history_cache = []
        for features, grid_thw, pose in zip(features_list, grid_thw_list, history_poses):
            compressed = self._compress_features(features, grid_thw)
            self.history_cache.append((compressed, pose))

    def _compute_history_cache_gtc(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        window_start: int,
    ):
        """
        Compute history cache using Global Token Clustering.
        
        Key insight: 
            In evaluation, predictions happen every num_future_steps.
            The vit_feature_cache already contains features for steps:
                [0, num_future_steps, 2*num_future_steps, ...]
            
            This exactly matches the training sampling strategy:
                history_step_ids = np.arange(0, current_start_abs, self.num_future_steps)
            
        So NO additional sampling is needed - just use all cached features
        where step_id < window_start.
        """
        # Collect all cached features for steps before window_start
        # These are exactly the history frames that training would sample
        history_features = []
        
        for step_id in sorted(self.vit_feature_cache.keys()):
            if step_id < window_start:
                features, grid_thw, pose = self.vit_feature_cache[step_id]
                history_features.append(features)
        
        if not history_features:
            self.history_cache = []
            return
        
        # Process all history features through GTC
        clustered = self.history_processor.process(
            frame_embeds_list=history_features,
            frame_grid_thws=[],  # Not used by GTC
        )
        
        # GTC returns a single clustered tensor, wrap in list for consistency
        self.history_cache = [(clustered, None)]
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from input_ids."""
        base_model = self.model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            return base_model.model.embed_tokens(input_ids)
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
            return base_model.model.language_model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embed_tokens in model")
    
    def _build_system_prompt_ids(
        self, 
        instruction: str, 
        history_token_counts: List[int],
        initial_token_count: int = 0,
    ) -> torch.Tensor:
        """
        Build and tokenize system prompt with unified memory token and optional first view.
        
        This method constructs the same format as dataset.py's system prompt:
        - First view: Uncompressed image at the start (optional, via system_prompt_setting)
        - History observations: Single unified <history_memory> block (no individual frames)
        
        Args:
            instruction: Navigation instruction
            history_token_counts: List of token counts for each history image (after compression)
            initial_token_count: Number of tokens for initial view image (0 if disabled)
            
        Returns:
            Token IDs for system prompt (including first view and history memory tokens)
        """
        # Select prompt template based on environment type
        if self.env_type == "satnav":
            template = PROMPT_TEMPLATE_SATNAV
        else:  # habitat or default
            template = PROMPT_TEMPLATE_HABITAT
        
        system_prompt = template.format(instruction=instruction)
        
        # Add initial view if enabled (must match dataset.py order: initial before history)
        if initial_token_count > 0:
            initial_tokens = CURRENT_IMAGE_TOKEN * initial_token_count
            initial_str = f'<|vision_start|>{initial_tokens}<|vision_end|>'
            system_prompt += f" This is your initial observation at the starting point of this journey: {initial_str}."
        
        if history_token_counts:
            # Unified memory mode: single <history_memory> block with total token count
            total_history_tokens = sum(history_token_counts)
            # All history embeddings are packed into a single unified block
            history_str = f'<|vision_start|>{HISTORY_MEMORY_TOKEN * total_history_tokens}<|vision_end|>'
            if self.memory_method == 'map':
                system_prompt += f" These are your explored map memories: {history_str}."
            else:
                system_prompt += f" These are your historical observations: {history_str}."
        
        # Format as system message
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Tokenize without generation prompt
        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors='pt',
            return_dict=True
        )

        if _debug_enabled() and self.memory_method == 'map' and self._debug_map_prompt_count < 6:
            token_ids = inputs['input_ids']
            history_positions = (token_ids[0] == self.history_memory_token_id).nonzero(as_tuple=True)[0]
            current_positions = (token_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            print(
                f"[MAP DEBUG][eval.prompt.system] rank={_debug_rank()} "
                f"history_token_counts={history_token_counts} initial_token_count={initial_token_count} "
                f"tokenized_len={token_ids.shape[1]} "
                f"history_positions={len(history_positions)} current_positions={len(current_positions)}"
            )
            print(
                f"[MAP DEBUG][eval.prompt.system] instruction={_preview_text(instruction, limit=180)}"
            )
            print(
                f"[MAP DEBUG][eval.prompt.system] prompt={_preview_text(system_prompt, limit=340)}"
            )
            self._debug_map_prompt_count += 1
        
        return inputs['input_ids'].to(self.device)
    
    def _build_user_turn_ids(
        self, 
        conjunction: str, 
        current_token_count: int,
        add_generation_prompt: bool = False
    ) -> torch.Tensor:
        """
        Build and tokenize a user turn.
        
        This method directly constructs the same format that template.py's replace_tag
        produces during training:
        - Training: dataset outputs "<image>" → replace_tag returns "<|vision_start|><current_image><|vision_end|>"
        - Inference: this method directly builds "<|vision_start|><current_image>...<|vision_end|>"
        
        This ensures prompt format consistency between training and inference,
        while enabling ROPE position encoding support for current observations.
        
        Args:
            conjunction: Conjunction phrase
            current_token_count: Number of tokens for current image
            add_generation_prompt: Whether to add generation prompt at the end
                                   (True for the final user turn before generation,
                                    False for completed turns in history)
            
        Returns:
            Token IDs for user turn
        """
        # Construct the same format as template.py's replace_tag output
        # Multiple <current_image> tokens based on actual VIT output token count
        current_tokens = CURRENT_IMAGE_TOKEN * current_token_count
        content = f"{conjunction}<|vision_start|>{current_tokens}<|vision_end|>."
        messages = [{'role': 'user', 'content': content}]
        
        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors='pt',
            return_dict=True
        )

        if _debug_enabled() and self.memory_method == 'map' and self._debug_map_user_prompt_count < 6:
            token_ids = inputs['input_ids']
            current_positions = (token_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            print(
                f"[MAP DEBUG][eval.prompt.user] rank={_debug_rank()} "
                f"generation_prompt={add_generation_prompt} "
                f"current_token_count={current_token_count} tokenized_len={token_ids.shape[1]} "
                f"current_positions={len(current_positions)}"
            )
            print(
                f"[MAP DEBUG][eval.prompt.user] content={_preview_text(content, limit=240)}"
            )
            self._debug_map_user_prompt_count += 1
        
        return inputs['input_ids'].to(self.device)
    
    def _build_assistant_turn_ids(self, response: str) -> torch.Tensor:
        """
        Build and tokenize an assistant turn.
        
        Args:
            response: Assistant's response text
            
        Returns:
            Token IDs for assistant turn
        """
        # Get just the assistant response tokens
        # We need to be careful to get only the assistant tokens, not the whole message
        messages = [{'role': 'assistant', 'content': response}]
        
        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors='pt',
            return_dict=True
        )
        
        return inputs['input_ids'].to(self.device)
    
    def _build_complete_prompt_embeds(
        self,
        instruction: str,
        current_image: Image.Image,
        conjunction: str,
        current_pose: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Build complete prompt embeddings for generation.
        
        This combines:
        1. System prompt with history images
        2. Overlap context from previous window (if any)
        3. Completed turns in current window
        4. New user turn with current image
        
        Args:
            instruction: Navigation instruction
            current_image: Current observation image
            conjunction: Conjunction for new user turn
            
        Returns:
            Tuple of (inputs_embeds, seq_len, current_vit_features, current_grid_thw)
        """
        # 1. Encode current image
        current_vit_features, current_grid_thw = self._encode_frame(current_image, pose=current_pose)
        current_token_count = current_vit_features.shape[0]
        
        # 2. Build system prompt with correct history token counts and optional initial view
        history_token_counts = [h[0].shape[0] for h in self.history_cache]
        initial_token_count = self.initial_features.shape[0] if self.initial_features is not None else 0
        system_ids = self._build_system_prompt_ids(
            instruction, history_token_counts,
            initial_token_count=initial_token_count,
        )
        system_embeds = self._get_text_embeddings(system_ids)
        
        # 3a. Replace initial view tokens with uncompressed features (if enabled)
        # Initial view uses <current_image> tokens in the system prompt
        # These positions come BEFORE any <current_image> tokens in user turns
        if self.initial_features is not None:
            init_positions = (system_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            init_count = self.initial_features.shape[0]
            if len(init_positions) >= init_count:
                init_positions_slice = init_positions[:init_count]
                init_embeds = self.initial_features.to(system_embeds.device, system_embeds.dtype)
                system_embeds[0, init_positions_slice] = init_embeds
                
                if os.environ.get('SWIFTVLN_DEBUG') and self._debug_initial_eval_count < 3:
                    print(f"[INITIAL DEBUG] _build_complete_prompt_embeds: "
                          f"Injected {init_count} initial tokens at positions "
                          f"[{init_positions_slice[0].item()}..{init_positions_slice[-1].item()}] "
                          f"in system prompt (total system_ids len={system_ids.shape[1]})")
            else:
                print(f"[INITIAL WARNING] Not enough <current_image> positions in system prompt! "
                      f"Found {len(init_positions)}, need {init_count}")
        
        # 3b. Replace unified history memory tokens with cached features
        history_injection_ok = True
        history_injection_msg = "no_history"
        history_embeds = None
        if self.history_cache:
            # Use history_memory_token_id for unified memory mode
            history_positions = (system_ids[0] == self.history_memory_token_id).nonzero(as_tuple=True)[0]
            history_embeds = torch.cat([h[0] for h in self.history_cache], dim=0)
            total_history_tokens = history_embeds.shape[0]
            
            if len(history_positions) >= total_history_tokens:
                history_positions_slice = history_positions[:total_history_tokens]
                history_embeds = history_embeds.to(system_embeds.device, system_embeds.dtype)
                system_embeds[0, history_positions_slice] = history_embeds
                history_injection_msg = (
                    f"ok positions=[{history_positions_slice[0].item()}.."
                    f"{history_positions_slice[-1].item()}]"
                )
            else:
                history_injection_ok = False
                history_injection_msg = (
                    f"failed positions={len(history_positions)} need={total_history_tokens}"
                )
                print(
                    f"[MAP WARNING][eval.embed] system history injection skipped: "
                    f"rank={_debug_rank()} {history_injection_msg}"
                )
        
        embeds_parts = [system_embeds]
        
        # 4. Add overlap context from the previous window
        if self.overlap_context is not None and self.overlap_context.input_ids is not None:
            overlap_embeds = self._get_text_embeddings(self.overlap_context.input_ids)
            
            # Replace current_image tokens with cached embeddings using index assignment
            if self.overlap_context.image_embeds:
                overlap_positions = (self.overlap_context.input_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
                overlap_img_embeds = torch.cat(self.overlap_context.image_embeds, dim=0)
                total_overlap_tokens = overlap_img_embeds.shape[0]
                
                if len(overlap_positions) >= total_overlap_tokens:
                    overlap_positions_slice = overlap_positions[:total_overlap_tokens]
                    overlap_img_embeds = overlap_img_embeds.to(overlap_embeds.device, overlap_embeds.dtype)
                    overlap_embeds[0, overlap_positions_slice] = overlap_img_embeds
            
            embeds_parts.append(overlap_embeds)
        
        # 5. Add completed turns in current window
        for turn in self.window_turns:
            turn_user_embeds = self._get_text_embeddings(turn.user_input_ids)
            # Replace image tokens using index assignment
            turn_positions = (turn.user_input_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            turn_token_count = turn.image_embed.shape[0]
            
            if len(turn_positions) >= turn_token_count:
                turn_positions_slice = turn_positions[:turn_token_count]
                turn_img = turn.image_embed.to(turn_user_embeds.device, turn_user_embeds.dtype)
                turn_user_embeds[0, turn_positions_slice] = turn_img
            
            embeds_parts.append(turn_user_embeds)
            
            # Add assistant response
            assistant_ids = self._build_assistant_turn_ids(turn.assistant_response)
            assistant_embeds = self._get_text_embeddings(assistant_ids)
            embeds_parts.append(assistant_embeds)
        
        # 6. Add new user turn (with generation prompt for model to generate response)
        new_user_ids = self._build_user_turn_ids(conjunction, current_token_count, add_generation_prompt=True)
        new_user_embeds = self._get_text_embeddings(new_user_ids)
        
        # Replace image tokens with current VIT features using index assignment
        cur_positions = (new_user_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
        
        if len(cur_positions) >= current_token_count:
            cur_positions_slice = cur_positions[:current_token_count]
            current_vit = current_vit_features.to(new_user_embeds.device, new_user_embeds.dtype)
            new_user_embeds[0, cur_positions_slice] = current_vit
        
        embeds_parts.append(new_user_embeds)
        
        # Concatenate all parts
        inputs_embeds = torch.cat(embeds_parts, dim=1)
        seq_len = inputs_embeds.shape[1]

        if _debug_enabled() and self.memory_method == 'map' and self._debug_map_embed_count < 6:
            overlap_len = (
                int(self.overlap_context.input_ids.shape[1])
                if self.overlap_context is not None and self.overlap_context.input_ids is not None
                else 0
            )
            completed_turn_count = len(self.window_turns)
            completed_turn_token_count = int(sum(turn.user_input_ids.shape[1] for turn in self.window_turns))
            history_positions = (system_ids[0] == self.history_memory_token_id).nonzero(as_tuple=True)[0]
            current_positions_system = (system_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            current_positions_new = (new_user_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            print(
                f"[MAP DEBUG][eval.embed] rank={_debug_rank()} "
                f"history_cache_tokens={history_token_counts} initial_tokens={initial_token_count} "
                f"system_len={system_ids.shape[1]} overlap_len={overlap_len} "
                f"completed_turns={completed_turn_count} completed_user_tokens={completed_turn_token_count} "
                f"new_user_len={new_user_ids.shape[1]} seq_len={seq_len} "
                f"history_injection={history_injection_msg}"
            )
            print(
                f"[MAP DEBUG][eval.embed] system_history_positions={len(history_positions)} "
                f"system_current_positions={len(current_positions_system)} "
                f"new_user_current_positions={len(current_positions_new)} "
                f"current_vit_tokens={current_token_count}"
            )
            if history_embeds is not None:
                print(
                    f"[MAP DEBUG][eval.embed] history_embed_stats={_tensor_debug_stats(history_embeds)} "
                    f"injection_ok={history_injection_ok}"
                )
            print(
                f"[MAP DEBUG][eval.embed] current_vit_stats={_tensor_debug_stats(current_vit_features)}"
            )
            self._debug_map_embed_count += 1
        
        return inputs_embeds, seq_len, current_vit_features, current_grid_thw
    
    def _save_turn_to_window(
        self,
        conjunction: str,
        response: str,
        vit_features: torch.Tensor,
    ):
        """
        Save a completed turn to the window context.
        
        Args:
            conjunction: Conjunction used
            response: Assistant's response
            vit_features: VIT features for this turn's image
        """
        token_count = vit_features.shape[0]
        user_ids = self._build_user_turn_ids(conjunction, token_count)
        
        turn = TurnContext(
            user_input_ids=user_ids,
            assistant_response=response,
            image_embed=vit_features,
        )
        self.window_turns.append(turn)
    
    def _prepare_overlap_context(self):
        """
        Prepare overlap context from the last overlap_turns of current window.
        
        This will be used as context for the next window.
        """
        if self.overlap_turns <= 0:
            self.overlap_context = None
            return

        if len(self.window_turns) < self.overlap_turns:
            self.overlap_context = None
            return
        
        # Get last overlap_turns
        overlap_turns = self.window_turns[-self.overlap_turns:]
        
        # Build input_ids for these turns
        ids_parts = []
        image_embeds = []
        
        for turn in overlap_turns:
            ids_parts.append(turn.user_input_ids)
            assistant_ids = self._build_assistant_turn_ids(turn.assistant_response)
            ids_parts.append(assistant_ids)
            image_embeds.append(turn.image_embed)
        
        overlap_ids = torch.cat(ids_parts, dim=1)
        
        self.overlap_context = OverlapContext(
            input_ids=overlap_ids,
            image_embeds=image_embeds,
        )
    
    def _slide_window(
        self,
        rgb_list: List[Image.Image],
        pose_list: List[List[float]],
        new_window_start: int,
        episode: Any,
    ):
        """
        Slide the window and update caches.
        
        Args:
            rgb_list: All collected RGB images
            new_window_start: Start step of new window
        """
        # 1. Prepare overlap context from current window
        self._prepare_overlap_context()
        
        # 2. Recompute history cache for new window
        self._compute_history_cache(rgb_list, pose_list, new_window_start, episode)
        if os.environ.get('SWIFTVLN_DEBUG') and self.memory_method == 'map':
            print(
                f"[MAP DEBUG][eval] slide_window -> new_window_start={new_window_start}, "
                f"history_cache={len(self.history_cache)}, overlap_turns={self.overlap_turns}"
            )

        # 3. Reset window state
        self.window_turns = []
        self.window_start_step = new_window_start
        self.current_window_idx += 1
    

    # =========================================================================
    # Debug Landmark Analysis
    # =========================================================================
    def _is_debug_landmark_enabled(self):
        """Check if debug_landmark mode is enabled."""
        return getattr(self.args, 'debug_landmark', False)
    
    def _init_debug_state(self, episode, instruction):
        """Initialize debug tracking state for an episode."""
        debug_state = {
            'episode_id': str(episode.episode_id),
            'scene_id': getattr(episode, 'scene_id', 'unknown'),
            'trajectory_type': getattr(episode, 'trajectory_type', 'unknown'),
            'instruction': instruction,
            'start_position': list(episode.start_position) if hasattr(episode, 'start_position') else None,
            'start_rotation': getattr(episode, 'start_rotation', None),
            'goal_position': list(episode.goals[0].position) if hasattr(episode, 'goals') and episode.goals else None,
            'reference_path': [list(p) for p in episode.reference_path] if hasattr(episode, 'reference_path') else None,
            'waypoints': [list(w) for w in episode.waypoints] if hasattr(episode, 'waypoints') else None,
            'aux_info': episode.aux_info if hasattr(episode, 'aux_info') else {},
            'steps': [],
            'model_outputs': [],
            'action_sequences': [],
        }
        return debug_state
    
    def _debug_log_step(self, debug_state, step_id, action, env_wrapper, 
                         output_text=None, action_seq_snapshot=None):
        """Log debug info for a single step."""
        try:
            agent_state = env_wrapper.get_agent_state()
            position = list(agent_state.position) if hasattr(agent_state, 'position') else None
            rotation = agent_state.rotation if hasattr(agent_state, 'rotation') else None
            
            # Get current distance to goal
            metrics = env_wrapper.get_metrics()
            distance_to_goal = metrics.get('distance_to_goal', None)
            
            action_names = {0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
            step_info = {
                'step_id': step_id,
                'action': action_names.get(action, f'UNKNOWN({action})'),
                'action_idx': action,
                'position': position,
                'rotation': float(rotation) if rotation is not None else None,
                'distance_to_goal': float(distance_to_goal) if distance_to_goal is not None else None,
            }
            debug_state['steps'].append(step_info)
        except Exception as e:
            debug_state['steps'].append({
                'step_id': step_id,
                'action_idx': action,
                'error': str(e)
            })
    
    def _debug_log_model_output(self, debug_state, step_id, output_text, action_seq):
        """Log model output and parsed actions."""
        action_names = {0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
        debug_state['model_outputs'].append({
            'at_step': step_id,
            'raw_output': output_text,
            'parsed_actions': [action_names.get(a, f'?{a}') for a in action_seq],
            'num_actions': len(action_seq),
        })
    
    def _debug_save_frames(self, debug_state, rgb_list, episode_id):
        """Save key RGB frames for debug analysis."""
        debug_dir = os.path.join(self.output_path, 'debug_landmark', str(episode_id))
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save frames: first, every 4th, and last
        save_indices = set([0])  # First frame
        save_indices.add(len(rgb_list) - 1)  # Last frame
        for i in range(0, len(rgb_list), 4):  # Every 4th
            save_indices.add(i)
        
        for idx in sorted(save_indices):
            if idx < len(rgb_list):
                frame_path = os.path.join(debug_dir, f'frame_{idx:04d}.jpg')
                rgb_list[idx].save(frame_path, quality=85)
        
        return debug_dir
    
    def _debug_analyze_trajectory(self, debug_state):
        """Analyze the trajectory for common failure patterns."""
        analysis = {}
        steps = debug_state['steps']
        if not steps:
            return analysis
        
        # Count action types
        action_counts = {}
        for s in steps:
            a = s.get('action', 'UNKNOWN')
            action_counts[a] = action_counts.get(a, 0) + 1
        analysis['action_counts'] = action_counts
        analysis['total_steps'] = len(steps)
        
        # Distance progression
        distances = [s.get('distance_to_goal') for s in steps if s.get('distance_to_goal') is not None]
        if distances:
            analysis['initial_distance'] = distances[0]
            analysis['final_distance'] = distances[-1]
            analysis['min_distance'] = min(distances)
            analysis['min_distance_step'] = distances.index(min(distances))
            analysis['distance_reduced'] = distances[0] - distances[-1]
            
            # Did agent ever get closer than 50m?
            analysis['ever_within_50m'] = min(distances) < 50
            analysis['ever_within_100m'] = min(distances) < 100
        
        # Detect turning behavior
        turn_segments = []
        current_segment = {'type': None, 'count': 0, 'start_step': 0}
        for i, s in enumerate(steps):
            action = s.get('action', '')
            if 'TURN' in action:
                if current_segment['type'] == 'turn':
                    current_segment['count'] += 1
                else:
                    if current_segment['type'] is not None:
                        turn_segments.append(dict(current_segment))
                    current_segment = {'type': 'turn', 'count': 1, 'start_step': i, 'direction': action}
            elif action == 'FORWARD':
                if current_segment['type'] == 'forward':
                    current_segment['count'] += 1
                else:
                    if current_segment['type'] is not None:
                        turn_segments.append(dict(current_segment))
                    current_segment = {'type': 'forward', 'count': 1, 'start_step': i}
            elif action == 'STOP':
                if current_segment['type'] is not None:
                    turn_segments.append(dict(current_segment))
                turn_segments.append({'type': 'stop', 'count': 1, 'start_step': i})
                current_segment = {'type': None, 'count': 0, 'start_step': i}
        if current_segment['type'] is not None:
            turn_segments.append(dict(current_segment))
        
        analysis['trajectory_segments'] = turn_segments
        
        # Compute rotation changes
        rotations = [s.get('rotation') for s in steps if s.get('rotation') is not None]
        if len(rotations) >= 2:
            total_rotation_change = 0
            for i in range(1, len(rotations)):
                diff = rotations[i] - rotations[i-1]
                # Normalize to [-180, 180]
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                total_rotation_change += diff
            analysis['total_rotation_change'] = total_rotation_change
            analysis['initial_rotation'] = rotations[0]
            analysis['final_rotation'] = rotations[-1]
        
        # Compare with reference path
        if debug_state.get('reference_path'):
            ref_path = debug_state['reference_path']
            analysis['reference_path_length'] = len(ref_path)
        if debug_state.get('waypoints'):
            analysis['num_waypoints'] = len(debug_state['waypoints'])
        
        # Early stop detection
        if len(steps) < 30 and analysis.get('final_distance', 0) > 50:
            analysis['likely_issue'] = 'EARLY_STOP'
        elif analysis.get('distance_reduced', 0) < 0:
            analysis['likely_issue'] = 'MOVING_AWAY_FROM_GOAL'
        elif action_counts.get('FORWARD', 0) < 5 and len(steps) > 10:
            analysis['likely_issue'] = 'TOO_MUCH_TURNING'
        else:
            analysis['likely_issue'] = 'DIRECTION_ERROR'
        
        return analysis
    
    def _debug_save_report(self, debug_state, debug_dir):
        """Save comprehensive debug report."""
        # Add trajectory analysis
        debug_state['trajectory_analysis'] = self._debug_analyze_trajectory(debug_state)
        
        report_path = os.path.join(debug_dir, 'debug_report.json')
        with open(report_path, 'w') as f:
            json_module.dump(debug_state, f, indent=2, ensure_ascii=False, default=str)
        
        # Also print summary
        analysis = debug_state['trajectory_analysis']
        print(f"\n{'='*70}")
        print(f"[DEBUG LANDMARK] Episode {debug_state['episode_id']} "
              f"({debug_state.get('trajectory_type', '?')})")
        print(f"{'='*70}")
        print(f"  Instruction: {debug_state['instruction'][:200]}")
        print(f"  Total steps: {analysis.get('total_steps', '?')}")
        print(f"  Actions: {analysis.get('action_counts', {})}")
        print(f"  Distance: initial={analysis.get('initial_distance', '?'):.1f}m "
              f"-> final={analysis.get('final_distance', '?'):.1f}m "
              f"(min={analysis.get('min_distance', '?'):.1f}m at step {analysis.get('min_distance_step', '?')})")
        if 'total_rotation_change' in analysis:
            print(f"  Rotation: {analysis.get('initial_rotation', 0):.1f} -> "
                  f"{analysis.get('final_rotation', 0):.1f} "
                  f"(total change: {analysis.get('total_rotation_change', 0):.1f} deg)")
        print(f"  Segments: {analysis.get('trajectory_segments', [])}")
        print(f"  Likely issue: {analysis.get('likely_issue', 'UNKNOWN')}")
        
        # Print model outputs
        print(f"  Model outputs:")
        for mo in debug_state.get('model_outputs', []):
            print(f"    Step {mo['at_step']}: {mo['raw_output'][:80]} -> {mo['parsed_actions']}")
        print(f"  Report saved to: {report_path}")
        print(f"{'='*70}\n")

    @torch.no_grad()
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode using VIT caching and overlap context.
        
        Flow:
        1. For each step, check if we need a new action prediction
        2. If action queue is empty:
           a. Check if we need to slide window
           b. Build complete prompt with cached features
           c. Generate assistant response
           d. Parse actions
           e. Update caches
        3. Execute action from queue
        
        Args:
            env_wrapper: Environment wrapper
            episode: Episode to evaluate
            env_idx: Environment index
            
        Returns:
            Evaluation metrics dictionary
        """
        # Initialize timing
        timing_stats = {
            'init': 0.0,
            'collect_observation': 0.0,
            'encode_current': 0.0,
            'build_embeds': 0.0,
            'model_generate': 0.0,
            'decode': 0.0,
            'parse_actions': 0.0,
            'update_cache': 0.0,
            'window_slide': 0.0,
            'visualization': 0.0,
            'env_step': 0.0,
            'satnav_topdown': 0.0,
            'trajectory_record': 0.0,
            'error_analysis': 0.0,
            'video_save': 0.0,
        }
        episode_start_time = time.time()
        
        # Initialization
        init_start = time.time()
        self.model.eval()
        self._reset_caches()
        
        observations = env_wrapper.reset(episode)
        instruction = env_wrapper.get_instruction(episode)
        episode_id = episode.episode_id
        timing_stats['init'] = time.time() - init_start
        
        if hasattr(episode, 'scene_id'):
            scene_id = episode.scene_id.split('/')[-2] if '/' in episode.scene_id else episode.scene_id
        else:
            scene_id = "unknown"
        
        rgb_list = []
        action_seq = []
        step_id = 0
        
        trajectory_recorder = TrajectoryRecorder()
        vis_frames = []
        rgb_frames = []
        topdown_frames = []
        
        max_steps = env_wrapper.max_steps
        
        # Track exception for graceful handling after video save
        episode_exception = None
        
        # Debug landmark tracking
        debug_state = None
        if self._is_debug_landmark_enabled():
            debug_state = self._init_debug_state(episode, instruction)
        
        try:
            while not env_wrapper.episode_over and step_id < max_steps:
                # 1. Collect current observation
                t0 = time.time()
                rgb = env_wrapper.get_rgb(observations)
                current_img = Image.fromarray(rgb).convert('RGB')
                rgb_list.append(current_img)
                current_pose = self._get_current_pose(env_wrapper, episode)
                self.pose_history.append(current_pose)
                timing_stats['collect_observation'] += time.time() - t0
                
                # 2. If action queue is empty, predict new actions
                if len(action_seq) == 0:
                    # Encode initial view image at step 0 (if initial prompt is enabled)
                    if step_id == 0 and self.system_prompt_setting == "initial":
                        self.initial_features, _ = self._encode_frame(current_img, pose=current_pose)
                        if os.environ.get('SWIFTVLN_DEBUG') and self._debug_initial_eval_count < 3:
                            print(f"[INITIAL DEBUG] Episode {episode_id} step=0: "
                                  f"Encoded initial frame -> {self.initial_features.shape[0]} tokens (uncompressed)")
                    
                    # Check if we need to slide window
                    if step_id > 0 and step_id % self.stride == 0 and step_id >= self.num_frames:
                        t0 = time.time()
                        new_window_start = step_id - self.num_overlap
                        self._slide_window(rgb_list, self.pose_history, new_window_start, episode)
                        timing_stats['window_slide'] += time.time() - t0
                    elif step_id == 0:
                        # First window
                        self.window_start_step = 0
                        self.current_window_idx = 0
                        if self.memory_method == 'map':
                            self._compute_history_cache(rgb_list, self.pose_history, 0, episode)
                    
                    try:
                        # Build complete prompt embeddings
                        t0 = time.time()
                        conjunction = random.choice(self.conjunctions)
                        inputs_embeds, seq_len, current_vit, current_grid_thw = self._build_complete_prompt_embeds(
                            instruction=instruction,
                            current_image=current_img,
                            conjunction=conjunction,
                            current_pose=current_pose,
                        )
                        timing_stats['build_embeds'] += time.time() - t0
                        
                        # Create attention mask and dummy input_ids
                        attention_mask = torch.ones(
                            (1, seq_len),
                            dtype=torch.long,
                            device=inputs_embeds.device
                        )
                        
                        pad_token_id = self.processor.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = self.processor.tokenizer.eos_token_id
                        dummy_input_ids = torch.full(
                            (1, seq_len),
                            pad_token_id,
                            dtype=torch.long,
                            device=inputs_embeds.device
                        )
                        
                        # Generate
                        t0 = time.time()
                        model_inputs = {
                            'input_ids': dummy_input_ids,
                            'inputs_embeds': inputs_embeds,
                            'attention_mask': attention_mask,
                        }
                        
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=64,
                            do_sample=False,
                            use_cache=True,
                        )
                        timing_stats['model_generate'] += time.time() - t0
                        
                        # Decode
                        t0 = time.time()
                        generated_ids = outputs[0][seq_len:]
                        output_text = self.processor.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True
                        ).strip()
                        timing_stats['decode'] += time.time() - t0
                        
                        # Parse actions
                        t0 = time.time()
                        action_seq = self.parse_actions(output_text)
                        timing_stats['parse_actions'] += time.time() - t0
                        
                        # Debug: log model output
                        if debug_state is not None:
                            self._debug_log_model_output(debug_state, step_id, output_text, list(action_seq))
                        
                        # Update caches
                        t0 = time.time()
                        self._save_turn_to_window(
                            conjunction=conjunction,
                            response=output_text,
                            vit_features=current_vit,
                        )
                        
                        # Cache VIT features for GTC/SegmentGTC (key is the step_id when this prediction was made)
                        if self.history_processor_type in ('gtc', 'segment_gtc'):
                            self.vit_feature_cache[step_id] = (current_vit, current_grid_thw, current_pose)
                        
                        timing_stats['update_cache'] += time.time() - t0
                        
                        if getattr(self.args, 'verbose', False):
                            print(f"Step {step_id}: window={self.current_window_idx}, "
                                  f"turns={len(self.window_turns)}, history={len(self.history_cache)}, "
                                  f"vit_cache={len(self.vit_feature_cache)}, output={output_text}")
                        
                    except Exception as e:
                        print(f"[Warning] Generation failed at step {step_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        action_seq = []
                    
                    if not action_seq:
                        action_seq = [0]  # Default to STOP
                
                # 3. Video frame collection
                if self.save_video:
                    t0 = time.time()
                    if self.env_type == "habitat":
                        frame = self._collect_habitat_frame(observations, instruction, env_wrapper)
                        if frame is not None:
                            vis_frames.append(frame)
                    elif self.env_type == "satnav":
                        rgb_frames.append(rgb.copy())
                    timing_stats['visualization'] += time.time() - t0
                
                # 4. Execute action
                t0 = time.time()
                action = action_seq.pop(0)
                observations, _ = env_wrapper.step(action)
                self.executed_actions.append(int(action))
                timing_stats['env_step'] += time.time() - t0
                step_id += 1
                
                # Debug: log step info
                if debug_state is not None:
                    self._debug_log_step(debug_state, step_id, action, env_wrapper)
                
                # 5. SatNav topdown frame
                if self.save_video and self.env_type == "satnav":
                    t0 = time.time()
                    self._collect_satnav_topdown(
                        env_wrapper, episode, action, step_id,
                        topdown_frames, rgb
                    )
                    timing_stats['satnav_topdown'] += time.time() - t0
                
                # 6. Trajectory recording (Habitat)
                if self.env_type == "habitat":
                    t0 = time.time()
                    try:
                        habitat_env = env_wrapper.env
                        agent_state = habitat_env.sim.get_agent_state()
                        trajectory_recorder.add_step(agent_state.position)
                    except Exception:
                        pass
                    timing_stats['trajectory_record'] += time.time() - t0
            
            metrics = env_wrapper.get_metrics()
            
            # Error analysis (Habitat)
            if self.env_type == "habitat":
                t0 = time.time()
                error_analysis = self._analyze_trajectory_errors(trajectory_recorder, episode, metrics)
                metrics.update(error_analysis)
                timing_stats['error_analysis'] = time.time() - t0
        
        except Exception as e:
            # Store exception info for logging
            episode_exception = e
            # Try to get metrics even on error (contains last known distance_to_goal)
            try:
                metrics = env_wrapper.get_metrics()
            except Exception:
                # Fallback metrics if env_wrapper.get_metrics() also fails
                metrics = {
                    'success': 0.0,
                    'spl': 0.0,
                    'distance_to_goal': float('inf'),
                    'oracle_success': 0.0,
                }
            # Ensure failure metrics
            metrics['success'] = 0.0
            metrics['spl'] = 0.0
            metrics['_error'] = str(e)
        
        finally:
            # Always save video (even on error) for debugging
            if self.save_video:
                t0 = time.time()
                if self.env_type == "habitat":
                    self._save_habitat_video(episode_id, vis_frames, metrics)
                elif self.env_type == "satnav":
                    self._save_satnav_video(episode_id, instruction, rgb_frames, topdown_frames, metrics)
                timing_stats['video_save'] = time.time() - t0
        
        # Calculate total episode time
        total_time = time.time() - episode_start_time
        
        # Print timing statistics if debug_timing flag is set
        if getattr(self.args, 'debug_timing', False):
            print(f"\n{'='*60}")
            print(f"Episode {episode_id} (Scene: {scene_id}) Timing Statistics:")
            print(f"{'='*60}")
            print(f"Total episode time: {total_time:.3f}s")
            print(f"Total steps: {step_id}")
            print(f"Windows used: {self.current_window_idx + 1}")
            print(f"History cache size: {len(self.history_cache)}")
            print(f"\nBreakdown by component:")
            sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1], reverse=True)
            for component, elapsed_time in sorted_stats:
                percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0.0
                print(f"  {component:20s}: {elapsed_time:8.3f}s ({percentage:5.1f}%)")
            print(f"{'='*60}\n")
        
        # Debug: save debug report and frames
        if debug_state is not None:
            try:
                debug_dir = self._debug_save_frames(debug_state, rgb_list, episode_id)
                self._debug_save_report(debug_state, debug_dir)
            except Exception as e:
                print(f"[Warning] Failed to save debug report: {e}")
        
        # Debug: initial strategy episode summary
        if os.environ.get('SWIFTVLN_DEBUG') and self._debug_initial_eval_count < 3:
            has_initial = self.initial_features is not None
            init_tokens = self.initial_features.shape[0] if has_initial else 0
            print(f"[INITIAL DEBUG] Episode {episode_id} summary: "
                  f"system_prompt_setting={self.system_prompt_setting}, "
                  f"initial_features={'YES (' + str(init_tokens) + ' tokens)' if has_initial else 'NO'}, "
                  f"steps={step_id}, windows={self.current_window_idx + 1}")
            self._debug_initial_eval_count += 1
        
        metrics['_timing_stats'] = timing_stats
        metrics['_total_time'] = total_time
        metrics['_step_count'] = step_id
        
        return metrics
