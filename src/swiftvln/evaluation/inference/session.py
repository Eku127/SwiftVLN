# Copyright (c) Alibaba, Inc. and its affiliates.
"""SwiftVLN windowed inference state and prompt construction.

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

Key state:
- history_cache: Compressed VIT features for global history frames (unified block)
- overlap_context: Cached input_ids + image embeddings from configured overlap turns

Window Sliding Strategy:
- Window size: num_frames (default 32 actions = 8 turns)
- Overlap: num_overlap (default 0 for current baseline)
- Stride: num_frames - num_overlap
"""

from dataclasses import dataclass, field
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from swiftvln.backends.specs import EnvironmentSpec
from swiftvln.modeling.constants import (
    CURRENT_IMAGE_TOKEN,
    DEFAULT_CONJUNCTIONS,
    HISTORY_MEMORY_TOKEN,
)
from swiftvln.modeling.history.compressor import HistoryTokenCompressor
from swiftvln.modeling.history import create_history_processor
from swiftvln.modeling.history.per_frame import (
    sample_per_frame_history_indices,
)
from swiftvln.modeling.embeddings import reconstruct_pose_from_actions
from swiftvln.modeling.memory.satnav_map import SatNavMapMemoryBuilder


def _derive_dataset_cache_dir(data_path_tmpl: str) -> Optional[str]:
    """Infer {dataset_root}/map_cache from a SatNav DATA_PATH template."""
    if not data_path_tmpl:
        return None

    parts = str(data_path_tmpl).split(os.sep)
    if "episodes" in parts:
        idx = parts.index("episodes")
        if idx > 0:
            dataset_root = os.sep.join(parts[:idx])
            if os.path.isabs(str(data_path_tmpl)) and not dataset_root.startswith(
                os.sep
            ):
                dataset_root = os.sep + dataset_root
            return os.path.join(os.path.abspath(dataset_root), "map_cache")

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


class SwiftVLNInferenceSession:
    """
    Stateful frame, history, prompt, and overlap pipeline for one evaluator.

    Key Features:
    - VIT feature caching: Avoid recomputing features for cached frames
    - Overlap context: Reuse configured overlap turns from previous window
    - Manual embedding construction: Bypass template for efficient inference

    Cache Architecture:
    - history_cache: List[Tensor] - Compressed features for NUM_HISTORY global frames
    - overlap_context: OverlapContext - Cached context from last window's overlap turns
    """

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        args: Any,
        config: Any,
        environment_spec: EnvironmentSpec,
        device: torch.device,
        num_history: int,
        num_future_steps: int,
        diagnostics: Any,
    ):
        self.model = model
        self.processor = processor
        self.args = args
        self.config = config
        self.environment_spec = environment_spec
        self.env_type = environment_spec.name
        self.device = device
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.diagnostics = diagnostics

        # ==========================================================================
        # Window parameters (shared by all history processor types)
        # ==========================================================================
        self.num_frames = getattr(self.args, "num_frames", 32)
        self.num_overlap = getattr(self.args, "num_overlap", 0)

        # Calculate derived window parameters
        self.stride = self.num_frames - self.num_overlap
        self.turns_per_window = self.num_frames // self.num_future_steps
        self.overlap_turns = self.num_overlap // self.num_future_steps
        self.new_turns_per_window = self.turns_per_window - self.overlap_turns

        # ==========================================================================
        # History processor configuration
        # ==========================================================================
        self.history_processor_type = getattr(
            self.args, "history_processor_type", "per_frame"
        )

        # Per-frame specific parameters
        self.compress_stride = getattr(self.args, "compress_stride", 2)
        self.use_tome = getattr(self.args, "use_tome", False)
        self.log_base = getattr(self.args, "log_base", 1.0)
        self.use_random = getattr(self.args, "use_random", False)

        # GTC specific parameters
        self.gtc_output_tokens = getattr(self.args, "gtc_output_tokens", 512)
        self.gtc_temperature = getattr(self.args, "gtc_temperature", 0.1)
        self.gtc_num_iterations = getattr(self.args, "gtc_num_iterations", 1)

        # Initialize history processor based on type
        if self.history_processor_type == "gtc":
            self.history_processor = create_history_processor(
                processor_type="gtc",
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None  # Not used for GTC
        elif self.history_processor_type == "segment_gtc":
            self.history_processor = create_history_processor(
                processor_type="segment_gtc",
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None  # Not used for Segment GTC
        else:
            # Per-frame compression (default)
            # Supports both uniform (log_base=1.0) and logarithmic (log_base>1.0) sampling
            compress_method = "tome" if self.use_tome else "pooling"
            self.history_processor = create_history_processor(
                processor_type="per_frame",
                compress_stride=self.compress_stride,
                compress_method=compress_method,
                num_history=self.num_history,
                log_base=self.log_base,
            )
            # Keep compressor for _compress_features method
            self.compressor = HistoryTokenCompressor(
                stride=self.compress_stride, method=compress_method
            )

        # ==========================================================================
        # System prompt setting
        # ==========================================================================
        self.system_prompt_setting = getattr(
            self.args, "system_prompt_setting", "vanilla"
        ).lower()
        self.memory_method = getattr(self.args, "memory_method", "history").lower()
        self.map_global_side_m = float(getattr(self.args, "map_global_side_m", 1000.0))
        self.map_local_side_m = float(getattr(self.args, "map_local_side_m", 400.0))
        self.map_render_px = int(getattr(self.args, "map_render_px", 448))
        self.map_mask_method = str(
            getattr(self.args, "map_mask_method", "dilate20")
        ).lower()
        self.map_builder: Optional[SatNavMapMemoryBuilder] = None
        if self.memory_method not in ("history", "map"):
            raise ValueError(f"Unsupported memory_method: {self.memory_method}")
        if self.memory_method == "map":
            if not self.environment_spec.supports_map_memory:
                raise ValueError(
                    "SwiftVLN memory_method=map currently supports only satnav "
                    f"(got {self.env_type})."
                )
            if self.history_processor_type != "per_frame":
                raise ValueError(
                    "SwiftVLN memory_method=map currently requires history_processor_type=per_frame."
                )
            if self.use_tome:
                raise ValueError(
                    "SwiftVLN memory_method=map currently requires use_tome=false."
                )
            # Map images are synthesized top-down views, not real camera frames,
            # so embedding enhancement must stay disabled to match training.
            if getattr(self.args, "embedding_mode", "none") != "none":
                raise ValueError(
                    "SwiftVLN memory_method=map requires embedding_mode=none."
                )
            # Derive cache dir from DATA_PATH so any dataset root name, such as
            # SatNav-v0.1 or a custom abcd directory, maps to {root}/map_cache.
            # SWIFTVLN_MAP_CACHE_DIR overrides this; "off" disables it.
            data_path_tmpl = getattr(self.config.DATASET, "DATA_PATH", "") or ""
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
        self.history_memory_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            HISTORY_MEMORY_TOKEN
        )
        self.current_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            CURRENT_IMAGE_TOKEN
        )
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)

        # Cache initialization (will be reset per episode)
        self.reset()

        # ==========================================================================
        # Embedding enhancement pipeline (auto-detect from model checkpoint)
        # ==========================================================================
        self.has_embed_enhance = (
            hasattr(self.model, "embed_enhance")
            and not self.model.embed_enhance.is_empty
        )
        # ==========================================================================
        # Print configuration
        # ==========================================================================
        print("[SwiftVLNInferenceSession] Initialized:")
        print(
            f"  Window: num_frames={self.num_frames}, num_overlap={self.num_overlap}, stride={self.stride}"
        )
        print(
            f"  Turns: per_window={self.turns_per_window}, overlap={self.overlap_turns}"
        )
        print(f"  History Processor: {self.history_processor.name}")
        if self.history_processor_type in ("gtc", "segment_gtc"):
            print(
                f"    Sampling: every {self.num_future_steps} frames (same as training)"
            )
        else:
            if self.use_random:
                sampling_type = "random"
            else:
                sampling_type = (
                    "uniform"
                    if self.log_base == 1.0
                    else f"logarithmic (b={self.log_base})"
                )
            print(f"    Sampling: {sampling_type}, {self.num_history} frames")
        print(f"  System Prompt: {self.system_prompt_setting}")
        if self.system_prompt_setting == "initial":
            print(
                "    [INITIAL] Initial view ENABLED: first frame (uncompressed) in system prompt"
            )
        print(f"  Memory Method: {self.memory_method}")
        if self.memory_method == "map":
            print(
                f"    map: global={self.map_global_side_m:.0f}m, "
                f"local={self.map_local_side_m:.0f}m, "
                f"render={self.map_render_px}px, mask={self.map_mask_method}"
            )
        if self.has_embed_enhance:
            print(
                f"  Embed Enhance: {self.model.embed_enhance} (auto-detected from checkpoint)"
            )

    def reset(self) -> None:
        """Reset all caches at the start of each episode."""
        # VIT feature caches
        self.history_cache: List[Tuple[torch.Tensor, Optional[List[float]]]] = []

        # For GTC: cache VIT features by step_id to avoid re-encoding
        # Key: step_id (prediction step), Value: (vit_features, grid_thw, pose)
        self.vit_feature_cache: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, Optional[List[float]]]
        ] = {}

        # Initial view features (uncompressed, for initial prompt setting)
        self.initial_features: Optional[torch.Tensor] = None

        # Overlap context from previous window
        self.overlap_context: Optional[OverlapContext] = None

        # Current window state
        self.window_turns: List[TurnContext] = []  # Turns in current window
        self.window_start_step: int = 0
        self.current_window_idx: int = 0
        # Pose tracking
        self.pose_history: List[List[float]] = []
        self.executed_actions: List[int] = [-1]  # Start with INITIAL action marker

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
        """Get current pose vector from action integral (consistent with training).

        We always use action integration here so that the pose representation
        matches exactly what ``reconstruct_pose_from_actions`` produces during
        training.
        """
        return self._get_pose_from_actions()

    def observe_pose(self) -> List[float]:
        """Record and return the action-integrated pose for the current frame."""
        pose = self.current_pose()
        self.pose_history.append(pose)
        return pose

    def encode_initial_view(
        self, image: Image.Image, pose: List[float]
    ) -> torch.Tensor:
        """Cache the uncompressed initial view used by the initial prompt mode."""
        self.initial_features, _ = self._encode_frame(image, pose=pose)
        return self.initial_features

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

    @staticmethod
    def _extract_visual_features(visual_res: Any) -> torch.Tensor:
        """Normalize Qwen-family visual outputs to pooled visual tokens."""
        if hasattr(visual_res, "pooler_output"):
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
            images=[image], return_tensors="pt"
        )

        pixel_values = (
            media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        )
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )

            # Apply embedding enhancement pipeline if enabled (auto-detected from checkpoint)
            if self.has_embed_enhance:
                t, h, w = image_grid_thw[0].tolist()
                h_m, w_m = int(h) // self.merge_size, int(w) // self.merge_size
                vit_features = self.model.embed_enhance(
                    vit_features, h_m, w_m, pose=pose
                )

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
            images=images, return_tensors="pt"
        )

        pixel_values = (
            media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        )
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            all_vit_features = self._extract_visual_features(
                self.model.visual(pixel_values, grid_thw=image_grid_thw)
            )

        # Split features by image
        merge_length = self.merge_size**2
        features_list = []
        grid_thw_list = []
        embed_idx = 0

        for i in range(len(images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            img_features = all_vit_features[embed_idx : embed_idx + num_tokens]
            embed_idx += num_tokens

            # Apply embedding enhancement pipeline if enabled (auto-detected from checkpoint)
            if self.has_embed_enhance:
                t, h, w = image_grid_thw[i].tolist()
                h_m, w_m = int(h) // self.merge_size, int(w) // self.merge_size
                pose_i = None
                if poses is not None and i < len(poses):
                    pose_i = poses[i]
                with torch.no_grad():
                    img_features = self.model.embed_enhance(
                        img_features, h_m, w_m, pose=pose_i
                    )

            features_list.append(img_features)
            grid_thw_list.append(image_grid_thw[i])

        return features_list, grid_thw_list

    def _compress_features(
        self, features: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
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
        if self.memory_method == "map":
            self._compute_history_cache_map(episode, window_start)
            return

        if window_start <= 0:
            self.history_cache = []
            return

        if self.history_processor_type in ("gtc", "segment_gtc"):
            self._compute_history_cache_gtc(rgb_list, pose_list, window_start)
        else:
            self._compute_history_cache_per_frame(rgb_list, pose_list, window_start)

    def _compute_history_cache_map(
        self,
        episode: Any,
        window_start: int,
    ):
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
        self.diagnostics.map_cache(self, episode, window_start, features_list)

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
        history_poses = [
            pose_list[i] if i < len(pose_list) else None for i in history_indices
        ]

        # Batch encode
        features_list, grid_thw_list = self._encode_batch_frames(
            history_images, poses=history_poses
        )

        # Compress each frame
        self.history_cache = []
        for features, grid_thw, pose in zip(
            features_list, grid_thw_list, history_poses
        ):
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
        if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            return base_model.model.embed_tokens(input_ids)
        elif hasattr(base_model, "model") and hasattr(
            base_model.model, "language_model"
        ):
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
        system_prompt = self.environment_spec.format_prompt(instruction)

        # Add initial view if enabled (must match dataset.py order: initial before history)
        if initial_token_count > 0:
            initial_tokens = CURRENT_IMAGE_TOKEN * initial_token_count
            initial_str = f"<|vision_start|>{initial_tokens}<|vision_end|>"
            system_prompt += f" This is your initial observation at the starting point of this journey: {initial_str}."

        if history_token_counts:
            # Unified memory mode: single <history_memory> block with total token count
            total_history_tokens = sum(history_token_counts)
            # All history embeddings are packed into a single unified block
            history_str = f"<|vision_start|>{HISTORY_MEMORY_TOKEN * total_history_tokens}<|vision_end|>"
            if self.memory_method == "map":
                system_prompt += (
                    f" These are your explored map memories: {history_str}."
                )
            else:
                system_prompt += (
                    f" These are your historical observations: {history_str}."
                )

        # Format as system message
        messages = [{"role": "system", "content": system_prompt}]

        # Tokenize without generation prompt
        inputs = self.processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt", return_dict=True
        )

        self.diagnostics.map_system_prompt(
            self,
            inputs,
            history_token_counts,
            initial_token_count,
            instruction,
            system_prompt,
        )

        return inputs["input_ids"].to(self.device)

    def _build_user_turn_ids(
        self,
        conjunction: str,
        current_token_count: int,
        add_generation_prompt: bool = False,
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
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )

        self.diagnostics.map_user_prompt(
            self,
            inputs,
            current_token_count,
            add_generation_prompt,
            content,
        )

        return inputs["input_ids"].to(self.device)

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
        messages = [{"role": "assistant", "content": response}]

        inputs = self.processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt", return_dict=True
        )

        return inputs["input_ids"].to(self.device)

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
        current_vit_features, current_grid_thw = self._encode_frame(
            current_image, pose=current_pose
        )
        current_token_count = current_vit_features.shape[0]

        # 2. Build system prompt with correct history token counts and optional initial view
        history_token_counts = [h[0].shape[0] for h in self.history_cache]
        initial_token_count = (
            self.initial_features.shape[0] if self.initial_features is not None else 0
        )
        system_ids = self._build_system_prompt_ids(
            instruction,
            history_token_counts,
            initial_token_count=initial_token_count,
        )
        system_embeds = self._get_text_embeddings(system_ids)

        # 3a. Replace initial view tokens with uncompressed features (if enabled)
        # Initial view uses <current_image> tokens in the system prompt
        # These positions come BEFORE any <current_image> tokens in user turns
        if self.initial_features is not None:
            init_positions = (system_ids[0] == self.current_image_token_id).nonzero(
                as_tuple=True
            )[0]
            init_count = self.initial_features.shape[0]
            if len(init_positions) >= init_count:
                init_positions_slice = init_positions[:init_count]
                init_embeds = self.initial_features.to(
                    system_embeds.device, system_embeds.dtype
                )
                system_embeds[0, init_positions_slice] = init_embeds

                self.diagnostics.initial_tokens_injected(
                    init_count,
                    init_positions_slice,
                    system_ids.shape[1],
                )
            else:
                print(
                    f"[INITIAL WARNING] Not enough <current_image> positions in system prompt! "
                    f"Found {len(init_positions)}, need {init_count}"
                )

        # 3b. Replace unified history memory tokens with cached features
        history_injection_ok = True
        history_injection_msg = "no_history"
        history_embeds = None
        if self.history_cache:
            # Use history_memory_token_id for unified memory mode
            history_positions = (system_ids[0] == self.history_memory_token_id).nonzero(
                as_tuple=True
            )[0]
            history_embeds = torch.cat([h[0] for h in self.history_cache], dim=0)
            total_history_tokens = history_embeds.shape[0]

            if len(history_positions) >= total_history_tokens:
                history_positions_slice = history_positions[:total_history_tokens]
                history_embeds = history_embeds.to(
                    system_embeds.device, system_embeds.dtype
                )
                system_embeds[0, history_positions_slice] = history_embeds
                history_injection_msg = (
                    f"ok positions=[{history_positions_slice[0].item()}.."
                    f"{history_positions_slice[-1].item()}]"
                )
            else:
                history_injection_ok = False
                history_injection_msg = f"failed positions={len(history_positions)} need={total_history_tokens}"
                self.diagnostics.map_history_injection_warning(history_injection_msg)

        embeds_parts = [system_embeds]

        # 4. Add overlap context from the previous window
        if (
            self.overlap_context is not None
            and self.overlap_context.input_ids is not None
        ):
            overlap_embeds = self._get_text_embeddings(self.overlap_context.input_ids)

            # Replace current_image tokens with cached embeddings using index assignment
            if self.overlap_context.image_embeds:
                overlap_positions = (
                    self.overlap_context.input_ids[0] == self.current_image_token_id
                ).nonzero(as_tuple=True)[0]
                overlap_img_embeds = torch.cat(self.overlap_context.image_embeds, dim=0)
                total_overlap_tokens = overlap_img_embeds.shape[0]

                if len(overlap_positions) >= total_overlap_tokens:
                    overlap_positions_slice = overlap_positions[:total_overlap_tokens]
                    overlap_img_embeds = overlap_img_embeds.to(
                        overlap_embeds.device, overlap_embeds.dtype
                    )
                    overlap_embeds[0, overlap_positions_slice] = overlap_img_embeds

            embeds_parts.append(overlap_embeds)

        # 5. Add completed turns in current window
        for turn in self.window_turns:
            turn_user_embeds = self._get_text_embeddings(turn.user_input_ids)
            # Replace image tokens using index assignment
            turn_positions = (
                turn.user_input_ids[0] == self.current_image_token_id
            ).nonzero(as_tuple=True)[0]
            turn_token_count = turn.image_embed.shape[0]

            if len(turn_positions) >= turn_token_count:
                turn_positions_slice = turn_positions[:turn_token_count]
                turn_img = turn.image_embed.to(
                    turn_user_embeds.device, turn_user_embeds.dtype
                )
                turn_user_embeds[0, turn_positions_slice] = turn_img

            embeds_parts.append(turn_user_embeds)

            # Add assistant response
            assistant_ids = self._build_assistant_turn_ids(turn.assistant_response)
            assistant_embeds = self._get_text_embeddings(assistant_ids)
            embeds_parts.append(assistant_embeds)

        # 6. Add new user turn (with generation prompt for model to generate response)
        new_user_ids = self._build_user_turn_ids(
            conjunction, current_token_count, add_generation_prompt=True
        )
        new_user_embeds = self._get_text_embeddings(new_user_ids)

        # Replace image tokens with current VIT features using index assignment
        cur_positions = (new_user_ids[0] == self.current_image_token_id).nonzero(
            as_tuple=True
        )[0]

        if len(cur_positions) >= current_token_count:
            cur_positions_slice = cur_positions[:current_token_count]
            current_vit = current_vit_features.to(
                new_user_embeds.device, new_user_embeds.dtype
            )
            new_user_embeds[0, cur_positions_slice] = current_vit

        embeds_parts.append(new_user_embeds)

        # Concatenate all parts
        inputs_embeds = torch.cat(embeds_parts, dim=1)
        seq_len = inputs_embeds.shape[1]

        self.diagnostics.map_complete_embeds(
            self,
            system_ids,
            new_user_ids,
            current_vit_features,
            current_token_count,
            history_token_counts,
            initial_token_count,
            history_injection_msg,
            history_embeds,
            history_injection_ok,
            seq_len,
        )

        return inputs_embeds, seq_len, current_vit_features, current_grid_thw

    def predict(
        self,
        instruction: str,
        current_image: Image.Image,
        current_pose: List[float],
        step_id: int,
        parse_actions: Callable[[str], List[int]],
        timing_stats: Dict[str, float],
    ) -> List[int]:
        """Generate and cache one action sequence for the current observation."""
        start_time = time.time()
        conjunction = random.choice(self.conjunctions)
        inputs_embeds, seq_len, current_vit, current_grid_thw = (
            self._build_complete_prompt_embeds(
                instruction=instruction,
                current_image=current_image,
                conjunction=conjunction,
                current_pose=current_pose,
            )
        )
        timing_stats["build_embeds"] += time.time() - start_time

        attention_mask = torch.ones(
            (1, seq_len),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        dummy_input_ids = torch.full(
            (1, seq_len),
            pad_token_id,
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        start_time = time.time()
        outputs = self.model.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )
        timing_stats["model_generate"] += time.time() - start_time

        start_time = time.time()
        generated_ids = outputs[0][seq_len:]
        output_text = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        timing_stats["decode"] += time.time() - start_time

        start_time = time.time()
        action_sequence = parse_actions(output_text)
        timing_stats["parse_actions"] += time.time() - start_time

        start_time = time.time()
        self._save_turn_to_window(
            conjunction=conjunction,
            response=output_text,
            vit_features=current_vit,
        )
        if self.history_processor_type in ("gtc", "segment_gtc"):
            self.vit_feature_cache[step_id] = (
                current_vit,
                current_grid_thw,
                current_pose,
            )
        timing_stats["update_cache"] += time.time() - start_time

        if getattr(self.args, "verbose", False):
            print(
                f"Step {step_id}: window={self.current_window_idx}, "
                f"turns={len(self.window_turns)}, history={len(self.history_cache)}, "
                f"vit_cache={len(self.vit_feature_cache)}, output={output_text}"
            )
        return action_sequence

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
        overlap_turns = self.window_turns[-self.overlap_turns :]

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

    def slide_window(
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
        self.diagnostics.map_slide_window(self, new_window_start)

        # 3. Reset window state
        self.window_turns = []
        self.window_start_step = new_window_start
        self.current_window_idx += 1
