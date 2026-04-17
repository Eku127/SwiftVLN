# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Dataset for Visual Language Navigation

This dataset provides the OverlapVLN-specific data flow with:
1. Sliding window sampling with configurable overlap
2. Loss masking for overlap turns (using ms-swift's message.loss field)

Key features:
- num_overlap: Number of overlapping actions between consecutive windows
- When num_overlap > 0, stride = num_frames - num_overlap
- For samples where start_idx > 0, first (num_overlap / num_future_steps) turns 
  have loss=0.0 (masked), acting as pure context
"""

import os
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset

from swiftvln.common.constants import (
    DEFAULT_ACTION_MAP,
    DEFAULT_CONJUNCTIONS,
    DEFAULT_IMAGE_TOKEN,
    HISTORY_MEMORY_TOKEN,
)
from swiftvln.common.embedding_enhancement import reconstruct_pose_from_actions
from swiftvln.model.map_memory import (
    SatNavMapMemoryBuilder,
    SatNavTrajectoryMetadataResolver,
)


def _debug_enabled() -> bool:
    return os.environ.get('OVERLAPVLN_DEBUG', '') != ''


def _debug_rank() -> int:
    raw = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    try:
        return int(raw)
    except ValueError:
        return 0


def _preview_text(text: str, limit: int = 220) -> str:
    text = str(text).replace('\n', '\\n')
    if len(text) <= limit:
        return text
    return text[:limit] + '...'


class OverlapVLNDataset(Dataset):
    """
    OverlapVLN Dataset with sliding window overlap and loss masking.
    
    Features:
    - Sliding window sampling with configurable overlap
    - Loss masking for overlap turns (first N turns don't compute loss)
    - Uses standard <image> tokens (template handles differentiation)
    
    Args:
        num_overlap: Number of overlapping actions between windows (default: 16)
                    When > 0, stride = num_frames - num_overlap
                    Set to 0 to disable overlap (original behavior)
        env_type: Environment type - 'habitat' (forward=0.25m) or 'satnav' (forward=10m)
    """
    
    def __init__(
        self,
        data_path: str,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        use_random: bool = False,
        max_samples: Optional[int] = None,
        num_overlap: int = 16,  # New parameter for overlap
        env_type: str = "habitat",  # New parameter for environment type
        history_processor_type: str = "per_frame",  # History sampling strategy
        log_base: float = 1.0,  # Sampling distribution (1.0=uniform, >1.0=logarithmic)
        system_prompt_setting: str = "vanilla",  # System prompt strategy: "vanilla" or "initial"
        memory_method: str = "history",
        map_global_side_m: float = 1000.0,
        map_local_side_m: float = 400.0,
        map_render_px: int = 384,
        map_mask_method: str = "dilate20",
    ):
        # Store num_overlap before calling super().__init__ 
        # because we need to override the data indexing logic
        self.num_overlap = num_overlap
        
        # Don't call parent __init__ directly, replicate the logic with our modifications
        # This is necessary because parent builds data_list in __init__
        Dataset.__init__(self)
        
        # VLN parameters
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.use_random = use_random
        self.max_samples = max_samples
        self.env_type = env_type.lower()
        self.history_processor_type = history_processor_type.lower()
        self.log_base = log_base  # Sampling distribution
        self.system_prompt_setting = system_prompt_setting.lower()  # "vanilla" or "initial"
        self.memory_method = memory_method.lower()
        self.map_global_side_m = float(map_global_side_m)
        self.map_local_side_m = float(map_local_side_m)
        self.map_render_px = int(map_render_px)
        self.map_mask_method = str(map_mask_method).lower()
        self.map_builder: Optional[SatNavMapMemoryBuilder] = None
        self._map_scenes_dir: Optional[str] = None
        self._debug_map_sample_count = 0
        self._debug_prompt_count = 0
        
        # Set forward distance based on environment type
        if self.env_type == "satnav":
            self.forward_distance = "10m"
        else:  # habitat or default
            self.forward_distance = "0.25m"
        
        # Validate overlap parameter
        if self.num_overlap < 0:
            raise ValueError(f"num_overlap must be >= 0, got {self.num_overlap}")
        if self.num_overlap >= self.num_frames:
            raise ValueError(f"num_overlap ({self.num_overlap}) must be < num_frames ({self.num_frames})")
        if self.memory_method not in ("history", "map"):
            raise ValueError(f"memory_method must be 'history' or 'map', got {self.memory_method}")
        if self.memory_method == "map":
            if self.env_type != "satnav":
                raise ValueError("memory_method=map currently supports only satnav.")
            if self.history_processor_type != "per_frame":
                raise ValueError("memory_method=map currently requires history_processor_type=per_frame.")
        
        # Calculate stride
        self.stride = self.num_frames - self.num_overlap
        
        # Load navigation data from multiple paths (comma-separated)
        self.video_folders = [p.strip() for p in data_path.split(',') if p.strip()]
        self.nav_data = []
        for vf in self.video_folders:
            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"Warning: {anno_path} not found, skipping...")
                continue

            map_resolver = None
            if self.memory_method == "map":
                map_resolver = SatNavTrajectoryMetadataResolver(vf)
                scenes_dir = os.path.abspath(map_resolver.scenes_dir)
                if self.map_builder is None:
                    # Default cache next to the dataset version (e.g.
                    # ver_260404/map_cache). Env var OVERLAPVLN_MAP_CACHE_DIR
                    # overrides this, and the sentinel value "off" disables
                    # caching entirely. See map_memory._resolve_cache_dir.
                    default_cache_dir = os.path.join(
                        os.path.abspath(map_resolver.dataset_root),
                        "map_cache",
                    )
                    self.map_builder = SatNavMapMemoryBuilder(
                        scenes_dir=scenes_dir,
                        global_side_m=self.map_global_side_m,
                        local_side_m=self.map_local_side_m,
                        render_px=self.map_render_px,
                        mask_method=self.map_mask_method,
                        cache_dir=default_cache_dir,
                    )
                    self._map_scenes_dir = scenes_dir
                elif scenes_dir != self._map_scenes_dir:
                    raise ValueError(
                        "memory_method=map expects a shared scenes dir across data roots, "
                        f"got {self._map_scenes_dir} vs {scenes_dir}"
                    )

            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
                if map_resolver is not None:
                    map_meta = map_resolver.resolve(tdata)
                    tdata['_map_scene_id'] = map_meta.scene_id
                    tdata['_map_episode_id'] = map_meta.episode_id
                    tdata['_map_start_position'] = map_meta.start_position
                    tdata['_map_start_rotation'] = map_meta.start_rotation
            self.nav_data += anno_json
            print(f"Loaded {len(anno_json)} episodes from {vf}")
        
        # Build data index with sliding window overlap
        # Format: (episode_id, instruction_id, start_frame)
        self.data_list = []
        skipped_samples = 0
        adjusted_samples = 0
        
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            actions = item.get('actions', [])
            actions_len = len(actions)
            
            if actions_len < 4:
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            for ins_id in range(len(instructions)):
                # Generate all potential start indices
                all_start_indices = list(range(0, actions_len, self.stride))
                
                for i, start_idx in enumerate(all_start_indices):
                    # Calculate actual action count for this sample
                    sample_actions = min(self.num_frames, actions_len - start_idx)
                    
                    # Calculate masked action count (only for non-first samples)
                    mask_actions = self.num_overlap if start_idx > 0 else 0
                    
                    # Calculate effective (trainable) action count
                    effective_actions = sample_actions - mask_actions
                    
                    actual_start_idx = start_idx
                    
                    # If effective actions too few, adjust start_idx to cover more
                    # This ensures end-of-episode data (including STOP) is trained
                    if effective_actions < self.num_future_steps:
                        # Adjust start_idx: from end backwards by num_frames
                        adjusted_start = max(0, actions_len - self.num_frames)
                        
                        # Check if previous sample already covers this range
                        if i > 0 and adjusted_start <= all_start_indices[i - 1]:
                            # Previous sample already covers the end, skip this one
                            skipped_samples += 1
                            continue
                        
                        # Use adjusted start_idx
                        actual_start_idx = adjusted_start
                        adjusted_samples += 1
                    
                    self.data_list.append((ep_id, ins_id, actual_start_idx))
        
        # Log overlap configuration
        if self.num_overlap > 0:
            print(f"[OverlapVLN] Sliding window: num_frames={num_frames}, "
                  f"num_overlap={num_overlap}, stride={self.stride}")
            print(f"[OverlapVLN] Loss masking: first {num_overlap // num_future_steps} turns "
                  f"will have loss=0.0 for samples with start_idx > 0")
            if adjusted_samples > 0:
                print(f"[OverlapVLN] Adjusted {adjusted_samples} end-of-episode samples "
                      f"to ensure STOP data is trained")
            if skipped_samples > 0:
                print(f"[OverlapVLN] Skipped {skipped_samples} redundant samples "
                      f"(already covered by previous sample)")
        else:
            print(f"[OverlapVLN] No overlap (stride={self.stride})")
        
        # Limit samples if max_samples is specified
        if self.max_samples is None:
            env_max_samples = os.getenv('VLN_MAX_SAMPLES')
            if env_max_samples is not None:
                try:
                    self.max_samples = int(env_max_samples)
                except ValueError:
                    self.max_samples = None
        
        if self.max_samples is not None and self.max_samples > 0:
            original_len = len(self.data_list)
            if original_len > self.max_samples:
                # Random sampling instead of simple truncation
                # This ensures better data diversity across episodes
                import random as _random
                _rng = _random.Random(42)  # Fixed seed for reproducibility
                self.data_list = _rng.sample(self.data_list, self.max_samples)
                # Sort by (ep_id, ins_id, start_idx) to maintain temporal order within episodes
                self.data_list.sort()
                print(f"[OverlapVLN] Random sampled {self.max_samples} from {original_len} samples (seed=42)")
            else:
                print(f"[OverlapVLN] Requested max_samples={self.max_samples} >= available {original_len}, using all samples")
        
        # Action vocabulary / prompt conjunctions
        self.idx2actions = DEFAULT_ACTION_MAP.copy()
        self.conjunctions = DEFAULT_CONJUNCTIONS.copy()
        
        # Statistics
        if self.num_overlap > 0:
            first_samples = sum(1 for _, _, start_idx in self.data_list if start_idx == 0)
            overlap_samples = len(self.data_list) - first_samples
            print(f"[OverlapVLN] Sample breakdown: {first_samples} first (full loss), "
                  f"{overlap_samples} overlap (partial loss)")
        
        # Debug counter for initial strategy verification
        self._debug_initial_count = 0
        
        print(f"OverlapVLNDataset initialized: {len(self.data_list)} samples from {len(self.nav_data)} episodes")
        print(f"  env_type={self.env_type}, forward_distance={self.forward_distance}")

    def __len__(self) -> int:
        return len(self.data_list)

    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to compact action symbols."""
        if len(actions) == 0:
            return "STOP"
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(int(action), "STOP")
            converted_sequence.append(act_text)
        return "".join(converted_sequence)
        print(f"  system_prompt_setting={self.system_prompt_setting}")
        print(f"  memory_method={self.memory_method}")
        if self.system_prompt_setting == "initial":
            print(f"  [INITIAL] Initial view ENABLED: first frame (uncompressed) will be added to system prompt")
        if self.memory_method == "map":
            print(
                f"  map: global={self.map_global_side_m:.0f}m, local={self.map_local_side_m:.0f}m, "
                f"render={self.map_render_px}px, mask={self.map_mask_method}"
            )
            if os.environ.get('OVERLAPVLN_DEBUG'):
                print(f"  [MAP DEBUG] scenes_dir={self._map_scenes_dir}")
        if self.history_processor_type == 'per_frame':
            print(f"  history: per_frame (h={self.num_history}, log_base={self.log_base})")
            # Debug: show sampling distribution
            if os.environ.get('OVERLAPVLN_DEBUG'):
                import math
                print(f"  [DEBUG] Sampling distribution preview (for 32 history frames -> {self.num_history} samples):")
                num_frames = 32
                num_samples = min(self.num_history, num_frames)
                indices = []
                for i in range(num_samples):
                    t_sample = i / (num_samples - 1) if num_samples > 1 else 1.0
                    t_frame = 1.0 - math.pow(1.0 - t_sample, self.log_base)
                    frame_idx = int(round(t_frame * (num_frames - 1)))
                    indices.append(frame_idx)
                print(f"  [DEBUG] Sampled indices: {indices}")
                if self.log_base == 1.0:
                    print(f"  [DEBUG] -> Uniform distribution (linear)")
                else:
                    print(f"  [DEBUG] -> Logarithmic distribution (more recent frames)")
        else:
            print(f"  history: {self.history_processor_type}")

    def _sample_history_frames(
        self,
        current_start_abs: int,
        num_video_frames: int,
    ) -> np.ndarray:
        """
        Sample history frame indices based on history_processor_type.
        
        Args:
            current_start_abs: Start index of current window (exclusive upper bound for history)
            num_video_frames: Total number of video frames
            
        Returns:
            np.ndarray of sorted frame indices for history
            
        Sampling strategies:
        - per_frame: Sample num_history frames using power transformation
          - num_history=0: Disable history sampling entirely (effective no-memory mode)
          - log_base=1.0: Uniform/linear sampling
          - log_base>1.0: Logarithmic sampling (more recent frames)
        - gtc: Sample with num_future_steps interval (denser, for cross-frame clustering)
        """
        import math
        
        available_history_indices = np.arange(0, current_start_abs)
        
        if len(available_history_indices) == 0:
            return np.array([], dtype=np.int32)
        
        if self.history_processor_type in ('gtc', 'segment_gtc'):
            # GTC/SegmentGTC: Sample with num_future_steps interval (same as current frame sampling)
            # This gives denser history coverage for cross-frame clustering
            history_step_ids = np.arange(0, current_start_abs, self.num_future_steps, dtype=np.int32)
            history_step_ids = np.clip(history_step_ids, 0, num_video_frames - 1)
            history_step_ids = np.unique(history_step_ids)
        else:
            # per_frame (default): Sample num_history frames with configurable distribution
            num_frames = current_start_abs
            num_samples = min(self.num_history, num_frames)
            
            if num_samples == 0:
                # per_frame + NUM_HISTORY=0 is the supported no-memory path:
                # no history frames are returned, so downstream prompt/template logic
                # will skip inserting <history_memory>.
                return np.array([], dtype=np.int32)
            
            if self.use_random:
                history_step_ids = np.random.choice(
                    available_history_indices,
                    size=num_samples,
                    replace=False
                )
                history_step_ids = np.sort(history_step_ids)
            else:
                # Use power transformation for flexible sampling
                # log_base=1.0: t_frame = t_sample (linear/uniform)
                # log_base>1.0: more samples at recent end
                indices = []
                for i in range(num_samples):
                    if num_samples == 1:
                        t_sample = 1.0  # Most recent
                    else:
                        t_sample = i / (num_samples - 1)
                    
                    # Power transformation: t_frame = 1 - (1 - t_sample)^log_base
                    t_frame = 1.0 - math.pow(1.0 - t_sample, self.log_base)
                    
                    # Map to frame index
                    frame_idx = int(round(t_frame * (num_frames - 1)))
                    frame_idx = max(0, min(num_frames - 1, frame_idx))
                    
                    if frame_idx not in indices:
                        indices.append(frame_idx)
                
                # Sort and ensure we have num_samples frames
                indices.sort()
                
                # Fill missing slots if duplicates removed
                while len(indices) < num_samples:
                    for k in range(num_frames):
                        if k not in indices:
                            indices.append(k)
                            indices.sort()
                            break
                    else:
                        break  # No more frames available
                
                history_step_ids = np.array(indices[:num_samples], dtype=np.int32)
                
                # Debug output (only for first few samples)
                if os.environ.get('OVERLAPVLN_DEBUG') and not hasattr(self, '_debug_sample_count'):
                    self._debug_sample_count = 0
                if os.environ.get('OVERLAPVLN_DEBUG') and self._debug_sample_count < 3:
                    print(f"  [DEBUG SAMPLE {self._debug_sample_count}] History sampling:")
                    print(f"    -> Available frames: 0-{num_frames-1} ({num_frames} total)")
                    print(f"    -> Sampling {num_samples} frames with log_base={self.log_base}")
                    print(f"    -> Sampled indices: {list(history_step_ids)}")
                    self._debug_sample_count += 1
            
            history_step_ids = np.clip(history_step_ids, 0, num_video_frames - 1)
        
        return history_step_ids

    def _build_map_memory_images(
        self,
        data: Dict[str, Any],
        raw_actions: List[int],
        window_start: int,
    ) -> List[Any]:
        if self.map_builder is None:
            raise RuntimeError("memory_method=map requested but map_builder is not initialized.")

        scene_id = data.get('_map_scene_id')
        start_position = data.get('_map_start_position')
        start_rotation = data.get('_map_start_rotation')
        if scene_id is None or start_position is None or start_rotation is None:
            raise KeyError("SatNav map metadata missing from annotation record.")

        step_size = 10.0 if self.env_type == 'satnav' else 0.25
        turn_angle = 15.0 if self.env_type == 'satnav' else 30.0
        map_images = self.map_builder.render_from_actions(
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=float(start_rotation),
            actions=raw_actions,
            window_start=window_start,
            step_size=step_size,
            turn_angle=turn_angle,
        )
        if _debug_enabled() and self._debug_map_sample_count < 6:
            image_sizes = [img.size for img in map_images]
            print(
                f"[MAP DEBUG][dataset] sample[{self._debug_map_sample_count}] "
                f"rank={_debug_rank()} "
                f"scene={scene_id} episode={data.get('_map_episode_id', 'unknown')} "
                f"window_start={window_start} actions={len(raw_actions)} "
                f"history_images={len(map_images)} sizes={image_sizes} "
                f"global_center_mode={getattr(self.map_builder, 'global_center_mode', 'unknown')}"
            )
            self._debug_map_sample_count += 1
        return map_images

    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample with overlap-aware loss masking.
        
        For samples where start_idx > 0 and num_overlap > 0:
        - First (num_overlap / num_future_steps) assistant turns have loss=0.0
        - These turns act as pure context (model sees them but doesn't train on them)
        
        Returns:
            dict with keys:
                - messages: List of conversation turns with <image> tokens
                            Assistant turns may have 'loss' field for masking
                - images: List of PIL Images (history first, then current)
                - num_history_images: Number of history images (for template)
        """
        # Get sample index
        ep_id, ins_id, start_idx = self.data_list[i]
        data = self.nav_data[ep_id]
        
        # Get video frames
        video_path = data['video']
        rgb_path = os.path.join(video_path, 'rgb')
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB frames not found: {rgb_path}")
        video_frames = sorted(os.listdir(rgb_path))
        num_video_frames = len(video_frames)
        
        if num_video_frames == 0:
            raise ValueError(f"No frames found in: {rgb_path}")
        
        # Get instruction
        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            instructions = [instructions]
        instruction = instructions[ins_id] if ins_id < len(instructions) else instructions[0]
        
        # Get raw episode actions and shifted actions for prediction targets.
        # Raw format typically includes INITIAL at index 0: [-1, a1, a2, ...].
        raw_actions = data.get('actions', [])
        if not isinstance(raw_actions, list):
            raw_actions = list(raw_actions)
        if len(raw_actions) == 0:
            raw_actions = [-1] + [0] * max(0, num_video_frames - 1)

        # Shift by 1 to predict the next action from current observation.
        actions = raw_actions[1:] + [0]
        actions_len = len(actions)

        # Reconstruct per-frame pose from raw actions (aligned with frame indices).
        # Pose format: [delta_forward, delta_right, sin(delta_heading), cos(delta_heading)].
        # SatNav defaults: step=10m, turn=15deg. Habitat defaults: step=0.25m, turn=30deg.
        step_size = 10.0 if self.env_type == 'satnav' else 0.25
        turn_angle = 15.0 if self.env_type == 'satnav' else 30.0
        frame_poses_all = reconstruct_pose_from_actions(
            raw_actions,
            step_size=step_size,
            turn_angle=turn_angle,
        )
        if frame_poses_all.shape[0] < num_video_frames:
            if frame_poses_all.shape[0] == 0:
                padding = np.zeros((num_video_frames, 4), dtype=np.float32)
            else:
                last_pose = frame_poses_all[-1:]
                repeat = num_video_frames - frame_poses_all.shape[0]
                padding = np.repeat(last_pose, repeat, axis=0)
            frame_poses_all = np.concatenate([frame_poses_all, padding], axis=0)
        elif frame_poses_all.shape[0] > num_video_frames:
            frame_poses_all = frame_poses_all[:num_video_frames]
        
        # Get time slice
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        if len(time_ids) == 0:
            time_ids = np.array([start_idx])
        current_actions = np.array(actions)[time_ids]
        
        # Sample current frames
        start_idx_abs = time_ids[0]
        end_idx_abs = time_ids[-1] + 1
        interval = self.num_future_steps
        
        sample_step_ids = np.arange(start_idx_abs, end_idx_abs, interval, dtype=np.int32)
        sample_step_ids = np.clip(sample_step_ids, 0, num_video_frames - 1)
        sample_step_ids = np.unique(sample_step_ids)
        
        if len(sample_step_ids) == 0:
            sample_step_ids = np.array([min(start_idx_abs, num_video_frames - 1)])
        
        sample_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in sample_step_ids]
        
        # Sample historical frames if not first segment
        history_frame_paths = []
        history_images = []
        history_step_ids = np.array([], dtype=np.int32)
        has_history = False
        if self.memory_method == "map":
            history_images = self._build_map_memory_images(data, raw_actions, start_idx_abs)
            has_history = len(history_images) > 0
        elif time_ids[0] != 0:
            current_start_abs = min(time_ids[0], num_video_frames)
            
            # Different sampling strategies based on history_processor_type
            history_step_ids = self._sample_history_frames(current_start_abs, num_video_frames)
            
            if len(history_step_ids) > 0:
                history_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in history_step_ids]
                has_history = True
        
        # Load initial view frame if system_prompt_setting is "initial"
        initial_frame_paths = []
        num_initial_images = 0
        if self.system_prompt_setting == "initial":
            initial_path = os.path.join(rgb_path, video_frames[0])
            initial_frame_paths = [initial_path]
            num_initial_images = 1
        
        # Load images as PIL Images
        # Order: history frames, initial frame, current frames
        from PIL import Image
        images = list(history_images)
        frame_poses: List[Optional[List[float]]] = [None] * len(history_images)

        history_files_to_load: List[str] = []
        if self.memory_method != "map":
            all_history_indices = list(history_step_ids.tolist())
            frame_poses.extend(frame_poses_all[idx].tolist() for idx in all_history_indices)
            history_files_to_load = history_frame_paths

        image_files_to_load = history_files_to_load + initial_frame_paths + sample_frame_paths
        non_history_indices: List[int] = []
        if num_initial_images > 0:
            non_history_indices.append(0)
        non_history_indices.extend(sample_step_ids.tolist())
        frame_poses.extend(frame_poses_all[idx].tolist() for idx in non_history_indices)

        for image_file in image_files_to_load:
            try:
                image = Image.open(image_file).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {image_file}: {e}")
                images.append(Image.new('RGB', (640, 480), color='black'))
        
        if len(images) == 0:
            raise ValueError(f"No images loaded for sample {i}")
        
        # Build conversation with standard <image> tokens
        # Use environment-specific forward distance
        system_prompt = (
            f"You are an autonomous navigation assistant. Your task is to {instruction}. "
            f"Based on your observations, output a sequence of actions using: "
            f"↑ (forward {self.forward_distance}), ← (turn left), → (turn right), or STOP (when goal is reached). "
            f"Output actions directly without explanation."
        )
        
        # Add initial view description if enabled
        # The initial view image uses standard <image> tag (uncompressed tokens)
        # It is placed BEFORE the history memory in the system prompt
        if self.system_prompt_setting == "initial":
            system_prompt += (
                " This is your initial observation at the starting point of this journey: <image>."
            )
        
        # Add history description with unified memory token.
        # If num_history_images == 0 (e.g. per_frame + NUM_HISTORY=0), the prompt
        # stays memory-free and no <history_memory> block is inserted.
        num_history_images = len(history_images) if self.memory_method == "map" else len(history_frame_paths)
        if has_history:
            if self.memory_method == "map":
                system_prompt += (
                    f" These are your explored map memories: "
                    f"<|vision_start|>{HISTORY_MEMORY_TOKEN}<|vision_end|>."
                )
            else:
                # Use unified <history_memory> token in vision wrapper
                # Template will expand this to the correct number of tokens based on compression
                system_prompt += f" These are your historical observations: <|vision_start|>{HISTORY_MEMORY_TOKEN}<|vision_end|>."
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Calculate number of turns to mask
        # Mask turns only for non-first samples (start_idx > 0) when overlap is enabled
        has_overlap = (start_idx > 0) and (self.num_overlap > 0)
        mask_turn_count = self.num_overlap // self.num_future_steps if has_overlap else 0
        
        # Build multi-turn dialogue with standard <image> tokens
        current_actions_list = list(current_actions)
        num_current_images = len(sample_frame_paths)
        
        action_idx = 0
        image_idx = 0
        turn_idx = 0
        
        while action_idx < len(current_actions_list) and image_idx < num_current_images:
            # User turn with standard <image> tag
            # Template's replace_tag will convert this to <current_image> with proper ROPE handling
            conjunction = random.choice(self.conjunctions)
            user_content = f"{conjunction}<image>."
            messages.append({'role': 'user', 'content': user_content})
            
            # Assistant turn with actions
            step_actions = current_actions_list[action_idx:action_idx + self.num_future_steps]
            if len(step_actions) == 0:
                step_actions = [0]  # STOP
            answer = self.actions2text(step_actions)
            
            # Build assistant message with optional loss masking
            assistant_msg = {'role': 'assistant', 'content': answer}
            if turn_idx < mask_turn_count:
                # Overlap turn - mask loss (pure context)
                assistant_msg['loss'] = 0.0
            messages.append(assistant_msg)
            
            action_idx += len(step_actions)
            image_idx += 1
            turn_idx += 1
        
        result = {
            'messages': messages,
            'images': images,
            'num_history_images': num_history_images,  # Metadata for template
            'num_initial_images': num_initial_images,  # Metadata for initial prompt
            'frame_poses': frame_poses,  # Per-image metadata, aligned with image order
            'memory_method': self.memory_method,
        }
        
        # Debug: log initial strategy details for first few samples
        if _debug_enabled() and self._debug_initial_count < 3:
            rank = _debug_rank()
            print(f"\n[INITIAL DEBUG] Rank={rank} Sample[{i}] (ep={ep_id}, ins={ins_id}, start={start_idx}):")
            print(f"  system_prompt_setting={self.system_prompt_setting}")
            print(f"  memory_method={self.memory_method}")
            print(f"  num_history_images={num_history_images}, num_initial_images={num_initial_images}, "
                  f"num_current_images={num_current_images}")
            if self.memory_method == "map":
                print(f"  [MAP] history image sizes: {[img.size for img in history_images]}")
            print(f"  total_images={len(images)} "
                  f"(expected: {num_history_images} + {num_initial_images} + {num_current_images} = "
                  f"{num_history_images + num_initial_images + num_current_images})")
            if self.system_prompt_setting == "initial":
                print(f"  [INITIAL] Initial frame path: {initial_frame_paths[0] if initial_frame_paths else 'NONE'}")
                print(f"  [INITIAL] Image order: [{num_history_images} history] + [1 initial] + [{num_current_images} current]")
                # Check system prompt contains initial observation text
                sys_content = messages[0].get('content', '')
                has_initial_tag = 'initial observation' in sys_content and '<image>' in sys_content
                print(f"  [INITIAL] System prompt has initial <image>: {has_initial_tag}")
            else:
                print(f"  [VANILLA] No initial frame (vanilla mode)")
            # Show system prompt (truncated)
            sys_content = messages[0].get('content', '')
            print(f"  system_prompt: {_preview_text(sys_content, limit=240)}")
            self._debug_initial_count += 1

        if _debug_enabled() and self._debug_prompt_count < 4:
            rank = _debug_rank()
            sys_content = messages[0].get('content', '')
            user_turns = [m for m in messages if m.get('role') == 'user']
            assistant_turns = [m for m in messages if m.get('role') == 'assistant']
            none_pose_count = sum(1 for pose in frame_poses if pose is None)
            print(
                f"[MAP DEBUG][dataset.prompt] sample[{self._debug_prompt_count}] "
                f"rank={rank} ep={ep_id} ins={ins_id} start={start_idx} "
                f"memory_method={self.memory_method} images={len(images)} "
                f"history={num_history_images} initial={num_initial_images} current={num_current_images} "
                f"frame_poses={len(frame_poses)} none_poses={none_pose_count}"
            )
            print(
                f"[MAP DEBUG][dataset.prompt] system tags: "
                f"<history_memory>={sys_content.count(HISTORY_MEMORY_TOKEN)} "
                f"<image>={sys_content.count(DEFAULT_IMAGE_TOKEN)} "
                f"map_phrase={'explored map memories' in sys_content}"
            )
            print(f"[MAP DEBUG][dataset.prompt] system_prompt={_preview_text(sys_content, limit=320)}")
            if user_turns:
                print(
                    f"[MAP DEBUG][dataset.prompt] first_user={_preview_text(user_turns[0].get('content', ''), limit=160)}"
                )
            if assistant_turns:
                print(
                    f"[MAP DEBUG][dataset.prompt] first_assistant={_preview_text(assistant_turns[0].get('content', ''), limit=160)}"
                )
            self._debug_prompt_count += 1

        return result
