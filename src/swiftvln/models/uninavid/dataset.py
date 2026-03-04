# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Dataset for Visual Language Navigation

This dataset implements the UniNaVid-style three-level memory architecture:
- Long-term memory: Older history frames (will be merged by similarity)
- Short-term memory: Recent history frames (compressed with pooling)
- Current observation: Current frame (full resolution)

Key features:
- Sampling strategy same as MonoVLN (guaranteed start/end points)
- Uses ALL available history frames (no limit)
- Random frame dropping augmentation (10% probability)
- Action format: forward/left/right/stop (Uni-NaVid style)
- Returns num_long_term_images, num_short_term_images metadata for template
"""

import os
import json
import random
import numpy as np
import torch
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

# Constants
DEFAULT_IMAGE_TOKEN = "<image>"


class UniNaVidDataset(Dataset):
    """
    UniNaVid Dataset for Visual Language Navigation training.
    
    Returns data in ms-swift standard format with single-turn dialogue:
    {
        'messages': [
            {'role': 'user', 'content': 'Task: ... History: <image>... Current: <image>...'},
            {'role': 'assistant', 'content': 'forward forward left stop'}
        ],
        'images': [PIL.Image, ...],  # long-term, short-term, then current
        'num_long_term_images': int,   # metadata for template
        'num_short_term_images': int,  # metadata for template
    }
    
    Args:
        data_path: Path(s) to VLN data directory containing annotations.json.
                   Supports multiple paths separated by comma.
        short_term_frames: Number of recent frames as short-term memory (default: 64)
        num_future_steps: Number of actions to predict per step (default: 4)
        samples_per_episode: Number of time points to sample per episode (default: 8)
        drop_frame_prob: Probability of dropping each history frame (default: 0.1)
        max_samples: Limit number of training samples
    """
    
    def __init__(
        self,
        data_path: str,
        short_term_frames: int = 64,
        num_future_steps: int = 4,
        samples_per_episode: int = 8,
        drop_frame_prob: float = 0.1,
        max_samples: Optional[int] = None,
        image_resize_stride: float = 1.0,
        history_frame_stride: int = 1,
        use_precomputed_features: bool = False,
        feature_cache_dir: Optional[str] = None,
        feature_cache_size: int = 100,
    ):
        super(UniNaVidDataset, self).__init__()
        
        # UniNaVid parameters
        self.short_term_frames = short_term_frames
        self.num_future_steps = num_future_steps
        self.samples_per_episode = samples_per_episode
        self.drop_frame_prob = drop_frame_prob
        self.max_samples = max_samples
        self.image_resize_stride = image_resize_stride
        self.history_frame_stride = history_frame_stride
        
        # Precomputed features parameters
        self.use_precomputed_features = use_precomputed_features
        self.feature_cache_dir = feature_cache_dir
        self.feature_cache_size = feature_cache_size
        
        # Load navigation data from multiple paths (comma-separated)
        self.video_folders = [p.strip() for p in data_path.split(',') if p.strip()]
        self.nav_data = []
        for vf in self.video_folders:
            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"Warning: {anno_path} not found, skipping...")
                continue
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
            print(f"Loaded {len(anno_json)} episodes from {vf}")
        
        # Build data index: (episode_id, instruction_id, current_idx)
        # Sampling strategy same as MonoVLN
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            actions = item.get('actions', [])
            actions_len = len(actions)
            
            if actions_len < self.num_future_steps:
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            for ins_id in range(len(instructions)):
                # Sampling with guaranteed start and end points
                # Start: index 0 (no history)
                # End: actions_len - num_future_steps (ensures STOP at the end)
                start_idx = 0
                end_idx = max(0, actions_len - self.num_future_steps)
                
                if end_idx <= start_idx:
                    # Episode too short, just use start
                    sample_indices = [start_idx]
                elif self.samples_per_episode <= 2:
                    sample_indices = [start_idx, end_idx]
                else:
                    # Middle samples: randomly selected from (start_idx+1, end_idx-1)
                    middle_count = self.samples_per_episode - 2
                    middle_range = list(range(start_idx + 1, end_idx))
                    
                    if len(middle_range) <= middle_count:
                        middle_indices = middle_range
                    else:
                        middle_indices = sorted(random.sample(middle_range, middle_count))
                    
                    sample_indices = [start_idx] + middle_indices + [end_idx]
                
                for idx in sample_indices:
                    self.data_list.append((ep_id, ins_id, idx))
        
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
            self.data_list = self.data_list[:self.max_samples]
            print(f"Limited dataset from {original_len} to {len(self.data_list)} samples (max_samples={self.max_samples})")
        
        # Action vocabulary (Uni-NaVid style: text format)
        self.idx2actions = {
            0: 'stop',
            1: 'forward',
            2: 'left',
            3: 'right',
        }
        
        # Setup precomputed features if enabled
        if self.use_precomputed_features:
            self._setup_precomputed_features()
        
        print(f"UniNaVidDataset initialized: {len(self.data_list)} samples from {len(self.nav_data)} episodes")
        print(f"  samples_per_episode={samples_per_episode}, short_term_frames={short_term_frames}")
        print(f"  num_future_steps={num_future_steps}, drop_frame_prob={drop_frame_prob}")
        print(f"  image_resize_stride={image_resize_stride}, history_frame_stride={history_frame_stride}")
        if self.use_precomputed_features:
            print(f"  use_precomputed_features=True")
            for vf, fd in self._feature_dirs.items():
                print(f"    feature_dir: {fd}")
            print(f"  feature_cache_size={self.feature_cache_size} episodes")
    
    def __len__(self):
        return len(self.data_list)
    
    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text (Uni-NaVid style: space-separated)."""
        if len(actions) == 0:
            return "stop"
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(int(action), "stop")
            converted_sequence.append(act_text)
        return ' '.join(converted_sequence)
    
    def _apply_random_drop(self, frame_paths: List[str]) -> List[str]:
        """
        Apply random frame dropping augmentation (Uni-NaVid style).
        
        Args:
            frame_paths: List of frame paths
            
        Returns:
            Filtered list with some frames randomly dropped
        """
        if self.drop_frame_prob <= 0 or len(frame_paths) == 0:
            return frame_paths
        
        # Keep frames with probability (1 - drop_frame_prob)
        kept = [p for p in frame_paths if random.random() > self.drop_frame_prob]
        
        # Ensure at least one frame is kept if original list was non-empty
        if len(kept) == 0 and len(frame_paths) > 0:
            kept = [random.choice(frame_paths)]
        
        return kept
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image by the configured stride to reduce VIT memory usage.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized PIL Image (or original if stride is 1.0)
            
        Examples:
            - stride=1.0: 640x480 -> 640x480 (no resize)
            - stride=1.5: 640x480 -> 427x320
            - stride=2.0: 640x480 -> 320x240
            - stride=3.0: 640x480 -> 213x160
        """
        if self.image_resize_stride <= 1.0:
            return image
        
        orig_w, orig_h = image.size
        new_w = int(orig_w / self.image_resize_stride)
        new_h = int(orig_h / self.image_resize_stride)
        
        # Ensure minimum size
        new_w = max(new_w, 28)  # Qwen2.5-VL minimum patch size
        new_h = max(new_h, 28)
        
        return image.resize((new_w, new_h), Image.BILINEAR)
    
    def _setup_precomputed_features(self):
        """
        Setup precomputed features loading with LRU cache.
        
        For multi-dataset support, each dataset uses its own features/ directory.
        Creates an LRU-cached method to load episode features.
        """
        # Build mapping from video_folder to its features directory
        self._feature_dirs = {}
        for video_folder in self.video_folders:
            feature_dir = os.path.join(video_folder, 'features')
            if os.path.exists(feature_dir):
                self._feature_dirs[video_folder] = feature_dir
            else:
                print(f"Warning: Feature directory not found: {feature_dir}")
        
        if not self._feature_dirs:
            raise FileNotFoundError(
                f"No feature directories found in any of the data paths.\n"
                f"Please run precompute_features.py first to generate ViT features."
            )
        
        # Create LRU-cached loader for episode features
        # We use a closure to make it an instance method with lru_cache
        @lru_cache(maxsize=self.feature_cache_size)
        def _load_episode_features_cached(feature_path: str) -> Dict[str, Any]:
            """Load precomputed features for an episode (LRU cached)."""
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
            # Load to CPU memory (will be moved to GPU during training)
            return torch.load(feature_path, map_location='cpu', weights_only=True)
        
        self._load_episode_features_cached = _load_episode_features_cached
        
        # Verify a few feature files exist
        sample_count = min(3, len(self.data_list))
        for i in range(sample_count):
            ep_id, _, _ = self.data_list[i]
            feature_path = self._get_feature_path(ep_id)
            if feature_path is None or not os.path.exists(feature_path):
                episode_id = self._get_episode_id(ep_id)
                print(f"Warning: Feature file not found for episode {episode_id}")
    
    def _get_episode_id(self, ep_idx: int) -> str:
        """
        Get the episode ID string for feature file lookup.
        
        Uses the video folder name as the episode identifier.
        """
        data = self.nav_data[ep_idx]
        video_path = data['video']
        # Extract episode name from video path (last directory component)
        episode_id = os.path.basename(video_path.rstrip('/'))
        return episode_id
    
    def _get_feature_path(self, ep_idx: int) -> Optional[str]:
        """
        Get the full path to the feature file for an episode.
        
        Determines which dataset the episode belongs to and returns
        the corresponding feature file path.
        """
        data = self.nav_data[ep_idx]
        video_path = data['video']
        episode_id = os.path.basename(video_path.rstrip('/'))
        
        # Find which video_folder this episode belongs to
        for video_folder, feature_dir in self._feature_dirs.items():
            # Check if video_path starts with this video_folder
            if video_path.startswith(video_folder):
                return os.path.join(feature_dir, f'{episode_id}.pt')
        
        # Fallback: try all feature directories
        for feature_dir in self._feature_dirs.values():
            feature_path = os.path.join(feature_dir, f'{episode_id}.pt')
            if os.path.exists(feature_path):
                return feature_path
        
        return None
    
    def _get_frame_features(
        self, 
        ep_idx: int, 
        frame_indices: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get precomputed features for specific frames from cache.
        
        Args:
            ep_idx: Episode index in nav_data
            frame_indices: List of frame indices to retrieve
            
        Returns:
            Tuple of (features_list, grid_thw_list)
            - features_list: List of [num_tokens, hidden_size] tensors
            - grid_thw_list: List of [3] tensors with grid dimensions
        """
        feature_path = self._get_feature_path(ep_idx)
        if feature_path is None:
            episode_id = self._get_episode_id(ep_idx)
            raise FileNotFoundError(f"Feature file not found for episode {episode_id}")
        
        episode_data = self._load_episode_features_cached(feature_path)
        
        features_list = []
        grid_thw_list = []
        
        vit_features = episode_data['vit_features']
        grid_thws = episode_data['grid_thw']
        
        for idx in frame_indices:
            if idx < len(vit_features):
                features_list.append(vit_features[idx])
                grid_thw_list.append(grid_thws[idx])
            else:
                # Frame index out of range - should not happen with proper data
                print(f"Warning: Frame index {idx} out of range for episode {episode_id}")
                # Return empty tensor as fallback
                features_list.append(torch.zeros(1, 2048, dtype=torch.bfloat16))
                grid_thw_list.append(torch.tensor([1, 1, 1], dtype=torch.int64))
        
        return features_list, grid_thw_list
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample in ms-swift standard format with single-turn dialogue.
        
        Returns:
            dict with keys (image mode):
                - messages: Single-turn dialogue [user, assistant]
                - images: List of PIL Images (long-term, short-term, then current)
                - num_long_term_images: Number of long-term memory images
                - num_short_term_images: Number of short-term memory images
            
            dict with keys (precomputed features mode):
                - messages: Single-turn dialogue [user, assistant]
                - precomputed_features: List of [num_tokens, hidden] tensors
                - precomputed_grid_thw: List of [3] tensors
                - num_long_term_images: Number of long-term memory images
                - num_short_term_images: Number of short-term memory images
        """
        # Get sample index
        ep_id, ins_id, current_idx = self.data_list[i]

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
        
        # Get actions (shift by 1 to predict next action)
        actions = data['actions'][1:] + [0]  # Append STOP at the end
        
        # Get action sequence starting from current_idx
        action_slice = actions[current_idx:current_idx + self.num_future_steps]
        if len(action_slice) == 0:
            action_slice = [0]  # STOP
        
        # Pad with STOP if needed
        while len(action_slice) < self.num_future_steps:
            action_slice.append(0)
        
        # Get ALL history frame indices (UniNaVid uses all history, not limited)
        history_frame_indices = []
        if current_idx > 0:
            for idx in range(current_idx):
                frame_idx = min(idx, num_video_frames - 1)
                history_frame_indices.append(frame_idx)
        
        # Apply temporal downsampling (before random drop)
        # history_frame_stride=2 means take every 2nd frame
        if self.history_frame_stride > 1 and len(history_frame_indices) > 0:
            history_frame_indices = history_frame_indices[::self.history_frame_stride]
        
        # Apply random frame dropping augmentation (for image mode only, features use indices)
        if not self.use_precomputed_features:
            # Convert to paths for random drop
            history_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in history_frame_indices]
            history_frame_paths = self._apply_random_drop(history_frame_paths)
            # Convert back to indices for classification
            history_frame_indices = []
            for path in history_frame_paths:
                frame_name = os.path.basename(path)
                if frame_name in video_frames:
                    history_frame_indices.append(video_frames.index(frame_name))
        else:
            # For precomputed features, apply random drop directly on indices
            if self.drop_frame_prob > 0 and len(history_frame_indices) > 0:
                kept = [idx for idx in history_frame_indices if random.random() > self.drop_frame_prob]
                if len(kept) == 0 and len(history_frame_indices) > 0:
                    kept = [random.choice(history_frame_indices)]
                history_frame_indices = kept
        
        # Classify frames into long-term and short-term memory
        total_history = len(history_frame_indices)
        if total_history <= self.short_term_frames:
            # All history fits in short-term memory
            long_term_indices = []
            short_term_indices = history_frame_indices
        else:
            # Split into long-term and short-term
            long_term_indices = history_frame_indices[:-self.short_term_frames]
            short_term_indices = history_frame_indices[-self.short_term_frames:]
        
        # Current frame
        current_frame_idx = min(current_idx, num_video_frames - 1)
        
        num_long_term_images = len(long_term_indices)
        num_short_term_images = len(short_term_indices)
        
        # Branch: load images OR precomputed features
        if self.use_precomputed_features:
            # Collect all frame indices: long-term, short-term, current
            all_frame_indices = long_term_indices + short_term_indices + [current_frame_idx]
            
            # Load precomputed features
            features_list, grid_thw_list = self._get_frame_features(ep_id, all_frame_indices)
            
            if len(features_list) == 0:
                raise ValueError(f"No features loaded for sample {i}")
        else:
            # Load images as PIL Images (original path)
            images = []
            
            # Load long-term memory images first
            for frame_idx in long_term_indices:
                image_file = os.path.join(rgb_path, video_frames[frame_idx])
                try:
                    image = Image.open(image_file).convert('RGB')
                    image = self._resize_image(image)
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Failed to load long-term image {image_file}: {e}")
                    images.append(Image.new('RGB', (640, 480), color='black'))
            
            # Load short-term memory images
            for frame_idx in short_term_indices:
                image_file = os.path.join(rgb_path, video_frames[frame_idx])
                try:
                    image = Image.open(image_file).convert('RGB')
                    image = self._resize_image(image)
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Failed to load short-term image {image_file}: {e}")
                    images.append(Image.new('RGB', (640, 480), color='black'))
            
            # Load current image
            current_frame_path = os.path.join(rgb_path, video_frames[current_frame_idx])
            try:
                current_image = Image.open(current_frame_path).convert('RGB')
                current_image = self._resize_image(current_image)
                images.append(current_image)
            except Exception as e:
                print(f"Warning: Failed to load current image {current_frame_path}: {e}")
                images.append(Image.new('RGB', (640, 480), color='black'))
            
            if len(images) == 0:
                raise ValueError(f"No images loaded for sample {i}")
        
        # Build image tokens for prompt
        long_term_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_long_term_images)])
        short_term_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_short_term_images)])
        total_history_count = num_long_term_images + num_short_term_images
        
        # Action definition (Uni-NaVid style)
        action_definition = (
            "You are an intelligent navigation robot. "
            "Your goal is to navigate to a target location based on the instruction.\n"
            "Available Actions:\n"
            "- forward: Move forward 0.25 meters.\n"
            "- left: Turn left 15 degrees.\n"
            "- right: Turn right 15 degrees.\n"
            "- stop: Use this ONLY when you have reached the goal."
        )
        
        if total_history_count > 0:
            # Build history description (Option B: natural language with layered description)
            if num_long_term_images > 0 and num_short_term_images > 0:
                history_section = (
                    f"### Earlier Observations ({num_long_term_images} frames)\n"
                    f"These images are from the beginning of your journey:\n"
                    f"{long_term_tokens}\n\n"
                    f"### Recent Observations ({num_short_term_images} frames)\n"
                    f"These are your most recent views before the current moment:\n"
                    f"{short_term_tokens}"
                )
            elif num_short_term_images > 0:
                history_section = (
                    f"### Recent Observations ({num_short_term_images} frames)\n"
                    f"These are your views from the journey so far:\n"
                    f"{short_term_tokens}"
                )
            else:
                history_section = (
                    f"### Earlier Observations ({num_long_term_images} frames)\n"
                    f"These images are from the beginning of your journey:\n"
                    f"{long_term_tokens}"
                )
            
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"{history_section}\n\n"
                f"### Current Observation\n"
                f"This is what you see right now:\n"
                f"{DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"Based on your journey history and current view, predict the next {self.num_future_steps} actions "
                f"(space-separated, e.g., forward forward left stop):"
            )
        else:
            # First observation at the starting point
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Current Observation\n"
                f"This is what you see at the starting point:\n"
                f"{DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"You are at the starting point. Analyze the instruction and the current view. "
                f"Predict the next {self.num_future_steps} actions (space-separated):"
            )
        
        # Assistant response: action sequence (Uni-NaVid style: space-separated)
        action_sequence = self.actions2text(action_slice)
        
        messages = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': action_sequence}
        ]
        
        # Return different keys based on mode
        if self.use_precomputed_features:
            # Precomputed features are passed via extra_kwargs to template
            # Use underscore prefix so they go into extra_kwargs (not a known StdTemplateInputs field)
            # Use tiny dummy images for replace_tag to work correctly (count-based)
            # These won't be processed by ViT since we use precomputed features
            num_total_images = len(features_list)
            dummy_image = Image.new('RGB', (28, 28), color='black')  # Minimum size for Qwen2.5-VL
            dummy_images = [dummy_image] * num_total_images
            return {
                'messages': messages,
                'images': dummy_images,  # Needed for replace_tag to count images
                '_precomputed_features': features_list,  # Will go into extra_kwargs
                '_precomputed_grid_thw': grid_thw_list,  # Will go into extra_kwargs
                'num_long_term_images': num_long_term_images,
                'num_short_term_images': num_short_term_images,
            }
        else:
            return {
                'messages': messages,
                'images': images,
                'num_long_term_images': num_long_term_images,
                'num_short_term_images': num_short_term_images,
            }
