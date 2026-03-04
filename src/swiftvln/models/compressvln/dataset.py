# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Dataset for Visual Language Navigation

This dataset extends StreamVLNDataset and uses standard <image> tokens.
The template's replace_tag method will convert them to <history_image> or
<current_image> based on the image index.

Key change from original design:
- Dataset uses standard <image> token (ms-swift compatible)
- Dataset provides num_history_images metadata
- Template handles the differentiation in replace_tag
"""

import os
import random
import torch
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from swiftvln.common.constants import DEFAULT_IMAGE_TOKEN
from swiftvln.models.streamvln.dataset import StreamVLNDataset


class CompressVLNDataset(StreamVLNDataset):
    """
    CompressVLN Dataset that uses standard <image> tokens.
    
    Inherits all logic from StreamVLNDataset. The differentiation between
    history and current images is handled by the template based on:
    - Image order: history images come first, then current images
    - num_history_images metadata in the returned dict
    
    The template's replace_tag method uses this info to return different
    tokens (<history_image> vs <current_image>).
    
    Supports precomputed ViT features for training acceleration:
    - use_precomputed_features: Load precomputed features instead of images
    - feature_cache_dir: Directory containing precomputed .pt files
    - feature_cache_size: LRU cache size for feature loading
    
    Args:
        env_type: Environment type - 'habitat' (forward=0.25m) or 'satnav' (forward=10m)
    """
    
    def __init__(self, *args, 
                 use_precomputed_features: bool = False,
                 feature_cache_dir: Optional[str] = None,
                 feature_cache_size: int = 100,
                 env_type: str = "habitat",
                 **kwargs):
        # Pass env_type to parent
        kwargs['env_type'] = env_type
        super().__init__(*args, **kwargs)
        
        # Precomputed features parameters
        self.use_precomputed_features = use_precomputed_features
        self.feature_cache_dir = feature_cache_dir
        self.feature_cache_size = feature_cache_size
        
        # Setup precomputed features if enabled
        if self.use_precomputed_features:
            self._setup_precomputed_features()
            print(f"[CompressVLN] use_precomputed_features=True")
            for vf, fd in self._feature_dirs.items():
                print(f"  feature_dir: {fd}")
            print(f"  feature_cache_size={self.feature_cache_size} episodes")
    
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
                episode_id = self._get_episode_id(ep_idx)
                print(f"Warning: Frame index {idx} out of range for episode {episode_id}")
                # Return empty tensor as fallback
                features_list.append(torch.zeros(1, 2048, dtype=torch.bfloat16))
                grid_thw_list.append(torch.tensor([1, 1, 1], dtype=torch.int64))
        
        return features_list, grid_thw_list
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample with standard <image> tokens.
        
        Returns:
            dict with keys:
                - messages: List of conversation turns with <image> tokens
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
        
        # Get actions (shift by 1 to predict next action)
        actions = data['actions'][1:] + [0]
        actions_len = len(actions)
        
        # Get time slice
        import numpy as np
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
        has_history = False
        if time_ids[0] != 0:
            current_start_abs = min(time_ids[0], num_video_frames)
            available_history_indices = np.arange(0, current_start_abs)
            num_to_sample = min(self.num_history, len(available_history_indices))
            
            if num_to_sample > 0:
                if self.use_random:
                    history_step_ids = np.random.choice(
                        available_history_indices,
                        size=num_to_sample,
                        replace=False
                    )
                    history_step_ids = np.sort(history_step_ids)
                else:
                    step = max(current_start_abs // self.num_history, 1)
                    history_step_ids = np.arange(0, current_start_abs, step)
                    if len(history_step_ids) > self.num_history:
                        indices = np.linspace(0, len(history_step_ids) - 1, self.num_history, dtype=int)
                        history_step_ids = history_step_ids[indices]
                
                history_step_ids = np.clip(history_step_ids, 0, num_video_frames - 1)
                history_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in history_step_ids]
                has_history = len(history_frame_paths) > 0
        
        # Branch: load images OR precomputed features
        if self.use_precomputed_features:
            # Collect all frame indices: history, then current
            all_frame_indices = list(history_step_ids) if has_history else []
            all_frame_indices.extend(sample_step_ids)
            
            # Load precomputed features
            features_list, grid_thw_list = self._get_frame_features(ep_id, all_frame_indices)
            
            if len(features_list) == 0:
                raise ValueError(f"No features loaded for sample {i}")
        else:
            # Load images as PIL Images
            from PIL import Image
            all_frame_paths = history_frame_paths + sample_frame_paths
            images = []
            for image_file in all_frame_paths:
                try:
                    image = Image.open(image_file).convert('RGB')
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Failed to load image {image_file}: {e}")
                    images.append(Image.new('RGB', (640, 480), color='black'))
            
            if len(images) == 0:
                raise ValueError(f"No images loaded for sample {i}")
        
        # Build conversation with standard <image> tokens
        # Use environment-specific forward distance from parent class
        system_prompt = (
            f"You are an autonomous navigation assistant. Your task is to {instruction}. "
            f"Based on your observations, output a sequence of actions using: "
            f"↑ (forward {self.forward_distance}), ← (turn left), → (turn right), or STOP (when goal is reached). "
            f"Output actions directly without explanation."
        )
        
        # Add history description with standard <image> tokens
        num_history_images = len(history_frame_paths)
        if has_history:
            # Use standard <image> token - template will convert to <history_image>
            history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
            system_prompt += f" These are your historical observations: {history_tokens}."
        
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Build multi-turn dialogue with standard <image> tokens
        current_actions_list = list(current_actions)
        num_current_images = len(sample_frame_paths)
        
        action_idx = 0
        image_idx = 0
        while action_idx < len(current_actions_list) and image_idx < num_current_images:
            # User turn with standard <image> token - template will convert to <current_image>
            conjunction = random.choice(self.conjunctions)
            user_content = f"{conjunction}{DEFAULT_IMAGE_TOKEN}."
            messages.append({'role': 'user', 'content': user_content})
            
            # Assistant turn with actions
            step_actions = current_actions_list[action_idx:action_idx + self.num_future_steps]
            if len(step_actions) == 0:
                step_actions = [0]  # STOP
            answer = self.actions2text(step_actions)
            messages.append({'role': 'assistant', 'content': answer})
            
            action_idx += len(step_actions)
            image_idx += 1
        
        # Return different keys based on mode
        if self.use_precomputed_features:
            # Precomputed features are passed via extra_kwargs to template
            # Use underscore prefix so they go into extra_kwargs (not a known StdTemplateInputs field)
            # Use tiny dummy images for replace_tag to work correctly (count-based)
            # These won't be processed by ViT since we use precomputed features
            from PIL import Image
            num_total_images = len(features_list)
            dummy_image = Image.new('RGB', (28, 28), color='black')  # Minimum size for Qwen2.5-VL
            dummy_images = [dummy_image] * num_total_images
            return {
                'messages': messages,
                'images': dummy_images,  # Needed for replace_tag to count images
                '_precomputed_features': features_list,  # Will go into extra_kwargs
                '_precomputed_grid_thw': grid_thw_list,  # Will go into extra_kwargs
                'num_history_images': num_history_images,  # Metadata for template
            }
        else:
            return {
                'messages': messages,
                'images': images,
                'num_history_images': num_history_images,  # Metadata for template
            }
