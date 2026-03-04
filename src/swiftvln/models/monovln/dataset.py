# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Dataset for Visual Language Navigation

This dataset is independent of StreamVLN and implements single-turn dialogue format.

Key features:
- Directly inherits from torch.utils.data.Dataset (no StreamVLN dependency)
- Single-turn dialogue: user message (task + history + current) → assistant (actions)
- Uniform sampling with guaranteed start (no history) and end (with STOP) points
- History frame sampling: use all if < num_history, else uniform sample

The template's replace_tag method will convert <image> tokens to <history_image> or
<current_image> based on the image index.
"""

import os
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
from torch.utils.data import Dataset

# Constants
DEFAULT_IMAGE_TOKEN = "<image>"


class MonoVLNDataset(Dataset):
    """
    MonoVLN Dataset for Visual Language Navigation training.
    
    Returns data in ms-swift standard format with single-turn dialogue:
    {
        'messages': [
            {'role': 'user', 'content': 'Task: ... History: <image>... Current: <image>...'},
            {'role': 'assistant', 'content': '↑↑←STOP'}
        ],
        'images': [PIL.Image, ...],  # history images first, then current image
        'num_history_images': int     # metadata for template
    }
    
    Args:
        data_path: Path(s) to VLN data directory containing annotations.json.
                   Supports multiple paths separated by comma.
        num_history: Maximum number of historical frames to sample
        num_future_steps: Number of actions to predict per step
        samples_per_episode: Number of time points to sample per episode
        max_samples: Limit number of training samples
    """
    
    def __init__(
        self,
        data_path: str,
        num_history: int = 8,
        num_future_steps: int = 4,
        samples_per_episode: int = 8,
        max_samples: Optional[int] = None,
    ):
        super(MonoVLNDataset, self).__init__()
        
        # VLN parameters (MonoVLN uses single-frame inference, no num_frames)
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.samples_per_episode = samples_per_episode
        self.max_samples = max_samples
        
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
                    middle_range = list(range(start_idx + 1, end_idx))  # [1, 2, ..., end_idx-1]
                    
                    if len(middle_range) <= middle_count:
                        # Not enough middle points, use all available
                        middle_indices = middle_range
                    else:
                        # Randomly sample middle_count points from middle_range
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
        
        # Action vocabulary
        self.idx2actions = {
            0: 'STOP',
            1: "↑",  # MOVE_FORWARD
            2: "←",  # TURN_LEFT
            3: "→",  # TURN_RIGHT
        }
        
        print(f"MonoVLNDataset initialized: {len(self.data_list)} samples from {len(self.nav_data)} episodes")
        print(f"  samples_per_episode={samples_per_episode}, num_history={num_history}, num_future_steps={num_future_steps}")
    
    def __len__(self):
        return len(self.data_list)
    
    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text symbols."""
        if len(actions) == 0:
            return "STOP"
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(int(action), "STOP")
            converted_sequence.append(act_text)
        return ''.join(converted_sequence)
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample in ms-swift standard format with single-turn dialogue.
        
        Returns:
            dict with keys:
                - messages: Single-turn dialogue [user, assistant]
                - images: List of PIL Images (history first, then current)
                - num_history_images: Number of history images (for template)
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
        actions_len = len(actions)
        
        # Get action sequence starting from current_idx
        action_slice = actions[current_idx:current_idx + self.num_future_steps]
        if len(action_slice) == 0:
            action_slice = [0]  # STOP
        
        # Pad with STOP if needed
        while len(action_slice) < self.num_future_steps:
            action_slice.append(0)
        
        # Sample history frames
        history_frame_paths = []
        if current_idx > 0:
            # Available history: frames from 0 to current_idx - 1
            available_count = current_idx
            
            if available_count <= self.num_history:
                # Use all available history frames
                history_indices = list(range(available_count))
            else:
                # Uniform sampling of num_history frames
                history_indices = np.linspace(0, available_count - 1, self.num_history, dtype=int)
                history_indices = list(history_indices)
            
            # Clip to valid range and get frame paths
            for idx in history_indices:
                frame_idx = min(idx, num_video_frames - 1)
                history_frame_paths.append(os.path.join(rgb_path, video_frames[frame_idx]))
        
        # Current frame (only 1 frame)
        current_frame_idx = min(current_idx, num_video_frames - 1)
        current_frame_path = os.path.join(rgb_path, video_frames[current_frame_idx])
        
        # Load images as PIL Images
        images = []
        
        # Load history images first
        for image_file in history_frame_paths:
            try:
                image = Image.open(image_file).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load history image {image_file}: {e}")
                images.append(Image.new('RGB', (640, 480), color='black'))
        
        num_history_images = len(images)
        
        # Load current image
        try:
            current_image = Image.open(current_frame_path).convert('RGB')
            images.append(current_image)
        except Exception as e:
            print(f"Warning: Failed to load current image {current_frame_path}: {e}")
            images.append(Image.new('RGB', (640, 480), color='black'))
        
        if len(images) == 0:
            raise ValueError(f"No images loaded for sample {i}")
        
        # Build single-turn dialogue (no system message)
        # User message contains: action definition, task, observations, prediction request
        history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
        
        # Action definition with specific movement parameters
        action_definition = (
            "You are an intelligent navigation robot. "
            "Your goal is to navigate to a target location based on the instruction.\n"
            "Available Actions:\n"
            "- ↑ (Forward): Move forward 0.25 meters.\n"
            "- ← (Left): Turn left 15 degrees.\n"
            "- → (Right): Turn right 15 degrees.\n"
            "- STOP: Use this ONLY when you have reached the goal."
        )
        
        if num_history_images > 0:
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Trajectory History\n"
                f"The following tokens represent your past views and actions:\n"
                f"{history_tokens}\n\n"
                f"### Current View\n"
                f"Current Observation: {DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"Based on the history and current view, predict the next {self.num_future_steps} actions sequence (e.g., ↑, ↑, →, STOP):"
            )
        else:
            # First observation at the starting point
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Current View\n"
                f"Current Observation: {DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"You are at the starting point. Analyze the instruction and the current view. "
                f"Predict the next {self.num_future_steps} actions sequence:"
            )
        
        # Assistant response: action sequence
        action_sequence = self.actions2text(action_slice)
        
        messages = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': action_sequence}
        ]
        
        return {
            'messages': messages,
            'images': images,
            'num_history_images': num_history_images,  # Metadata for template
        }
