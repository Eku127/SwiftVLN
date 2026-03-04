# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN Dataset Components

Provides common dataset utilities shared across all VLN variants.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    CURRENT_IMAGE_TOKEN,
    DEFAULT_ACTION_MAP,
    DEFAULT_CONJUNCTIONS,
    DEFAULT_IMAGE_TOKEN,
    HISTORY_MEMORY_TOKEN,
)


# ============================================================================
# Base Dataset Class
# ============================================================================

class BaseVLNDataset(Dataset):
    """
    Base class for VLN datasets.
    
    Provides common functionality:
    - Data loading from annotations.json
    - Action to text conversion
    - Environment-specific forward distance
    - Historical frame sampling
    
    Subclasses should implement:
    - _build_data_list(): Build the list of (episode_id, instruction_id, start_frame) tuples
    - __getitem__(): Return training sample
    """
    
    def __init__(
        self,
        data_path: str,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        use_random: bool = False,
        max_samples: Optional[int] = None,
        env_type: str = "habitat",
    ):
        super().__init__()
        
        # VLN parameters
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.use_random = use_random
        self.max_samples = max_samples
        self.env_type = env_type.lower()
        
        # Set forward distance based on environment type
        self.forward_distance = "10m" if self.env_type == "satnav" else "0.25m"
        
        # Load navigation data
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
        
        # Action vocabulary
        self.idx2actions = DEFAULT_ACTION_MAP.copy()
        
        # Prompt templates
        self.conjunctions = DEFAULT_CONJUNCTIONS.copy()
        
        # Build data index - to be implemented by subclasses
        self.data_list = []
        self._build_data_list()
        
        # Apply max_samples limit
        self._apply_max_samples()
    
    def _build_data_list(self):
        """
        Build the list of (episode_id, instruction_id, start_frame) tuples.
        
        Default implementation: non-overlapping windows.
        Subclasses can override for different sampling strategies.
        """
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            actions = item.get('actions', [])
            actions_len = len(actions)
            
            if actions_len < 4:
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            for ins_id in range(len(instructions)):
                # Segment trajectory into num_frames windows
                num_rounds = actions_len // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames))
    
    def _apply_max_samples(self):
        """Apply max_samples limit to data_list."""
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
                # Random sampling for better diversity
                import random as _random
                _rng = _random.Random(42)
                self.data_list = _rng.sample(self.data_list, self.max_samples)
                self.data_list.sort()
                print(f"Random sampled {self.max_samples} from {original_len} samples (seed=42)")
    
    def __len__(self):
        return len(self.data_list)
    
    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text symbols."""
        if len(actions) == 0:
            return "STOP"
        return ''.join(self.idx2actions.get(int(a), "STOP") for a in actions)
    
    def _load_images(self, frame_paths: List[str]) -> List[Image.Image]:
        """Load images from paths as PIL Images."""
        images = []
        for path in frame_paths:
            try:
                images.append(Image.open(path).convert('RGB'))
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
                images.append(Image.new('RGB', (640, 480), color='black'))
        return images
    
    def _sample_history_frames(
        self, 
        rgb_path: str, 
        video_frames: List[str],
        current_start: int,
    ) -> List[str]:
        """
        Sample historical frames before current_start.
        
        Args:
            rgb_path: Path to RGB frames directory
            video_frames: Sorted list of frame filenames
            current_start: Current window start index
            
        Returns:
            List of historical frame paths
        """
        if current_start <= 0:
            return []
        
        num_video_frames = len(video_frames)
        available_indices = np.arange(0, min(current_start, num_video_frames))
        num_to_sample = min(self.num_history, len(available_indices))
        
        if num_to_sample == 0:
            return []
        
        if self.use_random:
            history_ids = np.random.choice(available_indices, size=num_to_sample, replace=False)
            history_ids = np.sort(history_ids)
        else:
            step = max(current_start // self.num_history, 1)
            history_ids = np.arange(0, current_start, step)
            if len(history_ids) > self.num_history:
                indices = np.linspace(0, len(history_ids) - 1, self.num_history, dtype=int)
                history_ids = history_ids[indices]
        
        history_ids = np.clip(history_ids, 0, num_video_frames - 1)
        return [os.path.join(rgb_path, video_frames[idx]) for idx in history_ids]
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            dict with keys: messages, images, and optionally other metadata
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
