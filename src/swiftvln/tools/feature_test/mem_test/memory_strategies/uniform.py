# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Uniform Frame Memory Strategy

This is the baseline memory strategy that:
1. Uniformly samples N frames from history
2. Compresses each frame's VIT features using average pooling

This matches the default behavior of OverlapVLN training.
"""

import os
import sys
from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseMemoryStrategy

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = os.path.dirname(_current_dir)
_vln_dir = os.path.dirname(os.path.dirname(_mem_test_dir))
if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)


class UniformMemoryStrategy(BaseMemoryStrategy):
    """
    Uniform frame sampling memory strategy (baseline).
    
    This strategy:
    1. Uniformly samples num_history frames from available history
    2. Extracts VIT features for each frame
    3. Compresses features using spatial average pooling
    
    This is the default memory strategy used in OverlapVLN training,
    serving as the baseline for comparison.
    """
    
    @property
    def name(self) -> str:
        return "uniform"
    
    def select_history_frames(
        self,
        total_history: int,
    ) -> List[int]:
        """
        Uniformly select history frames.
        
        If total_history <= num_history, select all frames.
        Otherwise, uniformly sample num_history frames.
        
        Args:
            total_history: Total number of available history frames
            
        Returns:
            List of frame indices (sorted)
        """
        if total_history == 0:
            return []
        
        if total_history <= self.num_history:
            # Use all available frames
            return list(range(total_history))
        
        # Uniform sampling
        step = total_history / self.num_history
        indices = []
        for i in range(self.num_history):
            idx = int(i * step)
            idx = min(idx, total_history - 1)
            indices.append(idx)
        
        # Ensure sorted and unique
        indices = sorted(set(indices))
        
        return indices
    
    def compress_features(
        self,
        features: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress VIT features using spatial average pooling.
        
        Args:
            features: VIT features [num_tokens, hidden_size]
            grid_thw: Grid dimensions [t, h, w] after merge
            
        Returns:
            Compressed features [compressed_tokens, hidden_size]
        """
        if features.shape[0] == 0:
            return features
        
        t, h, w = grid_thw.tolist()
        t, h, w = int(t), int(h), int(w)
        
        # Reshape to spatial grid: [t*h*w, hidden] -> [t, h, w, hidden]
        hidden_size = features.shape[-1]
        
        # Handle case where tokens don't match expected grid
        expected_tokens = t * h * w
        actual_tokens = features.shape[0]
        
        if actual_tokens != expected_tokens:
            # Fallback: simple stride-based downsampling
            stride = self.compress_stride ** 2
            if stride > 1 and actual_tokens >= stride:
                indices = torch.arange(0, actual_tokens, stride, device=features.device)
                return features[indices]
            return features
        
        # Reshape to spatial grid
        features_grid = features.view(t, h, w, hidden_size)
        
        # Apply average pooling with stride
        stride = self.compress_stride
        if stride > 1 and h >= stride and w >= stride:
            # Permute to [t, hidden, h, w] for pooling
            features_grid = features_grid.permute(0, 3, 1, 2)  # [t, hidden, h, w]
            
            # Average pooling
            pooled = F.avg_pool2d(
                features_grid.float(),
                kernel_size=stride,
                stride=stride,
            )
            
            # Permute back and flatten
            pooled = pooled.permute(0, 2, 3, 1)  # [t, h', w', hidden]
            compressed = pooled.reshape(-1, hidden_size)
            
            return compressed.to(features.dtype)
        
        # No compression needed
        return features.reshape(-1, hidden_size)


# For backward compatibility and easy import
Uniform = UniformMemoryStrategy
