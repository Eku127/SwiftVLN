# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Uniform Frame Sampling + ToMe Compression Memory Strategy

This strategy:
1. Uniformly samples N frames from history (same as uniform)
2. Compresses each frame's VIT features using Grid-based ToMe (Token Merging)

ToMe uses soft k-means within spatial grid cells for better semantic preservation
compared to simple average pooling.
"""

import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .base import BaseMemoryStrategy

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = os.path.dirname(_current_dir)
_feature_test_dir = os.path.dirname(_mem_test_dir)
_vln_dir = os.path.dirname(_feature_test_dir)
_common_dir = os.path.join(_vln_dir, 'common')

if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)

# Import the compressor from common
from common.compressor import HistoryTokenCompressor


class UniformToMeMemoryStrategy(BaseMemoryStrategy):
    """
    Uniform frame sampling + ToMe compression strategy.
    
    This strategy:
    1. Uniformly samples num_history frames from available history
    2. Extracts VIT features for each frame
    3. Compresses features using Grid-based ToMe (Token Merging)
    
    ToMe benefits over pooling:
    - Better semantic preservation for small objects
    - Maintains spatial structure (left/right/top/bottom)
    - Uses attention-weighted aggregation instead of simple averaging
    """
    
    def __init__(
        self,
        num_history: int = 8,
        compress_stride: int = 2,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        grid_size: int = 2,  # ToMe-specific: grid division
    ):
        """
        Initialize the UniformToMe strategy.
        
        Args:
            num_history: Number of history frames to use
            compress_stride: Compression stride for ToMe
            device: Device to use for computation
            dtype: Data type for tensors
            grid_size: Grid size for ToMe (splits image into grid_size x grid_size regions)
        """
        super().__init__(num_history, compress_stride, device, dtype)
        self.grid_size = grid_size
        
        # Create compressor with ToMe method
        self.compressor = HistoryTokenCompressor(
            stride=compress_stride,
            method='tome',
            grid_size=grid_size
        )
    
    @property
    def name(self) -> str:
        return "uniform_tome"
    
    def select_history_frames(
        self,
        total_history: int,
    ) -> List[int]:
        """
        Uniformly select history frames (same as uniform strategy).
        
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
        Compress VIT features using Grid-based ToMe.
        
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
        
        # Use ToMe compression from common/compressor.py
        compressed, new_grid_thw = self.compressor.compress_grid_tome(
            features, grid_thw, self.compress_stride
        )
        
        return compressed.to(features.dtype)
    
    def get_compressed_token_count(self, grid_thw: torch.Tensor) -> int:
        """
        Get the number of tokens after ToMe compression.
        
        Args:
            grid_thw: Grid dimensions [t, h, w]
            
        Returns:
            Number of compressed tokens
        """
        t, h, w = grid_thw.tolist()
        return self.compressor.get_compressed_token_count(
            int(t), int(h), int(w), self.compress_stride
        )


# For backward compatibility and easy import
UniformToMe = UniformToMeMemoryStrategy
