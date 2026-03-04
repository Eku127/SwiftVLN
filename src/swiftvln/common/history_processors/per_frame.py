# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Per-Frame Compression Processor

Compresses each frame independently, then concatenates results.
Supports both average pooling and Grid-based ToMe.

Sampling Strategy:
- Uses power transformation for flexible frame sampling
- log_base=1.0: Uniform sampling (default, linear distribution)
- log_base>1.0: Logarithmic sampling (more recent frames preserved)

Formula: t_frame = 1 - (1 - t_sample)^log_base
- When log_base=1.0: t_frame = t_sample (perfectly linear/uniform)
- When log_base>1.0: More samples concentrated at recent end
"""

import math
import torch
from typing import List, Tuple, Literal

from .base import HistoryProcessor


class PerFrameCompressor(HistoryProcessor):
    """
    Per-frame compression processor with flexible sampling.
    
    Each frame is compressed independently using pooling or ToMe,
    then concatenated together. This is the default/baseline approach.
    
    Features:
    - Flexible sampling: uniform (log_base=1.0) or logarithmic (log_base>1.0)
    - Preserves temporal order (frames stay separate)
    - Backward compatible with existing code
    - Supports 'pooling' (fast) and 'tome' (better semantic preservation)
    
    Args:
        stride: Compression stride (2 = 4x compression per frame)
        method: 'pooling' or 'tome'
        num_history: Number of frames to sample from history
        log_base: Sampling distribution base (1.0=uniform, >1.0=log, more recent)
        grid_size: Grid size for ToMe method
    """
    
    def __init__(
        self,
        stride: int = 2,
        method: Literal['pooling', 'tome'] = 'pooling',
        num_history: int = 8,
        log_base: float = 1.0,
        grid_size: int = 2,
    ):
        """
        Initialize per-frame compressor.
        
        Args:
            stride: Compression stride (2 = 4x compression per frame)
            method: 'pooling' or 'tome'
            num_history: Number of frames to sample from history
            log_base: Sampling distribution (1.0=uniform, >1.0=logarithmic)
            grid_size: Grid size for ToMe method
        """
        # Lazy import to avoid circular dependency
        from ..compressor import HistoryTokenCompressor
        
        self.stride = stride
        self.method = method
        self.num_history = num_history
        self.log_base = log_base
        self.grid_size = grid_size
        self._compressor = HistoryTokenCompressor(
            stride=stride,
            method=method,
            grid_size=grid_size,
        )
    
    @property
    def name(self) -> str:
        sampling = "uniform" if self.log_base == 1.0 else f"log(b={self.log_base})"
        return f"PerFrame(h={self.num_history}, {sampling}, {self.method}, s={self.stride})"
    
    def sample_indices(self, num_frames: int, num_samples: int = None) -> List[int]:
        """
        Sample frame indices using power transformation.
        
        When log_base=1.0, this gives uniform/linear sampling.
        When log_base>1.0, this gives logarithmic sampling (more recent frames).
        
        Formula: t_frame = 1 - (1 - t_sample)^log_base
        
        Args:
            num_frames: Total number of available frames
            num_samples: Number of frames to sample (default: self.num_history)
            
        Returns:
            List of frame indices to keep (sorted, oldest to newest)
        """
        if num_samples is None:
            num_samples = self.num_history
        
        if num_samples >= num_frames:
            return list(range(num_frames))
        
        if num_samples <= 0:
            return []
        
        if num_samples == 1:
            # Only keep the most recent frame
            return [num_frames - 1]
        
        indices = []
        for i in range(num_samples):
            # Normalized position in sample space [0, 1]
            t_sample = i / (num_samples - 1)
            
            # Power transformation
            # log_base=1.0: t_frame = t_sample (linear/uniform)
            # log_base>1.0: more concentration at recent end
            t_frame = 1.0 - math.pow(1.0 - t_sample, self.log_base)
            
            # Map to frame index
            frame_idx = int(round(t_frame * (num_frames - 1)))
            frame_idx = max(0, min(num_frames - 1, frame_idx))
            
            if frame_idx not in indices:
                indices.append(frame_idx)
        
        # Sort indices (oldest to newest)
        indices.sort()
        
        # Ensure we have exactly num_samples frames
        # If duplicates were removed, add missing frames
        while len(indices) < num_samples:
            if len(indices) == 1:
                if indices[0] < num_frames - 1:
                    indices.append(num_frames - 1)
                else:
                    indices.insert(0, 0)
            else:
                # Find largest gap, preferring gaps closer to the end
                max_gap = 0
                max_gap_idx = len(indices) - 1
                for j in range(len(indices) - 1, 0, -1):
                    gap = indices[j] - indices[j-1]
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_idx = j
                
                # Check gap to the end
                gap_to_end = (num_frames - 1) - indices[-1]
                if gap_to_end > max_gap:
                    indices.append(num_frames - 1)
                else:
                    new_idx = (indices[max_gap_idx] + indices[max_gap_idx - 1]) // 2
                    if new_idx not in indices:
                        indices.insert(max_gap_idx, new_idx)
                    else:
                        for k in range(num_frames):
                            if k not in indices:
                                indices.append(k)
                                break
            
            indices.sort()
        
        return indices[:num_samples]
    
    def get_output_token_count(
        self,
        num_frames: int,
        frame_infos: List[Tuple[int, int, int]],
    ) -> int:
        """Sum of compressed tokens for each frame."""
        total = 0
        for t, h, w in frame_infos:
            tokens = self._compressor.get_compressed_token_count(t, h, w, self.stride)
            total += max(1, tokens)
        return total
    
    def process(
        self,
        frame_embeds_list: List[torch.Tensor],
        frame_grid_thws: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compress each frame independently and concatenate."""
        if not frame_embeds_list:
            return torch.empty(0, 0)
        
        compressed_list = []
        for embeds, grid_thw in zip(frame_embeds_list, frame_grid_thws):
            compressed, _ = self._compressor.compress(embeds, grid_thw, self.stride)
            compressed_list.append(compressed)
        
        if not compressed_list:
            hidden_size = frame_embeds_list[0].shape[-1]
            return torch.empty(0, hidden_size)
        
        return torch.cat(compressed_list, dim=0)
