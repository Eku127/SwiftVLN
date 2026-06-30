# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Per-Frame Compression Processor

Compresses each frame independently, then concatenates results.
Supports both average pooling and Grid-based ToMe.

Sampling Strategy:
- Supports random sampling without replacement for train/eval alignment
- Uses power transformation for flexible frame sampling
- log_base=1.0: Uniform sampling (default, linear distribution)
- log_base>1.0: Logarithmic sampling (more recent frames preserved)

Formula: t_frame = 1 - (1 - t_sample)^log_base
- When log_base=1.0: t_frame = t_sample (perfectly linear/uniform)
- When log_base>1.0: More samples concentrated at recent end
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Literal

from .base import HistoryProcessor


def sample_per_frame_history_indices(
    num_frames: int,
    num_samples: int,
    log_base: float = 1.0,
    use_random: bool = False,
) -> List[int]:
    """
    Sample sorted indices from the history prefix [0, num_frames).

    This helper intentionally mirrors the SwiftVLN train/eval per-frame
    sampling logic so both sides stay behaviorally aligned.
    """
    num_frames = int(num_frames)
    num_samples = int(num_samples)

    if num_frames <= 0 or num_samples <= 0:
        return []

    num_samples = min(num_samples, num_frames)
    if num_samples == num_frames:
        return list(range(num_frames))

    if use_random:
        sampled = np.random.choice(num_frames, size=num_samples, replace=False)
        sampled.sort()
        return sampled.astype(int).tolist()

    indices: List[int] = []
    for i in range(num_samples):
        if num_samples == 1:
            t_sample = 1.0
        else:
            t_sample = i / (num_samples - 1)

        t_frame = 1.0 - math.pow(1.0 - t_sample, log_base)
        frame_idx = int(round(t_frame * (num_frames - 1)))
        frame_idx = max(0, min(num_frames - 1, frame_idx))

        if frame_idx not in indices:
            indices.append(frame_idx)

    indices.sort()

    # Match the current train/eval fallback: if rounding produced duplicate
    # indices, backfill the earliest unused frames in ascending order.
    while len(indices) < num_samples:
        for k in range(num_frames):
            if k not in indices:
                indices.append(k)
                indices.sort()
                break
        else:
            break

    return indices[:num_samples]


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
        from .compressor import HistoryTokenCompressor
        
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
