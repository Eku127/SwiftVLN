# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Memory Redundancy Computation

This module computes the redundancy of memory tokens, measuring how much
information is duplicated/wasted in the memory representation.

Metrics:
1. Frame-level Redundancy (primary):
   - Red_NN: Nearest neighbor max cosine similarity
   - Red@K: Top-K average cosine similarity (more sensitive to clusters)
   - Red_P90/P95: 90th/95th percentile of pairwise similarities
   
2. Token-level Redundancy (secondary):
   - Same metrics but computed on all tokens

A lower redundancy indicates better memory utilization (more diverse frames).
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class RedundancyResult:
    """Result of redundancy computation."""
    # Frame-level metrics (primary)
    redundancy_frame: float      # NN max cosine (original)
    redundancy_frame_k5: float   # Top-5 average cosine
    redundancy_frame_mean: float # Mean pairwise cosine
    redundancy_frame_p90: float  # 90th percentile
    redundancy_frame_p95: float  # 95th percentile
    
    # Token-level metrics (secondary)
    redundancy_token: float      # NN max cosine
    redundancy_token_k5: float   # Top-5 average cosine
    
    # Metadata
    num_frames: int
    num_tokens: int
    frame_sim_min: float
    frame_sim_max: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'redundancy_frame': self.redundancy_frame,
            'redundancy_frame_k5': self.redundancy_frame_k5,
            'redundancy_frame_mean': self.redundancy_frame_mean,
            'redundancy_frame_p90': self.redundancy_frame_p90,
            'redundancy_frame_p95': self.redundancy_frame_p95,
            'redundancy_token': self.redundancy_token,
            'redundancy_token_k5': self.redundancy_token_k5,
            'num_frames': self.num_frames,
            'num_tokens': self.num_tokens,
            'frame_sim_min': self.frame_sim_min,
            'frame_sim_max': self.frame_sim_max,
        }


class RedundancyComputer:
    """
    Compute memory redundancy at frame level.
    
    Measures how similar the selected frames are to each other.
    High redundancy indicates the selected frames are too similar (wasteful).
    Low redundancy indicates diverse frame selection (efficient).
    
    Metrics computed:
    - NN (nearest neighbor): max similarity to other frames
    - @K (top-K average): more sensitive to "similarity clusters"
    - P90/P95 (percentiles): robust to outliers
    """
    
    def __init__(self, device: str = 'cuda', tokens_per_frame: int = 88):
        """
        Initialize redundancy computer.
        
        Args:
            device: Device for computation
            tokens_per_frame: Number of tokens per frame after compression (default: 88 for stride=2)
        """
        self.device = device
        self.tokens_per_frame = tokens_per_frame
    
    @torch.no_grad()
    def compute(
        self,
        memory_tokens: torch.Tensor,
        tokens_per_frame: int = None,
        normalize: bool = True,
        top_k: int = 5,
    ) -> RedundancyResult:
        """
        Compute redundancy metrics at both frame and token level.
        
        Args:
            memory_tokens: Memory features [num_frames * tokens_per_frame, hidden_size]
            tokens_per_frame: Override tokens per frame (if None, uses self.tokens_per_frame)
            normalize: Whether to L2-normalize before computing similarity
            top_k: K value for top-K average redundancy
            
        Returns:
            RedundancyResult with frame-level and token-level metrics
        """
        if tokens_per_frame is None:
            tokens_per_frame = self.tokens_per_frame
        
        num_tokens = memory_tokens.shape[0]
        num_frames = num_tokens // tokens_per_frame if tokens_per_frame > 0 else 0
        
        if num_frames < 2:
            return RedundancyResult(
                redundancy_frame=0.0,
                redundancy_frame_k5=0.0,
                redundancy_frame_mean=0.0,
                redundancy_frame_p90=0.0,
                redundancy_frame_p95=0.0,
                redundancy_token=0.0,
                redundancy_token_k5=0.0,
                num_frames=num_frames,
                num_tokens=num_tokens,
                frame_sim_min=0.0,
                frame_sim_max=0.0,
            )
        
        # Move to device and convert to float
        tokens = memory_tokens.to(self.device).float()
        
        # === Frame-level redundancy (primary metrics) ===
        # Reshape to [num_frames, tokens_per_frame, hidden_size]
        frames = tokens[:num_frames * tokens_per_frame].view(num_frames, tokens_per_frame, -1)
        
        # Mean pool each frame to get frame representation: [num_frames, hidden_size]
        frame_repr = frames.mean(dim=1)
        
        # Normalize frame representations
        if normalize:
            frame_repr = F.normalize(frame_repr, p=2, dim=-1)
        
        # Frame-level similarity matrix [num_frames, num_frames]
        frame_sim = torch.mm(frame_repr, frame_repr.t())
        
        # Mask diagonal for NN/top-K computation
        mask = torch.eye(num_frames, device=self.device, dtype=torch.bool)
        frame_sim_masked = frame_sim.masked_fill(mask, float('-inf'))
        
        # 1. NN redundancy: max similarity for each frame
        frame_nn_sim, _ = frame_sim_masked.max(dim=1)
        redundancy_frame = frame_nn_sim.mean().item()
        
        # 2. Top-K redundancy: average of top-K similarities for each frame
        k = min(top_k, num_frames - 1)  # Can't exceed num_frames - 1
        if k > 0:
            topk_sim, _ = frame_sim_masked.topk(k, dim=1)
            redundancy_frame_k5 = topk_sim.mean().item()
        else:
            redundancy_frame_k5 = redundancy_frame
        
        # 3. Mean pairwise similarity (excluding diagonal)
        upper_tri_mask = torch.triu(torch.ones(num_frames, num_frames, device=self.device, dtype=torch.bool), diagonal=1)
        upper_tri_values = frame_sim[upper_tri_mask]
        
        if len(upper_tri_values) > 0:
            redundancy_frame_mean = upper_tri_values.mean().item()
            frame_sim_max = upper_tri_values.max().item()
            frame_sim_min = upper_tri_values.min().item()
            
            # 4. Percentile-based redundancy (P90, P95)
            sorted_values = upper_tri_values.sort()[0]
            n = len(sorted_values)
            p90_idx = int(0.90 * n)
            p95_idx = int(0.95 * n)
            redundancy_frame_p90 = sorted_values[min(p90_idx, n-1)].item()
            redundancy_frame_p95 = sorted_values[min(p95_idx, n-1)].item()
        else:
            redundancy_frame_mean = 0.0
            frame_sim_max = 0.0
            frame_sim_min = 0.0
            redundancy_frame_p90 = 0.0
            redundancy_frame_p95 = 0.0
        
        # === Token-level redundancy (secondary, for reference) ===
        if normalize:
            tokens_norm = F.normalize(tokens, p=2, dim=-1)
        else:
            tokens_norm = tokens
        
        token_sim = torch.mm(tokens_norm, tokens_norm.t())
        token_mask = torch.eye(num_tokens, device=self.device, dtype=torch.bool)
        token_sim_masked = token_sim.masked_fill(token_mask, float('-inf'))
        
        # Token NN redundancy
        token_nn_sim, _ = token_sim_masked.max(dim=1)
        redundancy_token = token_nn_sim.mean().item()
        
        # Token Top-K redundancy
        token_k = min(top_k, num_tokens - 1)
        if token_k > 0:
            token_topk_sim, _ = token_sim_masked.topk(token_k, dim=1)
            redundancy_token_k5 = token_topk_sim.mean().item()
        else:
            redundancy_token_k5 = redundancy_token
        
        return RedundancyResult(
            redundancy_frame=redundancy_frame,
            redundancy_frame_k5=redundancy_frame_k5,
            redundancy_frame_mean=redundancy_frame_mean,
            redundancy_frame_p90=redundancy_frame_p90,
            redundancy_frame_p95=redundancy_frame_p95,
            redundancy_token=redundancy_token,
            redundancy_token_k5=redundancy_token_k5,
            num_frames=num_frames,
            num_tokens=num_tokens,
            frame_sim_min=frame_sim_min,
            frame_sim_max=frame_sim_max,
        )


def compute_redundancy(
    memory_tokens: torch.Tensor,
    device: str = 'cuda',
    tokens_per_frame: int = 88,
    normalize: bool = True,
) -> RedundancyResult:
    """
    Convenience function to compute redundancy.
    
    Args:
        memory_tokens: Memory features [num_frames * tokens_per_frame, hidden_size]
        device: Device for computation
        tokens_per_frame: Number of tokens per frame (default: 88 for stride=2)
        normalize: Whether to L2-normalize
        
    Returns:
        RedundancyResult with frame-level and token-level metrics
    """
    computer = RedundancyComputer(device, tokens_per_frame)
    return computer.compute(memory_tokens, tokens_per_frame, normalize)
