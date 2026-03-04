# Copyright (c) Alibaba, Inc. and its affiliates.
"""
History Coverage Computation

This module computes how well memory tokens represent the full history.
High coverage means the memory captures information from throughout the trajectory.

Metrics:
1. Max Coverage (original): Cov = (1/T) * sum_u(max_k cos(e_u, m_k))
2. Soft Coverage (LSE): SoftCov = (1/T) * sum_u((1/β) * log(sum_k exp(β * cos)))
3. Coverage@τ: Fraction of frames with max_cos > threshold τ

A higher coverage indicates better representation of the full history.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class CoverageResult:
    """Result of coverage computation."""
    # Primary metrics
    coverage: float          # Max cosine coverage (original)
    coverage_soft: float     # Soft coverage (log-sum-exp)
    coverage_tau70: float    # Coverage@τ=0.7 (fraction above threshold)
    coverage_tau80: float    # Coverage@τ=0.8
    
    # Statistics
    coverage_mean: float     # Mean of all pairwise similarities (not just max)
    coverage_p10: float      # 10th percentile (worst covered frames)
    coverage_p50: float      # Median coverage
    
    # Metadata
    per_frame_coverage: List[float]  # Max coverage for each history frame
    num_history_frames: int
    num_memory_tokens: int
    min_frame_coverage: float
    max_frame_coverage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coverage': self.coverage,
            'coverage_soft': self.coverage_soft,
            'coverage_tau70': self.coverage_tau70,
            'coverage_tau80': self.coverage_tau80,
            'coverage_mean': self.coverage_mean,
            'coverage_p10': self.coverage_p10,
            'coverage_p50': self.coverage_p50,
            'per_frame_coverage': self.per_frame_coverage,
            'num_history_frames': self.num_history_frames,
            'num_memory_tokens': self.num_memory_tokens,
            'min_frame_coverage': self.min_frame_coverage,
            'max_frame_coverage': self.max_frame_coverage,
        }


class CoverageComputer:
    """
    Compute history coverage by memory tokens.
    
    Measures how well the memory tokens represent the full history.
    Multiple metrics are computed:
    - Max coverage: original metric using max similarity
    - Soft coverage: log-sum-exp for smoother gradient
    - Coverage@τ: fraction of frames above threshold
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize coverage computer.
        
        Args:
            device: Device for computation
        """
        self.device = device
    
    @torch.no_grad()
    def compute(
        self,
        history_features: torch.Tensor,
        memory_tokens: torch.Tensor,
        normalize: bool = True,
        beta: float = 10.0,  # Temperature for soft coverage
    ) -> CoverageResult:
        """
        Compute history coverage with multiple metrics.
        
        Args:
            history_features: Features for all history frames
                              [num_frames, tokens_per_frame, hidden_size] or
                              [total_tokens, hidden_size]
            memory_tokens: Memory tokens [num_memory_tokens, hidden_size]
            normalize: Whether to L2-normalize before computing similarity
            beta: Temperature for soft coverage (higher = closer to max)
            
        Returns:
            CoverageResult with computed metrics
        """
        # Handle different input shapes
        if history_features.dim() == 3:
            # [num_frames, tokens_per_frame, hidden] -> aggregate per frame
            num_frames = history_features.shape[0]
            # Use mean pooling to get per-frame representation
            history_repr = history_features.mean(dim=1)  # [num_frames, hidden]
        elif history_features.dim() == 2:
            # Assume each row is a frame representation
            history_repr = history_features
            num_frames = history_features.shape[0]
        else:
            raise ValueError(f"Unexpected history_features shape: {history_features.shape}")
        
        num_memory_tokens = memory_tokens.shape[0]
        
        if num_frames == 0 or num_memory_tokens == 0:
            return CoverageResult(
                coverage=0.0,
                coverage_soft=0.0,
                coverage_tau70=0.0,
                coverage_tau80=0.0,
                coverage_mean=0.0,
                coverage_p10=0.0,
                coverage_p50=0.0,
                per_frame_coverage=[],
                num_history_frames=num_frames,
                num_memory_tokens=num_memory_tokens,
                min_frame_coverage=0.0,
                max_frame_coverage=0.0,
            )
        
        # Move to device and convert to float
        history_repr = history_repr.to(self.device).float()
        memory = memory_tokens.to(self.device).float()
        
        # Normalize if requested
        if normalize:
            history_repr = F.normalize(history_repr, p=2, dim=-1)
            memory = F.normalize(memory, p=2, dim=-1)
        
        # Compute similarity between each history frame and all memory tokens
        # [num_frames, num_memory_tokens]
        similarity = torch.mm(history_repr, memory.t())
        
        # 1. Max coverage (original): max similarity for each frame
        max_similarities, _ = similarity.max(dim=1)  # [num_frames]
        per_frame_coverage = max_similarities.cpu().tolist()
        coverage = max_similarities.mean().item()
        
        # 2. Soft coverage (log-sum-exp): smoother than max
        # SoftCov = (1/β) * log(sum_k exp(β * cos(e_u, m_k)))
        # Using logsumexp for numerical stability
        soft_max = torch.logsumexp(beta * similarity, dim=1) / beta  # [num_frames]
        coverage_soft = soft_max.mean().item()
        
        # 3. Coverage@τ: fraction of frames with max_cos > threshold
        coverage_tau70 = (max_similarities > 0.70).float().mean().item()
        coverage_tau80 = (max_similarities > 0.80).float().mean().item()
        
        # 4. Mean coverage (average over all pairs, not just max)
        coverage_mean = similarity.mean().item()
        
        # 5. Percentile-based coverage (P10, P50)
        sorted_max_sim = max_similarities.sort()[0]
        n = len(sorted_max_sim)
        p10_idx = int(0.10 * n)
        p50_idx = int(0.50 * n)
        coverage_p10 = sorted_max_sim[p10_idx].item() if n > 0 else 0.0
        coverage_p50 = sorted_max_sim[p50_idx].item() if n > 0 else 0.0
        
        return CoverageResult(
            coverage=coverage,
            coverage_soft=coverage_soft,
            coverage_tau70=coverage_tau70,
            coverage_tau80=coverage_tau80,
            coverage_mean=coverage_mean,
            coverage_p10=coverage_p10,
            coverage_p50=coverage_p50,
            per_frame_coverage=per_frame_coverage,
            num_history_frames=num_frames,
            num_memory_tokens=num_memory_tokens,
            min_frame_coverage=min(per_frame_coverage) if per_frame_coverage else 0.0,
            max_frame_coverage=max(per_frame_coverage) if per_frame_coverage else 0.0,
        )
    
    @torch.no_grad()
    def compute_from_raw_features(
        self,
        all_history_features: List[torch.Tensor],
        memory_tokens: torch.Tensor,
        selected_indices: Optional[List[int]] = None,
        normalize: bool = True,
    ) -> CoverageResult:
        """
        Compute coverage using raw VIT features for all history frames.
        
        This method is more accurate as it uses the original features
        rather than aggregated representations.
        
        Args:
            all_history_features: List of VIT features for each history frame
                                  Each tensor is [tokens_per_frame, hidden_size]
            memory_tokens: Memory tokens [num_memory_tokens, hidden_size]
            selected_indices: If provided, only compute coverage for these frames
            normalize: Whether to L2-normalize
            
        Returns:
            CoverageResult with computed metrics
        """
        if not all_history_features:
            return CoverageResult(
                coverage=0.0,
                coverage_soft=0.0,
                coverage_tau70=0.0,
                coverage_tau80=0.0,
                coverage_mean=0.0,
                coverage_p10=0.0,
                coverage_p50=0.0,
                per_frame_coverage=[],
                num_history_frames=0,
                num_memory_tokens=memory_tokens.shape[0] if memory_tokens is not None else 0,
                min_frame_coverage=0.0,
                max_frame_coverage=0.0,
            )
        
        # Aggregate each frame's features to a single vector
        frame_representations = []
        for features in all_history_features:
            if features is not None and features.shape[0] > 0:
                # Mean pool tokens within frame
                frame_repr = features.mean(dim=0)  # [hidden_size]
                frame_representations.append(frame_repr)
        
        if not frame_representations:
            return CoverageResult(
                coverage=0.0,
                coverage_soft=0.0,
                coverage_tau70=0.0,
                coverage_tau80=0.0,
                coverage_mean=0.0,
                coverage_p10=0.0,
                coverage_p50=0.0,
                per_frame_coverage=[],
                num_history_frames=0,
                num_memory_tokens=memory_tokens.shape[0],
                min_frame_coverage=0.0,
                max_frame_coverage=0.0,
            )
        
        # Stack into single tensor
        history_repr = torch.stack(frame_representations)  # [num_frames, hidden_size]
        
        return self.compute(history_repr, memory_tokens, normalize)


def compute_coverage(
    history_features: torch.Tensor,
    memory_tokens: torch.Tensor,
    device: str = 'cuda',
    normalize: bool = True,
) -> CoverageResult:
    """
    Convenience function to compute coverage.
    
    Args:
        history_features: Features for history frames
        memory_tokens: Memory tokens
        device: Device for computation
        normalize: Whether to L2-normalize
        
    Returns:
        CoverageResult with computed metrics
    """
    computer = CoverageComputer(device)
    return computer.compute(history_features, memory_tokens, normalize)
