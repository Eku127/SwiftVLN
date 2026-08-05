# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Segment GTC (Segment-wise Global Token Clustering) Processor

Divides history into K segments (default 8), applies GTC within each segment,
then concatenates to preserve coarse temporal order.

Key benefits:
- Preserves coarse-grained temporal order (which segment is early/late)
- Removes redundancy within each temporal segment
- Fixed output token count regardless of input
"""

import torch
from typing import List, Tuple

from .base import HistoryProcessor
from .gtc import soft_kmeans_step


class SegmentGTC(HistoryProcessor):
    """
    Segment-wise Global Token Clustering processor.
    
    Algorithm:
    1. Divide N history frames into K segments (default 8, uniform)
    2. For each segment: Apply GTC to cluster frames into output_tokens/K tokens
    3. Concatenate segments in chronological order (early → late)
    
    This preserves coarse temporal structure while removing intra-segment redundancy.
    
    Example (8 segments, 512 output tokens, 32 input frames):
        Segment 0: frames [0-3]   → GTC → 64 tokens
        Segment 1: frames [4-7]   → GTC → 64 tokens
        ...
        Segment 7: frames [28-31] → GTC → 64 tokens
        Output: [Seg0, Seg1, ..., Seg7] = 512 tokens (chronological order)
    """
    
    # Fixed settings
    NUM_SEGMENTS = 8  # Fixed 8 segments
    
    def __init__(
        self,
        output_tokens: int = 512,
        temperature: float = 0.1,
        num_iterations: int = 1,
    ):
        """
        Initialize Segment GTC processor.
        
        Args:
            output_tokens: Total output tokens (will be divided among segments)
            temperature: Softmax temperature for GTC soft assignment
            num_iterations: Number of soft k-means iterations per segment
        """
        self.output_tokens = output_tokens
        self.temperature = temperature
        self.num_iterations = num_iterations
        
        # Tokens per segment (may have remainder)
        self.tokens_per_segment = output_tokens // self.NUM_SEGMENTS
        self.remainder_tokens = output_tokens % self.NUM_SEGMENTS
    
    @property
    def name(self) -> str:
        return f"SegmentGTC(K={self.output_tokens}, S={self.NUM_SEGMENTS}, τ={self.temperature})"
    
    def get_output_token_count(
        self,
        num_frames: int,
        frame_infos: List[Tuple[int, int, int]],
    ) -> int:
        """Segment GTC outputs fixed tokens, capped by total input."""
        total_input = sum(t * h * w for t, h, w in frame_infos)
        return min(self.output_tokens, total_input)
    
    def process(
        self,
        frame_embeds_list: List[torch.Tensor],
        frame_grid_thws: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process history frames using segment-wise GTC.
        
        Args:
            frame_embeds_list: List of frame embeddings [num_tokens, hidden_size]
            frame_grid_thws: List of grid_thw tensors (not used, for interface compatibility)
            
        Returns:
            Clustered embeddings [output_tokens, hidden_size]
        """
        if not frame_embeds_list:
            return torch.empty(0, 0)
        
        num_frames = len(frame_embeds_list)
        
        # If very few frames, just concatenate and return (or apply global GTC)
        if num_frames <= self.NUM_SEGMENTS:
            X = torch.cat(frame_embeds_list, dim=0)
            M = X.shape[0]
            K = min(self.output_tokens, M)
            if K >= M:
                return X
            # Apply global GTC for small inputs
            return self._gtc_cluster(X, K)
        
        # Split frames into segments
        segments = self._split_into_segments(frame_embeds_list)
        
        # Process each segment with GTC
        segment_outputs = []
        for seg_idx, segment_frames in enumerate(segments):
            if not segment_frames:
                continue
            
            # Concatenate frames in this segment
            segment_embeds = torch.cat(segment_frames, dim=0)
            M = segment_embeds.shape[0]
            
            # Determine tokens for this segment
            # Distribute remainder to later segments (closer to current)
            tokens_for_seg = self.tokens_per_segment
            if seg_idx >= (self.NUM_SEGMENTS - self.remainder_tokens):
                tokens_for_seg += 1
            
            K = min(tokens_for_seg, M)
            
            if K >= M:
                # No compression needed for this segment
                segment_outputs.append(segment_embeds)
            else:
                # Apply GTC within segment
                clustered = self._gtc_cluster(segment_embeds, K)
                segment_outputs.append(clustered)
        
        if not segment_outputs:
            return torch.empty(0, 0)
        
        # Concatenate in chronological order (early → late)
        return torch.cat(segment_outputs, dim=0)
    
    def _split_into_segments(
        self,
        frame_embeds_list: List[torch.Tensor],
    ) -> List[List[torch.Tensor]]:
        """
        Split frames into K uniform segments.
        
        Args:
            frame_embeds_list: List of frame embeddings
            
        Returns:
            List of K segments, each containing a list of frame embeddings
        """
        num_frames = len(frame_embeds_list)
        frames_per_segment = num_frames // self.NUM_SEGMENTS
        remainder = num_frames % self.NUM_SEGMENTS
        
        segments = []
        start_idx = 0
        
        for seg_idx in range(self.NUM_SEGMENTS):
            # Distribute remainder frames to earlier segments
            seg_size = frames_per_segment
            if seg_idx < remainder:
                seg_size += 1
            
            end_idx = start_idx + seg_size
            segment_frames = frame_embeds_list[start_idx:end_idx]
            segments.append(segment_frames)
            start_idx = end_idx
        
        return segments
    
    def _gtc_cluster(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """
        Apply GTC clustering to reduce M tokens to K tokens.
        
        Args:
            X: Input tokens [M, d]
            K: Target number of output tokens
            
        Returns:
            Clustered tokens [K, d]
        """
        M, d = X.shape
        
        if K >= M:
            return X
        
        # Initialize centroids (uniform sampling)
        indices = torch.linspace(0, M - 1, K, dtype=torch.long, device=X.device)
        C = X[indices].clone()
        
        # Soft K-Means iterations
        for _ in range(self.num_iterations):
            C = soft_kmeans_step(X, C, self.temperature)
        
        return C
