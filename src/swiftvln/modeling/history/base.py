# Copyright (c) Alibaba, Inc. and its affiliates.
"""
History Processor Base Classes

This module defines the abstract interface for all history processors.
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class HistoryProcessorConfig:
    """
    Unified configuration for all history processor types.
    
    Each processor implementation uses only its relevant fields.
    """
    # Processor selection
    processor_type: str = 'per_frame'
    
    # Per-frame options (also affects sampling distribution)
    compress_stride: int = 2
    compress_method: str = 'pooling'  # 'pooling' or 'tome'
    num_history: int = 8              # Number of frames to sample
    log_base: float = 1.0             # Sampling distribution (1.0=uniform, >1.0=logarithmic)
    grid_size: int = 2
    
    # GTC options
    output_tokens: int = 512
    temperature: float = 0.1
    num_iterations: int = 1
    init_method: str = 'uniform'  # 'uniform' or 'pooled'


class HistoryProcessor(ABC):
    """
    Abstract base class for history information processors.
    
    All history processors must implement two core methods:
    1. get_output_token_count() - Predict output token count
    2. process() - Process embeddings
    
    This unified interface allows seamless switching between
    different processing strategies (per-frame, GTC, etc.)
    """
    
    @abstractmethod
    def get_output_token_count(
        self,
        num_frames: int,
        frame_infos: List[Tuple[int, int, int]],
    ) -> int:
        """
        Predict the number of output tokens after processing.
        
        Called during encoding phase to allocate placeholder tokens.
        
        Args:
            num_frames: Number of history frames
            frame_infos: List of (t, h, w) for each frame (after ViT merge)
            
        Returns:
            int: Total number of output tokens
        """
        pass
    
    @abstractmethod
    def process(
        self,
        frame_embeds_list: List[torch.Tensor],
        frame_grid_thws: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process history frame embeddings.
        
        Called during post-encoding phase to actually process embeddings.
        
        Args:
            frame_embeds_list: List of [num_tokens, hidden_size] tensors
            frame_grid_thws: List of [3] tensors with (t, h, w)
            
        Returns:
            Tensor: Processed embeddings [output_tokens, hidden_size]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return processor name for logging."""
        pass
