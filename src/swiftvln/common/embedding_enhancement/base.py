# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base class for Embedding Enhancement modules.

All embedding enhancements (pose, UAV adapter, temporal, etc.) should inherit
from BaseEmbeddingEnhancement and implement the forward() method.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEmbeddingEnhancement(nn.Module, ABC):
    """
    Abstract base class for all embedding enhancement modules.
    
    Each enhancement takes ViT features and augments them with additional
    information (spatial coordinates, pose, temporal position, etc.).
    
    The interface is designed so all enhancements can be composed in a
    pipeline via EmbeddingEnhancementPipeline.
    
    Args (forward):
        embed: Tensor of shape [N, D] where N = t * H * W
        H: Height of the feature grid (after ViT merge)
        W: Width of the feature grid (after ViT merge)
        **kwargs: Enhancement-specific extra info (e.g., pose, timestamp)
    
    Returns:
        Enhanced tensor of shape [N, D]
    """
    
    @abstractmethod
    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """Apply enhancement to ViT features."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return enhancement name for logging."""
        pass
