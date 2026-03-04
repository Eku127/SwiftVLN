# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Embedding Enhancement Pipeline.

A composable pipeline that chains multiple embedding enhancements.
Uses nn.ModuleDict so all parameters are automatically:
- Visible in model.parameters() (trained by optimizer)
- Serialized in model.state_dict() (saved/restored with checkpoints)
"""

import torch
import torch.nn as nn
from typing import List

from .base import BaseEmbeddingEnhancement


class EmbeddingEnhancementPipeline(nn.Module):
    """
    Composable pipeline for embedding enhancements.
    
    Chains multiple BaseEmbeddingEnhancement modules. Each module is applied
    sequentially to the input embeddings.
    
    Usage:
        pipeline = EmbeddingEnhancementPipeline()
        pipeline.add('pixel', PixelFeatureAugment(embed_dim=1536))
        pipeline.add('pose', PoseEmbedding(embed_dim=1536))
        
        # Apply all enhancements
        enhanced = pipeline(embed, H, W, pose=frame_pose)
    
    Benefits:
    - nn.ModuleDict auto-manages parameters, state_dict, device/dtype
    - No manual checkpoint restore logic needed
    - Adding new enhancements = one line
    """
    
    def __init__(self):
        super().__init__()
        self.enhancements = nn.ModuleDict()
    
    def add(self, name: str, module: BaseEmbeddingEnhancement):
        """
        Add an enhancement module to the pipeline.
        
        Args:
            name: Unique name for this enhancement (e.g., 'pixel', 'pose')
            module: Enhancement module instance
        """
        self.enhancements[name] = module
    
    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """
        Apply all enhancements sequentially.
        
        Args:
            embed: [N, D] ViT features
            H: Feature grid height (after ViT merge)
            W: Feature grid width (after ViT merge)
            **kwargs: Extra info passed to each enhancement (e.g., pose)
            
        Returns:
            Enhanced [N, D] tensor
        """
        for name, module in self.enhancements.items():
            embed = module(embed, H, W, **kwargs)
        return embed
    
    @property
    def is_empty(self) -> bool:
        """Check if pipeline has no enhancements."""
        return len(self.enhancements) == 0
    
    @property
    def enhancement_names(self) -> List[str]:
        """Get list of enhancement names."""
        return list(self.enhancements.keys())
    
    def __repr__(self):
        if self.is_empty:
            return "EmbeddingEnhancementPipeline(empty)"
        names = ', '.join(f'{k}: {v.name}' for k, v in self.enhancements.items())
        return f"EmbeddingEnhancementPipeline({names})"
