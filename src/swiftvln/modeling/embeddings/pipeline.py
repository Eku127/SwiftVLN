# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Embedding Enhancement Pipeline.

The historical class name and ModuleDict layout are retained for checkpoint
compatibility, but SwiftVLN now permits at most one enhancement mode.
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
    Mutually-exclusive container for one embedding enhancement.
    
    Contains either no module, one pose module, or one UAV module.
    
    Usage:
        pipeline = EmbeddingEnhancementPipeline()
        pipeline.add('pose', PoseEmbedding(embed_dim=1536))
        
        # Apply the selected enhancement
        enhanced = pipeline(embed, H, W, pose=frame_pose)
    
    Benefits:
    - nn.ModuleDict auto-manages parameters, state_dict, device/dtype
    - No manual checkpoint restore logic needed
    - Keeps historical checkpoint keys under ``enhancements.<name>``
    """
    
    def __init__(self):
        super().__init__()
        self.enhancements = nn.ModuleDict()
    
    def add(self, name: str, module: BaseEmbeddingEnhancement):
        """
        Add an enhancement module to the pipeline.
        
        Args:
            name: Unique name for this enhancement (e.g., 'pose', 'uav')
            module: Enhancement module instance
        """
        if self.enhancements:
            active = next(iter(self.enhancements))
            raise ValueError(
                "embedding modes are mutually exclusive; "
                f"cannot add {name!r} while {active!r} is active"
            )
        self.enhancements[name] = module
    
    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """
        Apply the selected enhancement, if any.
        
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
