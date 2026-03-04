# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Embedding Enhancement Module for VLN.

This module provides pluggable embedding enhancement layers that are applied
AFTER the visual encoder (model.visual) produces ViT features, but BEFORE
any history compression or token processing.

Architecture:
    EmbeddingEnhancementPipeline (container, nn.ModuleDict)
        ├── pixel: PixelFeatureAugment (Fourier + MLP position encoding)
        ├── pose: PoseEmbedding (future)
        └── temporal: TemporalEmbedding (future)

Usage:
    from swiftvln.common.embedding_enhancement import (
        EmbeddingEnhancementPipeline,
        PixelFeatureAugment,
        create_embedding_pipeline,
    )
    
    # Via factory (recommended):
    pipeline = create_embedding_pipeline(
        embed_dim=1536,
        use_pixel_embed=True,
        use_pose_embed=False,
    )
    
    # Manual:
    pipeline = EmbeddingEnhancementPipeline()
    pipeline.add('pixel', PixelFeatureAugment(embed_dim=1536))
"""

from .base import BaseEmbeddingEnhancement
from .pipeline import EmbeddingEnhancementPipeline
from .pixel_embed import PixelFeatureAugment
from .pose_embed import PoseEmbedding
from .pose_utils import reconstruct_pose_from_actions


def create_embedding_pipeline(
    embed_dim: int,
    use_pixel_embed: bool = False,
    use_pose_embed: bool = False,
    # Pixel embed hyperparameters
    pixel_num_bands: int = 6,
    pixel_hidden_dim: int = 256,
    pixel_beta: float = 1.0,
    # Pose embed hyperparameters
    pose_dim: int = 4,
    pose_hidden_dim: int = 256,
    pose_beta: float = 1.0,
    pose_fusion: str = 'additive',
    pose_norm_scale: float = 100.0,
) -> EmbeddingEnhancementPipeline:
    """
    Factory function to create an EmbeddingEnhancementPipeline.
    
    Creates the pipeline and adds requested enhancement modules.
    
    Args:
        embed_dim: ViT output embedding dimension (e.g., 1536 for Qwen2.5-VL-3B)
        use_pixel_embed: Enable pixel coordinate embedding (Fourier + MLP)
        use_pose_embed: Enable pose embedding (MLP, additive/FiLM)
        pixel_num_bands: Number of Fourier frequency bands (default: 6)
        pixel_hidden_dim: MLP hidden dimension (default: 256)
        pixel_beta: Scaling factor for pixel embedding (default: 1.0)
        pose_dim: Pose vector dimension (default: 4)
        pose_hidden_dim: Pose MLP hidden dimension (default: 256)
        pose_beta: Scaling factor for pose embedding (default: 1.0)
        pose_fusion: Pose fusion method: 'additive' or 'film'
        pose_norm_scale: tanh normalization scale for positional components
        
    Returns:
        Configured EmbeddingEnhancementPipeline instance
    """
    pipeline = EmbeddingEnhancementPipeline()
    
    if use_pixel_embed:
        pipeline.add('pixel', PixelFeatureAugment(
            embed_dim=embed_dim,
            num_bands=pixel_num_bands,
            hidden_dim=pixel_hidden_dim,
            beta=pixel_beta,
        ))

    if use_pose_embed:
        pipeline.add('pose', PoseEmbedding(
            embed_dim=embed_dim,
            pose_dim=pose_dim,
            hidden_dim=pose_hidden_dim,
            beta=pose_beta,
            fusion=pose_fusion,
            norm_scale=pose_norm_scale,
        ))
    
    return pipeline


__all__ = [
    'BaseEmbeddingEnhancement',
    'EmbeddingEnhancementPipeline',
    'PixelFeatureAugment',
    'PoseEmbedding',
    'reconstruct_pose_from_actions',
    'create_embedding_pipeline',
]
