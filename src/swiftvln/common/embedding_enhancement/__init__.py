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
from .uav_adapter import UAVAdapterEnhancement


def create_embedding_pipeline(
    embed_dim: int,
    use_pixel_embed: bool = False,
    use_pose_embed: bool = False,
    use_uav_adapter: bool = False,
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
    # UAV adapter hyperparameters
    uav_adapter_path: str = '',
    uav_adapter_type: str = 'transformer_v1',
    uav_adapter_apply_scope: str = 'all_images',
    uav_adapter_num_layers: int = 2,
    uav_adapter_num_heads: int = 8,
    uav_adapter_mlp_ratio: float = 4.0,
    uav_adapter_dropout: float = 0.0,
) -> EmbeddingEnhancementPipeline:
    """
    Factory function to create an EmbeddingEnhancementPipeline.
    
    Creates the pipeline and adds requested enhancement modules.
    
    Args:
        embed_dim: ViT output embedding dimension (e.g., 1536 for Qwen2.5-VL-3B)
        use_pixel_embed: Enable pixel coordinate embedding (Fourier + MLP)
        use_pose_embed: Enable pose embedding (MLP, additive/FiLM)
        use_uav_adapter: Enable Stage-A UAV adapter enhancement
        pixel_num_bands: Number of Fourier frequency bands (default: 6)
        pixel_hidden_dim: MLP hidden dimension (default: 256)
        pixel_beta: Scaling factor for pixel embedding (default: 1.0)
        pose_dim: Pose vector dimension (default: 4)
        pose_hidden_dim: Pose MLP hidden dimension (default: 256)
        pose_beta: Scaling factor for pose embedding (default: 1.0)
        pose_fusion: Pose fusion method: 'additive' or 'film'
        pose_norm_scale: tanh normalization scale for positional components
        uav_adapter_path: Optional external Stage-A checkpoint (.pt or output dir)
        uav_adapter_type: UAV adapter implementation type
        uav_adapter_apply_scope: Currently only 'all_images' is supported
        uav_adapter_num_layers: Random-init UAV adapter depth when no checkpoint is provided
        uav_adapter_num_heads: Random-init UAV adapter attention heads
        uav_adapter_mlp_ratio: Random-init UAV adapter MLP ratio
        uav_adapter_dropout: Random-init UAV adapter dropout
        
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

    if use_uav_adapter:
        pipeline.add('uav', UAVAdapterEnhancement(
            embed_dim=embed_dim,
            adapter_type=uav_adapter_type,
            checkpoint_path=uav_adapter_path,
            apply_scope=uav_adapter_apply_scope,
            num_layers=uav_adapter_num_layers,
            num_heads=uav_adapter_num_heads,
            mlp_ratio=uav_adapter_mlp_ratio,
            dropout=uav_adapter_dropout,
        ))
    
    return pipeline


__all__ = [
    'BaseEmbeddingEnhancement',
    'EmbeddingEnhancementPipeline',
    'PixelFeatureAugment',
    'PoseEmbedding',
    'UAVAdapterEnhancement',
    'reconstruct_pose_from_actions',
    'create_embedding_pipeline',
]
