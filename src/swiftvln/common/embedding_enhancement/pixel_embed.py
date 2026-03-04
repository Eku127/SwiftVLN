# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Pixel-level Coordinate Embedding Enhancement (MLP_xy).

This module adds learnable pixel-level coordinate embeddings to ViT features
using Fourier encoding + MLP. The enhancement is applied after the visual
encoder produces features but before any history compression.

Key Features:
- Fourier encoding of normalized pixel coordinates (x, y) in [-1, 1]
- MLP: Linear -> GELU -> Linear with zero-initialization on the last layer
- Zero-init ensures E_xy = 0 at initialization, preserving baseline behavior
- Coordinate tensors are cached per (H, W) to avoid repeated computation

Formula:
    V_aug = V + beta * E_xy
    where E_xy = MLP(FourierEncode(x, y))
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from .base import BaseEmbeddingEnhancement


class PixelFeatureAugment(BaseEmbeddingEnhancement):
    """
    Pixel-level coordinate embedding enhancement module.
    
    Takes ViT features and adds learnable position embeddings based on
    Fourier-encoded pixel coordinates.
    
    Args:
        embed_dim: Dimension of the input ViT embeddings (e.g., 1536 for Qwen2.5-VL-3B)
        num_bands: Number of Fourier frequency bands (default: 6)
        hidden_dim: Hidden dimension of the MLP (default: 256)
        beta: Scaling factor for the embedding (default: 1.0)
    
    Input:
        embed: Tensor of shape [N, D] where N = H * W, D = embed_dim
        H: Height of the feature grid
        W: Width of the feature grid
    
    Output:
        Tensor of shape [N, D] with enhanced features
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_bands: int = 6,
        hidden_dim: int = 256,
        beta: float = 1.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.beta = beta
        
        # Fourier encoding dimension: 2 (x, y) + 4 * num_bands (sin/cos for x and y)
        # For each coordinate: 1 + 2*num_bands (original + sin/cos pairs)
        # Total: 2 * (1 + 2*num_bands) = 2 + 4*num_bands
        fourier_dim = 2 + 4 * num_bands
        
        # MLP: Linear -> GELU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Zero-initialize the last layer to ensure E_xy = 0 at initialization
        # This preserves baseline behavior when first enabled
        self._zero_init_last_layer()
        
        # Cache for coordinate tensors: (H, W) -> [H*W, fourier_dim]
        self._coord_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Precompute frequency bands: [1, 2, 4, ..., 2^(L-1)] * pi
        # Using log-spaced frequencies for better multi-scale encoding
        freq_bands = 2.0 ** torch.arange(num_bands).float() * math.pi
        self.register_buffer('freq_bands', freq_bands)
        
        print(f"[PixelFeatureAugment] Initialized with:")
        print(f"  embed_dim={embed_dim}, num_bands={num_bands}")
        print(f"  hidden_dim={hidden_dim}, beta={beta}")
        print(f"  fourier_dim={fourier_dim}")
    
    def _zero_init_last_layer(self):
        """Zero-initialize the last linear layer's weight and bias."""
        # The last layer is mlp[2] (index 2 in Sequential)
        last_layer = self.mlp[2]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
    
    def _get_fourier_coords(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get Fourier-encoded coordinate tensor for grid of size (H, W).
        
        Uses caching to avoid recomputation for common grid sizes.
        
        Args:
            H: Grid height
            W: Grid width
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tensor of shape [H*W, fourier_dim] with Fourier-encoded coordinates
        """
        cache_key = (H, W)
        
        # Check cache (with device/dtype handling)
        if cache_key in self._coord_cache:
            cached = self._coord_cache[cache_key]
            if cached.device == device and cached.dtype == dtype:
                return cached
        
        # Generate normalized coordinates in [-1, 1]
        # y: varies along H dimension (rows)
        # x: varies along W dimension (columns)
        y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        
        # Create meshgrid: [H, W] each
        # Note: indexing='ij' means y varies along dim 0, x along dim 1
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten to [H*W]
        xx = xx.reshape(-1)  # [H*W]
        yy = yy.reshape(-1)  # [H*W]
        
        # Fourier encoding
        # freq_bands: [num_bands]
        # For each coordinate, compute: [coord, sin(w0*coord), cos(w0*coord), ..., sin(wL*coord), cos(wL*coord)]
        
        # Scale coordinates by frequency bands: [H*W, num_bands]
        freq_bands = self.freq_bands.to(device=device, dtype=dtype)
        xx_scaled = xx.unsqueeze(-1) * freq_bands  # [H*W, num_bands]
        yy_scaled = yy.unsqueeze(-1) * freq_bands  # [H*W, num_bands]
        
        # Compute sin and cos
        sin_xx = torch.sin(xx_scaled)  # [H*W, num_bands]
        cos_xx = torch.cos(xx_scaled)  # [H*W, num_bands]
        sin_yy = torch.sin(yy_scaled)  # [H*W, num_bands]
        cos_yy = torch.cos(yy_scaled)  # [H*W, num_bands]
        
        # Concatenate: [x, y, sin_x..., cos_x..., sin_y..., cos_y...]
        # Shape: [H*W, 2 + 4*num_bands]
        fourier_coords = torch.cat([
            xx.unsqueeze(-1),  # [H*W, 1]
            yy.unsqueeze(-1),  # [H*W, 1]
            sin_xx,            # [H*W, num_bands]
            cos_xx,            # [H*W, num_bands]
            sin_yy,            # [H*W, num_bands]
            cos_yy,            # [H*W, num_bands]
        ], dim=-1)
        
        # Cache the result
        self._coord_cache[cache_key] = fourier_coords
        
        return fourier_coords
    
    @property
    def name(self) -> str:
        return f"PixelFeatureAugment(bands={self.num_bands}, hidden={self.hidden_dim}, beta={self.beta})"
    
    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """
        Apply pixel coordinate embedding enhancement.
        
        Args:
            embed: Input embeddings of shape [N, D] where N = t * H * W
            H: Height of the feature grid (after ViT merge)
            W: Width of the feature grid (after ViT merge)
            **kwargs: Ignored (for pipeline compatibility)
            
        Returns:
            Enhanced embeddings of shape [N, D]
        """
        N, D = embed.shape
        
        # Handle temporal dimension if present
        # For Qwen2.5-VL, N = t * H * W where t is typically 1 for images
        expected_spatial = H * W
        
        if N == expected_spatial:
            # Single frame: direct processing
            t = 1
        elif N % expected_spatial == 0:
            # Multiple frames: process each frame identically
            t = N // expected_spatial
        else:
            # Unexpected size - just return unchanged (should not happen)
            import warnings
            warnings.warn(
                f"[PixelFeatureAugment] Unexpected embed size: N={N}, H={H}, W={W}, "
                f"expected N = t * {H*W}. Returning unchanged."
            )
            return embed
        
        # Get Fourier-encoded coordinates
        fourier_coords = self._get_fourier_coords(H, W, embed.device, embed.dtype)
        
        # Compute position embeddings via MLP: [H*W, D]
        pos_embed = self.mlp(fourier_coords)
        
        # Repeat for temporal dimension if needed: [t * H * W, D]
        if t > 1:
            pos_embed = pos_embed.repeat(t, 1)
        
        # Add scaled position embedding
        enhanced = embed + self.beta * pos_embed
        
        return enhanced
    
    def clear_cache(self):
        """Clear the coordinate cache to free memory."""
        self._coord_cache.clear()
