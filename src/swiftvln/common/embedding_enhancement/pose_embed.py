# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Pose embedding enhancement for VLN.

This module injects a per-image pose vector into all visual tokens of that
image. The pose vector is expected to be:
    [delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]

Fusion options:
- additive: embed = embed + beta * E_pose
- film:     embed = embed * (1 + beta * gamma) + beta * bias
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .base import BaseEmbeddingEnhancement


class PoseEmbedding(BaseEmbeddingEnhancement):
    """Per-image pose embedding with Additive or FiLM fusion."""

    def __init__(
        self,
        embed_dim: int,
        pose_dim: int = 4,
        hidden_dim: int = 256,
        beta: float = 1.0,
        fusion: str = 'additive',
        norm_scale: float = 100.0,
    ):
        super().__init__()
        fusion = fusion.lower()
        if fusion not in ('additive', 'film'):
            raise ValueError(f"Unsupported pose fusion method: {fusion}. Use 'additive' or 'film'.")

        self.embed_dim = embed_dim
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.fusion = fusion
        self.norm_scale = norm_scale

        out_dim = embed_dim if fusion == 'additive' else 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._zero_init_last_layer()

        print("[PoseEmbedding] Initialized with:")
        print(f"  embed_dim={embed_dim}, pose_dim={pose_dim}")
        print(f"  hidden_dim={hidden_dim}, beta={beta}")
        print(f"  fusion={fusion}, norm_scale={norm_scale}")

    def _zero_init_last_layer(self):
        """Zero-init last linear layer for stable baseline-equivalent start."""
        last_layer = self.mlp[2]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _to_pose_tensor(self, pose: object, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Convert pose input to a 1D tensor of shape [pose_dim]."""
        if pose is None:
            return None

        if isinstance(pose, torch.Tensor):
            p = pose.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(pose, np.ndarray):
            p = torch.from_numpy(pose).to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(pose, Sequence):
            p = torch.tensor(pose, device=device, dtype=dtype).reshape(-1)
        else:
            return None

        if p.numel() < self.pose_dim:
            pad = torch.zeros(self.pose_dim - p.numel(), device=device, dtype=dtype)
            p = torch.cat([p, pad], dim=0)
        elif p.numel() > self.pose_dim:
            p = p[:self.pose_dim]

        return p

    def _normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """tanh-normalize positional components and keep heading sin/cos unchanged."""
        normed = pose.clone()
        if self.pose_dim >= 2 and self.norm_scale and self.norm_scale > 0:
            normed[:2] = torch.tanh(normed[:2] / self.norm_scale)
        return normed

    @property
    def name(self) -> str:
        return (f"PoseEmbedding(fusion={self.fusion}, hidden={self.hidden_dim}, "
                f"beta={self.beta}, scale={self.norm_scale})")

    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """
        Apply pose embedding to one image's tokens.

        Args:
            embed: [N, D] image tokens
            H: feature map height (unused)
            W: feature map width (unused)
            pose: optional pose vector in kwargs
        """
        pose = kwargs.get('pose')
        pose_t = self._to_pose_tensor(pose, device=embed.device, dtype=embed.dtype)
        if pose_t is None:
            return embed

        pose_t = self._normalize_pose(pose_t).unsqueeze(0)  # [1, pose_dim]
        proj = self.mlp(pose_t).squeeze(0)  # [D] or [2D]

        if self.fusion == 'additive':
            return embed + self.beta * proj.unsqueeze(0)

        gamma, bias = torch.split(proj, self.embed_dim, dim=0)
        return embed * (1.0 + self.beta * gamma.unsqueeze(0)) + self.beta * bias.unsqueeze(0)
