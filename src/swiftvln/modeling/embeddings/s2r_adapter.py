"""Runtime-only sim-to-real adapter used by SwiftVLN embeddings.

Stage-A training and data-production code lives under ``tools/s2r``.  This
module intentionally contains only the small adapter architecture required to
load a Stage-A checkpoint during SwiftVLN training or evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    """A lightweight token transformer block for sim-to-real alignment."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~attention_mask
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class Sim2RealAdapter(nn.Module):
    """Token-level transformer adapter compatible with Stage-A checkpoints."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.blocks = nn.ModuleList(
            [
                AdapterBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


__all__ = ["Sim2RealAdapter"]
