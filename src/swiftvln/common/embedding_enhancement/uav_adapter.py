# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UAV adapter enhancement for OverlapVLN.

This module wraps the Stage-A sim-to-real adapter so it can be inserted into
the existing embed_enhance pipeline during OverlapVLN inference or finetuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from swiftvln.s2r.model import Sim2RealAdapter

from .base import BaseEmbeddingEnhancement


def resolve_stagea_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve a Stage-A checkpoint file from either a file or output directory."""
    if not checkpoint_path:
        raise ValueError("checkpoint_path must not be empty")

    path = Path(checkpoint_path)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Stage-A checkpoint path not found: {checkpoint_path}")

    candidates = [
        path / "best.pt",
        path / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    checkpoint_dir = path / "checkpoints"
    if checkpoint_dir.is_dir():
        step_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
        if step_checkpoints:
            return str(step_checkpoints[-1])

    raise FileNotFoundError(
        f"Could not resolve Stage-A checkpoint from directory: {checkpoint_path}"
    )


def load_stagea_adapter_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], str]:
    """Load adapter config and weights from a Stage-A checkpoint."""
    resolved_path = resolve_stagea_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    if "adapter_kwargs" not in checkpoint or "adapter_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported Stage-A checkpoint format: {resolved_path}. "
            "Expected adapter_kwargs and adapter_state_dict."
        )
    return checkpoint["adapter_kwargs"], checkpoint["adapter_state_dict"], resolved_path


class UAVAdapterEnhancement(BaseEmbeddingEnhancement):
    """Wrap a Stage-A Sim2RealAdapter as an embed_enhance module."""

    def __init__(
        self,
        embed_dim: int,
        adapter_type: str = "transformer_v1",
        checkpoint_path: str = "",
        apply_scope: str = "all_images",
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        adapter_type = str(adapter_type).strip().lower()
        if adapter_type != "transformer_v1":
            raise ValueError(f"Unsupported uav_adapter_type: {adapter_type}")
        if apply_scope != "all_images":
            raise ValueError(
                f"Unsupported uav_adapter_apply_scope: {apply_scope}. "
                "Currently only 'all_images' is supported."
            )

        self.embed_dim = int(embed_dim)
        self.adapter_type = adapter_type
        self.apply_scope = apply_scope
        self.loaded_checkpoint_path = ""

        self.core = Sim2RealAdapter(
            dim=self.embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        if checkpoint_path:
            self.load_external_checkpoint(checkpoint_path, strict=True)

    def _rebuild_core_from_kwargs(self, adapter_kwargs: Dict[str, object]) -> Sim2RealAdapter:
        kwargs = dict(adapter_kwargs)
        checkpoint_dim = int(kwargs.get("dim", self.embed_dim))
        if checkpoint_dim != self.embed_dim:
            raise ValueError(
                f"Stage-A checkpoint dim mismatch: checkpoint dim={checkpoint_dim}, "
                f"model embed_dim={self.embed_dim}"
            )
        return Sim2RealAdapter(**kwargs)

    def load_external_checkpoint(self, checkpoint_path: str, *, strict: bool = True) -> str:
        """Load weights from an external Stage-A checkpoint."""
        adapter_kwargs, adapter_state_dict, resolved_path = load_stagea_adapter_checkpoint(checkpoint_path)

        target_device, target_dtype = self._infer_device_dtype()
        rebuilt_core = self._rebuild_core_from_kwargs(adapter_kwargs).to(
            device=target_device,
            dtype=target_dtype,
        )
        incompatible = rebuilt_core.load_state_dict(adapter_state_dict, strict=strict)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                f"Unexpected Stage-A checkpoint mismatch for {resolved_path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        self.core = rebuilt_core
        self.loaded_checkpoint_path = resolved_path
        return resolved_path

    def _infer_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        try:
            parameter = next(self.core.parameters())
            return parameter.device, parameter.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32

    @property
    def name(self) -> str:
        source = Path(self.loaded_checkpoint_path).name if self.loaded_checkpoint_path else "random_init"
        return f"UAVAdapter(type={self.adapter_type}, scope={self.apply_scope}, source={source})"

    def forward(self, embed: torch.Tensor, H: int, W: int, **kwargs) -> torch.Tensor:
        """
        Apply the Stage-A adapter to a single image token sequence.

        Args:
            embed: [N, D] tokens for one image after the visual tower
            H: feature grid height (unused by the adapter)
            W: feature grid width (unused by the adapter)

        Returns:
            Enhanced [N, D] tensor
        """
        del H, W, kwargs

        if embed.ndim != 2:
            raise ValueError(f"Expected embed to have shape [N, D], got {tuple(embed.shape)}")
        if embed.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed hidden size {self.embed_dim}, got {embed.shape[-1]}"
            )

        attention_mask = torch.ones(
            (1, embed.shape[0]),
            device=embed.device,
            dtype=torch.bool,
        )
        adapted = self.core(embed.unsqueeze(0), attention_mask=attention_mask)
        return adapted.squeeze(0)
