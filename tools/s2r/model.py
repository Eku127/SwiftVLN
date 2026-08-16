"""Modeling utilities for Stage-A sim-to-real alignment."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from swiftvln.modeling.embeddings.s2r_adapter import Sim2RealAdapter


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve target device from a CLI string."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(device: torch.device, dtype_name: str = "auto") -> torch.dtype:
    """Resolve compute dtype for teacher loading."""
    if dtype_name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = mapping[key]
    if device.type != "cuda" and dtype != torch.float32:
        return torch.float32
    return dtype


def split_visual_embeddings(
    all_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int,
) -> List[torch.Tensor]:
    """Split concatenated Qwen visual embeddings into one tensor per image."""
    merge_length = merge_size ** 2
    outputs: List[torch.Tensor] = []
    offset = 0
    for i in range(image_grid_thw.shape[0]):
        num_tokens = int(image_grid_thw[i].prod().item() // merge_length)
        outputs.append(all_embeddings[offset:offset + num_tokens])
        offset += num_tokens
    return outputs


def pad_token_sequences(
    token_sequences: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length token sequences and build a validity mask."""
    if not token_sequences:
        raise ValueError("token_sequences must not be empty")

    casted = [tokens.to(device=device, dtype=dtype) for tokens in token_sequences]
    lengths = torch.tensor([tokens.shape[0] for tokens in casted], device=device, dtype=torch.long)
    padded = pad_sequence(casted, batch_first=True, padding_value=0.0)
    steps = torch.arange(padded.shape[1], device=device).unsqueeze(0)
    mask = steps < lengths.unsqueeze(1)
    return padded, mask


def masked_mean_pool(tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token features using a validity mask."""
    mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denom


class ProjectionHead(nn.Module):
    """Two-layer projection head shared across UAV/SAT pooled features."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TeacherVisionTower:
    """Frozen Qwen visual tower and paired processor."""

    visual: nn.Module
    processor: object
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    merge_size: int

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: torch.device,
        dtype_name: str = "auto",
    ) -> "TeacherVisionTower":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Import for side effects; this keeps future custom configs loadable.
        try:
            from swiftvln.modeling import register_swiftvln_models

            register_swiftvln_models()
        except Exception:
            pass

        load_dtype = resolve_dtype(device, dtype_name)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=load_dtype,
        )

        hidden_size = int(getattr(model.config, "hidden_size"))
        visual = model.visual
        visual.eval()
        for parameter in visual.parameters():
            parameter.requires_grad = False

        visual_dtype = load_dtype if device.type == "cuda" else torch.float32
        visual = visual.to(device=device, dtype=visual_dtype)
        merge_size = int(processor.image_processor.merge_size)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return cls(
            visual=visual,
            processor=processor,
            device=device,
            dtype=visual_dtype,
            hidden_size=hidden_size,
            merge_size=merge_size,
        )

    @torch.no_grad()
    def encode_images(self, images: Sequence[object]) -> List[torch.Tensor]:
        """Encode a list of PIL images into per-image token tensors."""
        media_inputs = self.processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values = media_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_grid_thw = media_inputs["image_grid_thw"].to(device=self.device)
        all_embeddings = self.visual(pixel_values, grid_thw=image_grid_thw)
        return split_visual_embeddings(all_embeddings, image_grid_thw, self.merge_size)
