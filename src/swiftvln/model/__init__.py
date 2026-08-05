# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SwiftVLN Module for ms-swift

This module provides VLN (Visual Language Navigation) training support with
history frame compression using Qwen2.5-VL as the base model.

Components:
    1. Model Registration - Registers SwiftVLN model types with ms-swift
    2. Template - Custom template with differentiated compression for history/current images
    3. Dataset - Extended VLN dataset using <history_memory>/<current_image> tokens
    4. Training Arguments - SwiftVLN-specific parameters (compress_stride, etc.)
    5. Compressor - 2D Average Pooling compression logic

Key features:
    - History frames compressed via configurable 2D average pooling
    - Current frames keep standard resolution
    - Compression applies in BOTH training and inference
    - Special tokens: <history_memory>, <current_image>

Usage:
    python src/swiftvln/model/trainer.py \\
        --custom_register_path src/swiftvln/model \\
        --model_type swiftvln_qwen2_5_vl \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --dataset /path/to/vln_data \\
        --compress_stride 2 \\
        ...

Architecture:
    Dataset returns: {'messages': [...], 'images': [PIL.Image, ...]}
                              ↓
    Template._encode(): Calculate different token counts for history/current
                              ↓
    Template._post_encode(): Apply pooling to history, keep current unchanged
                              ↓
    Model receives pre-fused inputs_embeds with compressed history
"""

# =============================================================================
# Template Registration (must be before model registration)
# =============================================================================

# Import and register the custom templates FIRST
from .template import (
    SwiftVLNQwen25VLTemplate,
    SwiftVLNQwen3VLTemplate,
    CURRENT_IMAGE_TOKEN,
)


# =============================================================================
# Model Registration
# =============================================================================

from swift.model import (
    MODEL_MAPPING,
    Model,
    ModelArch,
    ModelGroup,
    ModelMeta,
    register_model,
)

from .model import (
    SwiftVLNQwen25VLLoader,
    SwiftVLNQwen3VLLoader,
    SwiftVLNQwen25VLConfig,
    SwiftVLNQwen3VLConfig,
    SwiftVLNQwen25VLForConditionalGeneration,
    SwiftVLNQwen3VLForConditionalGeneration,
)

# Register SwiftVLN model based on Qwen2.5-VL (only if not already registered)
if "swiftvln_qwen2_5_vl" not in MODEL_MAPPING:
    model_meta_kwargs = dict(
        model_type="swiftvln_qwen2_5_vl",
        model_groups=[
            ModelGroup(
                [
                    Model("swiftvln-qwen2.5-vl-3b", "Qwen/Qwen2.5-VL-3B-Instruct"),
                    Model("swiftvln-qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
                ]
            )
        ],
        template="swiftvln_qwen2_5_vl",  # Use custom template with compression
        model_arch=ModelArch.qwen2_vl,
        architectures=["SwiftVLNQwen25VLForConditionalGeneration"],
        requires=["transformers>=4.49", "qwen_vl_utils>=0.0.6"],
        tags=["vision", "vln", "navigation", "compression"],
        is_multimodal=True,
    )
    model_meta_kwargs["loader"] = SwiftVLNQwen25VLLoader
    register_model(ModelMeta(**model_meta_kwargs))

if "swiftvln_qwen3_vl" not in MODEL_MAPPING:
    model_meta_kwargs = dict(
        model_type="swiftvln_qwen3_vl",
        model_groups=[
            ModelGroup(
                [
                    Model("swiftvln-qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct"),
                    Model("swiftvln-qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct"),
                    Model("swiftvln-qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct"),
                ]
            )
        ],
        template="swiftvln_qwen3_vl",
        model_arch=ModelArch.qwen3_vl,
        architectures=["SwiftVLNQwen3VLForConditionalGeneration"],
        requires=["transformers>=4.57", "qwen_vl_utils>=0.0.14", "decord"],
        tags=["vision", "vln", "navigation", "compression", "qwen3"],
        is_multimodal=True,
    )
    model_meta_kwargs["loader"] = SwiftVLNQwen3VLLoader
    register_model(ModelMeta(**model_meta_kwargs))


# =============================================================================
# Public API
# =============================================================================

from .dataset import SwiftVLNDataset
from .arguments import SwiftVLNTrainArguments
from swiftvln.common.history_processors.compressor import HistoryTokenCompressor

__all__ = [
    # Model
    "SwiftVLNQwen25VLLoader",
    "SwiftVLNQwen3VLLoader",
    "SwiftVLNQwen25VLConfig",
    "SwiftVLNQwen3VLConfig",
    "SwiftVLNQwen25VLForConditionalGeneration",
    "SwiftVLNQwen3VLForConditionalGeneration",
    # Template
    "SwiftVLNQwen25VLTemplate",
    "SwiftVLNQwen3VLTemplate",
    "CURRENT_IMAGE_TOKEN",
    # Dataset
    "SwiftVLNDataset",
    # Compression
    "HistoryTokenCompressor",
    # Training
    "SwiftVLNTrainArguments",
]

# Note: SwiftVLNSft and train_main are imported from trainer.py directly
# to avoid circular imports when running trainer.py as a script
