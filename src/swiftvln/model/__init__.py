# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Module for ms-swift

This module provides VLN (Visual Language Navigation) training support with
history frame compression using Qwen2.5-VL as the base model.

Components:
    1. Model Registration - Registers OverlapVLN model types with ms-swift
    2. Template - Custom template with differentiated compression for history/current images
    3. Dataset - Extended VLN dataset using <history_image>/<current_image> tokens
    4. Training Arguments - OverlapVLN-specific parameters (compress_stride, etc.)
    5. Compressor - 2D Average Pooling compression logic

Key features:
    - History frames compressed via configurable 2D average pooling
    - Current frames keep standard resolution
    - Compression applies in BOTH training and inference
    - Special tokens: <history_image>, <current_image>

Usage:
    python src/swiftvln/model/trainer.py \\
        --custom_register_path src/swiftvln/model \\
        --model_type overlapvln_qwen2_5_vl \\
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
    OverlapVLNQwen25VLTemplate,
    OverlapVLNQwen3VLTemplate,
    HISTORY_IMAGE_TOKEN,
    CURRENT_IMAGE_TOKEN,
)

print("[OverlapVLN] Templates registered successfully!")


# =============================================================================
# Model Registration
# =============================================================================

from swift.model import MODEL_MAPPING, Model, ModelArch, ModelGroup, ModelMeta, register_model

from .model import (
    OverlapVLNQwen25VLLoader,
    OverlapVLNQwen3VLLoader,
    OverlapVLNQwen25VLConfig,
    OverlapVLNQwen3VLConfig,
    OverlapVLNQwen25VLForConditionalGeneration,
    OverlapVLNQwen3VLForConditionalGeneration,
)

# Register OverlapVLN model based on Qwen2.5-VL (only if not already registered)
if 'overlapvln_qwen2_5_vl' not in MODEL_MAPPING:
    model_meta_kwargs = dict(
        model_type='overlapvln_qwen2_5_vl',
        model_groups=[
            ModelGroup([
                Model('overlapvln-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('overlapvln-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
            ])
        ],
        template='overlapvln_qwen2_5_vl',  # Use custom template with compression
        model_arch=ModelArch.qwen2_vl,
        architectures=['OverlapVLNQwen25VLForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
        tags=['vision', 'vln', 'navigation', 'compression'],
        is_multimodal=True,
    )
    model_meta_kwargs['loader'] = OverlapVLNQwen25VLLoader
    register_model(ModelMeta(**model_meta_kwargs))
    print("[OverlapVLN] Model 'overlapvln_qwen2_5_vl' registered successfully!")

if 'overlapvln_qwen3_vl' not in MODEL_MAPPING:
    model_meta_kwargs = dict(
        model_type='overlapvln_qwen3_vl',
        model_groups=[
            ModelGroup([
                Model('overlapvln-qwen3-vl-2b', 'Qwen/Qwen3-VL-2B-Instruct'),
                Model('overlapvln-qwen3-vl-4b', 'Qwen/Qwen3-VL-4B-Instruct'),
                Model('overlapvln-qwen3-vl-8b', 'Qwen/Qwen3-VL-8B-Instruct'),
            ])
        ],
        template='overlapvln_qwen3_vl',
        model_arch=ModelArch.qwen3_vl,
        architectures=['OverlapVLNQwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'vln', 'navigation', 'compression', 'qwen3'],
        is_multimodal=True,
    )
    model_meta_kwargs['loader'] = OverlapVLNQwen3VLLoader
    register_model(ModelMeta(**model_meta_kwargs))
    print("[OverlapVLN] Model 'overlapvln_qwen3_vl' registered successfully!")


# =============================================================================
# Public API
# =============================================================================

from .dataset import OverlapVLNDataset
from .arguments import OverlapVLNTrainArguments
from swiftvln.common import HistoryTokenCompressor

__all__ = [
    # Model
    'OverlapVLNQwen25VLLoader',
    'OverlapVLNQwen3VLLoader',
    'OverlapVLNQwen25VLConfig',
    'OverlapVLNQwen3VLConfig',
    'OverlapVLNQwen25VLForConditionalGeneration',
    'OverlapVLNQwen3VLForConditionalGeneration',
    # Template
    'OverlapVLNQwen25VLTemplate',
    'OverlapVLNQwen3VLTemplate',
    'HISTORY_IMAGE_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    # Dataset
    'OverlapVLNDataset',
    # Compression
    'HistoryTokenCompressor',
    # Training
    'OverlapVLNTrainArguments',
]

# Note: OverlapVLNSft and train_main are imported from trainer.py directly
# to avoid circular imports when running trainer.py as a script
