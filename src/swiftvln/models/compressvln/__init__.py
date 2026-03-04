# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Module for ms-swift

This module provides VLN (Visual Language Navigation) training support with
history frame compression using Qwen2.5-VL as the base model.

Components:
    1. Model Registration - Registers CompressVLN model type with ms-swift
    2. Template - Custom template with differentiated compression for history/current images
    3. Dataset - Extended VLN dataset using <history_image>/<current_image> tokens
    4. Training Arguments - CompressVLN-specific parameters (compress_stride, etc.)
    5. Compressor - 2D Average Pooling compression logic

Key features:
    - History frames compressed via configurable 2D average pooling
    - Current frames keep standard resolution
    - Compression applies in BOTH training and inference
    - Special tokens: <history_image>, <current_image>

Usage:
    python src/swiftvln/models/compressvln/trainer.py \\
        --custom_register_path src/swiftvln/models/compressvln \\
        --model_type compressvln_qwen2_5_vl \\
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

# Import and register the custom template FIRST
from .template import CompressVLNQwen25VLTemplate, HISTORY_IMAGE_TOKEN, CURRENT_IMAGE_TOKEN

print("[CompressVLN] Template 'compressvln_qwen2_5_vl' registered successfully!")


# =============================================================================
# Model Registration
# =============================================================================

from swift.llm import register_model, ModelMeta, ModelGroup, Model
from swift.llm.model import ModelArch, MODEL_MAPPING

from .model import (
    CompressVLNQwen25VLConfig,
    CompressVLNQwen25VLForConditionalGeneration,
    get_model_tokenizer_compressvln_qwen2_5_vl,
)

# Register CompressVLN model based on Qwen2.5-VL (only if not already registered)
if 'compressvln_qwen2_5_vl' not in MODEL_MAPPING:
    register_model(
        ModelMeta(
            model_type='compressvln_qwen2_5_vl',
            model_groups=[
                ModelGroup([
                    Model('compressvln-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                    Model('compressvln-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                ])
            ],
            template='compressvln_qwen2_5_vl',  # Use custom template with compression
            get_function=get_model_tokenizer_compressvln_qwen2_5_vl,
            model_arch=ModelArch.qwen2_vl,
            architectures=['CompressVLNQwen25VLForConditionalGeneration'],
            requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
            tags=['vision', 'vln', 'navigation', 'compression'],
            is_multimodal=True,
        )
    )
    print("[CompressVLN] Model 'compressvln_qwen2_5_vl' registered successfully!")


# =============================================================================
# Public API
# =============================================================================

from .dataset import CompressVLNDataset
from .arguments import CompressVLNTrainArguments
from swiftvln.common import HistoryTokenCompressor

__all__ = [
    # Model
    'CompressVLNQwen25VLConfig',
    'CompressVLNQwen25VLForConditionalGeneration',
    'get_model_tokenizer_compressvln_qwen2_5_vl',
    # Template
    'CompressVLNQwen25VLTemplate',
    'HISTORY_IMAGE_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    # Dataset
    'CompressVLNDataset',
    # Compression
    'HistoryTokenCompressor',
    # Training
    'CompressVLNTrainArguments',
]

# Note: CompressVLNSft and train_main are imported from trainer.py directly
# to avoid circular imports when running trainer.py as a script
