# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Module for ms-swift

This module provides VLN (Visual Language Navigation) training support with
history frame compression using Qwen2.5-VL as the base model.

MonoVLN uses single-turn dialogue format (vs StreamVLN's multi-turn):
- Input: Task instruction + history frames + current frame
- Output: Action sequence (num_future_steps actions)

Components:
    1. Model Registration - Registers MonoVLN model type with ms-swift
    2. Template - Custom template with differentiated compression for history/current images
    3. Dataset - Independent VLN dataset with single-turn dialogue format
    4. Training Arguments - MonoVLN-specific parameters (compress_stride, samples_per_episode, etc.)
    5. Compressor - 2D Average Pooling compression logic

Key features:
    - Single-turn dialogue format (user + assistant)
    - History frames compressed via configurable 2D average pooling
    - Current frame keeps standard resolution
    - Compression applies in BOTH training and inference
    - Special tokens: <history_image>, <current_image>

Usage:
    python src/swiftvln/models/monovln/trainer.py \\
        --custom_register_path src/swiftvln/models/monovln \\
        --model_type monovln_qwen2_5_vl \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --dataset /path/to/vln_data \\
        --compress_stride 2 \\
        --samples_per_episode 8 \\
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
from .template import MonoVLNQwen25VLTemplate, HISTORY_IMAGE_TOKEN, CURRENT_IMAGE_TOKEN

print("[MonoVLN] Template 'monovln_qwen2_5_vl' registered successfully!")


# =============================================================================
# Model Registration
# =============================================================================

from swift.llm import register_model, ModelMeta, ModelGroup, Model
from swift.llm.model import ModelArch, MODEL_MAPPING

from .model import (
    MonoVLNQwen25VLConfig,
    MonoVLNQwen25VLForConditionalGeneration,
    get_model_tokenizer_monovln_qwen2_5_vl,
)

# Register MonoVLN model based on Qwen2.5-VL (only if not already registered)
if 'monovln_qwen2_5_vl' not in MODEL_MAPPING:
    register_model(
        ModelMeta(
            model_type='monovln_qwen2_5_vl',
            model_groups=[
                ModelGroup([
                    Model('monovln-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                    Model('monovln-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                ])
            ],
            template='monovln_qwen2_5_vl',  # Use custom template with compression
            get_function=get_model_tokenizer_monovln_qwen2_5_vl,
            model_arch=ModelArch.qwen2_vl,
            architectures=['MonoVLNQwen25VLForConditionalGeneration'],
            requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
            tags=['vision', 'vln', 'navigation', 'compression', 'single-turn'],
            is_multimodal=True,
        )
    )
    print("[MonoVLN] Model 'monovln_qwen2_5_vl' registered successfully!")


# =============================================================================
# Public API
# =============================================================================

from .dataset import MonoVLNDataset
from .arguments import MonoVLNTrainArguments
from swiftvln.common.history_processors.compressor import HistoryTokenCompressor

__all__ = [
    # Model
    'MonoVLNQwen25VLConfig',
    'MonoVLNQwen25VLForConditionalGeneration',
    'get_model_tokenizer_monovln_qwen2_5_vl',
    # Template
    'MonoVLNQwen25VLTemplate',
    'HISTORY_IMAGE_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    # Dataset
    'MonoVLNDataset',
    # Compression
    'HistoryTokenCompressor',
    # Training
    'MonoVLNTrainArguments',
]

# Note: MonoVLNSft and train_main are imported from trainer.py directly
# to avoid circular imports when running trainer.py as a script
