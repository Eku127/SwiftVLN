# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Module for ms-swift

This module provides VLN (Visual Language Navigation) training support with
Uni-NaVid style three-level memory architecture using Qwen2.5-VL as the base model.

UniNaVid Memory Architecture:
- Long-term memory: Older history frames, merged by similarity (high compression)
- Short-term memory: Recent history frames, compressed with pooling
- Current observation: Current frame (no extra compression)

Components:
    1. Model Registration - Registers UniNaVid model type with ms-swift
    2. Template - Custom template with three-level compression and similarity merging
    3. Dataset - VLN dataset with three-level frame classification
    4. Arguments - UniNaVid-specific parameters (short_term_frames, similarity_threshold, etc.)

Key features:
    - Three-level memory architecture (following Uni-NaVid paper)
    - Long-term memory: pooling + mean + similarity-based merging
    - Short-term memory: pooling compression
    - Current frame: full resolution
    - Dynamic sequence trimming in _post_encode (N placeholders → M merged tokens)
    - Special tokens: <long_term_image>, <short_term_image>, <current_image>
    - Random frame dropping augmentation during training

Usage:
    # Training with ms-swift
    CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \\
        --custom_register_path src/swiftvln/models/uninavid \\
        --model_type uninavid_qwen2_5_vl \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --dataset /path/to/vln_data \\
        --short_term_frames 64 \\
        --similarity_threshold 0.985 \\
        --compress_stride 2 \\
        ...

Architecture:
    Dataset returns: {'messages': [...], 'images': [...], 
                      'num_long_term_images': N_lt, 'num_short_term_images': N_st}
                              ↓
    Template.replace_tag(): Classify images to <long/short_term/current_image>
                              ↓
    Template._encode(): Long-term: 1 placeholder/frame; Short-term/Current: pooled/full tokens
                              ↓
    Template._post_encode(): 
        - Visual encoding + Grid Pooling
        - Long-term: mean + similarity merge → M tokens
        - Trim sequence: N placeholders → M tokens
        - Fill embeddings
                              ↓
    Model receives pre-fused inputs_embeds with three-level compression
"""

# =============================================================================
# Template Registration (must be before model registration)
# =============================================================================

# Import and register the custom template FIRST
from .template import (
    UniNaVidQwen25VLTemplate, 
    LONG_TERM_IMAGE_TOKEN, 
    SHORT_TERM_IMAGE_TOKEN,
    CURRENT_IMAGE_TOKEN,
)

print("[UniNaVid] Template 'uninavid_qwen2_5_vl' registered successfully!")


# =============================================================================
# Model Registration
# =============================================================================

from swift.llm import register_model, ModelMeta, ModelGroup, Model
from swift.llm.model import ModelArch, MODEL_MAPPING

from .model import (
    UniNaVidQwen25VLConfig,
    UniNaVidQwen25VLForConditionalGeneration,
    get_model_tokenizer_uninavid_qwen2_5_vl,
)

# Register UniNaVid model based on Qwen2.5-VL (only if not already registered)
if 'uninavid_qwen2_5_vl' not in MODEL_MAPPING:
    register_model(
        ModelMeta(
            model_type='uninavid_qwen2_5_vl',
            model_groups=[
                ModelGroup([
                    Model('uninavid-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                    Model('uninavid-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                ])
            ],
            template='uninavid_qwen2_5_vl',  # Use custom template with three-level compression
            get_function=get_model_tokenizer_uninavid_qwen2_5_vl,
            model_arch=ModelArch.qwen2_vl,
            architectures=['UniNaVidQwen25VLForConditionalGeneration'],
            requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
            tags=['vision', 'vln', 'navigation', 'compression', 'single-turn', 'long-term-memory'],
            is_multimodal=True,
        )
    )
    print("[UniNaVid] Model 'uninavid_qwen2_5_vl' registered successfully!")


# =============================================================================
# Public API
# =============================================================================

from .dataset import UniNaVidDataset
from .arguments import UniNaVidArguments

__all__ = [
    # Model
    'UniNaVidQwen25VLConfig',
    'UniNaVidQwen25VLForConditionalGeneration',
    'get_model_tokenizer_uninavid_qwen2_5_vl',
    # Template
    'UniNaVidQwen25VLTemplate',
    'LONG_TERM_IMAGE_TOKEN',
    'SHORT_TERM_IMAGE_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    # Dataset
    'UniNaVidDataset',
    # Arguments
    'UniNaVidArguments',
]
