# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Model based on Qwen2.5-VL

This module provides a UniNaVid model that inherits from Qwen2.5-VL.
The key addition is registering custom special tokens for three-level
image compression (following Uni-NaVid's memory architecture):
- <long_term_image>: For long-term memory frames (highly compressed, similarity merged)
- <short_term_image>: For short-term memory frames (compressed with pooling)
- <current_image>: For current observation (no compression)

UniNaVid uses single-turn dialogue with Uni-NaVid style memory management:
- Long-term memory: Frames older than short_term_frames, merged by similarity
- Short-term memory: Recent short_term_frames frames, compressed with Grid Pooling
- Current frame: Full resolution for detailed observation

Architecture:
    UniNaVidQwen25VLForConditionalGeneration
            ↓ inherits
    Qwen2_5_VLForConditionalGeneration (from transformers)
"""

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

# Special tokens (must match dataset.py and template.py)
LONG_TERM_IMAGE_TOKEN = "<long_term_image>"
SHORT_TERM_IMAGE_TOKEN = "<short_term_image>"
CURRENT_IMAGE_TOKEN = "<current_image>"


class UniNaVidQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for UniNaVid model based on Qwen2.5-VL.
    
    Inherits all configuration from Qwen2_5_VLConfig.
    """
    model_type = "uninavid_qwen2_5_vl"


class UniNaVidQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    UniNaVid model for Visual Language Navigation based on Qwen2.5-VL.
    
    This model directly inherits from Qwen2_5_VLForConditionalGeneration.
    The ms-swift framework handles all multimodal processing through
    the UniNaVid template (UniNaVidQwen25VLTemplate).
    
    Key features:
    - Three-level memory architecture: long-term, short-term, current
    - Custom special tokens for differentiated image compression
    - Long-term memory: similarity-based merging (Uni-NaVid style)
    - Short-term memory: Grid Pooling compression
    - Current frame: Full resolution
    - All compression/merging logic is handled in the template's _post_encode()
    - Single-turn dialogue (no KV cache management needed)
    
    Note: reset() and reset_for_env() are kept for evaluator compatibility,
    but they are essentially no-ops for single-turn inference.
    """
    
    config_class = UniNaVidQwen25VLConfig

    def reset(self, env_num: int = 1):
        """
        Initialize for evaluation (required by evaluator).
        
        For UniNaVid's single-turn inference, this is essentially a no-op
        but kept for API compatibility with evaluator.
        
        Args:
            env_num: Number of parallel environments
        """
        self._env_num = env_num

    def reset_for_env(self, env_idx: int):
        """
        Reset for a single environment (required by evaluator).
        
        For UniNaVid's single-turn inference, this is a no-op
        but kept for API compatibility.
        
        Args:
            env_idx: Environment index to reset
        """
        pass  # No-op for single-turn inference


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    AutoConfig.register("uninavid_qwen2_5_vl", UniNaVidQwen25VLConfig)
    AutoModel.register(UniNaVidQwen25VLConfig, UniNaVidQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(UniNaVidQwen25VLConfig, UniNaVidQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_uninavid_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """
    Load UniNaVid Qwen2.5-VL model and processor.
    
    This function is called by ms-swift when loading the registered model.
    It adds custom special tokens (<long_term_image>, <short_term_image>, 
    <current_image>) to the tokenizer and resizes model embeddings if necessary.
    
    Args:
        model_dir: Path to the model directory (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')
        model_info: ModelInfo object from ms-swift
        model_kwargs: Additional kwargs for model loading
        load_model: Whether to load the model weights
        **kwargs: Additional arguments including attn_impl, torch_dtype, etc.
        
    Returns:
        tuple: (model, processor)
    """
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
    
    # Use UniNaVid's custom model class
    kwargs['automodel_class'] = UniNaVidQwen25VLForConditionalGeneration
    
    # Use the standard Qwen2.5-VL loader
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # Add custom special tokens for three-level image compression
    special_tokens_dict = {
        'additional_special_tokens': [LONG_TERM_IMAGE_TOKEN, SHORT_TERM_IMAGE_TOKEN, CURRENT_IMAGE_TOKEN]
    }
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added > 0:
        print(f"[UniNaVid] Added {num_added} special tokens to tokenizer:")
        print(f"  - {LONG_TERM_IMAGE_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(LONG_TERM_IMAGE_TOKEN)}")
        print(f"  - {SHORT_TERM_IMAGE_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(SHORT_TERM_IMAGE_TOKEN)}")
        print(f"  - {CURRENT_IMAGE_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)}")
        
        # Resize model embeddings only if tokenizer is larger than model vocab
        if model is not None:
            tokenizer_vocab_size = len(processor.tokenizer)
            model_vocab_size = model.config.vocab_size
            if tokenizer_vocab_size > model_vocab_size:
                model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"  - Resized model embeddings: {model_vocab_size} -> {tokenizer_vocab_size}")
            else:
                print(f"  - Model vocab ({model_vocab_size}) already covers tokenizer ({tokenizer_vocab_size}), no resize needed")
    
    return model, processor
