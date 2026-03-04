# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Model based on Qwen2.5-VL

This module provides a MonoVLN model that inherits from Qwen2.5-VL.
The key addition is registering custom special tokens for differentiated
image compression:
- <history_image>: For history frames (will be compressed)
- <current_image>: For current frames (no compression)

MonoVLN uses single-turn dialogue, so KV cache management is simplified
compared to StreamVLN's multi-turn approach.

Architecture:
    MonoVLNQwen25VLForConditionalGeneration
            ↓ inherits
    Qwen2_5_VLForConditionalGeneration (from transformers)
"""

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

# Special tokens (must match dataset.py and template.py)
HISTORY_IMAGE_TOKEN = "<history_image>"
CURRENT_IMAGE_TOKEN = "<current_image>"


class MonoVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for MonoVLN model based on Qwen2.5-VL.
    
    Inherits all configuration from Qwen2_5_VLConfig.
    """
    model_type = "monovln_qwen2_5_vl"


class MonoVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    MonoVLN model for Visual Language Navigation based on Qwen2.5-VL.
    
    This model directly inherits from Qwen2_5_VLForConditionalGeneration.
    The ms-swift framework handles all multimodal processing through
    the MonoVLN template (MonoVLNQwen25VLTemplate).
    
    Key features:
    - Custom special tokens for history/current image differentiation
    - All compression logic is handled in the template's _post_encode()
    - Single-turn dialogue (no KV cache management needed)
    
    Note: reset() and reset_for_env() are kept for evaluator compatibility,
    but they are essentially no-ops for single-turn inference.
    """
    
    config_class = MonoVLNQwen25VLConfig

    def reset(self, env_num: int = 1):
        """
        Initialize for evaluation (required by evaluator).
        
        For MonoVLN's single-turn inference, this is essentially a no-op
        but kept for API compatibility with evaluator.
        
        Args:
            env_num: Number of parallel environments
        """
        self._env_num = env_num

    def reset_for_env(self, env_idx: int):
        """
        Reset for a single environment (required by evaluator).
        
        For MonoVLN's single-turn inference, this is a no-op
        but kept for API compatibility.
        
        Args:
            env_idx: Environment index to reset
        """
        pass  # No-op for single-turn inference


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    AutoConfig.register("monovln_qwen2_5_vl", MonoVLNQwen25VLConfig)
    AutoModel.register(MonoVLNQwen25VLConfig, MonoVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(MonoVLNQwen25VLConfig, MonoVLNQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_monovln_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """
    Load MonoVLN Qwen2.5-VL model and processor.
    
    This function is called by ms-swift when loading the registered model.
    It adds custom special tokens (<history_image>, <current_image>) to the
    tokenizer and resizes model embeddings if necessary.
    
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
    
    # Use MonoVLN's custom model class
    kwargs['automodel_class'] = MonoVLNQwen25VLForConditionalGeneration
    
    # Use the standard Qwen2.5-VL loader
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # Add custom special tokens for differentiated image compression
    special_tokens_dict = {
        'additional_special_tokens': [HISTORY_IMAGE_TOKEN, CURRENT_IMAGE_TOKEN]
    }
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added > 0:
        print(f"[MonoVLN] Added {num_added} special tokens to tokenizer:")
        print(f"  - {HISTORY_IMAGE_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(HISTORY_IMAGE_TOKEN)}")
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
