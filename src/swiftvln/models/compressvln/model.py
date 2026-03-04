# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Model based on Qwen2.5-VL

This module provides a CompressVLN model that inherits from Qwen2.5-VL.
The key addition is registering custom special tokens for differentiated
image compression:
- <history_image>: For history frames (will be compressed)
- <current_image>: For current frames (no compression)

Architecture:
    CompressVLNQwen25VLForConditionalGeneration
            ↓ inherits
    Qwen2_5_VLForConditionalGeneration (from transformers)
"""

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

# Special tokens (must match dataset.py and template.py)
HISTORY_IMAGE_TOKEN = "<history_image>"
CURRENT_IMAGE_TOKEN = "<current_image>"


class CompressVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for CompressVLN model based on Qwen2.5-VL.
    
    Inherits all configuration from Qwen2_5_VLConfig.
    """
    model_type = "compressvln_qwen2_5_vl"


class CompressVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    CompressVLN model for Visual Language Navigation based on Qwen2.5-VL.
    
    This model directly inherits from Qwen2_5_VLForConditionalGeneration.
    The ms-swift framework handles all multimodal processing through
    the CompressVLN template (CompressVLNQwen25VLTemplate).
    
    Key features:
    - Custom special tokens for history/current image differentiation
    - All compression logic is handled in the template's _post_encode()
    - Streaming inference state management (KV cache)
    """
    
    config_class = CompressVLNQwen25VLConfig

    # =====================================================
    # Streaming Inference State Management
    # =====================================================
    # These methods implement the KV cache management for streaming
    # inference, following the original StreamVLN's approach.
    # The cache allows efficient multi-turn inference within
    # num_frames windows.
    
    def reset(self, env_num: int = 1):
        """
        Initialize KV cache for multiple environments.
        
        MUST be called before evaluation starts!
        
        Args:
            env_num: Number of parallel environments
        """
        self._env_num = env_num
        self.curr_t = [0] * env_num
        self.cache = [dict() for _ in range(env_num)]

    def reset_for_env(self, env_idx: int):
        """
        Reset KV cache for a single environment.
        
        Called at the start of each episode and every num_frames steps.
        
        Args:
            env_idx: Environment index to reset
        """
        # Auto-initialize if not done
        if not hasattr(self, 'cache') or not hasattr(self, 'curr_t'):
            self.reset(env_num=max(env_idx + 1, 1))
        
        # Expand if needed
        while env_idx >= len(self.cache):
            self.cache.append(dict())
            self.curr_t.append(0)
        
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = dict()

    def get_cache(self, env_idx: int):
        """
        Get past_key_values cache for a specific environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            past_key_values or None if not cached
        """
        if hasattr(self, 'cache') and env_idx < len(self.cache):
            return self.cache[env_idx].get('past_key_values', None)
        return None

    def update_cache(self, env_idx: int, past_key_values):
        """
        Update past_key_values cache for a specific environment.
        
        Args:
            env_idx: Environment index
            past_key_values: KV cache from model.generate()
        """
        # Auto-initialize if not done
        if not hasattr(self, 'cache') or not hasattr(self, 'curr_t'):
            self.reset(env_num=max(env_idx + 1, 1))
        
        # Expand if needed
        while env_idx >= len(self.cache):
            self.cache.append(dict())
            self.curr_t.append(0)
        
        self.cache[env_idx]['past_key_values'] = past_key_values
        self.curr_t[env_idx] += 1
    
    def get_step_count(self, env_idx: int) -> int:
        """
        Get current step count for an environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Current step count
        """
        if hasattr(self, 'curr_t') and env_idx < len(self.curr_t):
            return self.curr_t[env_idx]
        return 0


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    AutoConfig.register("compressvln_qwen2_5_vl", CompressVLNQwen25VLConfig)
    AutoModel.register(CompressVLNQwen25VLConfig, CompressVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(CompressVLNQwen25VLConfig, CompressVLNQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_compressvln_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """
    Load CompressVLN Qwen2.5-VL model and processor.
    
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
    
    # Use CompressVLN's custom model class
    kwargs['automodel_class'] = CompressVLNQwen25VLForConditionalGeneration
    
    # Use the standard Qwen2.5-VL loader
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # Add custom special tokens for differentiated image compression
    special_tokens_dict = {
        'additional_special_tokens': [HISTORY_IMAGE_TOKEN, CURRENT_IMAGE_TOKEN]
    }
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added > 0:
        print(f"[CompressVLN] Added {num_added} special tokens to tokenizer:")
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
