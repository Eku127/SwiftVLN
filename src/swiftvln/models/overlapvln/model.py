# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Model based on Qwen2.5-VL

This module provides a OverlapVLN model that inherits from Qwen2.5-VL.
The key addition is registering custom special tokens for differentiated
image compression:
- <history_image>: For history frames (will be compressed)
- <current_image>: For current frames (no compression)

Architecture:
    OverlapVLNQwen25VLForConditionalGeneration
            ↓ inherits
    Qwen2_5_VLForConditionalGeneration (from transformers)
"""

import json
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from swiftvln.common.constants import CURRENT_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN

# Special tokens (must match dataset.py and template.py)
HISTORY_IMAGE_TOKEN = "<history_image>"  # Legacy: per-frame token (deprecated)


class OverlapVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for OverlapVLN model based on Qwen2.5-VL.
    
    Inherits all configuration from Qwen2_5_VLConfig.
    """
    model_type = "overlapvln_qwen2_5_vl"


class OverlapVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    OverlapVLN model for Visual Language Navigation based on Qwen2.5-VL.
    
    This model directly inherits from Qwen2_5_VLForConditionalGeneration.
    The ms-swift framework handles all multimodal processing through
    the OverlapVLN template (OverlapVLNQwen25VLTemplate).
    
    Key features:
    - Custom special tokens for history/current image differentiation
    - All compression logic is handled in the template's _post_encode()
    - Streaming inference state management (KV cache)
    """
    
    config_class = OverlapVLNQwen25VLConfig

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
    AutoConfig.register("overlapvln_qwen2_5_vl", OverlapVLNQwen25VLConfig)
    AutoModel.register(OverlapVLNQwen25VLConfig, OverlapVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(OverlapVLNQwen25VLConfig, OverlapVLNQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_overlapvln_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """
    Load OverlapVLN Qwen2.5-VL model and processor.
    
    This function is called by ms-swift when loading the registered model.
    It adds custom special tokens (<history_image>, <current_image>) to the
    tokenizer and resizes model embeddings if necessary.
    
    Args:
        model_dir: Path to the model directory (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')
        model_info: ModelInfo object from ms-swift
        model_kwargs: Additional kwargs for model loading
        load_model: Whether to load the model weights
        **kwargs: Additional arguments including:
            - use_pixel_embed: bool, whether to enable pixel coordinate embedding enhancement
            - use_pose_embed: bool, whether to enable pose embedding enhancement
            - pose_fusion_method: str, pose fusion method ('additive' or 'film')
            - pose_norm_scale: float, tanh normalization scale for pose positions
            - attn_impl, torch_dtype, etc.
        
    Returns:
        tuple: (model, processor)
    """
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
    
    # Extract custom arguments before passing to parent loader
    use_pixel_embed = kwargs.pop('use_pixel_embed', False)
    use_pose_embed = kwargs.pop('use_pose_embed', False)
    pose_fusion_method = kwargs.pop('pose_fusion_method', 'additive')
    pose_norm_scale = kwargs.pop('pose_norm_scale', 100.0)
    
    # Use OverlapVLN's custom model class
    kwargs['automodel_class'] = OverlapVLNQwen25VLForConditionalGeneration
    
    # Use the standard Qwen2.5-VL loader
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # Add custom special tokens for differentiated image compression
    # Include both legacy per-frame token and new unified memory token
    special_tokens_dict = {
        'additional_special_tokens': [HISTORY_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN, CURRENT_IMAGE_TOKEN]
    }
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added > 0:
        print(f"[OverlapVLN] Added {num_added} special tokens to tokenizer:")
        print(f"  - {HISTORY_IMAGE_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(HISTORY_IMAGE_TOKEN)} (legacy)")
        print(f"  - {HISTORY_MEMORY_TOKEN}: {processor.tokenizer.convert_tokens_to_ids(HISTORY_MEMORY_TOKEN)} (unified)")
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
    
    # Create embedding enhancement pipeline
    # The pipeline uses nn.ModuleDict, so all parameters are automatically:
    # - Managed by the optimizer (trained)
    # - Serialized in state_dict (saved/restored with checkpoints)
    # - No manual checkpoint restore logic needed
    if model is not None:
        from swiftvln.common.embedding_enhancement import create_embedding_pipeline
        
        embed_dim = model.config.hidden_size
        
        model.embed_enhance = create_embedding_pipeline(
            embed_dim=embed_dim,
            use_pixel_embed=use_pixel_embed,
            use_pose_embed=use_pose_embed,
            pose_fusion=pose_fusion_method,
            pose_norm_scale=pose_norm_scale,
        )
        
        # Move to the same device/dtype as the visual encoder/model
        target_dtype = model.visual.dtype if hasattr(model, 'visual') and hasattr(model.visual, 'dtype') else None
        try:
            target_device = next(model.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            target_device = getattr(model, 'device', torch.device('cpu'))
        to_kwargs = {}
        if target_dtype is not None:
            to_kwargs['dtype'] = target_dtype
        if target_device is not None:
            to_kwargs['device'] = target_device
        if to_kwargs and not model.embed_enhance.is_empty:
            model.embed_enhance = model.embed_enhance.to(**to_kwargs)
        
        # Try to restore embed_enhance weights from local checkpoint (if present).
        # This is needed at eval time when loading a finetuned checkpoint that
        # contains trained enhancement parameters (e.g., pixel_embed weights).
        if not model.embed_enhance.is_empty and os.path.isdir(model_dir):
            _restore_enhancement_weights(model, model_dir)
        
        if not model.embed_enhance.is_empty:
            print(f"[OverlapVLN] Embedding enhancement pipeline: {model.embed_enhance}")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - Enhancements: {model.embed_enhance.enhancement_names}")
            print(f"  - Module will be trained and saved with checkpoints")
        
        # Backward compatibility: expose aliases without registering duplicate submodules.
        def _set_alias(alias_name: str, value) -> None:
            if hasattr(model, '_modules'):
                model._modules.pop(alias_name, None)
            model.__dict__[alias_name] = value

        if use_pixel_embed and 'pixel' in model.embed_enhance.enhancements:
            _set_alias('pixel_embed', model.embed_enhance.enhancements['pixel'])
        else:
            _set_alias('pixel_embed', None)
        if use_pose_embed and 'pose' in model.embed_enhance.enhancements:
            _set_alias('pose_embed', model.embed_enhance.enhancements['pose'])
        else:
            _set_alias('pose_embed', None)
    
    return model, processor


def _restore_enhancement_weights(model, model_dir: str):
    """
    Restore embed_enhance weights from a local checkpoint directory.
    
    Scans safetensors files for keys starting with 'embed_enhance.' and
    loads them into the model's embed_enhance pipeline.
    
    Args:
        model: Model with embed_enhance attribute
        model_dir: Path to checkpoint directory
    """
    prefix = 'embed_enhance.'
    enhancement_sd = {}
    
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    shard_files = []
    
    if os.path.isfile(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})
            shard_files = sorted({
                shard for name, shard in weight_map.items()
                if name.startswith(prefix)
            })
        except Exception as e:
            print(f"[OverlapVLN] Warning: failed to parse {index_path}: {e}")
    elif os.path.isfile(os.path.join(model_dir, 'model.safetensors')):
        shard_files = ['model.safetensors']
    
    if shard_files:
        try:
            from safetensors import safe_open
            for shard in shard_files:
                shard_path = os.path.join(model_dir, shard)
                with safe_open(shard_path, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            enhancement_sd[key.replace(prefix, '', 1)] = f.get_tensor(key)
        except Exception as e:
            print(f"[OverlapVLN] Warning: failed to read embed_enhance weights from safetensors: {e}")
    
    # Also try legacy keys (pixel_embed.*) for backward compatibility with old checkpoints
    if not enhancement_sd:
        legacy_prefix = 'pixel_embed.'
        legacy_shard_files = []
        if os.path.isfile(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                weight_map = index_data.get('weight_map', {})
                legacy_shard_files = sorted({
                    shard for name, shard in weight_map.items()
                    if name.startswith(legacy_prefix)
                })
            except Exception:
                pass
        elif os.path.isfile(os.path.join(model_dir, 'model.safetensors')):
            legacy_shard_files = ['model.safetensors']
        
        if legacy_shard_files:
            try:
                from safetensors import safe_open
                for shard in legacy_shard_files:
                    shard_path = os.path.join(model_dir, shard)
                    with safe_open(shard_path, framework='pt', device='cpu') as f:
                        for key in f.keys():
                            if key.startswith(legacy_prefix):
                                # Map legacy pixel_embed.* -> enhancements.pixel.*
                                new_key = 'enhancements.pixel.' + key.replace(legacy_prefix, '', 1)
                                enhancement_sd[new_key] = f.get_tensor(key)
                if enhancement_sd:
                    print(f"[OverlapVLN] Found legacy pixel_embed.* keys, remapping to embed_enhance.*")
            except Exception as e:
                print(f"[OverlapVLN] Warning: failed to read legacy pixel_embed weights: {e}")
    
    if enhancement_sd:
        missing, unexpected = model.embed_enhance.load_state_dict(enhancement_sd, strict=False)
        if missing or unexpected:
            print(f"[OverlapVLN] embed_enhance load_state_dict warnings:")
            if missing:
                print(f"  - missing_keys: {missing}")
            if unexpected:
                print(f"  - unexpected_keys: {unexpected}")
        else:
            print(f"[OverlapVLN] Restored embed_enhance weights from checkpoint ({len(enhancement_sd)} tensors)")
