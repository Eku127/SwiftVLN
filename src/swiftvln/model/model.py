# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SwiftVLN models based on Qwen VL families.

This module provides SwiftVLN model wrappers for Qwen2.5-VL and Qwen3-VL.
The key additions are registering custom special tokens for differentiated
image compression and attaching optional embedding enhancements:
- <history_image>: For history frames (will be compressed)
- <history_memory>: Unified history memory block
- <current_image>: For current frames (no compression)
"""

import json
import os
from types import MethodType

import torch
from swift.model.models.qwen import Qwen2_5VLLoader, Qwen3VLLoader
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from swiftvln.common.constants import CURRENT_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN

# Special tokens (must match dataset.py and template.py)
HISTORY_IMAGE_TOKEN = "<history_image>"  # Legacy: per-frame token (deprecated)
SWIFTVLN_SPECIAL_TOKENS = [HISTORY_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN, CURRENT_IMAGE_TOKEN]


class SwiftVLNStreamingMixin:
    """
    KV-cache state management shared by SwiftVLN Qwen-family wrappers.
    """

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


class SwiftVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """Configuration for SwiftVLN model based on Qwen2.5-VL."""

    model_type = "swiftvln_qwen2_5_vl"


class SwiftVLNQwen25VLForConditionalGeneration(
    SwiftVLNStreamingMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    """SwiftVLN model wrapper for Qwen2.5-VL."""

    config_class = SwiftVLNQwen25VLConfig


class SwiftVLNQwen3VLConfig(Qwen3VLConfig):
    """Configuration for SwiftVLN model based on Qwen3-VL."""

    model_type = "swiftvln_qwen3_vl"


class SwiftVLNQwen3VLForConditionalGeneration(
    SwiftVLNStreamingMixin,
    Qwen3VLForConditionalGeneration,
):
    """SwiftVLN model wrapper for Qwen3-VL."""

    config_class = SwiftVLNQwen3VLConfig


def _patch_qwen3_inputs_embeds_only_forward(qwen3_model) -> None:
    """
    Support the SwiftVLN path where visual features have already been injected
    into inputs_embeds before the Qwen3-VL submodel forward.
    """
    if getattr(qwen3_model, '_swiftvln_inputs_embeds_only_patch', False):
        return

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

    origin_forward = qwen3_model.forward

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        swiftvln_embeds_only = (
            inputs_embeds is not None
            and input_ids is None
            and pixel_values is None
            and pixel_values_videos is None
        )
        if not swiftvln_embeds_only:
            return origin_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        if position_ids is None:
            attention_mask_tensor = attention_mask
            if isinstance(attention_mask_tensor, dict):
                attention_mask_tensor = attention_mask_tensor.get('full_attention')
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask_tensor is not None:
                position_ids = attention_mask_tensor.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask_tensor == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(inputs_embeds.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                self.rope_deltas = max_position_ids + 1 - attention_mask_tensor.shape[-1]
            else:
                position_ids = (
                    torch.arange(seq_length, device=inputs_embeds.device)
                    .view(1, 1, -1)
                    .expand(3, batch_size, -1)
                )
                self.rope_deltas = torch.zeros(
                    [batch_size, 1],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    qwen3_model._swiftvln_origin_forward = origin_forward
    qwen3_model.forward = MethodType(forward, qwen3_model)
    qwen3_model._swiftvln_inputs_embeds_only_patch = True


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    AutoConfig.register("swiftvln_qwen2_5_vl", SwiftVLNQwen25VLConfig)
    AutoModel.register(SwiftVLNQwen25VLConfig, SwiftVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(SwiftVLNQwen25VLConfig, SwiftVLNQwen25VLForConditionalGeneration)
    AutoConfig.register("swiftvln_qwen3_vl", SwiftVLNQwen3VLConfig)
    AutoModel.register(SwiftVLNQwen3VLConfig, SwiftVLNQwen3VLForConditionalGeneration)
    AutoModelForCausalLM.register(SwiftVLNQwen3VLConfig, SwiftVLNQwen3VLForConditionalGeneration)
except Exception:
    pass


def _pop_embedding_options(kwargs):
    return {
        'use_pixel_embed': kwargs.pop('use_pixel_embed', False),
        'use_pose_embed': kwargs.pop('use_pose_embed', False),
        'use_uav_adapter': kwargs.pop('use_uav_adapter', False),
        'uav_adapter_path': kwargs.pop('uav_adapter_path', ''),
        'uav_adapter_type': kwargs.pop('uav_adapter_type', 'transformer_v1'),
        'uav_adapter_apply_scope': kwargs.pop('uav_adapter_apply_scope', 'all_images'),
        'pose_fusion_method': kwargs.pop('pose_fusion_method', 'additive'),
        'pose_norm_scale': kwargs.pop('pose_norm_scale', 100.0),
    }


def _attach_embedding_enhancement(model, model_dir: str, **options) -> None:
    if model is None:
        return

    from swiftvln.common.embedding_enhancement import create_embedding_pipeline

    use_pixel_embed = options['use_pixel_embed']
    use_pose_embed = options['use_pose_embed']
    use_uav_adapter = options['use_uav_adapter']
    uav_adapter_path = options['uav_adapter_path']
    uav_adapter_type = options['uav_adapter_type']
    uav_adapter_apply_scope = options['uav_adapter_apply_scope']
    pose_fusion_method = options['pose_fusion_method']
    pose_norm_scale = options['pose_norm_scale']

    embed_dim = model.config.hidden_size

    model.embed_enhance = create_embedding_pipeline(
        embed_dim=embed_dim,
        use_pixel_embed=use_pixel_embed,
        use_pose_embed=use_pose_embed,
        use_uav_adapter=use_uav_adapter,
        pose_fusion=pose_fusion_method,
        pose_norm_scale=pose_norm_scale,
        uav_adapter_path=uav_adapter_path,
        uav_adapter_type=uav_adapter_type,
        uav_adapter_apply_scope=uav_adapter_apply_scope,
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

    # Explicit external UAV adapter should win over local embed_enhance weights.
    if use_uav_adapter and uav_adapter_path and 'uav' in model.embed_enhance.enhancements:
        resolved_path = model.embed_enhance.enhancements['uav'].load_external_checkpoint(
            uav_adapter_path,
            strict=True,
        )
        print(f"[SwiftVLN] Loaded external UAV adapter from: {resolved_path}")

    if not model.embed_enhance.is_empty:
        print(f"[SwiftVLN] Embedding enhancement pipeline: {model.embed_enhance}")
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
    if use_uav_adapter and 'uav' in model.embed_enhance.enhancements:
        _set_alias('uav_adapter', model.embed_enhance.enhancements['uav'])
    else:
        _set_alias('uav_adapter', None)


class SwiftVLNQwen25VLLoader(Qwen2_5VLLoader):
    """ms-swift 4.x loader for the SwiftVLN Qwen2.5-VL model."""

    def __init__(self, *args, **kwargs):
        self._swiftvln_embedding_options = _pop_embedding_options(kwargs)
        new_special_tokens = list(kwargs.pop('new_special_tokens', None) or [])
        for token in SWIFTVLN_SPECIAL_TOKENS:
            if token not in new_special_tokens:
                new_special_tokens.append(token)
        kwargs['new_special_tokens'] = new_special_tokens
        super().__init__(*args, **kwargs)

    def get_model(self, model_dir: str, *args, **kwargs):
        self.auto_model_cls = self.auto_model_cls or SwiftVLNQwen25VLForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        _attach_embedding_enhancement(
            model,
            model_dir,
            **self._swiftvln_embedding_options,
        )
        return model


class SwiftVLNQwen3VLLoader(Qwen3VLLoader):
    """ms-swift 4.x loader for the SwiftVLN Qwen3-VL model."""

    def __init__(self, *args, **kwargs):
        self._swiftvln_embedding_options = _pop_embedding_options(kwargs)
        new_special_tokens = list(kwargs.pop('new_special_tokens', None) or [])
        for token in SWIFTVLN_SPECIAL_TOKENS:
            if token not in new_special_tokens:
                new_special_tokens.append(token)
        kwargs['new_special_tokens'] = new_special_tokens
        super().__init__(*args, **kwargs)

    def get_model(self, model_dir: str, *args, **kwargs):
        self.auto_model_cls = self.auto_model_cls or SwiftVLNQwen3VLForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        if model is not None and hasattr(model, 'model'):
            _patch_qwen3_inputs_embeds_only_forward(model.model)
        _attach_embedding_enhancement(
            model,
            model_dir,
            **self._swiftvln_embedding_options,
        )
        return model


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
            print(f"[SwiftVLN] Warning: failed to parse {index_path}: {e}")
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
            print(f"[SwiftVLN] Warning: failed to read embed_enhance weights from safetensors: {e}")
    
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
                    print(f"[SwiftVLN] Found legacy pixel_embed.* keys, remapping to embed_enhance.*")
            except Exception as e:
                print(f"[SwiftVLN] Warning: failed to read legacy pixel_embed weights: {e}")
    
    if enhancement_sd:
        missing, unexpected = model.embed_enhance.load_state_dict(enhancement_sd, strict=False)
        if missing or unexpected:
            print(f"[SwiftVLN] embed_enhance load_state_dict warnings:")
            if missing:
                print(f"  - missing_keys: {missing}")
            if unexpected:
                print(f"  - unexpected_keys: {unexpected}")
        else:
            print(f"[SwiftVLN] Restored embed_enhance weights from checkpoint ({len(enhancement_sd)} tensors)")
