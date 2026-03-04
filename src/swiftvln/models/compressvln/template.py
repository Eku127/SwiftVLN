# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Template with differentiated image compression.

This template extends Qwen2.5-VL template to support different compression rates
for history images vs current images:
- History images (<history_image>): Additional pooling (configurable stride)
- Current images (<current_image>): Standard merge_size=2 (4:1 compression)

Key differences from Navid template:
1. Configurable compress_stride (not fixed pool_size=2)
2. Compression applies in both training AND inference
3. Uses HistoryTokenCompressor from common.compressor
"""

from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from swift.llm.template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from swift.llm.template import register_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.base import to_device

from swiftvln.common import HistoryTokenCompressor

# Special tokens (must match dataset.py and model.py)
HISTORY_IMAGE_TOKEN = "<history_image>"
CURRENT_IMAGE_TOKEN = "<current_image>"

# Debug flag
DEBUG_COMPRESSION = False


class CompressVLNQwen25VLTemplate(Qwen2_5VLTemplate):
    """
    CompressVLN Template with configurable compression for history images.
    
    Features:
    - <history_image>: Configurable compression via pooling (default 4x with stride=2)
    - <current_image>: Standard compression (4:1) from Qwen2.5-VL
    - Compression applies in BOTH training and inference
    
    Args:
        compress_stride: Pooling stride for history image compression (default: 2)
    """
    
    # Token IDs will be set after processor initialization
    history_image_token_id: Optional[int] = None
    current_image_token_id: Optional[int] = None
    
    # Compression parameters (can be set externally)
    compress_stride: int = 2
    
    def __init__(self, *args, compress_stride: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_stride = compress_stride
        self.compressor = HistoryTokenCompressor(stride=compress_stride)
    
    def init_processor(self, processor) -> None:
        """Initialize processor and get custom special token IDs."""
        super().init_processor(processor)
        
        if processor is None:
            return
        
        # Get token IDs (tokens are added in model.py's get_model_tokenizer function)
        self.history_image_token_id = processor.tokenizer.convert_tokens_to_ids(HISTORY_IMAGE_TOKEN)
        self.current_image_token_id = processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        
        # Verify tokens exist
        if self.history_image_token_id != processor.tokenizer.unk_token_id:
            print(f"[CompressVLNTemplate] Using special tokens:")
            print(f"  - {HISTORY_IMAGE_TOKEN}: {self.history_image_token_id}")
            print(f"  - {CURRENT_IMAGE_TOKEN}: {self.current_image_token_id}")
            print(f"  - compress_stride: {self.compress_stride} ({self.compress_stride}x{self.compress_stride} pooling)")
            print(f"  - Standard image_token_id (<|image_pad|>): {self.image_token_id}")
    
    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override packing_row to raise an error if padding_free=true is used.
        
        CompressVLN uses custom image tokens which are incompatible with
        get_rope_index's expectation of standard <|image_pad|> tokens.
        """
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "[CompressVLNTemplate] ERROR: padding_free=true is NOT supported!\n\n"
            "CompressVLN uses custom tokens (<history_image>, <current_image>) which are\n"
            "incompatible with Qwen2.5-VL's get_rope_index function.\n\n"
            "Solution: Set padding_free=false in your training script.\n"
            "  - In shell script: PADDING_FREE=false\n"
            "  - In launch.json: \"--padding_free\", \"false\"\n"
            + "="*70
        )
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        Replace media tags with appropriate placeholders.
        
        For CompressVLN:
        - Dataset uses standard <image> tokens
        - Dataset provides num_history_images in extra_kwargs
        - This method converts <image> to <history_image> or <current_image> based on index
        
        History detection:
        - index < num_history_images → <history_image>
        - index >= num_history_images → <current_image>
        """
        from qwen_vl_utils import fetch_image
        
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            
            # Get num_history_images from dataset metadata
            num_history_images = inputs.extra_kwargs.get('num_history_images', 0)
            
            # Determine if this is a history or current image
            is_history = (index < num_history_images)
            
            if is_history:
                return ['<|vision_start|><history_image><|vision_end|>']
            else:
                return ['<|vision_start|><current_image><|vision_end|>']
        
        return super().replace_tag(media_type, index, inputs)
    
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        Encode inputs with different placeholder counts for history vs current images.
        
        History images: token_len = grid_thw.prod() // (merge_length * compress_stride^2)
        Current images: token_len = grid_thw.prod() // merge_length
        
        Supports two modes:
        1. Image mode: images are provided, pixel_values computed via image_processor
        2. Precomputed features mode: precomputed_features and precomputed_grid_thw are provided
        """
        from swift.llm.template.base import Template
        
        # Call grandparent's _encode to get basic encoding without image processing
        encoded = Template._encode(self, inputs)
        
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        
        images = inputs.images
        videos = inputs.videos
        
        # Check for precomputed features in extra_kwargs
        # These are passed with underscore prefix from dataset to go into extra_kwargs
        precomputed_features = inputs.extra_kwargs.get('_precomputed_features') if inputs.extra_kwargs else None
        precomputed_grid_thw = inputs.extra_kwargs.get('_precomputed_grid_thw') if inputs.extra_kwargs else None
        
        # Initialize shared variables
        image_grid_thw = None
        num_images = 0
        use_precomputed = False
        
        # Handle precomputed features mode
        if precomputed_features is not None and precomputed_grid_thw is not None:
            # Precomputed features mode: skip image_processor, use provided grid_thw
            import torch
            
            use_precomputed = True
            
            # Stack grid_thw into tensor
            image_grid_thw = torch.stack(precomputed_grid_thw, dim=0)
            
            # Store precomputed features for _post_encode
            encoded['_precomputed_features'] = precomputed_features
            encoded['_precomputed_grid_thw'] = image_grid_thw
            
            num_images = len(precomputed_features)
        
        # Process images with differentiated compression
        elif images:
            media_inputs = processor.image_processor(
                images=images, return_tensors='pt', do_resize=False
            )
            image_grid_thw = media_inputs['image_grid_thw']
            
            num_images = image_grid_thw.shape[0]
            encoded.update(media_inputs)
        
        # Process token expansion (shared logic for both modes)
        if image_grid_thw is not None and num_images > 0:
            merge_size = processor.image_processor.merge_size
            merge_length = merge_size ** 2
            
            # Find history and current image tokens
            history_idx_list = findall(input_ids, self.history_image_token_id)
            current_idx_list = findall(input_ids, self.current_image_token_id)
            
            num_history = len(history_idx_list)
            num_current = len(current_idx_list)
            num_images = image_grid_thw.shape[0]
            
            # Validate token count matches image count
            if num_history + num_current != num_images:
                print(f"[CompressVLN] WARNING: Token count mismatch! "
                      f"history_tokens={num_history}, current_tokens={num_current}, "
                      f"total_tokens={num_history + num_current}, actual_images={num_images}")
                # Adjust counts to match actual images
                if num_history > num_images:
                    num_history = num_images
                    num_current = 0
                elif num_history + num_current > num_images:
                    num_current = num_images - num_history
            
            # Store metadata for _post_encode
            encoded['_history_image_count'] = num_history
            encoded['_current_image_count'] = num_current
            encoded['_compress_stride'] = self.compress_stride
            
            # Process history images (higher compression)
            if history_idx_list and num_history > 0:
                # Only process the first num_history tokens
                history_to_process = history_idx_list[:num_history]
                
                def _get_history_tokens(i):
                    if i >= num_images:
                        return [self.history_image_token_id]  # Fallback
                    # Calculate token count with compression
                    t, h, w = image_grid_thw[i].tolist()
                    h_after_merge = int(h) // merge_size
                    w_after_merge = int(w) // merge_size
                    # Additional compression from pooling
                    h_after_pool = h_after_merge // self.compress_stride
                    w_after_pool = w_after_merge // self.compress_stride
                    token_len = int(t) * h_after_pool * w_after_pool
                    return [self.history_image_token_id] * max(1, token_len)
                
                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, history_to_process, _get_history_tokens
                )
            
            # Process current images (standard compression)
            if current_idx_list and num_current > 0:
                # Re-find indices after history expansion
                current_idx_list = findall(input_ids, self.current_image_token_id)
                # Only process the first num_current tokens
                current_to_process = current_idx_list[:num_current]
                
                def _get_current_tokens(i):
                    img_idx = num_history + i
                    if img_idx >= num_images:
                        return [self.current_image_token_id]  # Fallback
                    token_len = image_grid_thw[img_idx].prod() // merge_length
                    return [self.current_image_token_id] * int(token_len)
                
                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, current_to_process, _get_current_tokens
                )
        
        # Process videos (unchanged from parent)
        if videos:
            kwargs = {}
            if hasattr(processor, 'video_processor'):
                processor_func = processor.video_processor
            else:
                processor_func = processor.image_processor
                kwargs['images'] = None
            media_inputs = processor_func(
                videos=videos, return_tensors='pt', do_resize=False, **kwargs
            )
            video_grid_thw = media_inputs['video_grid_thw']
            merge_length = processor.image_processor.merge_size ** 2
            
            idx_list = findall(input_ids, self.video_token_id)
            
            if self.version == 'v2_5':
                fps = inputs.mm_processor_kwargs.get('fps', [])
                if fps:
                    media_inputs['second_per_grid_ts'] = [
                        processor.image_processor.temporal_patch_size / tmp for tmp in fps
                    ]
            
            def _get_video_tokens(i):
                token_len = video_grid_thw[i].prod() // merge_length
                return [self.video_token_id] * int(token_len)
            
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_video_tokens
            )
            
            encoded.update(media_inputs)
        
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        
        return encoded
    
    def _pool_embeddings_2d(self, embeddings: torch.Tensor, grid_thw: torch.Tensor, 
                           pool_size: int = 2) -> torch.Tensor:
        """
        Apply 2D average pooling to image embeddings.
        
        Args:
            embeddings: [num_tokens, hidden_size] - flattened image embeddings (after spatial merge)
            grid_thw: [3] tensor with [temporal, height, width] grid dimensions (after spatial merge)
            pool_size: pooling kernel size (e.g., 2 for 2x2 pooling)
        
        Returns:
            Pooled embeddings with reduced spatial dimensions
        """
        if isinstance(grid_thw, torch.Tensor):
            t, h, w = grid_thw.tolist()
        else:
            t, h, w = grid_thw
        
        t, h, w = int(t), int(h), int(w)
        hidden_size = embeddings.shape[-1]
        
        # Reshape to [t, h, w, hidden_size]
        embeddings = embeddings.view(t, h, w, hidden_size)
        
        # Permute to [t, hidden_size, h, w] for pooling
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
        
        # Apply 2D average pooling
        pooled = F.avg_pool2d(embeddings.float(), kernel_size=pool_size, stride=pool_size)
        pooled = pooled.to(embeddings.dtype)
        
        # Permute back to [t, h', w', hidden_size] and flatten
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        pooled = pooled.view(-1, hidden_size)
        
        return pooled
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process embeddings with differentiated handling for history vs current images.
        
        NOTE: Unlike Navid, compression applies in BOTH training and inference.
        
        History images: Apply additional pooling after visual encoding
        Current images: Use standard visual embeddings
        
        Supports precomputed features mode: skips model.visual() and uses cached features.
        """
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        base_model = self.get_base_model(model)
        
        # Get text embeddings
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        
        # Get metadata (may be lists after collation for batch processing)
        num_history_raw = inputs.pop('_history_image_count', 0)
        num_current_raw = inputs.pop('_current_image_count', 0)
        compress_stride_raw = inputs.pop('_compress_stride', self.compress_stride)
        
        # Check for precomputed features
        precomputed_features = inputs.pop('_precomputed_features', None)
        precomputed_grid_thw = inputs.pop('_precomputed_grid_thw', None)
        
        def _to_list(value):
            """Convert value to a flat list of integers."""
            if isinstance(value, int):
                return [value]
            if isinstance(value, list):
                result = []
                for v in value:
                    result.extend(_to_list(v))
                return result
            return [0]
        
        def _get_first(value):
            """Get first non-list value from potentially nested structure."""
            if isinstance(value, list):
                return _get_first(value[0]) if value else self.compress_stride
            return value
        
        # Get per-sample counts (important for batch_size > 1)
        history_counts = _to_list(num_history_raw)  # [sample1_history, sample2_history, ...]
        current_counts = _to_list(num_current_raw)  # [sample1_current, sample2_current, ...]
        compress_stride = _get_first(compress_stride_raw)
        
        # Ensure same number of samples
        num_samples = max(len(history_counts), len(current_counts))
        while len(history_counts) < num_samples:
            history_counts.append(0)
        while len(current_counts) < num_samples:
            current_counts.append(0)
        
        # Calculate totals for validation
        total_history = sum(history_counts)
        total_current = sum(current_counts)
        
        # Validate counts against actual image count
        if image_grid_thw is not None:
            num_images = image_grid_thw.shape[0]
            if total_history + total_current != num_images:
                if DEBUG_COMPRESSION:
                    print(f"[CompressVLN] WARNING: Image count mismatch! "
                          f"history={total_history}, current={total_current}, "
                          f"total={total_history + total_current}, actual={num_images}")
        
        if precomputed_features is not None:
            # Precomputed features mode: skip ViT, use features directly
            # Concatenate all features and move to device
            all_image_embeds = torch.cat([
                f.to(inputs_embeds.device, inputs_embeds.dtype) for f in precomputed_features
            ], dim=0)
            image_grid_thw = precomputed_grid_thw.to(inputs_embeds.device)
            
            if DEBUG_COMPRESSION:
                print(f"[CompressVLN] Using precomputed features: {all_image_embeds.shape}")
        
        elif pixel_values is None:
            # No images, handle training stability
            from PIL import Image
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(model.visual.dtype)
            image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            return {'inputs_embeds': inputs_embeds}
        
        else:
            # Original mode: Get all image embeddings from visual encoder
            dtype = model.visual.dtype
            pixel_values = pixel_values.type(dtype)
            all_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        
        merge_size = self.processor.image_processor.merge_size
        merge_length = merge_size ** 2
        
        if DEBUG_COMPRESSION:
            print("\n" + "="*70)
            print("[DEBUG] CompressVLN _post_encode: History Image Compression")
            print(f"  Batch size: {num_samples}")
            print(f"  Per-sample history counts: {history_counts}")
            print(f"  Per-sample current counts: {current_counts}")
            print(f"  Total images: {total_history + total_current} (history: {total_history}, current: {total_current})")
            print(f"  compress_stride: {compress_stride} ({compress_stride}x{compress_stride} pooling)")
            print(f"  merge_size: {merge_size} (Qwen2.5-VL visual encoder merge)")
            print(f"  all_image_embeds shape: {all_image_embeds.shape}")
        
        # Split embeddings by image, processing per-sample to maintain correct order
        # Image order in image_grid_thw: [sample1_all_images, sample2_all_images, ...]
        # Within each sample: [history_images, current_images]
        embed_idx = 0
        img_idx = 0
        history_embeds_list = []
        current_embeds_list = []
        
        total_history_tokens_before = 0
        total_history_tokens_after = 0
        total_current_tokens = 0
        
        for sample_idx in range(num_samples):
            sample_num_history = history_counts[sample_idx]
            sample_num_current = current_counts[sample_idx]
            
            if DEBUG_COMPRESSION:
                print(f"\n  Sample {sample_idx}: {sample_num_history} history + {sample_num_current} current")
            
            # Process history images for this sample (apply compression)
            for h_idx in range(sample_num_history):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                # Apply additional pooling for history images
                grid_after_merge = image_grid_thw[img_idx].clone()
                grid_after_merge[1] = grid_after_merge[1] // merge_size
                grid_after_merge[2] = grid_after_merge[2] // merge_size
                
                pooled_embeds = self._pool_embeddings_2d(
                    img_embeds, 
                    grid_after_merge,
                    pool_size=compress_stride
                )
                history_embeds_list.append(pooled_embeds)
                
                if DEBUG_COMPRESSION:
                    total_history_tokens_before += num_tokens
                    total_history_tokens_after += pooled_embeds.shape[0]
                    print(f"    History[{h_idx}] img_idx={img_idx}: {num_tokens} -> {pooled_embeds.shape[0]} tokens")
                
                img_idx += 1
            
            # Process current images for this sample (no compression)
            for c_idx in range(sample_num_current):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                current_embeds_list.append(img_embeds)
                
                if DEBUG_COMPRESSION:
                    total_current_tokens += num_tokens
                    print(f"    Current[{c_idx}] img_idx={img_idx}: {num_tokens} tokens (no compression)")
                
                img_idx += 1
        
        if DEBUG_COMPRESSION:
            print("\n" + "-"*70)
            print("  SUMMARY:")
            if total_history > 0:
                print(f"    History: {total_history_tokens_before} -> {total_history_tokens_after} tokens")
            print(f"    Current: {total_current_tokens} tokens")
            print("="*70 + "\n")
        
        # Replace history image tokens
        if history_embeds_list:
            history_embeds = torch.cat(history_embeds_list, dim=0)
            history_mask = (input_ids == self.history_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            history_embeds = history_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(history_mask, history_embeds)
        
        # Replace current image tokens
        if current_embeds_list:
            current_embeds = torch.cat(current_embeds_list, dim=0)
            current_mask = (input_ids == self.current_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            current_embeds = current_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(current_mask, current_embeds)
        
        result = {'inputs_embeds': inputs_embeds}
        if labels is not None:
            result['labels'] = labels
        return result
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate multimodal data including custom metadata and precomputed features."""
        res = super()._data_collator_mm_data(batch)
        
        # Collect custom metadata into lists
        for key in ['_history_image_count', '_current_image_count', '_compress_stride']:
            values = []
            for b in batch:
                val = b.pop(key, None)
                if val is not None:
                    values.append(val)
            if values:
                res[key] = values
        
        # Handle precomputed features (concatenate feature lists)
        precomputed_features = []
        precomputed_grid_thw = []
        for b in batch:
            feat = b.pop('_precomputed_features', None)
            grid = b.pop('_precomputed_grid_thw', None)
            if feat is not None:
                precomputed_features.extend(feat)
            if grid is not None:
                precomputed_grid_thw.append(grid)
        
        if precomputed_features:
            res['_precomputed_features'] = precomputed_features
            # Stack grid_thw tensors from all samples
            res['_precomputed_grid_thw'] = torch.cat(precomputed_grid_thw, dim=0)
        
        return res


# Register the custom template
register_template(
    QwenTemplateMeta(
        'compressvln_qwen2_5_vl',
        template_cls=CompressVLNQwen25VLTemplate,
    )
)

print("[CompressVLNTemplate] Template 'compressvln_qwen2_5_vl' registered successfully!")
