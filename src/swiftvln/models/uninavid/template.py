# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Template with three-level image compression (following Uni-NaVid architecture).

This template extends Qwen2.5-VL template to support three-level memory compression:
- Long-term memory (<long_term_image>): Pooling + mean + similarity-based merging
- Short-term memory (<short_term_image>): Pooling compression
- Current observation (<current_image>): No compression (full resolution)

Key features:
1. Three-level memory architecture following Uni-NaVid
2. Similarity-based merging for long-term memory (threshold configurable)
3. Dynamic sequence trimming in _post_encode (Plan B: N placeholders → M merged tokens)
4. Configurable short_term_frames threshold and similarity_threshold
5. Compression applies in both training AND inference
"""

import os
import sys
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from swift.llm.template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from swift.llm.template import register_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.base import to_device

# Special tokens (must match dataset.py and model.py)
LONG_TERM_IMAGE_TOKEN = "<long_term_image>"
SHORT_TERM_IMAGE_TOKEN = "<short_term_image>"
CURRENT_IMAGE_TOKEN = "<current_image>"

# Debug flag
DEBUG_COMPRESSION = False


class UniNaVidQwen25VLTemplate(Qwen2_5VLTemplate):
    """
    UniNaVid Template with three-level memory compression.
    
    Features:
    - <long_term_image>: High compression (pooling + mean + similarity merge)
    - <short_term_image>: Medium compression (pooling only)
    - <current_image>: No compression (full resolution)
    - Dynamic sequence trimming: N long-term placeholders → M merged tokens
    
    Args:
        compress_stride: Pooling stride for short-term image compression (default: 2)
        short_term_frames: Number of recent frames as short-term memory (default: 64)
        similarity_threshold: Cosine similarity threshold for merging (default: 0.985)
    """
    
    # Token IDs will be set after processor initialization
    long_term_image_token_id: Optional[int] = None
    short_term_image_token_id: Optional[int] = None
    current_image_token_id: Optional[int] = None
    
    # Compression parameters (can be set externally)
    compress_stride: int = 2
    short_term_frames: int = 64
    similarity_threshold: float = 0.985
    
    # Eval mode flag (when True, uses incremental processing in evaluator)
    eval_mode: bool = False
    
    def __init__(self, *args, 
                 compress_stride: int = 2, 
                 short_term_frames: int = 64,
                 similarity_threshold: float = 0.985,
                 eval_mode: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_stride = compress_stride
        self.short_term_frames = short_term_frames
        self.similarity_threshold = similarity_threshold
        self.eval_mode = eval_mode
    
    def init_processor(self, processor) -> None:
        """Initialize processor and get custom special token IDs."""
        super().init_processor(processor)
        
        if processor is None:
            return
        
        # Get token IDs (tokens are added in model.py's get_model_tokenizer function)
        self.long_term_image_token_id = processor.tokenizer.convert_tokens_to_ids(LONG_TERM_IMAGE_TOKEN)
        self.short_term_image_token_id = processor.tokenizer.convert_tokens_to_ids(SHORT_TERM_IMAGE_TOKEN)
        self.current_image_token_id = processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        
        # Verify tokens exist
        if self.long_term_image_token_id != processor.tokenizer.unk_token_id:
            print(f"[UniNaVidTemplate] Using special tokens:")
            print(f"  - {LONG_TERM_IMAGE_TOKEN}: {self.long_term_image_token_id}")
            print(f"  - {SHORT_TERM_IMAGE_TOKEN}: {self.short_term_image_token_id}")
            print(f"  - {CURRENT_IMAGE_TOKEN}: {self.current_image_token_id}")
            print(f"  - compress_stride: {self.compress_stride} ({self.compress_stride}x{self.compress_stride} pooling)")
            print(f"  - short_term_frames: {self.short_term_frames}")
            print(f"  - similarity_threshold: {self.similarity_threshold}")
            print(f"  - Standard image_token_id (<|image_pad|>): {self.image_token_id}")
    
    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override packing_row to raise an error if padding_free=true is used.
        
        UniNaVid uses custom image tokens which are incompatible with
        get_rope_index's expectation of standard <|image_pad|> tokens.
        """
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "[UniNaVidTemplate] ERROR: padding_free=true is NOT supported!\n\n"
            "UniNaVid uses custom tokens (<long_term_image>, <short_term_image>,\n"
            "<current_image>) which are incompatible with Qwen2.5-VL's\n"
            "get_rope_index function.\n\n"
            "Solution: Set padding_free=false in your training script.\n"
            "  - In shell script: PADDING_FREE=false\n"
            "  - In launch.json: \"--padding_free\", \"false\"\n"
            + "="*70
        )
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        Replace media tags with appropriate placeholders.
        
        For UniNaVid:
        - Image sequence: [history_0, ..., history_N-2, current_N-1]
        - Last image is always current_image
        - Classification based on short_term_frames parameter:
          - If history_count <= short_term_frames: all history is short-term
          - Otherwise: older frames are long-term, recent short_term_frames are short-term
        
        Classification by index:
        - index < num_long_term → <long_term_image>
        - index < num_long_term + num_short_term → <short_term_image>
        - index == total - 1 → <current_image>
        
        Note: In precomputed features mode, images may be None placeholders.
              fetch_image is skipped in that case.
        """
        from qwen_vl_utils import fetch_image
        
        if media_type == 'image':
            # fetch_image handles PIL Image, path, or URL
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            
            # Calculate classification based on short_term_frames
            total_images = len(inputs.images)
            history_count = total_images - 1  # Last one is current
            
            if history_count <= self.short_term_frames:
                # All history fits in short-term memory
                num_long_term = 0
                num_short_term = history_count
            else:
                # Split into long-term and short-term
                num_long_term = history_count - self.short_term_frames
                num_short_term = self.short_term_frames
            
            # Classify based on index
            if index < num_long_term:
                return ['<|vision_start|><long_term_image><|vision_end|>']
            elif index < num_long_term + num_short_term:
                return ['<|vision_start|><short_term_image><|vision_end|>']
            else:
                return ['<|vision_start|><current_image><|vision_end|>']
        
        return super().replace_tag(media_type, index, inputs)
    
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        Encode inputs with different placeholder counts for three image types.
        
        Long-term images: 1 token per frame (will be merged in _post_encode)
        Short-term images: token_len = grid_thw.prod() // (merge_length * compress_stride^2)
        Current images: token_len = grid_thw.prod() // merge_length (no extra compression)
        
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
        
        # Process images with three-level compression
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
            
            # Find all three types of image tokens
            long_term_idx_list = findall(input_ids, self.long_term_image_token_id)
            short_term_idx_list = findall(input_ids, self.short_term_image_token_id)
            current_idx_list = findall(input_ids, self.current_image_token_id)
            
            num_long_term = len(long_term_idx_list)
            num_short_term = len(short_term_idx_list)
            num_current = len(current_idx_list)
            num_images = image_grid_thw.shape[0]
            
            # Validate token count matches image count
            total_tokens = num_long_term + num_short_term + num_current
            if total_tokens != num_images:
                print(f"[UniNaVid] WARNING: Token count mismatch! "
                      f"long_term={num_long_term}, short_term={num_short_term}, "
                      f"current={num_current}, total={total_tokens}, actual_images={num_images}")
            
            # Store metadata for _post_encode
            encoded['_long_term_image_count'] = num_long_term
            encoded['_short_term_image_count'] = num_short_term
            encoded['_current_image_count'] = num_current
            encoded['_compress_stride'] = self.compress_stride
            encoded['_similarity_threshold'] = self.similarity_threshold
            
            # Process long-term images: 1 placeholder per frame (will be merged later)
            if long_term_idx_list and num_long_term > 0:
                long_term_to_process = long_term_idx_list[:num_long_term]
                
                def _get_long_term_tokens(i):
                    # Each long-term frame gets exactly 1 placeholder
                    # Actual embedding count (M) will be determined after merging
                    return [self.long_term_image_token_id] * 1
                
                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, long_term_to_process, _get_long_term_tokens
                )
            
            # Process short-term images (pooling compression)
            if short_term_idx_list and num_short_term > 0:
                # Re-find indices after long-term expansion
                short_term_idx_list = findall(input_ids, self.short_term_image_token_id)
                short_term_to_process = short_term_idx_list[:num_short_term]
                
                def _get_short_term_tokens(i):
                    img_idx = num_long_term + i
                    if img_idx >= num_images:
                        return [self.short_term_image_token_id]
                    # Calculate token count with compression
                    t, h, w = image_grid_thw[img_idx].tolist()
                    h_after_merge = int(h) // merge_size
                    w_after_merge = int(w) // merge_size
                    # Additional compression from pooling
                    h_after_pool = h_after_merge // self.compress_stride
                    w_after_pool = w_after_merge // self.compress_stride
                    token_len = int(t) * h_after_pool * w_after_pool
                    return [self.short_term_image_token_id] * max(1, token_len)
                
                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, short_term_to_process, _get_short_term_tokens
                )
            
            # Process current images (no extra compression)
            if current_idx_list and num_current > 0:
                # Re-find indices after previous expansions
                current_idx_list = findall(input_ids, self.current_image_token_id)
                current_to_process = current_idx_list[:num_current]
                
                def _get_current_tokens(i):
                    img_idx = num_long_term + num_short_term + i
                    if img_idx >= num_images:
                        return [self.current_image_token_id]
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
    
    def _merge_long_term_memory(self, embeddings_list: List[torch.Tensor], 
                                 threshold: float = 0.985) -> torch.Tensor:
        """
        Merge similar long-term memory frames using cosine similarity (Uni-NaVid style).
        
        Each frame's embedding is first averaged to get a single vector, then
        consecutive similar frames are merged using weighted averaging.
        
        Args:
            embeddings_list: List of [num_tokens, hidden_size] tensors, one per long-term frame
            threshold: Cosine similarity threshold for merging (default: 0.985)
        
        Returns:
            Merged embeddings [M, hidden_size] where M <= len(embeddings_list)
        """
        if not embeddings_list:
            return torch.tensor([])
        
        if len(embeddings_list) == 1:
            # Single frame: just take mean
            return embeddings_list[0].mean(dim=0, keepdim=True)
        
        # Convert each frame's embeddings to a single vector (mean pooling)
        frame_vectors = [emb.mean(dim=0, keepdim=True) for emb in embeddings_list]
        
        # Merge similar consecutive frames
        merged = [frame_vectors[0]]
        weights = [1]
        
        for i in range(1, len(frame_vectors)):
            current = frame_vectors[i]
            
            # Compute cosine similarity with last merged vector
            sim = F.cosine_similarity(merged[-1], current, dim=-1).mean().item()
            
            if sim > threshold:
                # Merge: weighted average
                new_weight = weights[-1] + 1
                merged[-1] = (merged[-1] * weights[-1] + current) / new_weight
                weights[-1] = new_weight
            else:
                # Don't merge: add as new entry
                merged.append(current)
                weights.append(1)
        
        # Concatenate all merged vectors
        return torch.cat(merged, dim=0)  # [M, hidden_size]
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process embeddings with three-level handling and dynamic sequence trimming.
        
        This is the core of UniNaVid's memory architecture:
        1. Long-term: pooling + mean + similarity merge → M tokens (from N placeholders)
        2. Short-term: pooling compression
        3. Current: no extra compression
        
        Key: Dynamic trimming removes N-M excess long-term placeholders.
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
        num_long_term_raw = inputs.pop('_long_term_image_count', 0)
        num_short_term_raw = inputs.pop('_short_term_image_count', 0)
        num_current_raw = inputs.pop('_current_image_count', 0)
        compress_stride_raw = inputs.pop('_compress_stride', self.compress_stride)
        similarity_threshold_raw = inputs.pop('_similarity_threshold', self.similarity_threshold)
        
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
        long_term_counts = _to_list(num_long_term_raw)
        short_term_counts = _to_list(num_short_term_raw)
        current_counts = _to_list(num_current_raw)
        compress_stride = _get_first(compress_stride_raw)
        similarity_threshold = _get_first(similarity_threshold_raw)
        
        # Ensure same number of samples
        num_samples = max(len(long_term_counts), len(short_term_counts), len(current_counts))
        while len(long_term_counts) < num_samples:
            long_term_counts.append(0)
        while len(short_term_counts) < num_samples:
            short_term_counts.append(0)
        while len(current_counts) < num_samples:
            current_counts.append(0)
        
        # Calculate totals
        total_long_term = sum(long_term_counts)
        total_short_term = sum(short_term_counts)
        total_current = sum(current_counts)
        
        # Check for precomputed features
        precomputed_features = inputs.pop('_precomputed_features', None)
        precomputed_grid_thw = inputs.pop('_precomputed_grid_thw', None)
        
        if precomputed_features is not None:
            # Precomputed features mode: skip ViT, use features directly
            # Concatenate all features and move to device
            all_image_embeds = torch.cat([
                f.to(inputs_embeds.device, inputs_embeds.dtype) for f in precomputed_features
            ], dim=0)
            image_grid_thw = precomputed_grid_thw.to(inputs_embeds.device)
            
            if DEBUG_COMPRESSION:
                print(f"[UniNaVid] Using precomputed features: {all_image_embeds.shape}")
        
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
            print("[DEBUG] UniNaVid _post_encode: Three-level compression")
            print(f"  Batch size: {num_samples}")
            print(f"  Long-term counts: {long_term_counts} (total: {total_long_term})")
            print(f"  Short-term counts: {short_term_counts} (total: {total_short_term})")
            print(f"  Current counts: {current_counts} (total: {total_current})")
            print(f"  compress_stride: {compress_stride}")
            print(f"  similarity_threshold: {similarity_threshold}")
        
        # Process embeddings per sample
        embed_idx = 0
        img_idx = 0
        long_term_embeds_per_sample = []  # List of merged embeddings per sample
        short_term_embeds_list = []
        current_embeds_list = []
        
        for sample_idx in range(num_samples):
            sample_num_long_term = long_term_counts[sample_idx]
            sample_num_short_term = short_term_counts[sample_idx]
            sample_num_current = current_counts[sample_idx]
            
            # Process long-term images: pooling + collect for merging
            sample_long_term_embeds = []
            for lt_idx in range(sample_num_long_term):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                # Apply pooling
                grid_after_merge = image_grid_thw[img_idx].clone()
                grid_after_merge[1] = grid_after_merge[1] // merge_size
                grid_after_merge[2] = grid_after_merge[2] // merge_size
                
                pooled_embeds = self._pool_embeddings_2d(
                    img_embeds, grid_after_merge, pool_size=compress_stride
                )
                sample_long_term_embeds.append(pooled_embeds)
                img_idx += 1
            
            # Merge long-term embeddings for this sample
            if sample_long_term_embeds:
                merged = self._merge_long_term_memory(sample_long_term_embeds, similarity_threshold)
                long_term_embeds_per_sample.append(merged)
            else:
                long_term_embeds_per_sample.append(None)
            
            # Process short-term images (pooling only)
            for st_idx in range(sample_num_short_term):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                # Apply pooling
                grid_after_merge = image_grid_thw[img_idx].clone()
                grid_after_merge[1] = grid_after_merge[1] // merge_size
                grid_after_merge[2] = grid_after_merge[2] // merge_size
                
                pooled_embeds = self._pool_embeddings_2d(
                    img_embeds, grid_after_merge, pool_size=compress_stride
                )
                short_term_embeds_list.append(pooled_embeds)
                img_idx += 1
            
            # Process current images (no extra compression)
            for c_idx in range(sample_num_current):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                current_embeds_list.append(img_embeds)
                img_idx += 1
        
        if DEBUG_COMPRESSION:
            for i, merged in enumerate(long_term_embeds_per_sample):
                if merged is not None:
                    print(f"  Sample {i}: {long_term_counts[i]} long-term frames -> {merged.shape[0]} merged tokens")
        
        # Now perform dynamic trimming for long-term placeholders
        # For each sample, we have N placeholders but only M merged tokens
        
        # Find all long-term placeholder positions
        lt_mask = (input_ids == self.long_term_image_token_id)
        
        # Calculate how many positions to remove
        total_placeholders = lt_mask.sum().item()
        total_merged = sum(m.shape[0] if m is not None else 0 for m in long_term_embeds_per_sample)
        num_to_remove = total_placeholders - total_merged
        
        if DEBUG_COMPRESSION:
            print(f"  Long-term: {total_placeholders} placeholders -> {total_merged} merged tokens")
            print(f"  Removing {num_to_remove} excess placeholders")
        
        if num_to_remove > 0:
            # Get positions to remove (remove the first N-M placeholders)
            lt_positions = lt_mask.nonzero(as_tuple=True)
            batch_indices = lt_positions[0]
            seq_indices = lt_positions[1]
            
            # Create keep mask
            keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            
            # Mark positions to remove (first num_to_remove long-term placeholders)
            positions_to_remove = seq_indices[:num_to_remove]
            keep_mask[positions_to_remove] = False
            
            # Trim inputs_embeds and labels
            inputs_embeds = inputs_embeds[:, keep_mask, :]
            if labels is not None:
                labels = labels[:, keep_mask]
            
            # Update input_ids for mask operations below
            input_ids = input_ids[:, keep_mask]
        
        # Now fill in the embeddings
        # Long-term: concatenate all merged embeddings
        if long_term_embeds_per_sample and any(m is not None for m in long_term_embeds_per_sample):
            all_long_term = torch.cat([m for m in long_term_embeds_per_sample if m is not None], dim=0)
            lt_mask_new = (input_ids == self.long_term_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            all_long_term = all_long_term.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(lt_mask_new, all_long_term)
        
        # Short-term: fill in pooled embeddings
        if short_term_embeds_list:
            all_short_term = torch.cat(short_term_embeds_list, dim=0)
            st_mask = (input_ids == self.short_term_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            all_short_term = all_short_term.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(st_mask, all_short_term)
        
        # Current: fill in full resolution embeddings
        if current_embeds_list:
            all_current = torch.cat(current_embeds_list, dim=0)
            cur_mask = (input_ids == self.current_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            all_current = all_current.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(cur_mask, all_current)
        
        if DEBUG_COMPRESSION:
            print(f"  Final inputs_embeds shape: {inputs_embeds.shape}")
            print("="*70 + "\n")
        
        result = {'inputs_embeds': inputs_embeds}
        if labels is not None:
            result['labels'] = labels
        return result
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate multimodal data including custom metadata and precomputed features."""
        res = super()._data_collator_mm_data(batch)
        
        # Collect custom metadata into lists
        for key in ['_long_term_image_count', '_short_term_image_count', '_current_image_count',
                    '_compress_stride', '_similarity_threshold']:
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
        'uninavid_qwen2_5_vl',
        template_cls=UniNaVidQwen25VLTemplate,
    )
)

print("[UniNaVidTemplate] Template 'uninavid_qwen2_5_vl' registered successfully!")
