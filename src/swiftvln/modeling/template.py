# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SwiftVLN Template with pluggable history processing.

This template extends Qwen2.5-VL to support different history information
processing strategies via the HistoryProcessor interface:
- <history_memory>: Unified token for all history frames (processed by HistoryProcessor)
- <current_image>: Standard Qwen2.5-VL compression (4:1)

Supported history processors:
- 'per_frame': Per-frame compression (pooling or ToMe) - default
- 'gtc': Global Token Clustering (cross-frame clustering)
"""

import os
from typing import Any, Dict, List, Literal, Optional

import torch

from swift.template import Template, register_template
from swift.template.base import to_device
from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.qwen import (
    Qwen2_5VLTemplate,
    Qwen3VLTemplate as SwiftQwen3VLTemplate,
    QwenTemplateMeta,
)
from swift.template.utils import Context, findall

from swiftvln.modeling.constants import CURRENT_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN
from swiftvln.modeling.history import (
    HistoryProcessor,
    create_history_processor,
    PerFrameCompressor,
    GlobalTokenClustering,
)

def _is_rank0() -> bool:
    raw = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    try:
        return int(raw) == 0
    except ValueError:
        return True


def _print_rank0(message: str) -> None:
    if _is_rank0():
        print(message)


class SwiftVLNTemplateMixin:
    """
    SwiftVLN Template with pluggable history processing.
    
    Features:
    - <history_memory>: Unified memory token for all history frames
    - <current_image>: Standard compression (4:1) from Qwen2.5-VL
    - Pluggable HistoryProcessor for different compression strategies
    
    Args:
        history_processor_type: 'per_frame' or 'gtc' (default: 'per_frame')
        
        Per-frame options:
        - num_history: Number of frames to sample (default: 8)
        - log_base: Sampling distribution, 1.0=uniform, >1.0=logarithmic (default: 1.0)
        - compress_stride: Compression stride (default: 2)
        - use_tome: Use ToMe instead of pooling (default: False)
        
        GTC options:
        - gtc_output_tokens: Fixed output tokens (default: 512)
        - gtc_temperature: Soft assignment temperature (default: 0.1)
        - gtc_num_iterations: K-means iterations (default: 1)
    """
    
    # Token IDs (set in init_processor)
    history_memory_token_id: Optional[int] = None
    current_image_token_id: Optional[int] = None
    
    # History processor
    history_processor: Optional[HistoryProcessor] = None
    
    def __init__(
        self, 
        *args, 
        history_processor_type: str = 'per_frame',
        # Per-frame options
        num_history: int = 8,
        log_base: float = 1.0,
        compress_stride: int = 2, 
        use_tome: bool = False,
        # GTC options
        gtc_output_tokens: int = 512,
        gtc_temperature: float = 0.1,
        gtc_num_iterations: int = 1,
        **kwargs
    ):
        # Store config for serialization/debugging (must be before super().__init__)
        self.history_processor_type = history_processor_type
        self.num_history = num_history
        self.log_base = log_base
        self.compress_stride = compress_stride
        self.use_tome = use_tome
        self.gtc_output_tokens = gtc_output_tokens
        self.gtc_temperature = gtc_temperature
        self.gtc_num_iterations = gtc_num_iterations
        
        # Create history processor BEFORE super().__init__()
        # because super().__init__() calls init_processor() which uses history_processor
        self.history_processor = create_history_processor(
            processor_type=history_processor_type,
            compress_stride=compress_stride,
            compress_method='tome' if use_tome else 'pooling',
            num_history=num_history,
            log_base=log_base,
            output_tokens=gtc_output_tokens,
            temperature=gtc_temperature,
            num_iterations=gtc_num_iterations,
        )
        
        super().__init__(*args, **kwargs)
    
    def init_processor(self, processor) -> None:
        """Initialize processor and get custom special token IDs."""
        super().init_processor(processor)
        
        if processor is None:
            return
        
        # Get token IDs (tokens are added in model.py's get_model_tokenizer function)
        self.history_memory_token_id = processor.tokenizer.convert_tokens_to_ids(HISTORY_MEMORY_TOKEN)
        self.current_image_token_id = processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        
        # Verify tokens exist and print a concise rank-0 configuration summary.
        if self.history_memory_token_id != processor.tokenizer.unk_token_id:
            _print_rank0("[SwiftVLNTemplate] Using special tokens (unified memory mode):")
            _print_rank0(f"  - {HISTORY_MEMORY_TOKEN}: {self.history_memory_token_id} (unified)")
            _print_rank0(f"  - {CURRENT_IMAGE_TOKEN}: {self.current_image_token_id}")
            _print_rank0(f"  - Standard image_token_id (<|image_pad|>): {self.image_token_id}")
            _print_rank0(f"  - History Processor: {self.history_processor.name}")
            
            # Print processor-specific info
            if isinstance(self.history_processor, PerFrameCompressor):
                method = "tome" if self.use_tome else "pool"
                _print_rank0(f"    - h={self.num_history}, b={self.log_base}, {method}, s={self.compress_stride}")
            elif isinstance(self.history_processor, GlobalTokenClustering):
                _print_rank0(f"    - output_tokens={self.gtc_output_tokens}, temperature={self.gtc_temperature}")
    
    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override packing_row to raise an error if padding_free=true is used.
        
        SwiftVLN uses custom image tokens which are incompatible with
        get_rope_index's expectation of standard <|image_pad|> tokens.
        """
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "[SwiftVLNTemplate] ERROR: padding_free=true is NOT supported!\n\n"
            "SwiftVLN uses custom tokens (<history_memory>, <current_image>) which are\n"
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
        
        Design:
        - History images: No <image> tags in prompt, they are represented by unified <history_memory> block
        - Current images: Use standard <image> tags, converted to <current_image> here for ROPE support
        
        Image index mapping (images list order):
        - index 0 ~ num_history_images-1: history images (no <image> tag, processed in _post_encode)
        - index >= num_history_images: current images (from <image> tags)
        
        This ensures:
        1. History: Unified semantic memory block (no ROPE needed for compressed features)
        2. Current: Standard ROPE position encoding for spatial understanding
        """
        from qwen_vl_utils import fetch_image
        
        if media_type == 'image':
            # Process image for visual encoder (required for all images)
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            
            # Get num_history_images from dataset metadata
            num_history_images = inputs.extra_kwargs.get('num_history_images', 0)
            
            # Determine if this is a history or current image
            is_history = (index < num_history_images)
            
            if is_history:
                # History images: no <image> tag in prompt for them
                # They are processed via the unified <history_memory> token in system prompt
                # Return empty - this shouldn't be called for history images in normal flow
                return []
            else:
                # Current images (including initial): convert <image> to our custom token with vision wrapper.
                # This enables ROPE position encoding while using our custom token.
                return [f'<|vision_start|>{CURRENT_IMAGE_TOKEN}<|vision_end|>']
        
        return super().replace_tag(media_type, index, inputs)
    
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        Encode inputs with unified memory token for history and per-image tokens for current.
        
        Unified Memory Mode:
        - <history_memory>: Single token expanded to total compressed tokens for ALL history images
        - <current_image>: Per-image tokens (standard compression)
        
        Token counts:
        - History: total_tokens = sum(compressed_tokens for each history image)
        - Current: token_len = grid_thw.prod() // merge_length
        """
        # Call grandparent's _encode to get basic encoding without image processing
        encoded = Template._encode(self, inputs)
        
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        
        images = inputs.images
        videos = inputs.videos
        
        # Get num_history_images from dataset metadata
        num_history_images = inputs.extra_kwargs.get('num_history_images', 0)
        num_initial_images = inputs.extra_kwargs.get('num_initial_images', 0)
        frame_poses = inputs.extra_kwargs.get('frame_poses', None)
        
        # Initialize shared variables
        image_grid_thw = None
        num_images = 0
        
        # Process images with differentiated compression
        if images:
            # Let Qwen processors resize Habitat frames to patch/merge-aligned sizes.
            media_inputs = processor.image_processor(
                images=images, return_tensors='pt'
            )
            image_grid_thw = media_inputs['image_grid_thw']
            
            num_images = image_grid_thw.shape[0]
            encoded.update(media_inputs)
        
        # Process token expansion with unified memory mode
        if image_grid_thw is not None and num_images > 0:
            merge_size = processor.image_processor.merge_size
            merge_length = merge_size ** 2
            
            # Find unified history memory token and current image tokens
            history_memory_idx_list = findall(input_ids, self.history_memory_token_id)
            current_idx_list = findall(input_ids, self.current_image_token_id)
            
            # Calculate actual counts
            num_history = min(num_history_images, num_images)
            num_current = len(current_idx_list)
            
            # Validate
            if num_history + num_current != num_images:
                print(f"[SwiftVLN] WARNING: Image count mismatch! "
                      f"num_history_images={num_history_images}, num_initial_images={num_initial_images}, "
                      f"current_tokens={num_current}, actual_images={num_images}")
                # Adjust counts
                if num_history > num_images:
                    num_history = num_images
                    num_current = 0
                elif num_history + num_current > num_images:
                    num_current = num_images - num_history
            
            # Store metadata for _post_encode (only essential counts)
            encoded['_history_image_count'] = num_history
            encoded['_current_image_count'] = num_current
            encoded['_num_initial_images'] = num_initial_images
            encoded['_frame_poses'] = frame_poses
            # Process unified history memory token
            if history_memory_idx_list and num_history > 0:
                # Calculate total tokens using HistoryProcessor
                # Build frame_infos: [(t, h, w), ...] for each history frame (after merge)
                frame_infos = []
                for i in range(num_history):
                    t, h, w = image_grid_thw[i].tolist()
                    h_after_merge = int(h) // merge_size
                    w_after_merge = int(w) // merge_size
                    frame_infos.append((int(t), h_after_merge, w_after_merge))
                
                # Use HistoryProcessor to calculate output token count
                total_history_tokens = self.history_processor.get_output_token_count(
                    num_frames=num_history,
                    frame_infos=frame_infos,
                )
                total_history_tokens = max(1, total_history_tokens)
                
                # Store total for _post_encode
                encoded['_total_history_tokens'] = total_history_tokens
                
                # Only process the first <history_memory> token (should be only one)
                history_to_process = history_memory_idx_list[:1]
                
                def _get_unified_history_tokens(i):
                    # Return ALL history tokens as a single unified block
                    return [self.history_memory_token_id] * total_history_tokens
                
                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, history_to_process, _get_unified_history_tokens
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
                videos=videos, return_tensors='pt', **kwargs
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
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process embeddings with differentiated handling for history vs current images.
        
        NOTE: Unlike Navid, compression applies in BOTH training and inference.
        
        History images: Apply additional pooling after visual encoding
        Current images: Use standard visual embeddings
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
        total_history_tokens_raw = inputs.pop('_total_history_tokens', None)
        inputs.pop('_num_initial_images', None)
        frame_poses_raw = inputs.pop('_frame_poses', None)
        
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
        
        # Get per-sample counts (important for batch_size > 1)
        history_counts = _to_list(num_history_raw)
        current_counts = _to_list(num_current_raw)
        
        # Ensure same number of samples
        num_samples = max(len(history_counts), len(current_counts))
        while len(history_counts) < num_samples:
            history_counts.append(0)
        while len(current_counts) < num_samples:
            current_counts.append(0)

        def _is_pose_vec(item):
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                return False
            return all(isinstance(v, (int, float)) for v in item[:4])

        def _is_pose_entry(item):
            return item is None or _is_pose_vec(item)

        def _is_pose_sample(item):
            if not isinstance(item, list):
                return False
            return all(_is_pose_entry(entry) for entry in item)

        def _to_pose_samples(value):
            """
            Convert collated/raw pose metadata to per-sample list.

            Output format:
                List[sample] where each sample is List[pose_vec]
            """
            if value is None:
                return []
            if hasattr(value, 'tolist') and not isinstance(value, list):
                try:
                    value = value.tolist()
                except Exception:
                    return []
            if not isinstance(value, list):
                return []
            if len(value) == 0:
                return []

            # Single-sample direct format: [pose_vec|None, pose_vec|None, ...]
            if _is_pose_sample(value):
                return [value]

            # Collated format: [[pose_vec, ...], [pose_vec, ...], ...]
            samples = []
            for item in value:
                if item is None:
                    samples.append([])
                    continue
                if hasattr(item, 'tolist') and not isinstance(item, list):
                    try:
                        item = item.tolist()
                    except Exception:
                        samples.append([])
                        continue
                if isinstance(item, list):
                    if _is_pose_sample(item):
                        samples.append(item)
                    elif _is_pose_vec(item):
                        samples.append([item])
                    else:
                        cleaned = []
                        for sub in item:
                            if _is_pose_entry(sub):
                                cleaned.append(sub)
                        samples.append(cleaned)
                else:
                    samples.append([])
            return samples

        pose_samples = _to_pose_samples(frame_poses_raw)
        while len(pose_samples) < num_samples:
            pose_samples.append([])

        # Flatten to per-image pose list in the same order as image_grid_thw:
        # [sample1_all_images, sample2_all_images, ...]
        flat_frame_poses = []
        for sample_idx in range(num_samples):
            expected_images = history_counts[sample_idx] + current_counts[sample_idx]
            sample_poses = pose_samples[sample_idx] if sample_idx < len(pose_samples) else []
            if len(sample_poses) < expected_images:
                sample_poses = sample_poses + [None] * (expected_images - len(sample_poses))
            elif len(sample_poses) > expected_images:
                sample_poses = sample_poses[:expected_images]
            flat_frame_poses.extend(sample_poses)
        
        # Calculate totals for validation
        total_history = sum(history_counts)
        total_current = sum(current_counts)
        
        # Validate counts against actual image count
        if image_grid_thw is not None:
            num_images = image_grid_thw.shape[0]
            if total_history + total_current != num_images:
                print(f"[SwiftVLN] WARNING: Image count mismatch! "
                      f"history={total_history}, current={total_current}, "
                      f"total={total_history + total_current}, actual={num_images}")
        
        if pixel_values is None:
            # No images, handle training stability
            from PIL import Image
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(model.visual.dtype)
            image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            return {'inputs_embeds': inputs_embeds}
        
        # Get all image embeddings from visual encoder. Qwen3-VL visual returns
        # a structure with pooler_output/deepstack_features; SwiftVLN consumes
        # the same pooled visual tokens as the standard placeholder path.
        dtype = model.visual.dtype
        pixel_values = pixel_values.type(dtype)
        visual_res = model.visual(pixel_values, grid_thw=image_grid_thw)
        if hasattr(visual_res, 'pooler_output'):
            all_image_embeds = visual_res.pooler_output
        elif isinstance(visual_res, tuple):
            all_image_embeds = visual_res[0]
        else:
            all_image_embeds = visual_res
        
        # --- Embedding Enhancement Pipeline ---
        # Apply the selected embedding enhancement to ALL images
        # This must happen BEFORE history compression so both history and current
        # images benefit from the learned enhancements
        if hasattr(model, 'embed_enhance') and not model.embed_enhance.is_empty:
            merge_size_for_enhance = self.processor.image_processor.merge_size
            enhanced = []
            offset = 0
            for i in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[i].tolist()
                # Grid dimensions after ViT's merge (same as what visual encoder outputs)
                h_m, w_m = int(h) // merge_size_for_enhance, int(w) // merge_size_for_enhance
                n_tokens = int(t) * h_m * w_m
                
                # Extract this image's embeddings
                img_embed = all_image_embeds[offset:offset + n_tokens]
                pose_i = flat_frame_poses[i] if i < len(flat_frame_poses) else None
                
                # Apply the selected enhancement via the checkpoint-compatible container
                img_embed = model.embed_enhance(img_embed, h_m, w_m, pose=pose_i)
                
                enhanced.append(img_embed)
                offset += n_tokens
            
            all_image_embeds = torch.cat(enhanced, dim=0)
        # --- End Embedding Enhancement Pipeline ---
        
        merge_size = self.processor.image_processor.merge_size
        merge_length = merge_size ** 2
        
        # Split embeddings by image, processing per-sample to maintain correct order
        # Image order in image_grid_thw: [sample1_all_images, sample2_all_images, ...]
        # Within each sample: [history_images, current_images]
        embed_idx = 0
        img_idx = 0
        
        # Collect raw history frame data for batch processing
        all_history_frame_embeds = []  # List of [num_tokens, hidden_size] per frame
        all_history_frame_grid_thws = []  # List of [3] grid_thw per frame
        current_embeds_list = []
        
        for sample_idx in range(num_samples):
            sample_num_history = history_counts[sample_idx]
            sample_num_current = current_counts[sample_idx]
            
            # Collect history frame embeddings (raw, before compression)
            for _ in range(sample_num_history):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                # Calculate grid dimensions after ViT merge
                grid_after_merge = image_grid_thw[img_idx].clone()
                grid_after_merge[1] = grid_after_merge[1] // merge_size
                grid_after_merge[2] = grid_after_merge[2] // merge_size
                
                all_history_frame_embeds.append(img_embeds)
                all_history_frame_grid_thws.append(grid_after_merge)

                img_idx += 1
            
            # Process current images for this sample (no compression)
            # Image order within current: [initial_image (if any), turn_images...]
            for _ in range(sample_num_current):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                current_embeds_list.append(img_embeds)

                img_idx += 1
        
        # Process history frames using HistoryProcessor
        # IMPORTANT: For GTC/SegmentGTC, must process each sample separately to match
        # the token counts calculated in _encode() for each sample
        history_embeds_list = []
        
        if all_history_frame_embeds:
            # Get per-sample token counts for validation
            if total_history_tokens_raw is not None:
                expected_tokens_per_sample = _to_list(total_history_tokens_raw)
            else:
                expected_tokens_per_sample = None
            
            # Rebuild per-sample history frame lists for separate processing
            # This is necessary because GTC/SegmentGTC should not mix frames across samples
            frame_idx = 0
            expected_idx = 0  # Separate index for expected_tokens_per_sample (only incremented for samples with history)
            for sample_idx in range(num_samples):
                sample_num_history = history_counts[sample_idx]
                if sample_num_history == 0:
                    continue
                
                # Extract this sample's history frames
                sample_frame_embeds = all_history_frame_embeds[frame_idx:frame_idx + sample_num_history]
                sample_frame_grid_thws = all_history_frame_grid_thws[frame_idx:frame_idx + sample_num_history]
                frame_idx += sample_num_history
                
                # Process this sample's history
                processed_sample = self.history_processor.process(
                    frame_embeds_list=sample_frame_embeds,
                    frame_grid_thws=sample_frame_grid_thws,
                )
                
                # Validate token count matches what _encode() calculated
                # Use expected_idx (not sample_idx) because expected_tokens_per_sample only contains
                # entries for samples WITH history frames (samples without history are skipped in collator)
                if expected_tokens_per_sample is not None and expected_idx < len(expected_tokens_per_sample):
                    expected = expected_tokens_per_sample[expected_idx]
                    actual = processed_sample.shape[0]
                    if expected != actual:
                        print(f"[SwiftVLN] WARNING: Sample {sample_idx} history token count mismatch; "
                              f"expected={expected}, actual={actual}. Adjusting embeddings to match placeholders.")
                        # Pad or truncate to match expected count
                        if actual < expected:
                            # Pad with zeros (shouldn't happen often)
                            padding = torch.zeros(expected - actual, processed_sample.shape[1], 
                                                  device=processed_sample.device, dtype=processed_sample.dtype)
                            processed_sample = torch.cat([processed_sample, padding], dim=0)
                        else:
                            # Truncate
                            processed_sample = processed_sample[:expected]
                
                expected_idx += 1  # Increment expected_idx for each sample with history
                history_embeds_list.append(processed_sample)
        
        # Replace unified history memory tokens
        # All history embeddings are concatenated into a single block
        if history_embeds_list:
            history_embeds = torch.cat(history_embeds_list, dim=0)
            # Use history_memory_token_id for unified memory mode
            history_mask = (input_ids == self.history_memory_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            history_embeds = history_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Validate before masked_scatter
            mask_true_count = (input_ids == self.history_memory_token_id).sum().item()
            embeds_count = history_embeds.shape[0]
            if mask_true_count != embeds_count:
                print("[SwiftVLN] CRITICAL: History token count mismatch before masked_scatter!")
                print(f"  mask_true_count={mask_true_count}, embeds_count={embeds_count}")
                # Try to fix by padding or truncating
                if embeds_count < mask_true_count:
                    padding = torch.zeros(mask_true_count - embeds_count, history_embeds.shape[1],
                                          device=history_embeds.device, dtype=history_embeds.dtype)
                    history_embeds = torch.cat([history_embeds, padding], dim=0)
                    print(f"  -> Padded history_embeds to {history_embeds.shape[0]}")
                else:
                    history_embeds = history_embeds[:mask_true_count]
                    print(f"  -> Truncated history_embeds to {history_embeds.shape[0]}")
            
            inputs_embeds = inputs_embeds.masked_scatter(history_mask, history_embeds)
        
        # Replace current image tokens (including initial image if enabled)
        if current_embeds_list:
            current_embeds = torch.cat(current_embeds_list, dim=0)
            current_mask = (input_ids == self.current_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            current_embeds = current_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Validate before masked_scatter
            mask_true_count = (input_ids == self.current_image_token_id).sum().item()
            embeds_count = current_embeds.shape[0]
            if mask_true_count != embeds_count:
                print("[SwiftVLN] CRITICAL: Current token count mismatch before masked_scatter!")
                print(f"  mask_true_count={mask_true_count}, embeds_count={embeds_count}")
            
            inputs_embeds = inputs_embeds.masked_scatter(current_mask, current_embeds)
        
        result = {'inputs_embeds': inputs_embeds}
        if labels is not None:
            result['labels'] = labels
        return result
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate multimodal data including custom metadata."""
        res = super()._data_collator_mm_data(batch)
        
        # Collect essential metadata into lists
        for key in ['_history_image_count', '_current_image_count', '_total_history_tokens', '_num_initial_images', '_frame_poses']:
            values = []
            for b in batch:
                val = b.pop(key, None)
                if val is not None:
                    values.append(val)
            if values:
                res[key] = values
        
        return res


class SwiftVLNQwen25VLTemplate(SwiftVLNTemplateMixin, Qwen2_5VLTemplate):
    """SwiftVLN template for Qwen2.5-VL."""


class SwiftVLNQwen3VLTemplate(SwiftVLNTemplateMixin, SwiftQwen3VLTemplate):
    """SwiftVLN template for Qwen3-VL."""


def register_swiftvln_templates() -> None:
    """Register both supported SwiftVLN templates with ms-swift."""
    register_template(
        QwenTemplateMeta(
            'swiftvln_qwen2_5_vl',
            template_cls=SwiftVLNQwen25VLTemplate,
        ),
        exist_ok=True,
    )
    register_template(
        QwenTemplateMeta(
            'swiftvln_qwen3_vl',
            template_cls=SwiftVLNQwen3VLTemplate,
            default_system=None,
            thinking_prefix='<think>\n',
        ),
        exist_ok=True,
    )
