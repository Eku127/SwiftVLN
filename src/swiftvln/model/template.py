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

from swiftvln.common.constants import CURRENT_IMAGE_TOKEN, HISTORY_MEMORY_TOKEN
from swiftvln.common.history_processors import (
    HistoryProcessor,
    create_history_processor,
    PerFrameCompressor,
    GlobalTokenClustering,
)

# Special tokens (must match dataset.py and model.py)
HISTORY_IMAGE_TOKEN = "<history_image>"  # Legacy: per-frame token (deprecated)

# Debug flag - set to True to see detailed GTC/SGTC processing info
DEBUG_COMPRESSION = False
# Debug flag - set to True to verify processor type in _encode
DEBUG_PROCESSOR_TYPE = False
# Debug flag - set via SWIFTVLN_DEBUG env var to verify initial strategy
DEBUG_INITIAL = os.environ.get('SWIFTVLN_DEBUG', '') != ''


def _debug_rank() -> int:
    raw = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    try:
        return int(raw)
    except ValueError:
        return 0


def _preview_list(values: List[int], limit: int = 8) -> str:
    if len(values) <= limit:
        return str(values)
    return str(values[:limit] + ['...'])


def _tensor_debug_stats(tensor: Optional[torch.Tensor], sample_limit: int = 4) -> str:
    if tensor is None:
        return "none"
    if not isinstance(tensor, torch.Tensor):
        return f"type={type(tensor).__name__}"
    if tensor.numel() == 0:
        return f"shape={tuple(tensor.shape)} empty"
    with torch.no_grad():
        flat = tensor.detach().float().cpu().reshape(-1)
        mean = float(flat.mean().item())
        std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
        checksum = float(flat.sum().item())
        abs_checksum = float(flat.abs().sum().item())
        l2 = float(torch.linalg.vector_norm(flat).item())
        sample = ", ".join(f"{v:.4f}" for v in flat[:sample_limit].tolist())
    return (
        f"shape={tuple(tensor.shape)} mean={mean:.6f} std={std:.6f} "
        f"sum={checksum:.6f} abs_sum={abs_checksum:.6f} l2={l2:.6f} "
        f"sample=[{sample}]"
    )


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
    history_image_token_id: Optional[int] = None
    history_memory_token_id: Optional[int] = None
    current_image_token_id: Optional[int] = None
    
    # History processor
    history_processor: Optional[HistoryProcessor] = None
    
    # Pixel embedding enhancement (set by trainer)
    use_pixel_embed: bool = False
    use_uav_adapter: bool = False
    
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
        self.history_image_token_id = processor.tokenizer.convert_tokens_to_ids(HISTORY_IMAGE_TOKEN)
        self.history_memory_token_id = processor.tokenizer.convert_tokens_to_ids(HISTORY_MEMORY_TOKEN)
        self.current_image_token_id = processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        
        # Verify tokens exist and print configuration
        if self.history_memory_token_id != processor.tokenizer.unk_token_id:
            print(f"[SwiftVLNTemplate] Using special tokens (unified memory mode):")
            print(f"  - {HISTORY_MEMORY_TOKEN}: {self.history_memory_token_id} (unified)")
            print(f"  - {HISTORY_IMAGE_TOKEN}: {self.history_image_token_id} (legacy)")
            print(f"  - {CURRENT_IMAGE_TOKEN}: {self.current_image_token_id}")
            print(f"  - Standard image_token_id (<|image_pad|>): {self.image_token_id}")
            print(f"  - History Processor: {self.history_processor.name}")
            
            # Print processor-specific info
            if isinstance(self.history_processor, PerFrameCompressor):
                method = "tome" if self.use_tome else "pool"
                print(f"    └─ h={self.num_history}, b={self.log_base}, {method}, s={self.compress_stride}")
                # Debug output
                if os.environ.get('SWIFTVLN_DEBUG'):
                    print(f"    [DEBUG] PerFrameCompressor configuration:")
                    print(f"      -> num_history: {self.num_history}")
                    print(f"      -> log_base: {self.log_base} ({'UNIFORM' if self.log_base == 1.0 else 'LOGARITHMIC'})")
                    print(f"      -> method: {method.upper()}")
                    print(f"      -> stride: {self.compress_stride} ({self.compress_stride**2}x compression)")
            elif isinstance(self.history_processor, GlobalTokenClustering):
                print(f"    └─ output_tokens={self.gtc_output_tokens}, τ={self.gtc_temperature}")
    
    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override packing_row to raise an error if padding_free=true is used.
        
        SwiftVLN uses custom image tokens which are incompatible with
        get_rope_index's expectation of standard <|image_pad|> tokens.
        """
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "[SwiftVLNTemplate] ERROR: padding_free=true is NOT supported!\n\n"
            "SwiftVLN uses custom tokens (<history_image>, <current_image>) which are\n"
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
            num_initial_images = inputs.extra_kwargs.get('num_initial_images', 0)
            
            # Determine if this is a history or current image
            is_history = (index < num_history_images)
            # Initial image is the first image after history images
            is_initial = (num_initial_images > 0 and index == num_history_images)
            
            if is_history:
                # History images: no <image> tag in prompt for them
                # They are processed via the unified <history_memory> token in system prompt
                # Return empty - this shouldn't be called for history images in normal flow
                return []
            else:
                # Current images (including initial): convert <image> to our custom token with vision wrapper
                # This enables ROPE position encoding while using our custom token
                # Debug: log initial image detection
                if DEBUG_INITIAL and is_initial:
                    if not hasattr(self, '_debug_initial_replace_count'):
                        self._debug_initial_replace_count = 0
                    if self._debug_initial_replace_count < 3:
                        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                        print(f"[INITIAL DEBUG] Rank={rank} replace_tag: index={index} is INITIAL image "
                              f"(num_history={num_history_images}, num_initial={num_initial_images}) "
                              f"-> <current_image> (uncompressed)")
                        self._debug_initial_replace_count += 1
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

            if DEBUG_INITIAL and getattr(self, 'memory_method', 'history') == 'map':
                if not hasattr(self, '_debug_map_tokenize_count'):
                    self._debug_map_tokenize_count = 0
                if self._debug_map_tokenize_count < 6:
                    history_positions_before = list(history_memory_idx_list[:8])
                    current_positions_before = list(current_idx_list[:8])
                    pose_entries = len(frame_poses) if isinstance(frame_poses, list) else 0
                    none_pose_entries = (
                        sum(1 for pose in frame_poses if pose is None)
                        if isinstance(frame_poses, list)
                        else 0
                    )
                    print(
                        f"[MAP DEBUG][template._encode.pre] Rank={_debug_rank()} "
                        f"input_len={len(input_ids)} num_images={num_images} "
                        f"num_history_images={num_history} num_initial_images={num_initial_images} "
                        f"num_current_placeholders={num_current} "
                        f"history_placeholder_count={len(history_memory_idx_list)} "
                        f"current_placeholder_count={len(current_idx_list)} "
                        f"frame_poses={pose_entries} none_poses={none_pose_entries} "
                        f"history_pos={_preview_list(history_positions_before)} "
                        f"current_pos={_preview_list(current_positions_before)}"
                    )
                    self._debug_map_tokenize_count += 1
            
            # Debug: log initial image in _encode
            if DEBUG_INITIAL and num_initial_images > 0:
                if not hasattr(self, '_debug_initial_encode_count'):
                    self._debug_initial_encode_count = 0
                if self._debug_initial_encode_count < 3:
                    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                    print(f"[INITIAL DEBUG] Rank={rank} _encode: num_images={num_images}, "
                          f"num_history={num_history}, num_initial={num_initial_images}, "
                          f"num_current_tokens={num_current} "
                          f"(expected: {num_history} + {num_initial_images} + {num_current - num_initial_images} current_turns = {num_images})")
                    # The initial image is at index num_history in the image list
                    # It produces a <current_image> token in the system prompt
                    # which is the first entry in current_idx_list
                    if current_idx_list:
                        print(f"[INITIAL DEBUG] Rank={rank} _encode: first <current_image> token at position {current_idx_list[0]} "
                              f"(this is the INITIAL image in system prompt, uncompressed)")
                    self._debug_initial_encode_count += 1
            
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
                
                # Debug: verify processor type and token calculation
                if DEBUG_PROCESSOR_TYPE or DEBUG_COMPRESSION:
                    total_input = sum(t * h * w for t, h, w in frame_infos)
                    processor_type = self.history_processor_type if hasattr(self, 'history_processor_type') else 'unknown'
                    print(f"\n[_encode DEBUG] HistoryProcessor Token Calculation:")
                    print(f"  - processor_type attr: {processor_type}")
                    print(f"  - processor.name: {self.history_processor.name}")
                    print(f"  - processor class: {type(self.history_processor).__name__}")
                    print(f"  - num_history_frames: {num_history}")
                    print(f"  - frame_infos: {frame_infos}")
                    print(f"  - total_input_tokens: {total_input}")
                    print(f"  - calculated_output: {total_history_tokens}")
                    if 'GTC' in self.history_processor.name or 'Segment' in self.history_processor.name:
                        print(f"  - GTC output_tokens setting: {getattr(self.history_processor, 'output_tokens', 'N/A')}")
                    print(f"  - RESULT: Will create {total_history_tokens} placeholder tokens")
                
                # Store total for _post_encode
                encoded['_total_history_tokens'] = total_history_tokens

                if DEBUG_INITIAL and getattr(self, 'memory_method', 'history') == 'map':
                    if not hasattr(self, '_debug_map_encode_count'):
                        self._debug_map_encode_count = 0
                    if self._debug_map_encode_count < 5:
                        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                        print(
                            f"[MAP DEBUG][template._encode] Rank={rank} "
                            f"num_history_images={num_history} frame_infos={frame_infos} "
                            f"unified_history_tokens={total_history_tokens}"
                        )
                        self._debug_map_encode_count += 1
                
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

            if DEBUG_INITIAL and getattr(self, 'memory_method', 'history') == 'map':
                if not hasattr(self, '_debug_map_expand_count'):
                    self._debug_map_expand_count = 0
                if self._debug_map_expand_count < 6:
                    history_token_count = sum(1 for token in input_ids if token == self.history_memory_token_id)
                    current_token_count = sum(1 for token in input_ids if token == self.current_image_token_id)
                    print(
                        f"[MAP DEBUG][template._encode.post] Rank={_debug_rank()} "
                        f"expanded_input_len={len(input_ids)} "
                        f"history_tokens={history_token_count} "
                        f"current_tokens={current_token_count} "
                        f"stored_total_history_tokens={encoded.get('_total_history_tokens', 0)}"
                    )
                    self._debug_map_expand_count += 1
        
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
        num_initial_raw = inputs.pop('_num_initial_images', 0)
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
        initial_counts = _to_list(num_initial_raw)
        
        # Ensure same number of samples
        num_samples = max(len(history_counts), len(current_counts))
        while len(history_counts) < num_samples:
            history_counts.append(0)
        while len(current_counts) < num_samples:
            current_counts.append(0)
        while len(initial_counts) < num_samples:
            initial_counts.append(0)

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
                if DEBUG_COMPRESSION:
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
        # Apply embedding enhancements (pixel, pose, etc.) to ALL images
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
                
                # Apply all enhancements via pipeline
                img_embed = model.embed_enhance(img_embed, h_m, w_m, pose=pose_i)
                
                enhanced.append(img_embed)
                offset += n_tokens
            
            all_image_embeds = torch.cat(enhanced, dim=0)
            
            if DEBUG_COMPRESSION:
                print(f"  [EmbedEnhance] Applied {model.embed_enhance.enhancement_names} to {image_grid_thw.shape[0]} images")
        # --- End Embedding Enhancement Pipeline ---
        
        merge_size = self.processor.image_processor.merge_size
        merge_length = merge_size ** 2
        
        if DEBUG_COMPRESSION:
            print("\n" + "="*70)
            print("[DEBUG] SwiftVLN _post_encode: History Processing")
            print(f"  Batch size: {num_samples}")
            print(f"  Per-sample history counts: {history_counts}")
            print(f"  Per-sample current counts: {current_counts}")
            print(f"  Total images: {total_history + total_current} (history: {total_history}, current: {total_current})")
            print(f"  History Processor: {self.history_processor.name}")
            print(f"  merge_size: {merge_size} (Qwen2.5-VL visual encoder merge)")
            print(f"  all_image_embeds shape: {all_image_embeds.shape}")
        
        # Split embeddings by image, processing per-sample to maintain correct order
        # Image order in image_grid_thw: [sample1_all_images, sample2_all_images, ...]
        # Within each sample: [history_images, current_images]
        embed_idx = 0
        img_idx = 0
        
        # Collect raw history frame data for batch processing
        all_history_frame_embeds = []  # List of [num_tokens, hidden_size] per frame
        all_history_frame_grid_thws = []  # List of [3] grid_thw per frame
        current_embeds_list = []
        
        total_history_tokens_before = 0
        total_current_tokens = 0
        
        for sample_idx in range(num_samples):
            sample_num_history = history_counts[sample_idx]
            sample_num_current = current_counts[sample_idx]
            
            if DEBUG_COMPRESSION:
                print(f"\n  Sample {sample_idx}: {sample_num_history} history + {sample_num_current} current")
            
            # Collect history frame embeddings (raw, before compression)
            for h_idx in range(sample_num_history):
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
                
                if DEBUG_COMPRESSION:
                    total_history_tokens_before += num_tokens
                    print(f"    History[{h_idx}] img_idx={img_idx}: {num_tokens} tokens (raw)")
                
                img_idx += 1
            
            sample_num_initial = initial_counts[sample_idx] if sample_idx < len(initial_counts) else 0
            
            # Process current images for this sample (no compression)
            # Image order within current: [initial_image (if any), turn_images...]
            for c_idx in range(sample_num_current):
                if img_idx >= image_grid_thw.shape[0]:
                    break
                num_tokens = int(image_grid_thw[img_idx].prod() // merge_length)
                img_embeds = all_image_embeds[embed_idx:embed_idx + num_tokens]
                embed_idx += num_tokens
                
                current_embeds_list.append(img_embeds)
                
                if DEBUG_COMPRESSION:
                    total_current_tokens += num_tokens
                    is_init_str = " [INITIAL]" if (c_idx == 0 and sample_num_initial > 0) else ""
                    print(f"    Current[{c_idx}] img_idx={img_idx}: {num_tokens} tokens (no compression){is_init_str}")
                
                img_idx += 1
        
        # Process history frames using HistoryProcessor
        # IMPORTANT: For GTC/SegmentGTC, must process each sample separately to match
        # the token counts calculated in _encode() for each sample
        history_embeds_list = []
        total_history_tokens_after = 0
        
        if all_history_frame_embeds:
            # Get per-sample token counts for validation
            if total_history_tokens_raw is not None:
                expected_tokens_per_sample = _to_list(total_history_tokens_raw)
            else:
                expected_tokens_per_sample = None
            
            if DEBUG_COMPRESSION:
                print(f"\n  [GTC/SGTC Processing] History Processor: {self.history_processor.name}")
                print(f"    Total history frames: {len(all_history_frame_embeds)}")
                print(f"    Expected tokens per sample: {expected_tokens_per_sample}")
                print(f"    Processing {num_samples} samples separately...")
            
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
                
                # Calculate input tokens for this sample
                input_tokens_this_sample = sum(e.shape[0] for e in sample_frame_embeds)
                
                if DEBUG_COMPRESSION:
                    print(f"\n    Sample {sample_idx}: {sample_num_history} history frames, {input_tokens_this_sample} input tokens")
                
                # Process this sample's history
                processed_sample = self.history_processor.process(
                    frame_embeds_list=sample_frame_embeds,
                    frame_grid_thws=sample_frame_grid_thws,
                )
                
                if DEBUG_COMPRESSION:
                    print(f"      -> Processed output: {processed_sample.shape[0]} tokens (compression: {input_tokens_this_sample / max(1, processed_sample.shape[0]):.2f}x)")
                
                # Validate token count matches what _encode() calculated
                # Use expected_idx (not sample_idx) because expected_tokens_per_sample only contains
                # entries for samples WITH history frames (samples without history are skipped in collator)
                if expected_tokens_per_sample is not None and expected_idx < len(expected_tokens_per_sample):
                    expected = expected_tokens_per_sample[expected_idx]
                    actual = processed_sample.shape[0]
                    if expected != actual:
                        if DEBUG_COMPRESSION:
                            print(f"[SwiftVLN] DEBUG: Sample {sample_idx} (expected_idx={expected_idx}) token mismatch! "
                                  f"expected={expected}, actual={actual}, diff={expected - actual}")
                        # Pad or truncate to match expected count
                        if actual < expected:
                            # Pad with zeros (shouldn't happen often)
                            padding = torch.zeros(expected - actual, processed_sample.shape[1], 
                                                  device=processed_sample.device, dtype=processed_sample.dtype)
                            processed_sample = torch.cat([processed_sample, padding], dim=0)
                            if DEBUG_COMPRESSION:
                                print(f"      -> Padded {expected - actual} tokens to match expected")
                        else:
                            # Truncate
                            processed_sample = processed_sample[:expected]
                            if DEBUG_COMPRESSION:
                                print(f"      -> Truncated {actual - expected} tokens to match expected")
                    else:
                        if DEBUG_COMPRESSION:
                            print(f"      -> Token count matches expected: {expected}")
                
                expected_idx += 1  # Increment expected_idx for each sample with history
                history_embeds_list.append(processed_sample)
                total_history_tokens_after += processed_sample.shape[0]
        
        if DEBUG_COMPRESSION:
            print("\n" + "-"*70)
            print("  SUMMARY:")
            if total_history > 0:
                print(f"    History: {total_history_tokens_before} -> {total_history_tokens_after} tokens")
                print(f"    Compression ratio: {total_history_tokens_before / max(1, total_history_tokens_after):.2f}x")
            print(f"    Current: {total_current_tokens} tokens")
            print("="*70 + "\n")
        
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
            if DEBUG_COMPRESSION:
                print(f"\n  [masked_scatter] History: mask_true={mask_true_count}, embeds={embeds_count}")
            injection_matched_before_fix = (mask_true_count == embeds_count)
            if mask_true_count != embeds_count:
                print(f"[SwiftVLN] CRITICAL: History token count mismatch before masked_scatter!")
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

            if DEBUG_INITIAL and getattr(self, 'memory_method', 'history') == 'map':
                if not hasattr(self, '_debug_map_post_encode_count'):
                    self._debug_map_post_encode_count = 0
                if self._debug_map_post_encode_count < 5:
                    rank = _debug_rank()
                    per_sample_tokens = [embed.shape[0] for embed in history_embeds_list]
                    per_sample_mask = [
                        int((input_ids[sample_idx] == self.history_memory_token_id).sum().item())
                        for sample_idx in range(input_ids.shape[0])
                    ]
                    per_sample_current_mask = [
                        int((input_ids[sample_idx] == self.current_image_token_id).sum().item())
                        for sample_idx in range(input_ids.shape[0])
                    ]
                    print(
                        f"[MAP DEBUG][template._post_encode] Rank={rank} "
                        f"history_counts={history_counts} per_sample_tokens={per_sample_tokens} "
                        f"per_sample_history_mask={per_sample_mask} "
                        f"per_sample_current_mask={per_sample_current_mask} "
                        f"mask_true={mask_true_count} embeds={embeds_count} "
                        f"matched_before_fix={injection_matched_before_fix}"
                    )
                    print(
                        f"[MAP DEBUG][template._post_encode] "
                        f"history_embed_stats={_tensor_debug_stats(history_embeds)}"
                    )
                    self._debug_map_post_encode_count += 1
            
            inputs_embeds = inputs_embeds.masked_scatter(history_mask, history_embeds)
        
        # Replace current image tokens (including initial image if enabled)
        if current_embeds_list:
            current_embeds = torch.cat(current_embeds_list, dim=0)
            current_mask = (input_ids == self.current_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            current_embeds = current_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Validate before masked_scatter
            mask_true_count = (input_ids == self.current_image_token_id).sum().item()
            embeds_count = current_embeds.shape[0]
            if DEBUG_COMPRESSION:
                print(f"  [masked_scatter] Current: mask_true={mask_true_count}, embeds={embeds_count}")
            if mask_true_count != embeds_count:
                print(f"[SwiftVLN] CRITICAL: Current token count mismatch before masked_scatter!")
                print(f"  mask_true_count={mask_true_count}, embeds_count={embeds_count}")
            
            # ============================================================
            # DEBUG: Verify initial token injection in multi-sample batch
            # ============================================================
            if DEBUG_INITIAL and any(ic > 0 for ic in initial_counts):
                if not hasattr(self, '_debug_initial_post_encode_count'):
                    self._debug_initial_post_encode_count = 0
                if self._debug_initial_post_encode_count < 5:
                    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                    print(f"\n{'='*70}")
                    print(f"[INITIAL DEBUG] Rank={rank} _post_encode: Initial Token Injection Verification")
                    print(f"  Batch size: {num_samples}")
                    print(f"  Per-sample: history={history_counts}, current={current_counts}, initial={initial_counts}")
                    print(f"  Total <current_image> mask positions: {mask_true_count}")
                    print(f"  Total current embeds to inject: {embeds_count}")
                    
                    # Per-sample position analysis
                    embed_offset = 0
                    for s_idx in range(num_samples):
                        s_num_init = initial_counts[s_idx] if s_idx < len(initial_counts) else 0
                        s_num_current = current_counts[s_idx] if s_idx < len(current_counts) else 0
                        s_num_turn = s_num_current - s_num_init  # turn images = current - initial
                        
                        # Count <current_image> positions in this sample's input_ids
                        if input_ids.dim() == 2 and s_idx < input_ids.shape[0]:
                            s_positions = (input_ids[s_idx] == self.current_image_token_id).nonzero(as_tuple=True)[0]
                            s_mask_count = len(s_positions)
                        else:
                            # 1D input_ids (single sample or already flattened)
                            s_positions = (input_ids.view(-1) == self.current_image_token_id).nonzero(as_tuple=True)[0]
                            s_mask_count = len(s_positions)
                        
                        # Calculate token counts for each current image in this sample
                        token_counts_str = []
                        for c_i in range(s_num_current):
                            idx = embed_offset + c_i
                            if idx < len(current_embeds_list):
                                tc = current_embeds_list[idx].shape[0]
                                label = "INITIAL" if (c_i == 0 and s_num_init > 0) else f"turn{c_i - s_num_init}"
                                token_counts_str.append(f"{label}:{tc}")
                        embed_offset += s_num_current
                        
                        print(f"  Sample {s_idx}: {s_num_current} current images "
                              f"({s_num_init} initial + {s_num_turn} turns), "
                              f"mask_positions={s_mask_count}, "
                              f"embeds=[{', '.join(token_counts_str)}]")
                    
                    # Verify total consistency
                    total_from_list = sum(e.shape[0] for e in current_embeds_list)
                    status = "OK" if mask_true_count == total_from_list else "MISMATCH"
                    print(f"  Verification: mask={mask_true_count} vs embeds={total_from_list} -> {status}")
                    print(f"{'='*70}\n")
                    self._debug_initial_post_encode_count += 1
            
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


# Register the custom templates
register_template(
    QwenTemplateMeta(
        'swiftvln_qwen2_5_vl',
        template_cls=SwiftVLNQwen25VLTemplate,
    ),
    exist_ok=True,
)

print("[SwiftVLNTemplate] Template 'swiftvln_qwen2_5_vl' registered successfully!")

register_template(
    QwenTemplateMeta(
        'swiftvln_qwen3_vl',
        template_cls=SwiftVLNQwen3VLTemplate,
        default_system=None,
        thinking_prefix='<think>\n',
    ),
    exist_ok=True,
)

print("[SwiftVLNTemplate] Template 'swiftvln_qwen3_vl' registered successfully!")
