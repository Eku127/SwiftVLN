# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Evaluator - Three-level memory with incremental feature caching

Implements the Uni-NaVid evaluation strategy:
1. Incremental feature caching (only encode new frame each step)
2. Three-level memory architecture (long-term, short-term, current)
3. Single-turn dialogue format (matching training)

Key difference from MonoVLN: UniNaVid caches ViT features to avoid recomputation.
"""

import os
import re
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image

# Import from common module
try:
    from swiftvln.common import (
        BaseVLNEvaluator,
        EnvWrapper,
        TrajectoryRecorder,
    )
except ImportError:
    from ..common import (
        BaseVLNEvaluator,
        EnvWrapper,
        TrajectoryRecorder,
    )


class UniNaVidEvaluator(BaseVLNEvaluator):
    """
    UniNaVid Evaluator with incremental feature caching.
    
    Key features:
    - Three-level memory: long-term (merged), short-term (pooled), current (full)
    - Incremental caching: only encode new frame each step
    - Uni-NaVid action format: forward/left/right/stop
    
    Memory architecture:
    - short_term_cache: List of pooled ViT features (max short_term_frames)
    - long_term_cache: Merged features from oldest frames
    - current: Full resolution features (not cached, always computed)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # UniNaVid-specific parameters (try to load from checkpoint config)
        self.short_term_frames = getattr(self.args, 'short_term_frames', 32)
        self.similarity_threshold = getattr(self.args, 'similarity_threshold', 0.985)
        self.compress_stride = getattr(self.args, 'compress_stride', 3)
        self.image_resize_stride = getattr(self.args, 'image_resize_stride', 1.5)
        
        # Override action mapping for UniNaVid (forward/left/right/stop)
        self.idx2actions = {
            0: 'stop',
            1: 'forward',
            2: 'left',
            3: 'right',
        }
        self.actions2idx = {v: k for k, v in self.idx2actions.items()}
        
        # Feature caches (reset per episode)
        self.short_term_cache: List[torch.Tensor] = []
        self.long_term_cache: Optional[torch.Tensor] = None
        self.long_term_weight: int = 1
        
        # Store ViT features of current frame (for caching after inference)
        self._current_vit_features: Optional[torch.Tensor] = None
        self._current_grid_thw: Optional[torch.Tensor] = None
        
        # Get merge_size from processor
        self.merge_size = self.processor.image_processor.merge_size if hasattr(self.processor.image_processor, 'merge_size') else 2
    
    def reset_cache(self):
        """Reset feature caches at the start of each episode."""
        self.short_term_cache = []
        self.long_term_cache = None
        self.long_term_weight = 1
        self._current_vit_features = None
        self._current_grid_thw = None
        self.pending_images = []  # Images collected since last inference
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to match training preprocessing."""
        if self.image_resize_stride <= 1.0:
            return image
        
        orig_w, orig_h = image.size
        new_w = int(orig_w / self.image_resize_stride)
        new_h = int(orig_h / self.image_resize_stride)
        
        # Ensure minimum size for Qwen2.5-VL
        new_w = max(new_w, 28)
        new_h = max(new_h, 28)
        
        return image.resize((new_w, new_h), Image.BILINEAR)
    
    def _pool_embeddings_2d(self, embeddings: torch.Tensor, grid_thw: torch.Tensor,
                           pool_size: int = 2) -> torch.Tensor:
        """Apply 2D average pooling to image embeddings."""
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
        
        # Permute back and flatten
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        pooled = pooled.view(-1, hidden_size)
        
        return pooled
    
    def _merge_to_long_term(self, new_frame_features: torch.Tensor):
        """
        Merge a single frame's features into long-term memory.
        
        Args:
            new_frame_features: Pooled features from one frame [num_tokens, hidden_size]
        """
        # First mean pool to single vector
        new_single = new_frame_features.mean(dim=0, keepdim=True)  # [1, hidden_size]
        
        if self.long_term_cache is None:
            # First long-term frame
            self.long_term_cache = new_single
            self.long_term_weight = 1
        else:
            # Check similarity with last merged vector
            sim = F.cosine_similarity(
                self.long_term_cache[-1:], new_single, dim=-1
            ).mean().item()
            
            if sim > self.similarity_threshold:
                # Merge with weighted average
                new_weight = self.long_term_weight + 1
                self.long_term_cache[-1:] = (
                    self.long_term_cache[-1:] * self.long_term_weight + new_single
                ) / new_weight
                self.long_term_weight = new_weight
            else:
                # Append as new token
                self.long_term_cache = torch.cat([self.long_term_cache, new_single], dim=0)
                self.long_term_weight = 1
    
    def update_cache(self, vit_features: torch.Tensor, grid_thw: torch.Tensor):
        """
        Update feature caches after inference.
        
        The current frame's features are compressed and added to short-term cache.
        If short-term overflows, oldest frame is merged into long-term.
        
        Args:
            vit_features: ViT features of current frame [num_tokens, hidden_size]
            grid_thw: Grid dimensions [3] for the current frame
        """
        # Get grid after spatial merge
        grid_after_merge = grid_thw.clone()
        grid_after_merge[1] = grid_after_merge[1] // self.merge_size
        grid_after_merge[2] = grid_after_merge[2] // self.merge_size
        
        # Pool current frame features for storage
        pooled = self._pool_embeddings_2d(
            vit_features, grid_after_merge, pool_size=self.compress_stride
        )
        
        # Check if short-term is full
        if len(self.short_term_cache) >= self.short_term_frames:
            # Move oldest frame to long-term
            oldest = self.short_term_cache.pop(0)
            self._merge_to_long_term(oldest)
        
        # Add current (now becomes history) to short-term
        self.short_term_cache.append(pooled)
    
    def parse_actions(self, output: str) -> List[int]:
        """
        Parse action sequence from model output (UniNaVid format: forward/left/right/stop).
        
        Args:
            output: Model output text
            
        Returns:
            List of action indices
        """
        # Normalize output
        output_lower = output.lower()
        
        # Match forward, left, right, stop
        action_patterns = r'\b(forward|left|right|stop)\b'
        matches = re.findall(action_patterns, output_lower)
        
        actions = []
        for match in matches:
            if match in self.actions2idx:
                actions.append(self.actions2idx[match])
        
        return actions
    
    def build_single_turn_messages(
        self,
        instruction: str,
        num_long_term: int,
        short_term_token_counts: List[int],
        current_token_count: int,
    ) -> str:
        """
        Build prompt content matching training format (Option B).
        
        Args:
            instruction: Navigation instruction
            num_long_term: Number of long-term memory tokens (merged, 1 token each)
            short_term_token_counts: List of token counts for each short-term frame
            current_token_count: Number of tokens for current frame
            
        Returns:
            User content string
        """
        # Build long-term tokens: each merged frame = 1 token
        # Format: <|vision_start|><long_term_image><|vision_end|> for each
        long_term_tokens = ' '.join(['<|vision_start|><long_term_image><|vision_end|>' for _ in range(num_long_term)])
        
        # Build short-term tokens: each frame may have multiple tokens (after pooling)
        # Format: <|vision_start|><short_term_image>...<short_term_image><|vision_end|> per frame
        short_term_frame_strs = []
        for token_count in short_term_token_counts:
            inner_tokens = '<short_term_image>' * token_count
            short_term_frame_strs.append(f'<|vision_start|>{inner_tokens}<|vision_end|>')
        short_term_tokens = ' '.join(short_term_frame_strs)
        
        # Build current token: full resolution
        current_inner = '<current_image>' * current_token_count
        current_token = f'<|vision_start|>{current_inner}<|vision_end|>'
        
        num_short_term = len(short_term_token_counts)
        total_history = num_long_term + num_short_term
        
        # Action definition
        action_definition = (
            "You are an intelligent navigation robot. "
            "Your goal is to navigate to a target location based on the instruction.\n"
            "Available Actions:\n"
            "- forward: Move forward 0.25 meters.\n"
            "- left: Turn left 15 degrees.\n"
            "- right: Turn right 15 degrees.\n"
            "- stop: Use this ONLY when you have reached the goal."
        )
        
        if total_history > 0:
            # Build history section
            if num_long_term > 0 and num_short_term > 0:
                history_section = (
                    f"### Earlier Observations ({num_long_term} frames)\n"
                    f"These images are from the beginning of your journey:\n"
                    f"{long_term_tokens}\n\n"
                    f"### Recent Observations ({num_short_term} frames)\n"
                    f"These are your most recent views before the current moment:\n"
                    f"{short_term_tokens}"
                )
            elif num_short_term > 0:
                history_section = (
                    f"### Recent Observations ({num_short_term} frames)\n"
                    f"These are your views from the journey so far:\n"
                    f"{short_term_tokens}"
                )
            else:
                history_section = (
                    f"### Earlier Observations ({num_long_term} frames)\n"
                    f"These images are from the beginning of your journey:\n"
                    f"{long_term_tokens}"
                )
            
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"{history_section}\n\n"
                f"### Current Observation\n"
                f"This is what you see right now:\n"
                f"{current_token}\n\n"
                f"### Prediction\n"
                f"Based on your journey history and current view, predict the next {self.num_future_steps} actions "
                f"(space-separated, e.g., forward forward left stop):"
            )
        else:
            # First observation
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Current Observation\n"
                f"This is what you see at the starting point:\n"
                f"{current_token}\n\n"
                f"### Prediction\n"
                f"You are at the starting point. Analyze the instruction and the current view. "
                f"Predict the next {self.num_future_steps} actions (space-separated):"
            )
        
        return user_content
    
    def _encode_current_frame(self, current_image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode current frame through ViT.
        
        Args:
            current_image: PIL Image of current observation
            
        Returns:
            Tuple of (vit_features, grid_thw)
        """
        # Resize image
        current_image = self._resize_image(current_image)
        
        # Process through image processor
        media_inputs = self.processor.image_processor(
            images=[current_image], return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        # Get ViT features
        with torch.no_grad():
            vit_features = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        
        return vit_features, image_grid_thw[0]
    
    def _encode_batch_frames(self, images: List[Image.Image]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Batch encode multiple frames through ViT.
        
        This is more efficient than encoding one frame at a time because:
        1. GPU batch processing is more efficient
        2. Fewer kernel launch overhead
        
        Args:
            images: List of PIL Images to encode
            
        Returns:
            Tuple of (list of vit_features, list of grid_thw) for each image
        """
        if not images:
            return [], []
        
        # Resize all images
        resized_images = [self._resize_image(img) for img in images]
        
        # Process through image processor (batch)
        media_inputs = self.processor.image_processor(
            images=resized_images, return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        # Get ViT features for all images at once
        with torch.no_grad():
            all_vit_features = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        
        # Split features by image
        merge_length = self.merge_size ** 2
        features_list = []
        grid_thw_list = []
        embed_idx = 0
        
        for i in range(len(images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            img_features = all_vit_features[embed_idx:embed_idx + num_tokens]
            embed_idx += num_tokens
            features_list.append(img_features)
            grid_thw_list.append(image_grid_thw[i])
        
        return features_list, grid_thw_list
    
    def _process_pending_images_to_cache(self, features_list: List[torch.Tensor], 
                                          grid_thw_list: List[torch.Tensor]):
        """
        Process pending images (except the last one) and add them to short-term cache.
        
        Args:
            features_list: List of ViT features for each pending image (excluding current)
            grid_thw_list: List of grid_thw for each pending image (excluding current)
        """
        for features, grid_thw in zip(features_list, grid_thw_list):
            self.update_cache(features, grid_thw)
    
    def _build_inputs_embeds(
        self,
        instruction: str,
        current_vit_features: torch.Tensor,
        current_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Build inputs_embeds by combining cached features and current frame.
        
        Args:
            instruction: Navigation instruction
            current_vit_features: ViT features of current frame
            current_grid_thw: Grid dimensions of current frame
            
        Returns:
            Tuple of (inputs_embeds, sequence_length)
        """
        # Calculate number of long-term tokens (after merging, each = 1 token)
        num_long_term = 0 if self.long_term_cache is None else self.long_term_cache.shape[0]
        
        # Calculate token counts for each short-term frame (already pooled in cache)
        short_term_token_counts = [st_feat.shape[0] for st_feat in self.short_term_cache]
        
        # Current frame token count (full resolution)
        current_token_count = current_vit_features.shape[0]
        
        # Build prompt with correct token counts
        user_content = self.build_single_turn_messages(
            instruction=instruction,
            num_long_term=num_long_term,
            short_term_token_counts=short_term_token_counts,
            current_token_count=current_token_count,
        )
        
        messages = [{'role': 'user', 'content': user_content}]
        
        # Tokenize
        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True
        )
        input_ids = inputs['input_ids'].to(self.device)
        
        # Get text embeddings
        base_model = self.model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embed_tokens in model")
        
        # Get token IDs for special tokens
        long_term_token_id = self.processor.tokenizer.convert_tokens_to_ids('<long_term_image>')
        short_term_token_id = self.processor.tokenizer.convert_tokens_to_ids('<short_term_image>')
        current_token_id = self.processor.tokenizer.convert_tokens_to_ids('<current_image>')
        
        # Fill in long-term embeddings (each merged frame = 1 token)
        if self.long_term_cache is not None and num_long_term > 0:
            lt_positions = (input_ids[0] == long_term_token_id).nonzero(as_tuple=True)[0]
            if len(lt_positions) >= num_long_term:
                # Batch assignment: more efficient than loop
                lt_positions_slice = lt_positions[:num_long_term]
                lt_cache_device = self.long_term_cache[:num_long_term].to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds[0, lt_positions_slice] = lt_cache_device
        
        # Fill in short-term embeddings (each frame may have multiple consecutive tokens)
        if len(self.short_term_cache) > 0:
            st_positions = (input_ids[0] == short_term_token_id).nonzero(as_tuple=True)[0]
            total_st_tokens = sum(short_term_token_counts)
            
            if len(st_positions) >= total_st_tokens:
                # Concatenate all short-term features
                st_features_list = [
                    st_feat.to(inputs_embeds.device, inputs_embeds.dtype)
                    for st_feat in self.short_term_cache
                ]
                st_features_concat = torch.cat(st_features_list, dim=0)  # [total_tokens, hidden_size]
                
                # Batch assignment using advanced indexing
                st_positions_slice = st_positions[:total_st_tokens]
                inputs_embeds[0, st_positions_slice] = st_features_concat
        
        # Fill in current frame embeddings (full resolution, consecutive tokens)
        cur_positions = (input_ids[0] == current_token_id).nonzero(as_tuple=True)[0]
        
        if len(cur_positions) >= current_token_count:
            # Batch assignment: more efficient than loop
            cur_positions_slice = cur_positions[:current_token_count]
            current_feat_device = current_vit_features[:current_token_count].to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds[0, cur_positions_slice] = current_feat_device
        
        return inputs_embeds, inputs_embeds.shape[1]
    
    @torch.no_grad()
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode using incremental feature caching.
        
        Incremental processing flow per step:
        1. Encode current frame through ViT (only this frame)
        2. Combine with cached long-term and short-term features
        3. Build inputs_embeds and run LLM inference
        4. Update cache: current -> short-term (-> long-term if overflow)
        5. Execute actions
        
        Args:
            env_wrapper: Environment wrapper
            episode: Episode to evaluate
            env_idx: Environment index
            
        Returns:
            Evaluation metrics dictionary
        """
        # Initialize timing
        timing_stats = {
            'init': 0.0,
            'collect_observation': 0.0,
            'encode_current': 0.0,
            'build_embeds': 0.0,
            'model_generate': 0.0,
            'decode': 0.0,
            'parse_actions': 0.0,
            'update_cache': 0.0,
            'visualization': 0.0,
            'env_step': 0.0,
            'satnav_topdown': 0.0,
            'trajectory_record': 0.0,
            'error_analysis': 0.0,
            'video_save': 0.0,
        }
        episode_start_time = time.time()
        
        # Initialization
        init_start = time.time()
        self.model.eval()
        self.reset_cache()
        
        observations = env_wrapper.reset(episode)
        instruction = env_wrapper.get_instruction(episode)
        episode_id = episode.episode_id
        timing_stats['init'] = time.time() - init_start
        
        if hasattr(episode, 'scene_id'):
            scene_id = episode.scene_id.split('/')[-2] if '/' in episode.scene_id else episode.scene_id
        else:
            scene_id = "unknown"
        
        action_seq = []
        step_id = 0
        
        trajectory_recorder = TrajectoryRecorder()
        vis_frames = []
        rgb_frames = []
        topdown_frames = []
        
        max_steps = env_wrapper.max_steps
        
        while not env_wrapper.episode_over and step_id < max_steps:
            # 1. Collect current observation
            t0 = time.time()
            rgb = env_wrapper.get_rgb(observations)
            current_img = Image.fromarray(rgb).convert('RGB')
            self.pending_images.append(current_img)
            timing_stats['collect_observation'] += time.time() - t0
            
            # 2. If action queue is empty, predict new actions
            if len(action_seq) == 0:
                try:
                    # Batch encode all pending images (more efficient than one-by-one)
                    t0 = time.time()
                    if len(self.pending_images) > 1:
                        # Batch encode: history frames + current frame
                        features_list, grid_thw_list = self._encode_batch_frames(self.pending_images)
                        
                        # Process history frames (all except last) -> add to short-term cache
                        history_features = features_list[:-1]
                        history_grid_thw = grid_thw_list[:-1]
                        self._process_pending_images_to_cache(history_features, history_grid_thw)
                        
                        # Current frame (last one) - keep full resolution
                        current_vit_features = features_list[-1]
                        current_grid_thw = grid_thw_list[-1]
                    else:
                        # Only current frame (first step or after stop)
                        current_vit_features, current_grid_thw = self._encode_current_frame(self.pending_images[0])
                    
                    # Clear pending images after encoding
                    self.pending_images = []
                    timing_stats['encode_current'] += time.time() - t0
                    
                    # Build inputs_embeds with cached features
                    t0 = time.time()
                    inputs_embeds, seq_len = self._build_inputs_embeds(
                        instruction=instruction,
                        current_vit_features=current_vit_features,
                        current_grid_thw=current_grid_thw,
                    )
                    
                    # Create attention mask
                    attention_mask = torch.ones(
                        (1, seq_len),
                        dtype=torch.long,
                        device=inputs_embeds.device
                    )
                    
                    # Create dummy input_ids
                    pad_token_id = self.processor.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = self.processor.tokenizer.eos_token_id
                    dummy_input_ids = torch.full(
                        (1, seq_len),
                        pad_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device
                    )
                    timing_stats['build_embeds'] += time.time() - t0
                    
                    # Generate
                    t0 = time.time()
                    model_inputs = {
                        'input_ids': dummy_input_ids,
                        'inputs_embeds': inputs_embeds,
                        'attention_mask': attention_mask,
                    }
                    
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        use_cache=True,
                    )
                    timing_stats['model_generate'] += time.time() - t0
                    
                    # Decode
                    t0 = time.time()
                    generated_ids = outputs[0][seq_len:]
                    output_text = self.processor.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()
                    timing_stats['decode'] += time.time() - t0
                    
                    # Parse actions
                    t0 = time.time()
                    action_seq = self.parse_actions(output_text)
                    timing_stats['parse_actions'] += time.time() - t0
                    
                    # Update cache: current frame becomes history
                    t0 = time.time()
                    self.update_cache(current_vit_features, current_grid_thw)
                    timing_stats['update_cache'] += time.time() - t0
                    
                    if getattr(self.args, 'verbose', False):
                        num_lt = 0 if self.long_term_cache is None else self.long_term_cache.shape[0]
                        num_st = len(self.short_term_cache)
                        print(f"Step {step_id}: LT={num_lt}, ST={num_st}, pending={len(self.pending_images)}, output={output_text}")
                    
                except Exception as e:
                    print(f"[Warning] Generation failed at step {step_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    action_seq = []
                    self.pending_images = []  # Clear pending on error
                
                if not action_seq:
                    action_seq = [0]  # Default to STOP
            
            # 3. Video frame collection
            if self.save_video:
                t0 = time.time()
                if self.env_type == "habitat":
                    frame = self._collect_habitat_frame(observations, instruction, env_wrapper)
                    if frame is not None:
                        vis_frames.append(frame)
                elif self.env_type == "satnav":
                    rgb_frames.append(rgb.copy())
                timing_stats['visualization'] += time.time() - t0
            
            # 4. Execute action
            t0 = time.time()
            action = action_seq.pop(0)
            observations, _ = env_wrapper.step(action)
            timing_stats['env_step'] += time.time() - t0
            step_id += 1
            
            # 5. SatNav topdown frame
            if self.save_video and self.env_type == "satnav":
                t0 = time.time()
                self._collect_satnav_topdown(
                    env_wrapper, episode, action, step_id,
                    topdown_frames, rgb
                )
                timing_stats['satnav_topdown'] += time.time() - t0
            
            # 6. Trajectory recording (Habitat)
            if self.env_type == "habitat":
                t0 = time.time()
                try:
                    habitat_env = env_wrapper.env
                    agent_state = habitat_env.sim.get_agent_state()
                    trajectory_recorder.add_step(agent_state.position)
                except Exception:
                    pass
                timing_stats['trajectory_record'] += time.time() - t0
        
        metrics = env_wrapper.get_metrics()
        
        # Error analysis (Habitat)
        if self.env_type == "habitat":
            t0 = time.time()
            error_analysis = self._analyze_trajectory_errors(trajectory_recorder, episode, metrics)
            metrics.update(error_analysis)
            timing_stats['error_analysis'] = time.time() - t0
        
        # Save video
        if self.save_video:
            t0 = time.time()
            if self.env_type == "habitat":
                self._save_habitat_video(episode_id, vis_frames, metrics)
            elif self.env_type == "satnav":
                self._save_satnav_video(episode_id, instruction, rgb_frames, topdown_frames, metrics)
            timing_stats['video_save'] = time.time() - t0
        
        # Calculate total episode time
        total_time = time.time() - episode_start_time
        
        # Print timing statistics if debug_timing flag is set
        if getattr(self.args, 'debug_timing', False):
            print(f"\n{'='*60}")
            print(f"Episode {episode_id} (Scene: {scene_id}) Timing Statistics:")
            print(f"{'='*60}")
            print(f"Total episode time: {total_time:.3f}s")
            print(f"Total steps: {step_id}")
            print(f"Long-term tokens: {0 if self.long_term_cache is None else self.long_term_cache.shape[0]}")
            print(f"Short-term frames: {len(self.short_term_cache)}")
            print(f"\nBreakdown by component:")
            sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1], reverse=True)
            for component, elapsed_time in sorted_stats:
                percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0.0
                print(f"  {component:20s}: {elapsed_time:8.3f}s ({percentage:5.1f}%)")
            print(f"{'='*60}\n")
        
        metrics['_timing_stats'] = timing_stats
        metrics['_total_time'] = total_time
        metrics['_step_count'] = step_id
        
        return metrics
