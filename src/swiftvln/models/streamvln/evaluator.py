# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Evaluator for VLN tasks (Habitat and SatNav).

This evaluator implements streaming inference for Visual Language Navigation,
adapted for ms-swift's StreamVLNQwen25VL model (RGB-only, multiple <image> tokens).

Complete Multi-turn Dialogue Inference:
=======================================

Each inference call builds COMPLETE messages matching training format:
1. System prompt (with historical <image> tokens if applicable)
2. All previous user/assistant turns within the current window
3. Current user turn (waiting for model response)

Window Management:
- Every num_frames steps, reset window state and re-sample history
- Within window, accumulate dialogue turns with cached responses
"""

import os
import time
import torch
import random
import warnings
from typing import Any, List, Dict, Tuple
from PIL import Image

# Import from common module
try:
    from swiftvln.common import (
        BaseVLNEvaluator,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        TrajectoryRecorder,
        append_text_to_image,
    )
except ImportError:
    from ..common import (
        BaseVLNEvaluator,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        TrajectoryRecorder,
        append_text_to_image,
    )

from habitat.utils.visualizations.utils import observations_to_image

# Environment-specific prompt templates
# Forward distance differs by environment: habitat=0.25m, satnav=10m
PROMPT_TEMPLATE_HABITAT = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Based on your observations, output a sequence of actions using: "
    "↑ (forward 0.25m), ← (turn left), → (turn right), or STOP (when goal is reached). "
    "Output actions directly without explanation."
)

PROMPT_TEMPLATE_SATNAV = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Based on your observations, output a sequence of actions using: "
    "↑ (forward 10m), ← (turn left), → (turn right), or STOP (when goal is reached). "
    "Output actions directly without explanation."
)


class StreamVLNEvaluator(BaseVLNEvaluator):
    """
    StreamVLN Evaluator implementing multi-turn dialogue streaming inference.
    
    Key features:
    - RGB-only input (no depth/gps/compass)
    - Historical frames represented by multiple <image> tokens
    - Window-based inference with num_frames window size
    - Dialogue format consistent with training (dataset.py)
    - Supports multiple environments (Habitat, SatNav)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # StreamVLN-specific: num_frames for window management
        self.num_frames = getattr(self.args, 'num_frames', 32)
        
        # Prompt templates (same as dataset.py)
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

    def build_system_prompt(self, instruction: str, num_history_images: int = 0) -> str:
        """
        Build system prompt consistent with training (dataset.py).
        
        Args:
            instruction: Navigation instruction text
            num_history_images: Number of historical observation images
            
        Returns:
            System prompt string
        """
        # Select prompt template based on environment type
        if self.env_type == "satnav":
            template = PROMPT_TEMPLATE_SATNAV
        else:  # habitat or default
            template = PROMPT_TEMPLATE_HABITAT
        
        system_prompt = template.format(instruction=instruction)
        
        if num_history_images > 0:
            history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
            system_prompt += f" These are your historical observations: {history_tokens}."
        
        return system_prompt

    def build_complete_messages(
        self,
        instruction: str,
        rgb_list: List[Image.Image],
        window_start: int,
        current_step: int,
        window_responses: List[str],
        window_conjunctions: List[str],
    ) -> Tuple[List[Dict], List[Image.Image]]:
        """
        Build COMPLETE multi-turn messages for inference (matching training format).
        
        Args:
            instruction: Navigation instruction
            rgb_list: All collected RGB images
            window_start: Start index of current window
            current_step: Current step index
            window_responses: Cached assistant responses for previous turns in this window
            window_conjunctions: Cached conjunctions used for previous turns
            
        Returns:
            Tuple of (messages, images)
        """
        # 1. Sample historical frames (from before current window)
        history_images = []
        if window_start > 0:
            history_indices = self.sample_history_indices(window_start, self.num_history)
            history_images = [rgb_list[i] for i in history_indices]
        
        # 2. Build system prompt with history
        system_prompt = self.build_system_prompt(
            instruction,
            num_history_images=len(history_images)
        )
        
        # 3. Start with system message
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # 4. Build multi-turn dialogue for current window
        window_images = []
        
        steps_in_window = current_step - window_start + 1
        num_turns = (steps_in_window - 1) // self.num_future_steps + 1
        
        for i in range(num_turns):
            step_idx = window_start + i * self.num_future_steps
            
            # Get or generate conjunction for this turn
            if i < len(window_conjunctions):
                conjunction = window_conjunctions[i]
            else:
                conjunction = random.choice(self.conjunctions)
            
            # User turn with image
            messages.append({'role': 'user', 'content': f"{conjunction}{DEFAULT_IMAGE_TOKEN}."})
            window_images.append(rgb_list[step_idx])
            
            # Assistant turn (except for the last turn which we're generating)
            if i < num_turns - 1:
                if i < len(window_responses):
                    messages.append({'role': 'assistant', 'content': window_responses[i]})
                else:
                    messages.append({'role': 'assistant', 'content': '↑'})
        
        # 5. Combine images: history + window
        images = history_images + window_images
        
        return messages, images

    @torch.no_grad()
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode using complete multi-turn dialogue inference.
        
        Args:
            env_wrapper: Unified environment wrapper
            episode: Episode to evaluate
            env_idx: Environment index for multi-env support
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize timing statistics
        timing_stats = {
            'init': 0.0,
            'collect_observation': 0.0,
            'build_messages': 0.0,
            'encode': 0.0,
            'model_generate': 0.0,
            'decode': 0.0,
            'parse_actions': 0.0,
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
        self.model.reset_for_env(env_idx)
        
        observations = env_wrapper.reset(episode)
        instruction = env_wrapper.get_instruction(episode)
        episode_id = episode.episode_id
        timing_stats['init'] = time.time() - init_start
        
        if hasattr(episode, 'scene_id'):
            scene_id = episode.scene_id.split('/')[-2] if '/' in episode.scene_id else episode.scene_id
        else:
            scene_id = "unknown"
        
        # State tracking
        rgb_list = []
        action_seq = []
        step_id = 0
        
        # Trajectory recording for error analysis
        trajectory_recorder = TrajectoryRecorder()
        
        # Video generation
        vis_frames = []
        rgb_frames = []
        topdown_frames = []
        
        # Window state for complete multi-turn dialogue
        window_start = 0
        window_responses = []
        window_conjunctions = []
        
        max_steps = env_wrapper.max_steps
        
        # Track exception for graceful handling after video save
        episode_exception = None
        
        try:
            while not env_wrapper.episode_over and step_id < max_steps:
                # 1. Collect observation
                t0 = time.time()
                rgb = env_wrapper.get_rgb(observations)
                current_img = Image.fromarray(rgb).convert('RGB')
                rgb_list.append(current_img)
                timing_stats['collect_observation'] += time.time() - t0
                
                # 2. If action queue is empty, predict new actions
                if len(action_seq) == 0:
                    # Check if we need to start a new window
                    if step_id % self.num_frames == 0:
                        window_start = step_id
                        window_responses = []
                        window_conjunctions = []
                    
                    try:
                        # Build COMPLETE multi-turn messages
                        t0 = time.time()
                        messages, images = self.build_complete_messages(
                            instruction=instruction,
                            rgb_list=rgb_list,
                            window_start=window_start,
                            current_step=step_id,
                            window_responses=window_responses,
                            window_conjunctions=window_conjunctions,
                        )
                        timing_stats['build_messages'] += time.time() - t0
                        
                        # Record the conjunction used
                        last_user_content = messages[-1]['content']
                        for conj in self.conjunctions:
                            if last_user_content.startswith(conj):
                                window_conjunctions.append(conj)
                                break
                        else:
                            window_conjunctions.append(self.conjunctions[0])
                        
                        # Encode complete messages with all images
                        t0 = time.time()
                        encoded = self.template.encode({'messages': messages, 'images': images})
                        input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                        
                        model_inputs = {'input_ids': input_ids}
                        
                        if 'pixel_values' in encoded:
                            pixel_values = encoded['pixel_values']
                            if isinstance(pixel_values, torch.Tensor):
                                model_inputs['pixel_values'] = pixel_values.to(self.device).to(self.model.dtype)
                            else:
                                model_inputs['pixel_values'] = torch.tensor(pixel_values).to(self.device).to(self.model.dtype)
                        
                        if 'image_grid_thw' in encoded:
                            grid_thw = encoded['image_grid_thw']
                            if isinstance(grid_thw, torch.Tensor):
                                model_inputs['image_grid_thw'] = grid_thw.to(self.device)
                            else:
                                model_inputs['image_grid_thw'] = torch.tensor(grid_thw).to(self.device)
                        timing_stats['encode'] += time.time() - t0
                        
                        # Generate
                        t0 = time.time()
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=64,
                            do_sample=False,
                            use_cache=True,
                        )
                        timing_stats['model_generate'] += time.time() - t0
                        
                        # Decode
                        t0 = time.time()
                        generated_ids = outputs[0][input_ids.shape[1]:]
                        output_text = self.processor.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True
                        ).strip()
                        timing_stats['decode'] += time.time() - t0
                        
                        # Cache response
                        window_responses.append(output_text)
                        
                        # Parse actions
                        t0 = time.time()
                        action_seq = self.parse_actions(output_text)
                        timing_stats['parse_actions'] += time.time() - t0
                        
                        if getattr(self.args, 'verbose', False):
                            print(f"Step {step_id}: {output_text} -> {action_seq}")
                        
                    except Exception as e:
                        print(f"[Warning] Generation failed at step {step_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        action_seq = []
                    
                    if not action_seq:
                        action_seq = [0]
                
                # 3. Collect visualization frame
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
                
                # 5. Collect SatNav topdown frame AFTER step
                if self.save_video and self.env_type == "satnav":
                    t0 = time.time()
                    self._collect_satnav_topdown(
                        env_wrapper, episode, action, step_id,
                        topdown_frames, rgb
                    )
                    timing_stats['satnav_topdown'] += time.time() - t0
                
                # 6. Record agent position for error analysis (Habitat only)
                if self.env_type == "habitat":
                    t0 = time.time()
                    try:
                        habitat_env = env_wrapper.env
                        agent_state = habitat_env.sim.get_agent_state()
                        trajectory_recorder.add_step(agent_state.position)
                    except Exception:
                        pass
                    timing_stats['trajectory_record'] += time.time() - t0
            
            # Get final metrics (only if no exception)
            metrics = env_wrapper.get_metrics()
            
            # Perform error analysis (Habitat only)
            if self.env_type == "habitat":
                t0 = time.time()
                error_analysis = self._analyze_trajectory_errors(
                    trajectory_recorder, episode, metrics
                )
                metrics.update(error_analysis)
                timing_stats['error_analysis'] = time.time() - t0
        
        except Exception as e:
            # Store exception info for logging
            episode_exception = e
            # Try to get metrics even on error (contains last known distance_to_goal)
            try:
                metrics = env_wrapper.get_metrics()
            except Exception:
                # Fallback metrics if env_wrapper.get_metrics() also fails
                metrics = {
                    'success': 0.0,
                    'spl': 0.0,
                    'distance_to_goal': float('inf'),
                    'oracle_success': 0.0,
                }
            # Ensure failure metrics
            metrics['success'] = 0.0
            metrics['spl'] = 0.0
            metrics['_error'] = str(e)
        
        finally:
            # Always save video (even on error) for debugging
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
            print(f"\nBreakdown by component:")
            sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1], reverse=True)
            for component, elapsed_time in sorted_stats:
                percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0.0
                print(f"  {component:20s}: {elapsed_time:8.3f}s ({percentage:5.1f}%)")
            print(f"{'='*60}\n")
        
        # Attach timing statistics to metrics
        metrics['_timing_stats'] = timing_stats
        metrics['_total_time'] = total_time
        metrics['_step_count'] = step_id
        
        return metrics


# Backward compatibility alias
VLNEvaluator = StreamVLNEvaluator


def run_evaluation(
    model_path: str,
    habitat_config_path: str,
    eval_split: str = 'val_unseen',
    num_frames: int = 32,
    num_history: int = 8,
    num_future_steps: int = 4,
    output_dir: str = './results/eval',
    save_video: bool = False,
):
    """
    Run VLN evaluation with the given parameters.
    """
    import argparse
    
    args = argparse.Namespace(
        model_path=model_path,
        habitat_config_path=habitat_config_path,
        eval_split=eval_split,
        num_frames=num_frames,
        num_history=num_history,
        num_future_steps=num_future_steps,
        output_dir=output_dir,
        save_video=save_video,
    )
    
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateType
    
    model, processor = get_model_tokenizer(
        model_id_or_path=model_path,
        model_type='streamvln_qwen2_5_vl',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    template = get_template(
        template_type=TemplateType.qwen2_5_vl,
        processor=processor,
        model=model
    )
    
    model.reset(env_num=1)
    
    evaluator = StreamVLNEvaluator(
        config_path=habitat_config_path,
        model=model,
        processor=processor,
        template=template,
        args=args
    )
    
    return evaluator
