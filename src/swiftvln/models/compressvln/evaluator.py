# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Evaluator - 支持历史帧压缩的多轮对话评估器

继承 BaseVLNEvaluator，使用多轮对话格式进行流式评估，
同时支持评估时的历史帧压缩（与训练一致）。
"""

import os
import time
import torch
import random
from typing import Any, Dict, List, Tuple
from PIL import Image

# Import from common module
try:
    from swiftvln.common import (
        BaseVLNEvaluator,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        TrajectoryRecorder,
    )
except ImportError:
    from ..common import (
        BaseVLNEvaluator,
        EnvWrapper,
        DEFAULT_IMAGE_TOKEN,
        TrajectoryRecorder,
    )

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


class CompressVLNEvaluator(BaseVLNEvaluator):
    """
    CompressVLN 评估器
    
    继承 BaseVLNEvaluator，使用多轮对话格式进行流式评估，
    同时在评估时应用历史帧压缩（与训练一致）。
    
    关键特性:
    - 多轮对话格式（与 StreamVLN 相同）
    - 历史帧压缩（通过 template._post_encode 实现）
    - 窗口管理（num_frames 窗口大小）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # CompressVLN-specific: num_frames for window management
        self.num_frames = getattr(self.args, 'num_frames', 32)
        
        # CompressVLN-specific: compress_stride for history compression
        self.compress_stride = getattr(self.args, 'compress_stride', 2)
        
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
    ) -> Tuple[List[Dict], List[Image.Image], int]:
        """
        构建完整的多轮对话消息，并返回历史图片数量。
        
        Returns:
            Tuple of (messages, images, num_history_images)
        """
        # 1. 采样历史帧（当前窗口之前）
        history_images = []
        if window_start > 0:
            history_indices = self.sample_history_indices(window_start, self.num_history)
            history_images = [rgb_list[i] for i in history_indices]
        
        num_history_images = len(history_images)
        
        # 2. 构建系统 prompt
        system_prompt = self.build_system_prompt(
            instruction,
            num_history_images=num_history_images
        )
        
        # 3. 系统消息
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # 4. 构建当前窗口的多轮对话
        window_images = []
        steps_in_window = current_step - window_start + 1
        num_turns = (steps_in_window - 1) // self.num_future_steps + 1
        
        for i in range(num_turns):
            step_idx = window_start + i * self.num_future_steps
            
            if i < len(window_conjunctions):
                conjunction = window_conjunctions[i]
            else:
                conjunction = random.choice(self.conjunctions)
            
            messages.append({'role': 'user', 'content': f"{conjunction}{DEFAULT_IMAGE_TOKEN}."})
            window_images.append(rgb_list[step_idx])
            
            if i < num_turns - 1:
                if i < len(window_responses):
                    messages.append({'role': 'assistant', 'content': window_responses[i]})
                else:
                    messages.append({'role': 'assistant', 'content': '↑'})
        
        # 5. 合并图片：历史 + 当前窗口
        images = history_images + window_images
        
        return messages, images, num_history_images

    @torch.no_grad()
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        评估单个 episode，使用多轮对话格式并应用历史帧压缩。
        
        关键修改：在调用 template.encode() 时传递 num_history_images，
        然后通过 _post_encode 应用压缩。
        
        Args:
            env_wrapper: 环境包装器
            episode: Episode 对象
            env_idx: 环境索引
            
        Returns:
            评估指标字典
        """
        # Initialize timing statistics
        timing_stats = {
            'init': 0.0,
            'collect_observation': 0.0,
            'build_messages': 0.0,
            'encode': 0.0,
            'post_encode': 0.0,
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
        
        rgb_list = []
        action_seq = []
        step_id = 0
        
        trajectory_recorder = TrajectoryRecorder()
        vis_frames = []
        rgb_frames = []
        topdown_frames = []
        
        window_start = 0
        window_responses = []
        window_conjunctions = []
        
        max_steps = env_wrapper.max_steps
        
        # Track exception for graceful handling after video save
        episode_exception = None
        
        try:
            while not env_wrapper.episode_over and step_id < max_steps:
                # 1. 收集观察
                t0 = time.time()
                rgb = env_wrapper.get_rgb(observations)
                current_img = Image.fromarray(rgb).convert('RGB')
                rgb_list.append(current_img)
                timing_stats['collect_observation'] += time.time() - t0
                
                # 2. 如果动作队列为空，预测新动作
                if len(action_seq) == 0:
                    if step_id % self.num_frames == 0:
                        window_start = step_id
                        window_responses = []
                        window_conjunctions = []
                    
                    try:
                        # 构建消息并获取历史图片数量
                        t0 = time.time()
                        messages, images, num_history_images = self.build_complete_messages(
                            instruction=instruction,
                            rgb_list=rgb_list,
                            window_start=window_start,
                            current_step=step_id,
                            window_responses=window_responses,
                            window_conjunctions=window_conjunctions,
                        )
                        
                        # 记录 conjunction
                        last_user_content = messages[-1]['content']
                        for conj in self.conjunctions:
                            if last_user_content.startswith(conj):
                                window_conjunctions.append(conj)
                                break
                        else:
                            window_conjunctions.append(self.conjunctions[0])
                        timing_stats['build_messages'] += time.time() - t0
                        
                        # ========== 关键修改 ==========
                        # 传递 num_history_images 给 template.encode()
                        t0 = time.time()
                        encoded = self.template.encode({
                            'messages': messages,
                            'images': images,
                            'num_history_images': num_history_images,  # 关键！
                        })
                        
                        # 准备 inputs 用于 _post_encode
                        input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                        
                        # 构建 post_encode 的输入
                        post_encode_inputs = {
                            'input_ids': input_ids,
                            '_history_image_count': num_history_images,
                            '_current_image_count': len(images) - num_history_images,
                            '_compress_stride': self.template.compress_stride,
                        }
                        
                        if 'pixel_values' in encoded:
                            pixel_values = encoded['pixel_values']
                            if isinstance(pixel_values, torch.Tensor):
                                post_encode_inputs['pixel_values'] = pixel_values.to(self.device).to(self.model.dtype)
                            else:
                                post_encode_inputs['pixel_values'] = torch.tensor(pixel_values).to(self.device).to(self.model.dtype)
                        
                        if 'image_grid_thw' in encoded:
                            grid_thw = encoded['image_grid_thw']
                            if isinstance(grid_thw, torch.Tensor):
                                post_encode_inputs['image_grid_thw'] = grid_thw.to(self.device)
                            else:
                                post_encode_inputs['image_grid_thw'] = torch.tensor(grid_thw).to(self.device)
                        timing_stats['encode'] += time.time() - t0
                        
                        # 调用 _post_encode 获取 inputs_embeds（关键！）
                        t0 = time.time()
                        post_encoded = self.template._post_encode(self.model, post_encode_inputs)
                        
                        inputs_embeds = post_encoded['inputs_embeds']
                        seq_len = inputs_embeds.shape[1]
                        
                        # 创建 attention_mask
                        attention_mask = torch.ones(
                            (1, seq_len), 
                            dtype=torch.long, 
                            device=inputs_embeds.device
                        )
                        
                        # 创建 dummy input_ids
                        pad_token_id = self.processor.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = self.processor.tokenizer.eos_token_id
                        dummy_input_ids = torch.full(
                            (1, seq_len),
                            pad_token_id,
                            dtype=torch.long,
                            device=inputs_embeds.device
                        )
                        timing_stats['post_encode'] += time.time() - t0
                        
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
                        
                        window_responses.append(output_text)
                        
                        t0 = time.time()
                        action_seq = self.parse_actions(output_text)
                        timing_stats['parse_actions'] += time.time() - t0
                        
                        if getattr(self.args, 'verbose', False):
                            print(f"Step {step_id}: history={num_history_images}, output={output_text}")
                        
                    except Exception as e:
                        print(f"[Warning] Generation failed at step {step_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        action_seq = []
                    
                    if not action_seq:
                        action_seq = [0]
                
                # 3. 视频帧收集
                if self.save_video:
                    t0 = time.time()
                    if self.env_type == "habitat":
                        frame = self._collect_habitat_frame(observations, instruction, env_wrapper)
                        if frame is not None:
                            vis_frames.append(frame)
                    elif self.env_type == "satnav":
                        rgb_frames.append(rgb.copy())
                    timing_stats['visualization'] += time.time() - t0
                
                # 4. 执行动作
                t0 = time.time()
                action = action_seq.pop(0)
                observations, _ = env_wrapper.step(action)
                timing_stats['env_step'] += time.time() - t0
                step_id += 1
                
                # 5. SatNav 视频帧
                if self.save_video and self.env_type == "satnav":
                    t0 = time.time()
                    self._collect_satnav_topdown(
                        env_wrapper, episode, action, step_id,
                        topdown_frames, rgb
                    )
                    timing_stats['satnav_topdown'] += time.time() - t0
                
                # 6. 轨迹记录（Habitat）
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
            
            # 误差分析（Habitat）
            if self.env_type == "habitat":
                t0 = time.time()
                error_analysis = self._analyze_trajectory_errors(trajectory_recorder, episode, metrics)
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
        
        metrics['_timing_stats'] = timing_stats
        metrics['_total_time'] = total_time
        metrics['_step_count'] = step_id
        
        return metrics
