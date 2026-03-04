# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Evaluator - 支持历史帧压缩的单轮对话评估器

继承 BaseVLNEvaluator，使用单轮对话格式进行自回归式评估:
1. 单轮对话格式（与训练一致）
2. 评估时的历史帧压缩
"""

import os
import time
import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from PIL import Image

from swiftvln.common import (
    BaseVLNEvaluator,
    EnvWrapper,
    DEFAULT_IMAGE_TOKEN,
    TrajectoryRecorder,
)


class MonoVLNEvaluator(BaseVLNEvaluator):
    """
    MonoVLN 评估器
    
    继承 BaseVLNEvaluator，使用单轮对话格式进行自回归式评估：
    - 每步输入：历史帧 + 当前观测
    - 输出：num_future_steps 个 action
    - 格式与训练时的 MonoVLNDataset 一致
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # MonoVLN-specific: compress_stride for history compression
        self.compress_stride = getattr(self.args, 'compress_stride', 2)
    
    def build_single_turn_messages(
        self,
        instruction: str,
        history_images: List[Image.Image],
        current_image: Image.Image,
    ) -> Tuple[List[Dict], List[Image.Image], int]:
        """
        构建单轮对话消息（与训练格式一致）
        
        Args:
            instruction: 导航指令
            history_images: 历史观测图像列表
            current_image: 当前观测图像
            
        Returns:
            Tuple of (messages, images, num_history_images)
        """
        num_history_images = len(history_images)
        
        # Build user content (matching dataset format)
        history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
        
        # Action definition with specific movement parameters
        action_definition = (
            "You are an intelligent navigation robot. "
            "Your goal is to navigate to a target location based on the instruction.\n"
            "Available Actions:\n"
            "- ↑ (Forward): Move forward 0.25 meters.\n"
            "- ← (Left): Turn left 15 degrees.\n"
            "- → (Right): Turn right 15 degrees.\n"
            "- STOP: Use this ONLY when you have reached the goal."
        )
        
        if num_history_images > 0:
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Trajectory History\n"
                f"The following tokens represent your past views and actions:\n"
                f"{history_tokens}\n\n"
                f"### Current View\n"
                f"Current Observation: {DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"Based on the history and current view, predict the next {self.num_future_steps} actions sequence (e.g., ↑, ↑, →, STOP):"
            )
        else:
            # First observation at the starting point
            user_content = (
                f"{action_definition}\n\n"
                f"### Navigation Task\n"
                f"Instruction: {instruction}\n\n"
                f"### Current View\n"
                f"Current Observation: {DEFAULT_IMAGE_TOKEN}\n\n"
                f"### Prediction\n"
                f"You are at the starting point. Analyze the instruction and the current view. "
                f"Predict the next {self.num_future_steps} actions sequence:"
            )
        
        messages = [{'role': 'user', 'content': user_content}]
        
        # Images: history first, then current
        images = history_images + [current_image]
        
        return messages, images, num_history_images
    
    def sample_history_images(
        self,
        rgb_list: List[Image.Image],
        current_step: int,
    ) -> List[Image.Image]:
        """
        从已收集的观测中采样历史帧
        
        Args:
            rgb_list: 所有已收集的 RGB 图像
            current_step: 当前步骤索引
            
        Returns:
            采样的历史图像列表
        """
        if current_step <= 0:
            return []
        
        available_count = current_step
        
        if available_count <= self.num_history:
            # Use all available history
            return rgb_list[:available_count]
        else:
            # Uniform sampling
            indices = np.linspace(0, available_count - 1, self.num_history, dtype=int)
            return [rgb_list[i] for i in indices]
    
    @torch.no_grad()
    def eval_episode(self, env_wrapper: EnvWrapper, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        评估单个 episode，使用单轮对话格式
        
        每步：
        1. 收集当前观测
        2. 采样历史帧
        3. 构建单轮对话消息
        4. 生成 action 序列
        5. 执行 action
        
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
        
        rgb_list = []  # 所有已收集的观测
        action_seq = []  # 待执行的 action 队列
        step_id = 0
        
        trajectory_recorder = TrajectoryRecorder()
        vis_frames = []
        rgb_frames = []
        topdown_frames = []
        
        max_steps = env_wrapper.max_steps
        
        while not env_wrapper.episode_over and step_id < max_steps:
            # 1. 收集观察
            t0 = time.time()
            rgb = env_wrapper.get_rgb(observations)
            current_img = Image.fromarray(rgb).convert('RGB')
            rgb_list.append(current_img)
            timing_stats['collect_observation'] += time.time() - t0
            
            # 2. 如果动作队列为空，预测新动作
            if len(action_seq) == 0:
                try:
                    # 采样历史帧
                    t0 = time.time()
                    history_images = self.sample_history_images(rgb_list, step_id)
                    
                    # 构建单轮对话消息
                    messages, images, num_history_images = self.build_single_turn_messages(
                        instruction=instruction,
                        history_images=history_images,
                        current_image=current_img,
                    )
                    timing_stats['build_messages'] += time.time() - t0
                    
                    # 传递 num_history_images 给 template.encode()
                    t0 = time.time()
                    encoded = self.template.encode({
                        'messages': messages,
                        'images': images,
                        'num_history_images': num_history_images,
                    })
                    
                    # 准备 inputs 用于 _post_encode
                    input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                    
                    post_encode_inputs = {
                        'input_ids': input_ids,
                        '_history_image_count': num_history_images,
                        '_current_image_count': 1,  # MonoVLN 只有 1 帧当前图像
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
                    
                    # 调用 _post_encode 获取 inputs_embeds
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
                    action_seq = [0]  # Default to STOP
            
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
        
        # 保存视频
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
