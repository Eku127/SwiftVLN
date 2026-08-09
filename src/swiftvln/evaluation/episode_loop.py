# Copyright (c) Alibaba, Inc. and its affiliates.
"""SwiftVLN evaluation orchestration.

The evaluator wires together environment services, a windowed inference session,
and optional diagnostics. The episode state machine lives in
``EnvironmentEpisodeLoop`` so prompt/history logic stays independent from
environment reset/step/video concerns.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from swiftvln.backends.base import EnvWrapper
from swiftvln.backends.factory import create_backend
from swiftvln.evaluation.diagnostics import DiagnosticsObserver
from swiftvln.evaluation.inference import SwiftVLNInferenceSession


def _next_window_start(
    *,
    step_id: int,
    window_start_step: int,
    num_frames: int,
    stride: int,
) -> Optional[int]:
    """Return the next aligned window start once its boundary is reached."""
    if step_id < window_start_step + num_frames:
        return None
    return window_start_step + stride


def _new_timing_stats() -> Dict[str, float]:
    return {
        "init": 0.0,
        "collect_observation": 0.0,
        "encode_current": 0.0,
        "build_embeds": 0.0,
        "model_generate": 0.0,
        "decode": 0.0,
        "parse_actions": 0.0,
        "update_cache": 0.0,
        "window_slide": 0.0,
        "visualization": 0.0,
        "env_step": 0.0,
        "satnav_topdown": 0.0,
        "video_save": 0.0,
    }


class EnvironmentEpisodeLoop:
    """Run one reset/predict/step episode against an environment wrapper."""

    def __init__(self, evaluator: "SwiftVLNEvaluator"):
        self.evaluator = evaluator

    @torch.no_grad()
    def run(
        self,
        env_wrapper: EnvWrapper,
        episode: Any,
    ) -> Dict[str, Any]:
        evaluator = self.evaluator
        environment = evaluator.environment
        session = evaluator.inference
        diagnostics = evaluator.diagnostics
        timing_stats = _new_timing_stats()
        episode_start_time = time.time()

        init_start = time.time()
        evaluator.model.eval()
        environment.set_seed()
        session.reset()
        observations = env_wrapper.reset(episode)
        instruction = env_wrapper.get_instruction(episode)
        episode_id = episode.episode_id
        timing_stats["init"] = time.time() - init_start

        scene_id = self._scene_id(episode)
        rgb_list: List[Image.Image] = []
        action_sequence: List[int] = []
        step_id = 0
        video_state = environment.create_video_state()
        try:
            while not env_wrapper.episode_over and step_id < env_wrapper.max_steps:
                collect_start = time.time()
                rgb = env_wrapper.get_rgb(observations)
                current_image = Image.fromarray(rgb).convert("RGB")
                rgb_list.append(current_image)
                current_pose = session.observe_pose()
                timing_stats["collect_observation"] += time.time() - collect_start

                if not action_sequence:
                    if step_id == 0 and session.system_prompt_setting == "initial":
                        initial_features = session.encode_initial_view(
                            current_image,
                            current_pose,
                        )
                        diagnostics.initial_frame_encoded(
                            episode_id,
                            initial_features,
                        )

                    next_window_start = _next_window_start(
                        step_id=step_id,
                        window_start_step=session.window_start_step,
                        num_frames=session.num_frames,
                        stride=session.stride,
                    )
                    while next_window_start is not None:
                        slide_start = time.time()
                        session.slide_window(
                            rgb_list,
                            session.pose_history,
                            next_window_start,
                            episode,
                        )
                        timing_stats["window_slide"] += time.time() - slide_start
                        next_window_start = _next_window_start(
                            step_id=step_id,
                            window_start_step=session.window_start_step,
                            num_frames=session.num_frames,
                            stride=session.stride,
                        )
                    if step_id == 0:
                        session.start_first_window(rgb_list, episode)

                    try:
                        action_sequence = session.predict(
                            instruction=instruction,
                            current_image=current_image,
                            current_pose=current_pose,
                            step_id=step_id,
                            parse_actions=environment.parse_actions,
                            timing_stats=timing_stats,
                        )
                    except Exception as exc:
                        print(f"[Warning] Generation failed at step {step_id}: {exc}")
                        traceback.print_exc()
                        action_sequence = []

                    if not action_sequence:
                        action_sequence = [0]

                if environment.save_video:
                    visualization_start = time.time()
                    environment.capture_before_step(
                        video_state,
                        observations=observations,
                        instruction=instruction,
                        env_wrapper=env_wrapper,
                        episode=episode,
                        rgb=rgb,
                    )
                    timing_stats["visualization"] += time.time() - visualization_start

                step_start = time.time()
                action = action_sequence.pop(0)
                observations, _ = env_wrapper.step(action)
                session.record_action(action)
                timing_stats["env_step"] += time.time() - step_start
                step_id += 1
                if environment.save_video and environment.captures_after_step:
                    topdown_start = time.time()
                    environment.capture_after_step(
                        video_state,
                        env_wrapper=env_wrapper,
                        episode=episode,
                        action=action,
                        step_id=step_id,
                        rgb_fallback=rgb,
                    )
                    timing_stats["satnav_topdown"] += time.time() - topdown_start

            metrics = env_wrapper.get_metrics()
        except Exception as exc:
            metrics = self._failure_metrics(env_wrapper, exc)
        finally:
            if environment.save_video:
                video_start = time.time()
                environment.save_episode_video(
                    video_state,
                    episode_id=episode_id,
                    instruction=instruction,
                    metrics=metrics,
                )
                timing_stats["video_save"] = time.time() - video_start

        total_time = time.time() - episode_start_time
        diagnostics.print_timing(
            episode_id,
            scene_id,
            session,
            timing_stats,
            total_time,
            step_id,
        )
        diagnostics.initial_episode_summary(episode_id, session, step_id)

        metrics["_timing_stats"] = timing_stats
        metrics["_total_time"] = total_time
        metrics["_step_count"] = step_id
        return metrics

    @staticmethod
    def _scene_id(episode: Any) -> str:
        raw_scene_id = getattr(episode, "scene_id", None)
        if raw_scene_id is None:
            return "unknown"
        return raw_scene_id.split("/")[-2] if "/" in raw_scene_id else raw_scene_id

    @staticmethod
    def _failure_metrics(env_wrapper: EnvWrapper, exc: Exception) -> Dict[str, Any]:
        try:
            metrics = env_wrapper.get_metrics()
        except Exception:
            metrics = {
                "success": 0.0,
                "spl": 0.0,
                "distance_to_goal": float("inf"),
                "oracle_success": 0.0,
            }
        metrics["success"] = 0.0
        metrics["spl"] = 0.0
        metrics["_error"] = str(exc)
        return metrics


class SwiftVLNEvaluator:
    """Compose environment services with SwiftVLN inference and episode flow."""

    def __init__(
        self,
        config_path: str,
        model: Any,
        processor: Any,
        args: Any,
        env_type: str = "habitat",
    ):
        self.args = args
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.environment = create_backend(env_type, config_path, args)
        self.diagnostics = DiagnosticsObserver(args, self.environment.output_path)
        self.environment.set_seed()
        self.inference = SwiftVLNInferenceSession(
            model=model,
            processor=processor,
            args=args,
            config=self.environment.config,
            environment_spec=self.environment.spec,
            device=self.device,
            num_history=getattr(args, "num_history", 8),
            num_future_steps=getattr(args, "num_future_steps", 4),
            diagnostics=self.diagnostics,
        )
        self.episode_loop = EnvironmentEpisodeLoop(self)

    def create_environment(self) -> EnvWrapper:
        """Construct the configured environment wrapper for this evaluator."""
        return self.environment.create_wrapper()

    def eval_episode(
        self,
        env_wrapper: EnvWrapper,
        episode: Any,
        env_idx: int = 0,
    ) -> Dict[str, Any]:
        del env_idx
        return self.episode_loop.run(env_wrapper, episode)
