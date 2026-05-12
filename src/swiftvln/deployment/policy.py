from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from swiftvln.common.constants import (
    CURRENT_IMAGE_TOKEN,
    DEFAULT_ACTION_MAP,
    DEFAULT_CONJUNCTIONS,
    HISTORY_MEMORY_TOKEN,
    PROMPT_TEMPLATE_HABITAT,
    PROMPT_TEMPLATE_SATNAV,
)
from swiftvln.common.embedding_enhancement import reconstruct_pose_from_actions
from swiftvln.common.history_processors import HistoryTokenCompressor
from swiftvln.common.history_processors.per_frame import sample_per_frame_history_indices
from swiftvln.deployment.model_resolver import SwiftVLNDeploySpec


DEFAULT_DEPLOY_SEED = 42


@dataclass
class OverlapContext:
    input_ids: Optional[torch.Tensor] = None
    image_embeds: List[torch.Tensor] = field(default_factory=list)


@dataclass
class TurnContext:
    user_input_ids: torch.Tensor
    assistant_response: str
    image_embed: torch.Tensor


@dataclass(frozen=True)
class InferenceResult:
    raw_action_text: str
    actions: List[int]
    step_id: int
    conjunction: str


class SwiftVLNBaselinePolicy:
    """Baseline-only swiftvln deploy policy aligned with eval behavior."""

    def __init__(self, model: Any, processor: Any, spec: SwiftVLNDeploySpec):
        self.model = model
        self.processor = processor
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_type = spec.env_type

        self.num_frames = spec.num_frames
        self.num_future_steps = spec.num_future_steps
        self.num_overlap = spec.num_overlap
        self.num_history = spec.num_history
        self.log_base = spec.log_base
        self.use_random = spec.use_random
        self.compress_stride = spec.compress_stride

        self.stride = self.num_frames - self.num_overlap
        self.turns_per_window = self.num_frames // self.num_future_steps
        self.overlap_turns = self.num_overlap // self.num_future_steps

        self.idx2actions = DEFAULT_ACTION_MAP.copy()
        self.actions2idx = {text: idx for idx, text in self.idx2actions.items()}
        self._action_regex = re.compile("|".join(re.escape(text) for text in self.actions2idx))
        self.conjunctions = DEFAULT_CONJUNCTIONS.copy()
        self.history_memory_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            HISTORY_MEMORY_TOKEN
        )
        self.current_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            CURRENT_IMAGE_TOKEN
        )
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.compressor = HistoryTokenCompressor(stride=self.compress_stride, method="pooling")
        self.has_embed_enhance = (
            hasattr(self.model, "embed_enhance") and not self.model.embed_enhance.is_empty
        )

        self.reset()

    def reset(self) -> None:
        random.seed(DEFAULT_DEPLOY_SEED)
        np.random.seed(DEFAULT_DEPLOY_SEED)
        torch.manual_seed(DEFAULT_DEPLOY_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(DEFAULT_DEPLOY_SEED)
            torch.cuda.manual_seed_all(DEFAULT_DEPLOY_SEED)

        if hasattr(self.model, "reset"):
            self.model.reset(env_num=1)
        elif hasattr(self.model, "reset_for_env"):
            self.model.reset_for_env(0)

        self.history_cache: List[Tuple[torch.Tensor, Optional[List[float]]]] = []
        self.initial_features: Optional[torch.Tensor] = None
        self.overlap_context: Optional[OverlapContext] = None
        self.window_turns: List[TurnContext] = []
        self.window_start_step = 0
        self.current_window_idx = 0
        self.pose_history: List[List[float]] = []
        self.executed_actions: List[int] = [-1]
        self.rgb_list: List[Image.Image] = []

    @property
    def step_id(self) -> int:
        return max(len(self.rgb_list) - 1, 0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "window_start_step": self.window_start_step,
            "current_window_idx": self.current_window_idx,
            "history_cache_size": len(self.history_cache),
            "history_token_counts": [int(item[0].shape[0]) for item in self.history_cache],
            "window_turns": len(self.window_turns),
            "rgb_count": len(self.rgb_list),
            "pose_count": len(self.pose_history),
            "executed_actions": [int(action) for action in self.executed_actions if int(action) >= 0],
        }

    def close(self) -> None:
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        self.model = None
        self.processor = None
        del model
        del processor
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def record_completed_action(self, action: int) -> None:
        self.executed_actions.append(int(action))

    def observe_image(self, image: Image.Image) -> List[float]:
        current_image = image.copy()
        self.rgb_list.append(current_image)
        pose = self._get_pose_from_actions()
        self.pose_history.append(pose)
        return pose

    def parse_actions(self, output: str) -> List[int]:
        matches = self._action_regex.findall(output or "")
        return [self.actions2idx[match] for match in matches]

    def infer_actions(self, instruction: str, current_image: Image.Image) -> InferenceResult:
        step_id = self.step_id
        current_pose = self.pose_history[-1] if self.pose_history else self._get_pose_from_actions()

        if step_id > 0 and step_id % self.stride == 0 and step_id >= self.num_frames:
            new_window_start = step_id - self.num_overlap
            self._slide_window(new_window_start)
        elif step_id == 0:
            self.window_start_step = 0
            self.current_window_idx = 0

        conjunction = random.choice(self.conjunctions)
        inputs_embeds, seq_len, current_vit, _ = self._build_complete_prompt_embeds(
            instruction=instruction,
            current_image=current_image,
            conjunction=conjunction,
            current_pose=current_pose,
        )

        attention_mask = torch.ones(
            (1, seq_len),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        dummy_input_ids = torch.full(
            (1, seq_len),
            pad_token_id,
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0][seq_len:]
        raw_action_text = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        actions = self.parse_actions(raw_action_text)
        if not actions:
            actions = [0]

        self._save_turn_to_window(
            conjunction=conjunction,
            response=raw_action_text,
            vit_features=current_vit,
        )
        return InferenceResult(
            raw_action_text=raw_action_text,
            actions=actions,
            step_id=step_id,
            conjunction=conjunction,
        )

    def _get_pose_from_actions(self) -> List[float]:
        step_size = 10.0 if self.env_type == "satnav" else 0.25
        turn_angle = 15.0 if self.env_type == "satnav" else 30.0
        poses = reconstruct_pose_from_actions(
            self.executed_actions,
            step_size=step_size,
            turn_angle=turn_angle,
        )
        if poses.shape[0] == 0:
            return [0.0, 0.0, 0.0, 1.0]
        return poses[-1].tolist()

    def _encode_frame(
        self,
        image: Image.Image,
        pose: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        media_inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            vit_features = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            if self.has_embed_enhance:
                _, h, w = image_grid_thw[0].tolist()
                h_m = int(h) // self.merge_size
                w_m = int(w) // self.merge_size
                vit_features = self.model.embed_enhance(vit_features, h_m, w_m, pose=pose)

        return vit_features, image_grid_thw[0]

    def _encode_batch_frames(
        self,
        images: Sequence[Image.Image],
        poses: Optional[Sequence[Optional[List[float]]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not images:
            return [], []

        media_inputs = self.processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values = media_inputs["pixel_values"].to(self.device).type(self.model.dtype)
        image_grid_thw = media_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            all_vit_features = self.model.visual(pixel_values, grid_thw=image_grid_thw)

        merge_length = self.merge_size**2
        features_list: List[torch.Tensor] = []
        grid_thw_list: List[torch.Tensor] = []
        embed_idx = 0

        for index in range(len(images)):
            num_tokens = int(image_grid_thw[index].prod() // merge_length)
            img_features = all_vit_features[embed_idx : embed_idx + num_tokens]
            embed_idx += num_tokens

            if self.has_embed_enhance:
                _, h, w = image_grid_thw[index].tolist()
                h_m = int(h) // self.merge_size
                w_m = int(w) // self.merge_size
                pose = poses[index] if poses is not None and index < len(poses) else None
                with torch.no_grad():
                    img_features = self.model.embed_enhance(img_features, h_m, w_m, pose=pose)

            features_list.append(img_features)
            grid_thw_list.append(image_grid_thw[index])

        return features_list, grid_thw_list

    def _compress_features(self, features: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_after_merge = grid_thw.clone()
        grid_after_merge[1] = grid_after_merge[1] // self.merge_size
        grid_after_merge[2] = grid_after_merge[2] // self.merge_size
        compressed, _ = self.compressor.compress(
            features,
            grid_after_merge,
            stride=self.compress_stride,
        )
        return compressed

    def _compute_history_cache_per_frame(self, window_start: int) -> None:
        available_history_frames = min(window_start, len(self.rgb_list))
        if self.num_history <= 0 or available_history_frames <= 0:
            self.history_cache = []
            return

        history_indices = sample_per_frame_history_indices(
            num_frames=available_history_frames,
            num_samples=self.num_history,
            log_base=self.log_base,
            use_random=self.use_random,
        )
        if not history_indices:
            self.history_cache = []
            return

        history_images = [self.rgb_list[index] for index in history_indices]
        history_poses = [
            self.pose_history[index] if index < len(self.pose_history) else None
            for index in history_indices
        ]
        features_list, grid_thw_list = self._encode_batch_frames(history_images, poses=history_poses)
        self.history_cache = []
        for features, grid_thw, pose in zip(features_list, grid_thw_list, history_poses):
            compressed = self._compress_features(features, grid_thw)
            self.history_cache.append((compressed, pose))

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_model = self.model
        if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            return base_model.model.embed_tokens(input_ids)
        if hasattr(base_model, "model") and hasattr(base_model.model, "language_model"):
            return base_model.model.language_model.embed_tokens(input_ids)
        raise ValueError("Cannot find embed_tokens in model")

    def _build_system_prompt_ids(
        self,
        instruction: str,
        history_token_counts: List[int],
    ) -> torch.Tensor:
        template = (
            PROMPT_TEMPLATE_SATNAV if self.env_type == "satnav" else PROMPT_TEMPLATE_HABITAT
        )
        system_prompt = template.format(instruction=instruction)
        if history_token_counts:
            total_history_tokens = sum(history_token_counts)
            history_str = (
                f"<|vision_start|>{HISTORY_MEMORY_TOKEN * total_history_tokens}<|vision_end|>"
            )
            system_prompt += f" These are your historical observations: {history_str}."

        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs["input_ids"].to(self.device)

    def _build_user_turn_ids(
        self,
        conjunction: str,
        current_token_count: int,
        add_generation_prompt: bool = False,
    ) -> torch.Tensor:
        current_tokens = CURRENT_IMAGE_TOKEN * current_token_count
        content = f"{conjunction}<|vision_start|>{current_tokens}<|vision_end|>."
        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs["input_ids"].to(self.device)

    def _build_assistant_turn_ids(self, response: str) -> torch.Tensor:
        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": response}],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs["input_ids"].to(self.device)

    def _build_complete_prompt_embeds(
        self,
        instruction: str,
        current_image: Image.Image,
        conjunction: str,
        current_pose: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        current_vit_features, current_grid_thw = self._encode_frame(current_image, pose=current_pose)
        current_token_count = current_vit_features.shape[0]

        history_token_counts = [history[0].shape[0] for history in self.history_cache]
        system_ids = self._build_system_prompt_ids(instruction, history_token_counts)
        system_embeds = self._get_text_embeddings(system_ids)

        if self.history_cache:
            history_positions = (
                system_ids[0] == self.history_memory_token_id
            ).nonzero(as_tuple=True)[0]
            history_embeds = torch.cat([history[0] for history in self.history_cache], dim=0)
            total_history_tokens = history_embeds.shape[0]
            if len(history_positions) >= total_history_tokens:
                history_positions_slice = history_positions[:total_history_tokens]
                history_embeds = history_embeds.to(system_embeds.device, system_embeds.dtype)
                system_embeds[0, history_positions_slice] = history_embeds

        embeds_parts = [system_embeds]

        if self.overlap_context is not None and self.overlap_context.input_ids is not None:
            overlap_embeds = self._get_text_embeddings(self.overlap_context.input_ids)
            if self.overlap_context.image_embeds:
                overlap_positions = (
                    self.overlap_context.input_ids[0] == self.current_image_token_id
                ).nonzero(as_tuple=True)[0]
                overlap_img_embeds = torch.cat(self.overlap_context.image_embeds, dim=0)
                total_overlap_tokens = overlap_img_embeds.shape[0]
                if len(overlap_positions) >= total_overlap_tokens:
                    overlap_positions_slice = overlap_positions[:total_overlap_tokens]
                    overlap_img_embeds = overlap_img_embeds.to(
                        overlap_embeds.device,
                        overlap_embeds.dtype,
                    )
                    overlap_embeds[0, overlap_positions_slice] = overlap_img_embeds
            embeds_parts.append(overlap_embeds)

        for turn in self.window_turns:
            turn_user_embeds = self._get_text_embeddings(turn.user_input_ids)
            turn_positions = (
                turn.user_input_ids[0] == self.current_image_token_id
            ).nonzero(as_tuple=True)[0]
            turn_token_count = turn.image_embed.shape[0]
            if len(turn_positions) >= turn_token_count:
                turn_positions_slice = turn_positions[:turn_token_count]
                turn_img = turn.image_embed.to(turn_user_embeds.device, turn_user_embeds.dtype)
                turn_user_embeds[0, turn_positions_slice] = turn_img
            embeds_parts.append(turn_user_embeds)

            assistant_ids = self._build_assistant_turn_ids(turn.assistant_response)
            assistant_embeds = self._get_text_embeddings(assistant_ids)
            embeds_parts.append(assistant_embeds)

        new_user_ids = self._build_user_turn_ids(
            conjunction,
            current_token_count,
            add_generation_prompt=True,
        )
        new_user_embeds = self._get_text_embeddings(new_user_ids)
        current_positions = (
            new_user_ids[0] == self.current_image_token_id
        ).nonzero(as_tuple=True)[0]
        if len(current_positions) >= current_token_count:
            current_positions_slice = current_positions[:current_token_count]
            current_vit = current_vit_features.to(new_user_embeds.device, new_user_embeds.dtype)
            new_user_embeds[0, current_positions_slice] = current_vit
        embeds_parts.append(new_user_embeds)

        inputs_embeds = torch.cat(embeds_parts, dim=1)
        seq_len = inputs_embeds.shape[1]
        return inputs_embeds, seq_len, current_vit_features, current_grid_thw

    def _save_turn_to_window(
        self,
        conjunction: str,
        response: str,
        vit_features: torch.Tensor,
    ) -> None:
        token_count = vit_features.shape[0]
        user_ids = self._build_user_turn_ids(conjunction, token_count)
        self.window_turns.append(
            TurnContext(
                user_input_ids=user_ids,
                assistant_response=response,
                image_embed=vit_features,
            )
        )

    def _prepare_overlap_context(self) -> None:
        if self.overlap_turns <= 0 or len(self.window_turns) < self.overlap_turns:
            self.overlap_context = None
            return

        overlap_turns = self.window_turns[-self.overlap_turns :]
        ids_parts: List[torch.Tensor] = []
        image_embeds: List[torch.Tensor] = []
        for turn in overlap_turns:
            ids_parts.append(turn.user_input_ids)
            ids_parts.append(self._build_assistant_turn_ids(turn.assistant_response))
            image_embeds.append(turn.image_embed)

        overlap_ids = torch.cat(ids_parts, dim=1)
        self.overlap_context = OverlapContext(input_ids=overlap_ids, image_embeds=image_embeds)

    def _slide_window(self, new_window_start: int) -> None:
        self._prepare_overlap_context()
        self._compute_history_cache_per_frame(new_window_start)
        self.window_turns = []
        self.window_start_step = new_window_start
        self.current_window_idx += 1
