# Copyright (c) Alibaba, Inc. and its affiliates.
"""Prompt tokenization and multimodal embedding assembly for inference."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from PIL import Image

from swiftvln.modeling.constants import (
    CURRENT_IMAGE_TOKEN,
    HISTORY_MEMORY_TOKEN,
)


class PromptConstructionMixin:
    """Build training-aligned prompts from session-owned cached features."""

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Resolve the Qwen-family text embedding layer."""
        base_model = self.model
        if hasattr(base_model, "model") and hasattr(
            base_model.model,
            "embed_tokens",
        ):
            return base_model.model.embed_tokens(input_ids)
        if hasattr(base_model, "model") and hasattr(
            base_model.model,
            "language_model",
        ):
            return base_model.model.language_model.embed_tokens(input_ids)
        raise ValueError("Cannot find embed_tokens in model")

    def _build_system_prompt_ids(
        self,
        instruction: str,
        history_token_counts: List[int],
        initial_token_count: int = 0,
    ) -> torch.Tensor:
        """Tokenize the system prompt and its unified memory block."""
        system_prompt = self.environment_spec.format_prompt(instruction)
        if initial_token_count > 0:
            initial_tokens = CURRENT_IMAGE_TOKEN * initial_token_count
            initial_str = (
                f"<|vision_start|>{initial_tokens}<|vision_end|>"
            )
            system_prompt += (
                " This is your initial observation at the starting point of "
                f"this journey: {initial_str}."
            )

        if history_token_counts:
            total_history_tokens = sum(history_token_counts)
            history_str = (
                f"<|vision_start|>"
                f"{HISTORY_MEMORY_TOKEN * total_history_tokens}"
                f"<|vision_end|>"
            )
            if self.memory_method == "map":
                system_prompt += (
                    f" These are your explored map memories: {history_str}."
                )
            else:
                system_prompt += (
                    " These are your historical observations: "
                    f"{history_str}."
                )

        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        self.diagnostics.map_system_prompt(
            self,
            inputs,
            history_token_counts,
            initial_token_count,
            instruction,
            system_prompt,
        )
        return inputs["input_ids"].to(self.device)

    def _build_user_turn_ids(
        self,
        conjunction: str,
        current_token_count: int,
        add_generation_prompt: bool = False,
    ) -> torch.Tensor:
        """Tokenize a user turn containing the current observation."""
        current_tokens = CURRENT_IMAGE_TOKEN * current_token_count
        content = (
            f"{conjunction}<|vision_start|>"
            f"{current_tokens}<|vision_end|>."
        )
        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        self.diagnostics.map_user_prompt(
            self,
            inputs,
            current_token_count,
            add_generation_prompt,
            content,
        )
        return inputs["input_ids"].to(self.device)

    def _build_assistant_turn_ids(self, response: str) -> torch.Tensor:
        """Tokenize one completed assistant response."""
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
        """Assemble system, overlap, completed-turn, and current embeddings."""
        current_vit_features, current_grid_thw = self._encode_frame(
            current_image,
            pose=current_pose,
        )
        current_token_count = current_vit_features.shape[0]

        history_token_counts = [
            history[0].shape[0] for history in self.history_cache
        ]
        initial_token_count = (
            self.initial_features.shape[0]
            if self.initial_features is not None
            else 0
        )
        system_ids = self._build_system_prompt_ids(
            instruction,
            history_token_counts,
            initial_token_count=initial_token_count,
        )
        system_embeds = self._get_text_embeddings(system_ids)

        if self.initial_features is not None:
            initial_positions = (
                system_ids[0] == self.current_image_token_id
            ).nonzero(as_tuple=True)[0]
            initial_count = self.initial_features.shape[0]
            if len(initial_positions) >= initial_count:
                initial_positions_slice = initial_positions[:initial_count]
                initial_embeds = self.initial_features.to(
                    system_embeds.device,
                    system_embeds.dtype,
                )
                system_embeds[0, initial_positions_slice] = initial_embeds
                self.diagnostics.initial_tokens_injected(
                    initial_count,
                    initial_positions_slice,
                    system_ids.shape[1],
                )
            else:
                print(
                    "[INITIAL WARNING] Not enough <current_image> positions in "
                    f"system prompt! Found {len(initial_positions)}, "
                    f"need {initial_count}"
                )

        history_injection_ok = True
        history_injection_msg = "no_history"
        history_embeds = None
        if self.history_cache:
            history_positions = (
                system_ids[0] == self.history_memory_token_id
            ).nonzero(as_tuple=True)[0]
            history_embeds = torch.cat(
                [history[0] for history in self.history_cache],
                dim=0,
            )
            total_history_tokens = history_embeds.shape[0]
            if len(history_positions) >= total_history_tokens:
                history_positions_slice = history_positions[
                    :total_history_tokens
                ]
                history_embeds = history_embeds.to(
                    system_embeds.device,
                    system_embeds.dtype,
                )
                system_embeds[0, history_positions_slice] = history_embeds
                history_injection_msg = (
                    f"ok positions=[{history_positions_slice[0].item()}.."
                    f"{history_positions_slice[-1].item()}]"
                )
            else:
                history_injection_ok = False
                history_injection_msg = (
                    f"failed positions={len(history_positions)} "
                    f"need={total_history_tokens}"
                )
                self.diagnostics.map_history_injection_warning(
                    history_injection_msg
                )

        embeds_parts = [system_embeds]
        if (
            self.overlap_context is not None
            and self.overlap_context.input_ids is not None
        ):
            overlap_embeds = self._get_text_embeddings(
                self.overlap_context.input_ids
            )
            if self.overlap_context.image_embeds:
                overlap_positions = (
                    self.overlap_context.input_ids[0]
                    == self.current_image_token_id
                ).nonzero(as_tuple=True)[0]
                overlap_image_embeds = torch.cat(
                    self.overlap_context.image_embeds,
                    dim=0,
                )
                total_overlap_tokens = overlap_image_embeds.shape[0]
                if len(overlap_positions) >= total_overlap_tokens:
                    overlap_positions_slice = overlap_positions[
                        :total_overlap_tokens
                    ]
                    overlap_image_embeds = overlap_image_embeds.to(
                        overlap_embeds.device,
                        overlap_embeds.dtype,
                    )
                    overlap_embeds[
                        0,
                        overlap_positions_slice,
                    ] = overlap_image_embeds
            embeds_parts.append(overlap_embeds)

        for turn in self.window_turns:
            turn_user_embeds = self._get_text_embeddings(
                turn.user_input_ids
            )
            turn_positions = (
                turn.user_input_ids[0] == self.current_image_token_id
            ).nonzero(as_tuple=True)[0]
            turn_token_count = turn.image_embed.shape[0]
            if len(turn_positions) >= turn_token_count:
                turn_positions_slice = turn_positions[:turn_token_count]
                turn_image = turn.image_embed.to(
                    turn_user_embeds.device,
                    turn_user_embeds.dtype,
                )
                turn_user_embeds[0, turn_positions_slice] = turn_image
            embeds_parts.append(turn_user_embeds)

            assistant_ids = self._build_assistant_turn_ids(
                turn.assistant_response
            )
            embeds_parts.append(self._get_text_embeddings(assistant_ids))

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
            current_vit = current_vit_features.to(
                new_user_embeds.device,
                new_user_embeds.dtype,
            )
            new_user_embeds[0, current_positions_slice] = current_vit
        embeds_parts.append(new_user_embeds)

        inputs_embeds = torch.cat(embeds_parts, dim=1)
        seq_len = inputs_embeds.shape[1]
        self.diagnostics.map_complete_embeds(
            self,
            system_ids,
            new_user_ids,
            current_vit_features,
            current_token_count,
            history_token_counts,
            initial_token_count,
            history_injection_msg,
            history_embeds,
            history_injection_ok,
            seq_len,
        )
        return (
            inputs_embeds,
            seq_len,
            current_vit_features,
            current_grid_thw,
        )


__all__ = ["PromptConstructionMixin"]
