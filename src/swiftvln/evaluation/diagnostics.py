"""Optional evaluation diagnostics kept outside the inference and episode loop."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from swiftvln.evaluation.inference import SwiftVLNInferenceSession


def _debug_enabled() -> bool:
    return bool(os.environ.get("SWIFTVLN_DEBUG"))


def _debug_rank() -> int:
    raw = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(raw)
    except ValueError:
        return 0


def _preview_text(text: str, limit: int = 260) -> str:
    normalized = str(text).replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


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
        sample = ", ".join(f"{value:.4f}" for value in flat[:sample_limit].tolist())
    return (
        f"shape={tuple(tensor.shape)} mean={mean:.6f} std={std:.6f} "
        f"sum={checksum:.6f} abs_sum={abs_checksum:.6f} l2={l2:.6f} "
        f"sample=[{sample}]"
    )


class DiagnosticsObserver:
    """Collect optional map, landmark, initial-view, and timing diagnostics."""

    def __init__(self, args: Any, output_path: str):
        self.args = args
        self.output_path = output_path
        self._initial_eval_count = 0
        self._map_eval_count = 0
        self._map_prompt_count = 0
        self._map_user_prompt_count = 0
        self._map_embed_count = 0

    def map_cache(
        self,
        session: "SwiftVLNInferenceSession",
        episode: Any,
        window_start: int,
        features_list: List[torch.Tensor],
    ) -> None:
        if not (_debug_enabled() and self._map_eval_count < 6):
            return

        raw_token_counts = [int(features.shape[0]) for features in features_list]
        token_counts = [int(item[0].shape[0]) for item in session.history_cache]
        map_labels = ["global", "local"]
        print(
            f"[MAP DEBUG][eval] cache[{self._map_eval_count}] "
            f"rank={_debug_rank()} "
            f"episode={getattr(episode, 'episode_id', 'unknown')} "
            f"window_start={window_start} executed_actions={len(session.executed_actions)} "
            f"map_images={len(features_list)} raw_tokens={raw_token_counts} "
            f"compressed_tokens={token_counts}"
        )
        for index, (features, cache_item) in enumerate(
            zip(features_list, session.history_cache)
        ):
            label = map_labels[index] if index < len(map_labels) else f"map{index}"
            print(
                f"[MAP DEBUG][eval] cache[{self._map_eval_count}].{label} "
                f"raw={_tensor_debug_stats(features)} "
                f"compressed={_tensor_debug_stats(cache_item[0])}"
            )
        self._map_eval_count += 1

    def map_system_prompt(
        self,
        session: "SwiftVLNInferenceSession",
        inputs: Dict[str, torch.Tensor],
        history_token_counts: List[int],
        initial_token_count: int,
        instruction: str,
        system_prompt: str,
    ) -> None:
        if not (
            _debug_enabled()
            and session.memory_method == "map"
            and self._map_prompt_count < 6
        ):
            return

        token_ids = inputs["input_ids"]
        history_positions = (token_ids[0] == session.history_memory_token_id).nonzero(
            as_tuple=True
        )[0]
        current_positions = (token_ids[0] == session.current_image_token_id).nonzero(
            as_tuple=True
        )[0]
        print(
            f"[MAP DEBUG][eval.prompt.system] rank={_debug_rank()} "
            f"history_token_counts={history_token_counts} "
            f"initial_token_count={initial_token_count} "
            f"tokenized_len={token_ids.shape[1]} "
            f"history_positions={len(history_positions)} "
            f"current_positions={len(current_positions)}"
        )
        print(
            "[MAP DEBUG][eval.prompt.system] "
            f"instruction={_preview_text(instruction, limit=180)}"
        )
        print(
            "[MAP DEBUG][eval.prompt.system] "
            f"prompt={_preview_text(system_prompt, limit=340)}"
        )
        self._map_prompt_count += 1

    def map_user_prompt(
        self,
        session: "SwiftVLNInferenceSession",
        inputs: Dict[str, torch.Tensor],
        current_token_count: int,
        add_generation_prompt: bool,
        content: str,
    ) -> None:
        if not (
            _debug_enabled()
            and session.memory_method == "map"
            and self._map_user_prompt_count < 6
        ):
            return

        token_ids = inputs["input_ids"]
        current_positions = (token_ids[0] == session.current_image_token_id).nonzero(
            as_tuple=True
        )[0]
        print(
            f"[MAP DEBUG][eval.prompt.user] rank={_debug_rank()} "
            f"generation_prompt={add_generation_prompt} "
            f"current_token_count={current_token_count} "
            f"tokenized_len={token_ids.shape[1]} "
            f"current_positions={len(current_positions)}"
        )
        print(
            f"[MAP DEBUG][eval.prompt.user] content={_preview_text(content, limit=240)}"
        )
        self._map_user_prompt_count += 1

    def map_complete_embeds(
        self,
        session: "SwiftVLNInferenceSession",
        system_ids: torch.Tensor,
        new_user_ids: torch.Tensor,
        current_vit_features: torch.Tensor,
        current_token_count: int,
        history_token_counts: List[int],
        initial_token_count: int,
        history_injection_msg: str,
        history_embeds: Optional[torch.Tensor],
        history_injection_ok: bool,
        seq_len: int,
    ) -> None:
        if not (
            _debug_enabled()
            and session.memory_method == "map"
            and self._map_embed_count < 6
        ):
            return

        overlap_len = (
            int(session.overlap_context.input_ids.shape[1])
            if session.overlap_context is not None
            and session.overlap_context.input_ids is not None
            else 0
        )
        completed_turn_token_count = int(
            sum(turn.user_input_ids.shape[1] for turn in session.window_turns)
        )
        history_positions = (system_ids[0] == session.history_memory_token_id).nonzero(
            as_tuple=True
        )[0]
        system_current_positions = (
            system_ids[0] == session.current_image_token_id
        ).nonzero(as_tuple=True)[0]
        new_current_positions = (
            new_user_ids[0] == session.current_image_token_id
        ).nonzero(as_tuple=True)[0]
        print(
            f"[MAP DEBUG][eval.embed] rank={_debug_rank()} "
            f"history_cache_tokens={history_token_counts} "
            f"initial_tokens={initial_token_count} system_len={system_ids.shape[1]} "
            f"overlap_len={overlap_len} completed_turns={len(session.window_turns)} "
            f"completed_user_tokens={completed_turn_token_count} "
            f"new_user_len={new_user_ids.shape[1]} seq_len={seq_len} "
            f"history_injection={history_injection_msg}"
        )
        print(
            "[MAP DEBUG][eval.embed] "
            f"system_history_positions={len(history_positions)} "
            f"system_current_positions={len(system_current_positions)} "
            f"new_user_current_positions={len(new_current_positions)} "
            f"current_vit_tokens={current_token_count}"
        )
        if history_embeds is not None:
            print(
                "[MAP DEBUG][eval.embed] "
                f"history_embed_stats={_tensor_debug_stats(history_embeds)} "
                f"injection_ok={history_injection_ok}"
            )
        print(
            "[MAP DEBUG][eval.embed] "
            f"current_vit_stats={_tensor_debug_stats(current_vit_features)}"
        )
        self._map_embed_count += 1

    def map_slide_window(
        self, session: "SwiftVLNInferenceSession", new_window_start: int
    ) -> None:
        if _debug_enabled() and session.memory_method == "map":
            print(
                "[MAP DEBUG][eval] slide_window -> "
                f"new_window_start={new_window_start}, "
                f"history_cache={len(session.history_cache)}, "
                f"overlap_turns={session.overlap_turns}"
            )

    def initial_frame_encoded(
        self,
        episode_id: Any,
        initial_features: torch.Tensor,
    ) -> None:
        if _debug_enabled() and self._initial_eval_count < 3:
            print(
                f"[INITIAL DEBUG] Episode {episode_id} step=0: "
                f"Encoded initial frame -> {initial_features.shape[0]} tokens "
                "(uncompressed)"
            )

    def initial_tokens_injected(
        self,
        token_count: int,
        positions: torch.Tensor,
        system_length: int,
    ) -> None:
        if _debug_enabled() and self._initial_eval_count < 3:
            print(
                "[INITIAL DEBUG] build_prompt_embeddings: "
                f"Injected {token_count} initial tokens at positions "
                f"[{positions[0].item()}..{positions[-1].item()}] "
                f"in system prompt (total system_ids len={system_length})"
            )

    @staticmethod
    def map_history_injection_warning(message: str) -> None:
        print(
            "[MAP WARNING][eval.embed] system history injection skipped: "
            f"rank={_debug_rank()} {message}"
        )

    def initial_episode_summary(
        self,
        episode_id: Any,
        session: "SwiftVLNInferenceSession",
        step_count: int,
    ) -> None:
        if not (_debug_enabled() and self._initial_eval_count < 3):
            return
        has_initial = session.initial_features is not None
        initial_tokens = session.initial_features.shape[0] if has_initial else 0
        initial_summary = f"YES ({initial_tokens} tokens)" if has_initial else "NO"
        print(
            f"[INITIAL DEBUG] Episode {episode_id} summary: "
            f"system_prompt_setting={session.system_prompt_setting}, "
            f"initial_features={initial_summary}, steps={step_count}, "
            f"windows={session.current_window_idx + 1}"
        )
        self._initial_eval_count += 1

    def print_timing(
        self,
        episode_id: Any,
        scene_id: str,
        session: "SwiftVLNInferenceSession",
        timing_stats: Dict[str, float],
        total_time: float,
        step_count: int,
    ) -> None:
        if not getattr(self.args, "debug_timing", False):
            return
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_id} (Scene: {scene_id}) Timing Statistics:")
        print("=" * 60)
        print(f"Total episode time: {total_time:.3f}s")
        print(f"Total steps: {step_count}")
        print(f"Windows used: {session.current_window_idx + 1}")
        print(f"History cache size: {len(session.history_cache)}")
        print("\nBreakdown by component:")
        for component, elapsed_time in sorted(
            timing_stats.items(), key=lambda item: item[1], reverse=True
        ):
            percentage = elapsed_time / total_time * 100 if total_time > 0 else 0.0
            print(f"  {component:20s}: {elapsed_time:8.3f}s ({percentage:5.1f}%)")
        print(f"{'=' * 60}\n")
