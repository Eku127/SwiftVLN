# Copyright (c) Alibaba, Inc. and its affiliates.
"""Stateful SwiftVLN inference-session orchestration.

Prompt construction, visual encoding, and sliding-window state live in sibling
modules.  This facade retains the established public class and method surface.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

from swiftvln.backends.specs import EnvironmentSpec
from swiftvln.modeling.constants import (
    CURRENT_IMAGE_TOKEN,
    DEFAULT_CONJUNCTIONS,
    HISTORY_MEMORY_TOKEN,
)
from swiftvln.modeling.history import create_history_processor
from swiftvln.modeling.history.compressor import HistoryTokenCompressor
from swiftvln.modeling.memory.satnav_map import SatNavMapMemoryBuilder

from .encoding import VisualEncodingMixin
from .prompt import PromptConstructionMixin
from .window import OverlapContext, TurnContext, WindowStateMixin


def _derive_dataset_cache_dir(data_path_tmpl: str) -> Optional[str]:
    """Infer ``{dataset_root}/map_cache`` from a SatNav data template."""
    if not data_path_tmpl:
        return None

    parts = str(data_path_tmpl).split(os.sep)
    if "episodes" in parts:
        index = parts.index("episodes")
        if index > 0:
            dataset_root = os.sep.join(parts[:index])
            if os.path.isabs(str(data_path_tmpl)) and not dataset_root.startswith(
                os.sep
            ):
                dataset_root = os.sep + dataset_root
            return os.path.join(
                os.path.abspath(dataset_root),
                "map_cache",
            )
    return None


class SwiftVLNInferenceSession(
    PromptConstructionMixin,
    VisualEncodingMixin,
    WindowStateMixin,
):
    """Coordinate one episode's model, memory, prompt, and window state."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        args: Any,
        config: Any,
        environment_spec: EnvironmentSpec,
        device: torch.device,
        num_history: int,
        num_future_steps: int,
        diagnostics: Any,
    ):
        self.model = model
        self.processor = processor
        self.args = args
        self.config = config
        self.environment_spec = environment_spec
        self.env_type = environment_spec.name
        self.device = device
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.diagnostics = diagnostics

        self.num_frames = getattr(self.args, "num_frames", 32)
        self.num_overlap = getattr(self.args, "num_overlap", 0)
        self.stride = self.num_frames - self.num_overlap
        self.turns_per_window = self.num_frames // self.num_future_steps
        self.overlap_turns = self.num_overlap // self.num_future_steps
        self.new_turns_per_window = (
            self.turns_per_window - self.overlap_turns
        )

        self.history_processor_type = getattr(
            self.args,
            "history_processor_type",
            "per_frame",
        )
        self.compress_stride = getattr(self.args, "compress_stride", 2)
        self.use_tome = getattr(self.args, "use_tome", False)
        self.log_base = getattr(self.args, "log_base", 1.0)
        self.use_random = getattr(self.args, "use_random", False)
        self.gtc_output_tokens = getattr(
            self.args,
            "gtc_output_tokens",
            512,
        )
        self.gtc_temperature = getattr(
            self.args,
            "gtc_temperature",
            0.1,
        )
        self.gtc_num_iterations = getattr(
            self.args,
            "gtc_num_iterations",
            1,
        )

        if self.history_processor_type == "gtc":
            self.history_processor = create_history_processor(
                processor_type="gtc",
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None
        elif self.history_processor_type == "segment_gtc":
            self.history_processor = create_history_processor(
                processor_type="segment_gtc",
                output_tokens=self.gtc_output_tokens,
                temperature=self.gtc_temperature,
                num_iterations=self.gtc_num_iterations,
            )
            self.compressor = None
        else:
            compress_method = "tome" if self.use_tome else "pooling"
            self.history_processor = create_history_processor(
                processor_type="per_frame",
                compress_stride=self.compress_stride,
                compress_method=compress_method,
                num_history=self.num_history,
                log_base=self.log_base,
            )
            self.compressor = HistoryTokenCompressor(
                stride=self.compress_stride,
                method=compress_method,
            )

        self.system_prompt_setting = getattr(
            self.args,
            "system_prompt_setting",
            "vanilla",
        ).lower()
        self.memory_method = getattr(
            self.args,
            "memory_method",
            "history",
        ).lower()
        self.map_global_side_m = float(
            getattr(self.args, "map_global_side_m", 1000.0)
        )
        self.map_local_side_m = float(
            getattr(self.args, "map_local_side_m", 400.0)
        )
        self.map_render_px = int(
            getattr(self.args, "map_render_px", 448)
        )
        self.map_mask_method = str(
            getattr(self.args, "map_mask_method", "dilate20")
        ).lower()
        self.map_builder: Optional[SatNavMapMemoryBuilder] = None
        if self.memory_method not in ("history", "map"):
            raise ValueError(
                f"Unsupported memory_method: {self.memory_method}"
            )
        if self.memory_method == "map":
            if not self.environment_spec.supports_map_memory:
                raise ValueError(
                    "SwiftVLN memory_method=map currently supports only satnav "
                    f"(got {self.env_type})."
                )
            if self.history_processor_type != "per_frame":
                raise ValueError(
                    "SwiftVLN memory_method=map currently requires "
                    "history_processor_type=per_frame."
                )
            if self.use_tome:
                raise ValueError(
                    "SwiftVLN memory_method=map currently requires "
                    "use_tome=false."
                )
            if getattr(self.args, "embedding_mode", "none") != "none":
                raise ValueError(
                    "SwiftVLN memory_method=map requires embedding_mode=none."
                )
            data_path_template = (
                getattr(self.config.DATASET, "DATA_PATH", "") or ""
            )
            default_map_cache_dir = _derive_dataset_cache_dir(
                str(data_path_template)
            )
            self.map_builder = SatNavMapMemoryBuilder(
                scenes_dir=self.config.DATASET.SCENES_DIR,
                global_side_m=self.map_global_side_m,
                local_side_m=self.map_local_side_m,
                render_px=self.map_render_px,
                mask_method=self.map_mask_method,
                hfov=float(self.config.SIMULATOR.RGB_SENSOR.HFOV),
                sensor_width=int(
                    self.config.SIMULATOR.RGB_SENSOR.WIDTH
                ),
                sensor_height=int(
                    self.config.SIMULATOR.RGB_SENSOR.HEIGHT
                ),
                cache_dir=default_map_cache_dir,
            )

        self.conjunctions = DEFAULT_CONJUNCTIONS.copy()
        self.history_memory_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(
                HISTORY_MEMORY_TOKEN
            )
        )
        self.current_image_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(
                CURRENT_IMAGE_TOKEN
            )
        )
        self.merge_size = getattr(
            self.processor.image_processor,
            "merge_size",
            2,
        )
        self.reset()

        self.has_embed_enhance = (
            hasattr(self.model, "embed_enhance")
            and not self.model.embed_enhance.is_empty
        )
        self._print_configuration()

    def _print_configuration(self) -> None:
        print("[SwiftVLNInferenceSession] Initialized:")
        print(
            f"  Window: num_frames={self.num_frames}, "
            f"num_overlap={self.num_overlap}, stride={self.stride}"
        )
        print(
            f"  Turns: per_window={self.turns_per_window}, "
            f"overlap={self.overlap_turns}"
        )
        print(f"  History Processor: {self.history_processor.name}")
        if self.history_processor_type in ("gtc", "segment_gtc"):
            print(
                f"    Sampling: every {self.num_future_steps} frames "
                "(same as training)"
            )
        else:
            if self.use_random:
                sampling_type = "random"
            else:
                sampling_type = (
                    "uniform"
                    if self.log_base == 1.0
                    else f"logarithmic (b={self.log_base})"
                )
            print(
                f"    Sampling: {sampling_type}, {self.num_history} frames"
            )
        print(f"  System Prompt: {self.system_prompt_setting}")
        if self.system_prompt_setting == "initial":
            print(
                "    [INITIAL] Initial view ENABLED: first frame "
                "(uncompressed) in system prompt"
            )
        print(f"  Memory Method: {self.memory_method}")
        if self.memory_method == "map":
            print(
                f"    map: global={self.map_global_side_m:.0f}m, "
                f"local={self.map_local_side_m:.0f}m, "
                f"render={self.map_render_px}px, "
                f"mask={self.map_mask_method}"
            )
        if self.has_embed_enhance:
            print(
                "  Embed Enhance: "
                f"{self.model.embed_enhance} "
                "(auto-detected from checkpoint)"
            )

    def predict(
        self,
        instruction: str,
        current_image: Image.Image,
        current_pose: List[float],
        step_id: int,
        parse_actions: Callable[[str], List[int]],
        timing_stats: Dict[str, float],
    ) -> List[int]:
        """Generate and cache one action sequence for the current observation."""
        start_time = time.time()
        conjunction = random.choice(self.conjunctions)
        inputs_embeds, seq_len, current_vit, current_grid_thw = (
            self._build_complete_prompt_embeds(
                instruction=instruction,
                current_image=current_image,
                conjunction=conjunction,
                current_pose=current_pose,
            )
        )
        timing_stats["build_embeds"] += time.time() - start_time

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

        start_time = time.time()
        outputs = self.model.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=getattr(self.args, "max_new_tokens", 64),
            do_sample=False,
            use_cache=True,
        )
        timing_stats["model_generate"] += time.time() - start_time

        start_time = time.time()
        generated_ids = outputs[0][seq_len:]
        output_text = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        timing_stats["decode"] += time.time() - start_time

        start_time = time.time()
        action_sequence = parse_actions(output_text)
        timing_stats["parse_actions"] += time.time() - start_time

        start_time = time.time()
        self._save_turn_to_window(
            conjunction=conjunction,
            response=output_text,
            vit_features=current_vit,
        )
        if self.history_processor_type in ("gtc", "segment_gtc"):
            self.vit_feature_cache[step_id] = (
                current_vit,
                current_grid_thw,
                current_pose,
            )
        timing_stats["update_cache"] += time.time() - start_time

        if getattr(self.args, "verbose", False):
            print(
                f"Step {step_id}: window={self.current_window_idx}, "
                f"turns={len(self.window_turns)}, "
                f"history={len(self.history_cache)}, "
                f"vit_cache={len(self.vit_feature_cache)}, "
                f"output={output_text}"
            )
        return action_sequence


__all__ = [
    "OverlapContext",
    "SwiftVLNInferenceSession",
    "TurnContext",
]
