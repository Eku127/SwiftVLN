import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from action_formats import (
    ACTION_TO_NAME,
    COMPACT,
    ORIGINAL,
    action_to_text,
    get_normalized_original_action_vector,
    get_original_action_vector,
    resolve_action_format,
)
from action_tokenizer import ActionTokenizer
from prompting import build_openfly_prompt, build_openfly_prompt_with_answer


IGNORE_INDEX = -100


def _frame_path(image_folder: str, video_dir: str, frame_idx: int) -> str:
    return os.path.join(image_folder, video_dir, "rgb", f"{frame_idx:03d}.jpg")


def _safe_open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _sample_history_indices(current_idx: int) -> list[int]:
    if current_idx <= 1:
        return [1, 1]
    if current_idx == 2:
        return [1, 1]
    return [current_idx - 1, current_idx - 2]


def _select_steps_head_stop_stride(actions: list[int], head_keep: int, fwd_stride: int) -> list[int]:
    """NaVILA-style sampling: keep head, stop, turns, and strided forward steps."""
    last_step = len(actions) - 1
    if last_step <= 0:
        return []

    kept: set[int] = set()

    for step_idx in range(1, min(head_keep + 1, last_step + 1)):
        if int(actions[step_idx]) in ACTION_TO_NAME:
            kept.add(step_idx)

    if int(actions[last_step]) in ACTION_TO_NAME:
        kept.add(last_step)

    mid_start = head_keep + 1
    mid_end = last_step - 1
    if fwd_stride > 0 and mid_start <= mid_end:
        consecutive_forward = 0
        for step_idx in range(mid_start, mid_end + 1):
            action = int(actions[step_idx])
            if action not in ACTION_TO_NAME:
                consecutive_forward = 0
                continue
            if action == 1:
                if consecutive_forward % fwd_stride == 0:
                    kept.add(step_idx)
                consecutive_forward += 1
            else:
                kept.add(step_idx)
                consecutive_forward = 0

    return sorted(kept)


class SatNavOpenFlyDataset(Dataset):
    """SatNav step-wise next-action supervision for OpenFly."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        action_format: str = COMPACT,
        max_episodes: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            episodes = json.load(f)

        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        head_keep_env = os.getenv("SATNAV_HEAD_KEEP", "").strip()
        sample_stride_env = os.getenv("SATNAV_SAMPLE_STRIDE", "").strip()
        stop_repeat_env = os.getenv("SATNAV_STOP_REPEAT", "").strip()
        head_keep = int(head_keep_env) if head_keep_env else None
        sample_stride = int(sample_stride_env) if sample_stride_env else None
        stop_repeat = int(stop_repeat_env) if stop_repeat_env else 1

        self.image_folder = image_folder
        self.action_format = resolve_action_format(action_format)
        self.samples: list[dict[str, Any]] = []
        self.action_counts = {action: 0 for action in ACTION_TO_NAME}

        for episode in episodes:
            instruction = episode["instructions"][0]
            video_dir = episode["video"]
            actions = episode["actions"]
            episode_id = str(episode.get("id", ""))
            trajectory_id = str(episode.get("trajectory_id", ""))

            if head_keep is not None:
                fwd_stride = sample_stride if sample_stride is not None and sample_stride > 1 else 1
                step_indices = _select_steps_head_stop_stride(actions, head_keep, fwd_stride)
            else:
                step_indices = list(range(1, len(actions)))

            for step_idx in step_indices:
                action = int(actions[step_idx])
                if action not in ACTION_TO_NAME:
                    continue

                current_frame_idx = step_idx
                history_indices = _sample_history_indices(current_frame_idx)
                frame_paths = [
                    _frame_path(image_folder, video_dir, current_frame_idx),
                    _frame_path(image_folder, video_dir, history_indices[0]),
                    _frame_path(image_folder, video_dir, history_indices[1]),
                ]

                self.samples.append(
                    {
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "step_idx": step_idx,
                        "instruction": instruction,
                        "frame_paths": frame_paths,
                        "action": action,
                        "answer": action_to_text(action),
                        "raw_action_vector": get_original_action_vector(action),
                        "normalized_action_vector": get_normalized_original_action_vector(action),
                    }
                )
                self.action_counts[action] += 1
                if action == 0 and stop_repeat > 1:
                    for _ in range(stop_repeat - 1):
                        self.samples.append(self.samples[-1].copy())
                        self.action_counts[action] += 1
                        if max_samples is not None and len(self.samples) >= max_samples:
                            break
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        print(
            "SatNavOpenFlyDataset: "
            f"samples={len(self.samples)}, "
            f"action_format={self.action_format}, "
            f"max_episodes={max_episodes}, "
            f"max_samples={max_samples}, "
            f"head_keep={head_keep}, "
            f"sample_stride={sample_stride}, "
            f"stop_repeat={stop_repeat}, "
            f"action_counts={self.action_counts}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        images = [_safe_open_image(path) for path in sample["frame_paths"]]
        return {
            **sample,
            "images": images,
            "prompt_text": build_openfly_prompt(sample["instruction"]),
            "full_text": (
                build_openfly_prompt_with_answer(sample["instruction"], sample["answer"])
                if self.action_format == COMPACT
                else None
            ),
        }


@dataclass
class OpenFlyDataCollator:
    processor: Any
    model_max_length: int
    pad_token_id: int
    action_format: str = COMPACT

    def __post_init__(self) -> None:
        self.action_format = resolve_action_format(self.action_format)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer) if self.action_format == ORIGINAL else None

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        pixel_values = []

        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor

        for instance in instances:
            if self.action_format == COMPACT:
                full_text = instance["full_text"]
            else:
                answer = self.action_tokenizer(instance["normalized_action_vector"])
                full_text = build_openfly_prompt_with_answer(instance["instruction"], answer)

            prompt_ids = tokenizer(
                instance["prompt_text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.model_max_length,
            ).input_ids
            full_ids = tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.model_max_length,
            ).input_ids

            sample_labels = full_ids.copy()
            masked_prefix = min(len(prompt_ids), len(sample_labels))
            for idx in range(masked_prefix):
                sample_labels[idx] = IGNORE_INDEX

            input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            labels.append(torch.tensor(sample_labels, dtype=torch.long))

            sample_pixels = image_processor(images=instance["images"], return_tensors="pt")["pixel_values"]
            pixel_values.append(sample_pixels)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)
        pixel_values = torch.stack(pixel_values, dim=0)

        return {
            "input_ids": input_ids[:, : self.model_max_length],
            "labels": labels[:, : self.model_max_length],
            "attention_mask": attention_mask[:, : self.model_max_length],
            "pixel_values": pixel_values,
        }
