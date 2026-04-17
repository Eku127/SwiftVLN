import json
import os
from pathlib import Path
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
from prompting import (
    DEFAULT_ACTION_HISTORY_LIMIT,
    build_openfly_prompt,
    build_openfly_prompt_with_answer,
)


IGNORE_INDEX = -100
DEFAULT_ORIGINAL_DIM_LOSS_WEIGHTS = (0.4, 1.2, 1.2, 1.2)
DEFAULT_HEAD_KEEP = 7
DEFAULT_SAMPLE_STRIDE = 3
DEFAULT_STOP_REPEAT = 2
DEFAULT_STOP_HISTORY_AUG = 1
DEFAULT_STOP_WINDOW = 2
DEFAULT_TAIL_KEEP = 5


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


def _build_stop_history_variants(current_idx: int, variant_count: int) -> list[list[int]]:
    variant_count = max(int(variant_count), 1)
    variants: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    base = _sample_history_indices(current_idx)

    def _add_variant(first_idx: int, second_idx: int) -> None:
        pair = (max(1, first_idx), max(1, second_idx))
        if pair not in seen:
            seen.add(pair)
            variants.append([pair[0], pair[1]])

    _add_variant(base[0], base[1])
    for shift in range(1, variant_count):
        _add_variant(current_idx - shift - 1, current_idx - shift - 2)
        if len(variants) >= variant_count:
            break

    return variants


def _safe_env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else int(default)


def _normalize_trajectory_type(value: str | None) -> str:
    if not value:
        return "unknown"
    return str(value)


def _dataset_root_from_annotations(data_path: str) -> Path:
    annotations_path = Path(data_path).resolve()
    if annotations_path.name != "annotations.json":
        return annotations_path.parent
    return annotations_path.parent.parent


def _episode_payload_to_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return episodes
        return []
    if isinstance(payload, list):
        return payload
    return []


def _load_trajectory_type_lookup(data_path: str) -> dict[str, str]:
    dataset_root = _dataset_root_from_annotations(data_path)
    lookup: dict[str, str] = {}
    episode_files = sorted((dataset_root / "episodes").glob("**/all_episodes.json"))
    for episode_path in episode_files:
        try:
            with open(episode_path, "r", encoding="utf-8") as f:
                episodes = _episode_payload_to_list(json.load(f))
        except Exception:
            continue

        for episode in episodes:
            trajectory_id = str(episode.get("trajectory_id", ""))
            if not trajectory_id:
                continue
            lookup[trajectory_id] = _normalize_trajectory_type(episode.get("trajectory_type"))
    return lookup


def _load_sampling_policy() -> dict[str, int]:
    return {
        "sample_stride": _safe_env_int("SATNAV_SAMPLE_STRIDE", DEFAULT_SAMPLE_STRIDE),
        "stop_repeat": _safe_env_int("SATNAV_STOP_REPEAT", DEFAULT_STOP_REPEAT),
        "stop_window": _safe_env_int("SATNAV_STOP_WINDOW", DEFAULT_STOP_WINDOW),
        "tail_keep": _safe_env_int("SATNAV_TAIL_KEEP", DEFAULT_TAIL_KEEP),
    }


def _select_steps_head_stop_stride(actions: list[int], head_keep: int, fwd_stride: int, tail_keep: int) -> list[int]:
    """Keep head, near-goal tail, all turns, and strided forward steps in the middle."""
    last_step = len(actions) - 1
    if last_step <= 0:
        return []

    kept: set[int] = set()

    for step_idx in range(1, min(head_keep + 1, last_step + 1)):
        if int(actions[step_idx]) in ACTION_TO_NAME:
            kept.add(step_idx)

    if int(actions[last_step]) in ACTION_TO_NAME:
        kept.add(last_step)

    tail_start = max(head_keep + 1, last_step - max(int(tail_keep), 0))
    for step_idx in range(tail_start, last_step):
        if int(actions[step_idx]) in ACTION_TO_NAME:
            kept.add(step_idx)

    mid_start = head_keep + 1
    mid_end = min(last_step - 1, tail_start - 1)
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


def _is_stop_window_step(step_idx: int, last_step: int, stop_window: int) -> bool:
    if stop_window <= 0:
        return step_idx == last_step
    return step_idx >= max(1, last_step - stop_window + 1)


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
        head_keep = int(head_keep_env) if head_keep_env else DEFAULT_HEAD_KEEP
        stop_history_aug_env = os.getenv("SATNAV_STOP_HISTORY_AUG", "").strip()
        stop_history_aug = int(stop_history_aug_env) if stop_history_aug_env else DEFAULT_STOP_HISTORY_AUG
        history_limit_env = os.getenv("OPENFLY_ACTION_HISTORY_LIMIT", "").strip()
        action_history_limit = int(history_limit_env) if history_limit_env else DEFAULT_ACTION_HISTORY_LIMIT
        trajectory_type_lookup = _load_trajectory_type_lookup(data_path)
        sampling_policy = _load_sampling_policy()

        self.image_folder = image_folder
        self.action_format = resolve_action_format(action_format)
        self.action_history_limit = action_history_limit
        self.samples: list[dict[str, Any]] = []
        self.action_counts = {action: 0 for action in ACTION_TO_NAME}
        self.trajectory_type_counts: dict[str, int] = {}

        for episode in episodes:
            instruction = episode["instructions"][0]
            video_dir = episode["video"]
            actions = episode["actions"]
            episode_id = str(episode.get("id", ""))
            trajectory_id = str(episode.get("trajectory_id", ""))
            trajectory_type = trajectory_type_lookup.get(trajectory_id, "unknown")
            sample_stride = sampling_policy["sample_stride"]
            stop_repeat = sampling_policy["stop_repeat"]
            stop_window = sampling_policy["stop_window"]
            tail_keep = sampling_policy["tail_keep"]
            step_indices = _select_steps_head_stop_stride(actions, head_keep, sample_stride, tail_keep)
            last_step = len(actions) - 1

            for step_idx in step_indices:
                source_action = int(actions[step_idx])
                if source_action not in ACTION_TO_NAME:
                    continue
                action = 0 if _is_stop_window_step(step_idx, last_step, stop_window) else source_action

                current_frame_idx = step_idx
                answer = action_to_text(action)
                raw_action_vector = get_original_action_vector(action)
                normalized_action_vector = get_normalized_original_action_vector(action)

                prior_action_history: list[int] = []
                for prior in actions[:step_idx]:
                    prior_int = int(prior)
                    if prior_int in ACTION_TO_NAME:
                        prior_action_history.append(prior_int)

                if action == 0:
                    history_variants = _build_stop_history_variants(current_frame_idx, stop_history_aug)
                    repeat_count = max(stop_repeat, 1)
                else:
                    history_variants = [_sample_history_indices(current_frame_idx)]
                    repeat_count = 1

                for repeat_idx in range(repeat_count):
                    history_indices = history_variants[repeat_idx % len(history_variants)]
                    frame_paths = [
                        _frame_path(image_folder, video_dir, current_frame_idx),
                        _frame_path(image_folder, video_dir, history_indices[0]),
                        _frame_path(image_folder, video_dir, history_indices[1]),
                    ]

                    self.samples.append(
                        {
                            "episode_id": episode_id,
                            "trajectory_id": trajectory_id,
                            "trajectory_type": trajectory_type,
                            "step_idx": step_idx,
                            "instruction": instruction,
                            "frame_paths": frame_paths,
                            "action": action,
                            "source_action": source_action,
                            "answer": answer,
                            "raw_action_vector": raw_action_vector,
                            "normalized_action_vector": normalized_action_vector,
                            "history_variant_idx": repeat_idx % len(history_variants),
                            "action_history": list(prior_action_history),
                        }
                    )
                    self.action_counts[action] += 1
                    self.trajectory_type_counts[trajectory_type] = self.trajectory_type_counts.get(trajectory_type, 0) + 1
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
            f"stop_history_aug={stop_history_aug}, "
            f"sampling_policy={sampling_policy}, "
            f"action_counts={self.action_counts}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        images = [_safe_open_image(path) for path in sample["frame_paths"]]
        action_history = sample.get("action_history", [])
        return {
            **sample,
            "images": images,
            "prompt_text": build_openfly_prompt(
                sample["instruction"],
                action_history=action_history,
                history_limit=self.action_history_limit,
            ),
            "full_text": (
                build_openfly_prompt_with_answer(
                    sample["instruction"],
                    sample["answer"],
                    action_history=action_history,
                    history_limit=self.action_history_limit,
                )
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
        self.original_supervised_dims = 4
        loss_weight_env = os.getenv("OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS", "").strip()
        if loss_weight_env:
            parsed = [float(item.strip()) for item in loss_weight_env.split(",") if item.strip()]
            if len(parsed) != self.original_supervised_dims:
                raise ValueError(
                    "OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS must provide exactly "
                    f"{self.original_supervised_dims} comma-separated values"
                )
            self.original_dim_loss_weights = tuple(parsed)
        else:
            self.original_dim_loss_weights = DEFAULT_ORIGINAL_DIM_LOSS_WEIGHTS

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        pixel_values = []
        loss_weights = []

        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor

        history_limit_env = os.getenv("OPENFLY_ACTION_HISTORY_LIMIT", "").strip()
        try:
            history_limit = int(history_limit_env) if history_limit_env else None
        except ValueError:
            history_limit = None

        for instance in instances:
            if self.action_format == COMPACT:
                full_text = instance["full_text"]
            else:
                answer = self.action_tokenizer(instance["normalized_action_vector"])
                full_text = build_openfly_prompt_with_answer(
                    instance["instruction"],
                    answer,
                    action_history=instance.get("action_history", []),
                    history_limit=history_limit,
                )

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
            sample_loss_weights = [0.0] * len(full_ids)
            masked_prefix = min(len(prompt_ids), len(sample_labels))
            for idx in range(masked_prefix):
                sample_labels[idx] = IGNORE_INDEX

            if self.action_format == ORIGINAL:
                for idx in range(masked_prefix, len(sample_labels)):
                    sample_labels[idx] = IGNORE_INDEX

                answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
                action_dim_count = len(instance["normalized_action_vector"])
                extra_prefix_tokens = max(len(answer_ids) - action_dim_count, 0)
                supervised_dim_count = min(self.original_supervised_dims, action_dim_count)
                supervise_start = masked_prefix + extra_prefix_tokens
                supervise_end = min(supervise_start + supervised_dim_count, len(sample_labels))
                for idx in range(supervise_start, supervise_end):
                    sample_labels[idx] = full_ids[idx]
                    sample_loss_weights[idx] = float(self.original_dim_loss_weights[idx - supervise_start])

            input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            labels.append(torch.tensor(sample_labels, dtype=torch.long))
            if self.action_format == ORIGINAL:
                loss_weights.append(torch.tensor(sample_loss_weights, dtype=torch.float32))

            sample_pixels = image_processor(images=instance["images"], return_tensors="pt")["pixel_values"]
            pixel_values.append(sample_pixels)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)
        pixel_values = torch.stack(pixel_values, dim=0)

        batch = {
            "input_ids": input_ids[:, : self.model_max_length],
            "labels": labels[:, : self.model_max_length],
            "attention_mask": attention_mask[:, : self.model_max_length],
            "pixel_values": pixel_values,
        }
        if self.action_format == ORIGINAL:
            batch["loss_weights"] = pad_sequence(loss_weights, batch_first=True, padding_value=0.0)[
                :, : self.model_max_length
            ]

        return batch
