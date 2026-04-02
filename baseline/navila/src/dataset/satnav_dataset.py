"""
SatNav dataset adapter for NaVILA training.

This keeps NaVILA's original navigation training formulation:
  - input: historical observations + current observation
  - prompt: original NaVILA navigation prompt
  - target: one natural-language next action

Unlike the upstream repo, which consumes pre-expanded VLNCE samples with
`frames + q + a`, this adapter expands SatNav `annotations.json` on the fly.
"""

import copy
import json
import os
import re
import sys
import zlib
from collections import Counter
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

_NAVILA_ROOT = "/mnt/data1/home/jiangjiajun/workspace/NaVILA"
if _NAVILA_ROOT not in sys.path:
    sys.path.insert(0, _NAVILA_ROOT)

from llava.data.dataset import preprocess
from llava.mm_utils import process_image, vlnce_frame_sampling


# Action text targets must match NaVILA's evaluation regex patterns:
#   stop    -> r"\bstop\b"
#   forward -> r"\bis move forward\b"  (distance fallback: 25 cm = 1 step)
#   left    -> r"\bis turn left\b"     + r"turn left (\d+) degree"
#   right   -> r"\bis turn right\b"    + r"turn right (\d+) degree"
# SatNav outdoor forward is ~10 m per step; the "cm" in evaluation fallback
# to 25 cm (1 VLN-CE step) is irrelevant for SatNav eval.
ACTION_TO_TEXT = {
    0: "The next action is stop.",
    1: "The next action is move forward 10 meters.",
    2: "The next action is turn left 15 degree.",
    3: "The next action is turn right 15 degree.",
}

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    'of historical observations {history_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, or stop if the task is completed."
)


def _normalize_instruction(text: str) -> str:
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+", " ", text)
    # Match original NaVILA: capitalize first letter and letters after ". "
    text = text.capitalize()
    text = re.sub(r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), text)
    return text


def _sample_kept(
    episode_id: str,
    trajectory_id: str,
    step_idx: int,
    sample_ratio: float | None,
    sample_stride: int | None,
) -> bool:
    if sample_stride is not None and sample_stride > 1:
        if step_idx % sample_stride != 0:
            return False

    if sample_ratio is not None and sample_ratio < 1.0:
        sample_key = f"{episode_id}|{trajectory_id}|{step_idx}"
        sample_hash = zlib.crc32(sample_key.encode("utf-8")) & 0xFFFFFFFF
        return (sample_hash / 0xFFFFFFFF) < sample_ratio

    return True


def _select_steps_head_stop_stride(actions: list, head_keep: int, fwd_stride: int) -> list[int]:
    """Select step indices using head-stop-stride strategy.

    Regions:
    - Head  [1, head_keep]: keep all valid steps (unique <8-frame input distribution)
    - Stop  [last_step]:    always keep (rare action, only 1 per episode)
    - Middle (head_keep+1, last_step-1):
        - turn left / turn right: always keep (decision-critical, minority class)
        - move forward: keep every fwd_stride-th consecutive forward step
          (resets the counter on each non-forward gap)

    Handles edge cases where head/stop regions overlap for short episodes.
    """
    last_step = len(actions) - 1
    if last_step <= 0:
        return []

    kept: set[int] = set()

    # Head: steps 1..head_keep (clamped to episode length)
    for i in range(1, min(head_keep + 1, last_step + 1)):
        if actions[i] in ACTION_TO_TEXT:
            kept.add(i)

    # Stop: the last step (always stop in SatNav)
    if actions[last_step] in ACTION_TO_TEXT:
        kept.add(last_step)

    # Middle: (head_keep+1)..(last_step-1)
    # turns are always kept; consecutive forward runs are strided independently.
    mid_start = head_keep + 1
    mid_end = last_step - 1
    if fwd_stride > 0 and mid_start <= mid_end:
        consecutive_fwd = 0
        for i in range(mid_start, mid_end + 1):
            a = actions[i]
            if a not in ACTION_TO_TEXT:
                consecutive_fwd = 0
                continue
            if a == 1:  # move forward: stride
                if consecutive_fwd % fwd_stride == 0:
                    kept.add(i)
                consecutive_fwd += 1
            else:  # turn left / turn right: always keep, reset forward counter
                kept.add(i)
                consecutive_fwd = 0

    return sorted(kept)


class SatNavNaVILADataset(Dataset):
    """Expand SatNav episodes into NaVILA-style next-action supervision."""

    def __init__(self, data_path, image_folder, tokenizer, data_args, training_args):
        super().__init__()
        with open(data_path) as f:
            episodes = json.load(f)

        max_episodes_env = os.getenv("SATNAV_MAX_EPISODES", "").strip()
        max_samples_env = os.getenv("SATNAV_MAX_SAMPLES", "").strip()
        sample_ratio_env = os.getenv("SATNAV_SAMPLE_RATIO", "").strip()
        sample_stride_env = os.getenv("SATNAV_SAMPLE_STRIDE", "").strip()
        head_keep_env = os.getenv("SATNAV_HEAD_KEEP", "").strip()
        stop_repeat_env = os.getenv("SATNAV_STOP_REPEAT", "").strip()
        max_episodes = int(max_episodes_env) if max_episodes_env else None
        max_samples = int(max_samples_env) if max_samples_env else None
        sample_ratio = float(sample_ratio_env) if sample_ratio_env else None
        sample_stride = int(sample_stride_env) if sample_stride_env else None
        stop_repeat = int(stop_repeat_env) if stop_repeat_env else 1
        # head_keep enables the "head + stop + strided middle" sampling mode.
        # When set, SATNAV_SAMPLE_STRIDE is used as the middle-section stride.
        head_keep = int(head_keep_env) if head_keep_env else None
        total_episodes = len(episodes)

        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder
        self.samples = []
        self.action_counts = Counter()

        for episode in episodes:
            episode_id = str(episode.get("id", ""))
            trajectory_id = str(episode.get("trajectory_id", ""))
            instruction = _normalize_instruction(episode["instructions"][0])
            video_dir = episode["video"]
            actions = episode["actions"]

            if head_keep is not None:
                # Head-stop-stride mode: keep trajectory start, stop action,
                # all turn steps, and stride forward steps in the middle.
                fwd_stride = sample_stride if (sample_stride is not None and sample_stride > 1) else 1
                step_indices = _select_steps_head_stop_stride(actions, head_keep, fwd_stride)
            else:
                # Legacy mode: iterate all steps and apply ratio/stride filters.
                step_indices = [
                    i for i in range(1, len(actions))
                    if actions[i] in ACTION_TO_TEXT
                    and _sample_kept(
                        episode_id=episode_id,
                        trajectory_id=trajectory_id,
                        step_idx=i,
                        sample_ratio=sample_ratio,
                        sample_stride=sample_stride,
                    )
                ]

            for step_idx in step_indices:
                frame_paths = [
                    os.path.join(image_folder, video_dir, "rgb", f"{frame_idx:03d}.jpg")
                    for frame_idx in range(1, step_idx + 1)
                ]
                action = actions[step_idx]
                sample = {
                    "episode_id": episode.get("id"),
                    "trajectory_id": episode.get("trajectory_id"),
                    "step_idx": step_idx,
                    "instruction": instruction,
                    "frame_paths": frame_paths,
                    "answer": ACTION_TO_TEXT[action],
                    "action": action,
                }
                repeat = stop_repeat if action == 0 else 1
                for _ in range(repeat):
                    self.samples.append(sample.copy())
                    self.action_counts[action] += 1
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        print(
            "SatNavNaVILADataset: "
            f"episodes={len(episodes)}/{total_episodes}, "
            f"samples={len(self.samples)}, "
            f"max_episodes={max_episodes}, max_samples={max_samples}, "
            f"sample_ratio={sample_ratio}, sample_stride={sample_stride}, "
            f"head_keep={head_keep}, stop_repeat={stop_repeat}, "
            f"action_counts={dict(self.action_counts)}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [len(sample["frame_paths"]) + 100 for sample in self.samples]

    @property
    def modality_lengths(self):
        return [len(sample["frame_paths"]) + 100 for sample in self.samples]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(index)
        except Exception as exc:
            sample = self.samples[index]
            raise RuntimeError(
                "Failed to load SatNav sample "
                f"index={index}, episode_id={sample.get('episode_id')}, "
                f"trajectory_id={sample.get('trajectory_id')}, step_idx={sample.get('step_idx')}"
            ) from exc

    def _get_item(self, index) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        sampled_frames = vlnce_frame_sampling(
            list(sample["frame_paths"]),
            num_frames=self.data_args.num_video_frames,
        )
        image_tensor = torch.stack(
            [process_image(image, self.data_args, image_folder=None) for image in sampled_frames]
        )

        history_count = max(self.data_args.num_video_frames - 1, 0)
        prompt = PROMPT_TEMPLATE.format(
            history_tokens="<image>\n" * history_count,
            instruction=sample["instruction"],
        )
        conversation = [[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": sample["answer"]},
        ]]

        data_dict = preprocess(
            copy.deepcopy(conversation),
            self.tokenizer,
            has_image=True,
        )
        data_dict = {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "image": image_tensor,
        }
        return data_dict
