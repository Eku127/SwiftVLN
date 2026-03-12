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


class SatNavNaVILADataset(Dataset):
    """Expand SatNav episodes into NaVILA-style next-action supervision."""

    def __init__(self, data_path, image_folder, tokenizer, data_args, training_args):
        super().__init__()
        with open(data_path) as f:
            episodes = json.load(f)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder
        self.samples = []

        for episode in episodes:
            instruction = _normalize_instruction(episode["instructions"][0])
            video_dir = episode["video"]
            actions = episode["actions"]

            for step_idx in range(1, len(actions)):
                action = actions[step_idx]
                if action not in ACTION_TO_TEXT:
                    continue

                frame_paths = [
                    os.path.join(image_folder, video_dir, "rgb", f"{frame_idx:03d}.jpg")
                    for frame_idx in range(1, step_idx + 1)
                ]
                self.samples.append(
                    {
                        "episode_id": episode.get("id"),
                        "trajectory_id": episode.get("trajectory_id"),
                        "step_idx": step_idx,
                        "instruction": instruction,
                        "frame_paths": frame_paths,
                        "answer": ACTION_TO_TEXT[action],
                    }
                )

        print(
            f"SatNavNaVILADataset: loaded {len(episodes)} episodes -> {len(self.samples)} next-action samples",
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
