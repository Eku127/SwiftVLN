"""
SatNavUniNaVidDataset: SatNav trajectory dataset for Uni-NaVid fine-tuning.

Reads SatNav annotations.json + JPEG frames directly (no mp4 conversion).
Each training sample is a 4-action window of an episode, with all historical
frames from episode start to the current step as video input.

The dataset produces exactly the same output interface as Uni-NaVid's
LazySupervisedDataset so the rest of the training pipeline (model, trainer,
collator) works unchanged.
"""

import os
import sys
import copy
import json
import math
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

_UNINAVID_ROOT = "/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
if _UNINAVID_ROOT not in sys.path:
    sys.path.insert(0, _UNINAVID_ROOT)

from uninavid.constants import DEFAULT_IMAGE_TOKEN, NAVIGATION_IDENTIFIER
from uninavid.train.train import (
    preprocess,
    preprocess_multimodal,
    duplicate_with_probability,
    random_color_jitter,
    rank0_print,
)

ACTION_MAP = {1: "forward", 2: "left", 3: "right", 0: "stop"}

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given {nav_id} {img_token}. "
    "Your assigned task is: '{instruction}'. "
    "Analyze this series of images to determine your next four actions. "
    "The predicted action should be one of the following: "
    "forward, left, right, or stop."
)


def format_action_target(actions):
    """Build a structured 4-step target sequence.

    Plain `"forward forward forward forward"` targets are extremely short and
    heavily repeated on SatNav. Adding explicit step markers keeps eval
    compatibility because the evaluator extracts action words with regex, while
    giving training a denser and less degenerate supervision signal.
    """
    return " ".join(f"{idx}. {ACTION_MAP[action]}" for idx, action in enumerate(actions, start=1))


class SatNavUniNaVidDataset(Dataset):
    """SatNav windowed dataset for Uni-NaVid training.

    Args:
        data_path:    Path to SatNav annotations.json
        tokenizer:    HuggingFace tokenizer (set by train())
        data_args:    DataArguments (video_folder, image_processor, etc.)
        window_size:  Number of actions per training sample (default 4)
    """

    def __init__(self, data_path, tokenizer, data_args, window_size=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.window_size = window_size

        with open(data_path) as f:
            episodes = json.load(f)

        self.samples = []
        for ep in episodes:
            real_actions = ep["actions"][1:]  # skip INITIAL (-1)
            instruction = ep["instructions"][0]
            video_dir = ep["video"]  # e.g. "images/Geneva-1_satnav_000000"

            for w_start in range(0, len(real_actions), window_size):
                w_end = min(w_start + window_size, len(real_actions))
                action_window = list(real_actions[w_start:w_end])
                while len(action_window) < window_size:
                    action_window.append(0)  # pad with STOP

                self.samples.append({
                    "video_dir": video_dir,
                    "instruction": instruction,
                    "actions": action_window,
                    "n_history_frames": w_start + 1,
                    "episode_id": ep["id"],
                    "window_idx": w_start // window_size,
                })

        rank0_print(
            f"SatNavUniNaVidDataset: {len(episodes)} episodes -> "
            f"{len(self.samples)} windowed samples (window_size={window_size})"
        )

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [s["n_history_frames"] + 100 for s in self.samples]

    @property
    def modality_lengths(self):
        return [s["n_history_frames"] + 100 for s in self.samples]

    # ------------------------------------------------------------------

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                return self._load_sample(self.samples[i])
            except Exception as e:
                attempt += 1
                print(f"[SatNavDataset] Error loading sample {i}: {e}, retrying...")
                i = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed to load sample after {max_attempt} attempts")

    # ------------------------------------------------------------------

    def _load_sample(self, sample):
        video_folder = self.data_args.video_folder
        frame_dir = os.path.join(video_folder, sample["video_dir"], "rgb")

        # --- 1. Load JPEG frames ---
        frames = []
        for fi in range(1, sample["n_history_frames"] + 1):
            path = os.path.join(frame_dir, f"{fi:03d}.jpg")
            frames.append(np.array(Image.open(path).convert("RGB")))
        video = np.stack(frames, axis=0)  # (N, H, W, 3)

        # --- 2. Navigation augmentation (matches original LazySupervisedDataset) ---
        if len(video) > 1:
            last_idx = len(video) - 1
            max_drop = math.ceil(0.1 * last_idx)
            n_sample = last_idx - random.randint(0, max_drop)
            sampled = sorted(random.sample(range(last_idx), max(n_sample, 0)))
            sampled.append(last_idx)
            sampled = duplicate_with_probability(sampled, 0.03)
            video = video[sampled]
            video = random_color_jitter(video)

        # --- 3. CLIP image processor ---
        processor = self.data_args.image_processor
        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]

        # --- 4. Build conversation ---
        action_text = format_action_target(sample["actions"])
        prompt = PROMPT_TEMPLATE.format(
            nav_id=NAVIGATION_IDENTIFIER,
            img_token=DEFAULT_IMAGE_TOKEN,
            instruction=sample["instruction"],
        )
        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": action_text},
        ]

        # --- 5. Multimodal preprocessing (moves <image> to front, etc.) ---
        sources = preprocess_multimodal(
            copy.deepcopy([conversations]),
            self.data_args,
        )

        # --- 6. Tokenize ---
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt,
            video_or_not=True,
        )

        prompt_out = data_dict.get("prompt", None)
        input_ids = data_dict["input_ids"][0]

        # --- 7. Fix labels: preprocess_imgsp_v1 uses +6 offset for navigation
        # special tokens, but each special token string tokenizes to 2 sub-tokens
        # (space + token), so the net addition is +12, not +6. This causes the
        # computed instruction_len to be too small and the response tokens to be
        # fully masked. We recompute labels by finding the ASSISTANT separator.
        labels = self._build_labels(input_ids)

        data_dict = dict(input_ids=input_ids, labels=labels)
        data_dict["image"] = image
        if prompt_out is not None:
            data_dict["prompt"] = prompt_out

        return data_dict

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Rebuild labels by finding the ASSISTANT response boundary.

        Everything before (and including) the 'ASSISTANT:' separator is
        masked to IGNORE_INDEX (-100). The response tokens are kept.

        Note: preprocess_imgsp_v1 uses +6 offset for navigation special tokens,
        but each special-token string tokenizes to 2 sub-tokens (space + token ID),
        so the net addition is +12. This makes instruction_len too small and the
        trailing mask too early, masking the entire response. We bypass that logic
        and find the ASSISTANT separator directly in input_ids instead.
        """
        IGNORE_INDEX = -100
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # "ASSISTANT:" (without leading space) = [319, 1799, 9047, 13566, 29901]
        # We search for the token sequence without the leading space because
        # the assembled vicuna prompt ends with ". ASSISTANT:" where "." may
        # merge differently with the space during tokenization.
        sep_ids = self.tokenizer(
            "ASSISTANT:", return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        n, s = len(input_ids), len(sep_ids)
        for i in range(n - s):
            if (input_ids[i:i + s] == sep_ids).all():
                # response starts right after "ASSISTANT:"
                response_start = i + s
                labels[response_start:] = input_ids[response_start:]
                return labels

        rank0_print(
            "[SatNavDataset] WARNING: ASSISTANT separator not found in input_ids; "
            "labels remain all -100 for this sample."
        )
        return labels
