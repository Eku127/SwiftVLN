"""
SatNavActionDataset: SatNav trajectory dataset for StreamVLN training.

Rewritten from StreamVLN's VLNActionDataset (streamvln/dataset/vln_action_dataset.py),
keeping the same logic but adapted for SatNav data format:
  - SatNav actions: -1=INIT (skip), 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
  - FORWARD step size: 10 meters (vs 25 centimeters in R2R)
  - actions[0] is always -1 (INIT placeholder, skip it)
  - actions[-1] is always 0 (STOP, already present, no need to append)
  - video dir structure: <video_folder>/<video>/rgb/*.jpg  (same as StreamVLN)
"""
import os
import sys

# Ensure StreamVLN repo is importable
_STREAMVLN_ROOT = "/mnt/data1/home/jiangjiajun/workspace/StreamVLN"
if _STREAMVLN_ROOT not in sys.path:
    sys.path.insert(0, _STREAMVLN_ROOT)

# streamvln/model/stream_video_vln.py uses `from utils.utils import ...`
# which resolves relative to StreamVLN/streamvln/, so add that dir too.
_STREAMVLN_SUBPKG = os.path.join(_STREAMVLN_ROOT, "streamvln")
if _STREAMVLN_SUBPKG not in sys.path:
    sys.path.insert(0, _STREAMVLN_SUBPKG)

import json
import copy
import random

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Sequence
from PIL import Image
from functools import partial

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava import conversation as conversation_lib

from streamvln.utils.utils import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
)
from streamvln.args import DataArguments

# Import tokenization helpers from the official training script
# (preprocess_qwen is defined inside streamvln_train.py; we replicate it here
#  by importing from the dataset module which has its own copy)
from streamvln.dataset.vln_action_dataset import (
    preprocess,
    pad_tensors,
)


class SatNavActionDataset(Dataset):
    """
    Dataset for SatNav trajectory data, following the same structure as
    StreamVLN's VLNActionDataset.

    annotations.json format per entry:
        {
            "id": int,
            "trajectory_id": int,
            "steps": int,
            "video": "images/<scene_satnav_id>",   # relative to video_folder
            "instructions": [str, ...],
            "actions": [-1, 1, 1, ..., 0]          # -1=INIT, 0=STOP, 1=FWD, 2=L, 3=R
        }
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        task_id: int = 0,
    ):
        super(SatNavActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        # Support comma-separated list of video folders (same as original)
        self.video_folder = data_args.video_folder.split(",")

        # Load all annotations
        self.nav_data = []
        for vf in self.video_folder:
            anno_path = os.path.join(vf, "annotations.json")
            anno_json = json.load(open(anno_path, "r"))
            for tdata in anno_json:
                # Prepend video folder so video path is absolute
                tdata["video"] = os.path.join(vf, tdata["video"])
            self.nav_data += anno_json

        # Build data_list: (ep_id, ins_id, start_idx, valid_idx)
        # Each entry represents one training chunk of up to num_frames actions.
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item["instructions"]
            # SatNav: actions[0] == -1 (INIT), skip it; actions[-1] == 0 (STOP)
            # Effective actions start from index 1
            actions = item["actions"][1:]  # skip INIT; STOP is already included
            actions_len = len(actions)

            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]

            for ins_id in range(len(instructions)):
                valid_idx = 0
                # remove_init_turns not applicable for SatNav (no pure-rotation start)
                if self.remove_init_turns:
                    valid_idx = self._clean_initial_rotations(instructions[ins_id], actions)

                if actions_len - valid_idx < 4:
                    continue

                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        # Action index → symbol mapping (same as original)
        self.idx2actions = {
            "0": "STOP",
            "1": "↑",
            "2": "←",
            "3": "→",
        }

        # Random conjunction templates (same as original)
        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]
        self.act_conjunctions = [
            "and then ",
            "after that ",
            "next ",
            "the next action is ",
            "followed by ",
            "leading to ",
            "continuing ",
            "subsequently ",
            "proceeding to ",
        ]

        # Navigation prompt — adapted for SatNav (10 meters forward, 15 deg turn)
        prompt = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 10 meters, or STOP."
        )
        answer = ""
        self.conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ]

    def __len__(self):
        return len(self.data_list)

    @property
    def task(self):
        return self.task_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_initial_rotations(self, instruction: str, actions: List[int]) -> int:
        """
        Skip leading turns that don't correspond to the instruction start.
        Mirrors original VLNActionDataset.clean_initial_rotations logic.
        For SatNav, this is typically not needed but kept for completeness.
        """
        valid_idx = 0
        for i, action in enumerate(actions):
            if action == 1:  # FORWARD
                valid_idx = i
                break
        return valid_idx

    def actions2text(self, actions: List[int]) -> str:
        """Convert a list of action indices to a text string of symbols."""
        converted = []
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if isinstance(act_text, list):
                act_text = random.choice(act_text)
            converted.append(act_text)
        return "".join(converted)

    def prepare_conversation(
        self,
        conversation: List[Dict],
        actions: List[int],
    ) -> List[Dict]:
        """
        Build interleaved (image, action) conversation turns.
        Same structure as original VLNActionDataset.prepare_conversation.
        """
        i = 0
        sources = []
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i : i + self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            source[1]["value"] = answer
            i += len(step_actions)
            sources.extend(source)
        return sources

    # ------------------------------------------------------------------
    # Dataset __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, i):
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        data = self.nav_data[ep_id]
        video_path = data["video"]
        video_frames = sorted(os.listdir(os.path.join(video_path, "rgb")))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        # SatNav: skip INIT (-1 at index 0), STOP (0) is already at the end
        actions = data["actions"][1:]  # shape: [steps]
        actions_len = len(actions)

        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0, f"Empty time_ids for ep_id={ep_id}, start_idx={start_idx}"

        actions_chunk = np.array(actions)[time_ids]

        start_idx_frame = time_ids[0] + valid_idx
        end_idx_frame = time_ids[-1] + 1 + valid_idx
        interval = self.num_future_steps
        sample_step_ids = np.arange(start_idx_frame, end_idx_frame, interval, dtype=np.int32)
        sample_frames = [
            os.path.join(video_path, "rgb", video_frames[idx])
            for idx in sample_step_ids
        ]

        if time_ids[0] != 0:
            history_step_ids = np.arange(
                0 + valid_idx,
                time_ids[0] + valid_idx,
                max(time_ids[0] // self.num_history, 1),
            )
            history_frames = [
                os.path.join(video_path, "rgb", video_frames[idx])
                for idx in history_step_ids
            ]
        else:
            history_frames = []

        # Load and process images
        images = []
        for image_file in history_frames + sample_frames:
            image = Image.open(image_file).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
            image = self.image_processor.preprocess(
                images=image, return_tensors="pt"
            )["pixel_values"][0]  # [3, H, W]
            images.append(image)

        images = torch.stack(images)  # [T, 3, H, W]

        # Build conversation
        sources = copy.deepcopy(self.conversations)

        if start_idx != 0:
            sources[0]["value"] += (
                f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
            )

        sources[0]["value"] = sources[0]["value"].replace(
            "<instruction>.", instructions[ins_id]
        )
        interleave_sources = self.prepare_conversation(sources, list(actions_chunk))

        data_dict = preprocess([interleave_sources], self.tokenizer, has_image=True)

        return (
            data_dict["input_ids"][0],
            data_dict["labels"][0],
            images,
            torch.tensor(time_ids),
            self.task,
        )


# ------------------------------------------------------------------
# Collate function (same logic as original collate_fn)
# ------------------------------------------------------------------

def satnav_collate_fn(batch, tokenizer):
    """
    Collate a list of (input_ids, labels, images, time_ids, task_type) tuples.
    Identical logic to StreamVLN's collate_fn in vln_action_dataset.py.
    """
    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(
        *batch
    )

    # Pad text sequences
    input_ids_batch = pad_sequence(
        input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_batch = pad_sequence(
        labels_batch, batch_first=True, padding_value=IGNORE_INDEX
    )

    input_ids_batch = input_ids_batch[:, : tokenizer.model_max_length]
    labels_batch = labels_batch[:, : tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)

    img_lens = np.array([img.size(0) for img in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(
            time_ids_batch, batch_first=True, padding_value=-1
        )

    image_batch = pad_tensors(image_batch, img_lens)

    return {
        "images": image_batch,
        "time_ids": time_ids_batch,
        "attention_mask": attention_mask,
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "task_type": task_type_batch,
    }


def make_satnav_collate_fn(tokenizer):
    """Return a collate_fn with tokenizer bound via partial."""
    return partial(satnav_collate_fn, tokenizer=tokenizer)
