"""
Training entry point for NaVILA fine-tuning on SatNav data.

This keeps the upstream NaVILA training stack intact and only swaps in a
SatNav-specific data module.
"""

import os
import sys
from unittest import mock

_NAVILA_ROOT = os.path.abspath(
    os.path.expanduser(os.environ.get("NAVILA_REPO", "../NaVILA"))
)
if _NAVILA_ROOT not in sys.path:
    sys.path.insert(0, _NAVILA_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

import llava.train.train as train_mod
from dataset.satnav_dataset import SatNavNaVILADataset
from llava.data.dataset import DataCollatorForSupervisedDataset
from llava.train.transformer_normalize_monkey_patch import patched_normalize


def _patched_make_supervised_data_module(tokenizer, data_args, training_args):
    train_dataset = SatNavNaVILADataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    training_args.sample_lens = [len(train_dataset)]
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    return {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }


def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()


if __name__ == "__main__":
    train_mod.make_supervised_data_module = _patched_make_supervised_data_module
    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=__len__),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=__iter__),
    ):
        train_mod.train()
