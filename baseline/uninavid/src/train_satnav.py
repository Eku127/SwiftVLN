"""
Training entry point for Uni-NaVid fine-tuning on SatNav data.

Does NOT modify the original Uni-NaVid repo.  Instead it:
  1. Adds Uni-NaVid repo to sys.path
  2. Applies the flash-attention monkey patch
  3. Monkey-patches make_supervised_data_module() to use SatNavUniNaVidDataset
  4. Calls the original train() function

Run via:  deepspeed baseline/uninavid/src/train_satnav.py  [args ...]
Or via:   bash baseline/uninavid/scripts/train_satnav.sh
"""

import os
import sys

# ------------------------------------------------------------------
# 1. Make Uni-NaVid and this baseline importable
# ------------------------------------------------------------------
_UNINAVID_ROOT = "/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
if _UNINAVID_ROOT not in sys.path:
    sys.path.insert(0, _UNINAVID_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

# ------------------------------------------------------------------
# 2. Flash-attention monkey patch
# NOTE: The Uni-NaVid flash-attn patch was written for flash_attn 1.x API
# (unpad_input returns 4 values), but our environment has flash_attn 2.x
# (returns 5 values).  Applying the broken patch causes a ValueError at
# the first training step.  We skip the patch and rely on transformers'
# built-in scaled_dot_product_attention, which is fast enough on H100
# and avoids all version-mismatch issues.
# ------------------------------------------------------------------
# (patch intentionally skipped)

# ------------------------------------------------------------------
# 3. Monkey-patch the data module builder
# ------------------------------------------------------------------
import uninavid.train.train as _train_mod
from dataset.satnav_dataset import SatNavUniNaVidDataset


def _make_satnav_data_module(tokenizer, data_args):
    """Drop-in replacement that loads SatNav windowed data."""
    train_dataset = SatNavUniNaVidDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = _train_mod.DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


_train_mod.make_supervised_data_module = _make_satnav_data_module

# ------------------------------------------------------------------
# 4. Run training (all model/trainer logic from original repo)
# ------------------------------------------------------------------
from uninavid.train.train import train

if __name__ == "__main__":
    train()
