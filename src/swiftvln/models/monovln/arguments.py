# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Training Arguments

Defines training arguments for MonoVLN with single-turn dialogue format
and episode-based sampling.
"""

from dataclasses import dataclass, field
from typing import Optional

from swift.llm import TrainArguments


@dataclass
class MonoVLNTrainArguments(TrainArguments):
    """
    MonoVLN training arguments.
    
    Extends TrainArguments with VLN-specific and compression parameters.
    """
    
    # VLN parameters (MonoVLN uses single-frame inference, no num_frames needed)
    num_history: int = field(
        default=8,
        metadata={"help": "Maximum number of history frames to sample"}
    )
    
    num_future_steps: int = field(
        default=4,
        metadata={"help": "Number of actions to predict per step"}
    )
    
    vln_max_samples: int = field(
        default=0,
        metadata={"help": "Maximum number of training samples (0 = use all)"}
    )
    
    # MonoVLN-specific parameters
    samples_per_episode: int = field(
        default=8,
        metadata={
            "help": "Number of time points to sample per episode. "
                    "Samples are uniformly distributed with guaranteed start (no history) "
                    "and end (with STOP action) points."
        }
    )
    
    # Compression parameters
    compress_stride: int = field(
        default=2,
        metadata={
            "help": "Pooling stride for history frame compression. "
                    "stride=2 gives 4x compression (256->64 tokens), "
                    "stride=3 gives 9x compression (256->28 tokens), "
                    "stride=4 gives 16x compression (256->16 tokens)."
        }
    )
