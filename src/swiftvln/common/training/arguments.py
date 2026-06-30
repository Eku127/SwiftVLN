# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN Training Arguments

Provides common training arguments used across all VLN variants (StreamVLN, CompressVLN, SwiftVLN).
"""

from dataclasses import dataclass, field
from typing import Optional

from swift.arguments import SftArguments as TrainArguments


@dataclass
class BaseVLNTrainArguments(TrainArguments):
    """
    Base VLN training arguments.

    Contains parameters shared across all VLN variants:
    - Basic VLN parameters (num_frames, num_history, etc.)
    """
    
    # ============================================================================
    # VLN Core Parameters
    # ============================================================================
    num_frames: int = field(
        default=32, 
        metadata={"help": "Window size for trajectory segmentation"}
    )
    num_history: int = field(
        default=8, 
        metadata={"help": "Number of historical frames to sample"}
    )
    num_future_steps: int = field(
        default=4, 
        metadata={"help": "Number of actions to predict per round"}
    )
    use_random: bool = field(
        default=False, 
        metadata={"help": "Use random sampling for history (false=uniform)"}
    )
    # Environment type: determines forward distance in prompts
    vln_env_type: str = field(
        default="satnav",
        metadata={
            "help": "VLN environment type: 'habitat' (forward=0.25m) or 'satnav' (forward=10m). "
                    "This affects the action description in prompts."
        }
    )

    # Limit dataset size
    vln_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of VLN training samples"}
    )
