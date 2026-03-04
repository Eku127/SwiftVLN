# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN Training Arguments

Provides common training arguments used across all VLN variants (StreamVLN, CompressVLN, OverlapVLN).
Includes QA mixed training parameters that can be shared.
"""

from dataclasses import dataclass, field
from typing import Optional

from swift.llm import TrainArguments


@dataclass
class BaseVLNTrainArguments(TrainArguments):
    """
    Base VLN training arguments.
    
    Contains parameters shared across all VLN variants:
    - Basic VLN parameters (num_frames, num_history, etc.)
    - QA mixed training parameters
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
        default="habitat",
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
    
    # ============================================================================
    # QA Mixed Training Parameters
    # ============================================================================
    qa_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to QA dataset (jsonl format) for mixed training with VLN data. "
                    "The QA dataset should be in ms-swift standard format with 'messages' and 'images' fields. "
                    "When provided, VLN and QA data will be interleaved during training. "
                    "Example: /path/to/qa_swift.jsonl"
        }
    )
    
    qa_ratio: float = field(
        default=0.2,
        metadata={
            "help": "Ratio of QA samples in each training batch when mixed training is enabled. "
                    "For example, 0.2 means 20%% QA samples and 80%% VLN samples. "
                    "Only effective when qa_dataset is provided. Default: 0.2"
        }
    )
    
    qa_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of QA samples to use. If None or 0, use all available samples. "
                    "Useful for balancing dataset sizes when VLN data is much larger/smaller than QA data."
        }
    )
