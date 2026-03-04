# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Training Arguments

Extends StreamVLN training arguments with compression-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional

from swiftvln.common.training.arguments import BaseVLNTrainArguments


@dataclass
class CompressVLNTrainArguments(BaseVLNTrainArguments):
    """
    CompressVLN training arguments.
    
    Extends BaseVLNTrainArguments with compression-specific parameters.
    All base VLN parameters (num_frames, num_history, etc.) and QA parameters are inherited.
    """
    
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
    
    # Precomputed features parameters
    use_precomputed_features: bool = field(
        default=False,
        metadata={
            "help": "Whether to use precomputed ViT features instead of images. "
                    "When enabled, ViT forward is skipped and features are loaded from disk."
        }
    )
    
    feature_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing precomputed features (.pt files). "
                    "If None, auto-inferred as {data_path}/features. "
                    "Each episode should have a corresponding {episode_id}.pt file."
        }
    )
    
    feature_cache_size: int = field(
        default=100,
        metadata={
            "help": "LRU cache size for episode features (number of episodes). "
                    "Higher values use more CPU memory but reduce disk I/O. "
                    "100 episodes ≈ 16GB CPU memory for 640x480 images."
        }
    )
