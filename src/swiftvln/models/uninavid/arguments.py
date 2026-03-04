# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Arguments

Defines UniNaVid-specific parameters for training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UniNaVidArguments:
    """
    Arguments specific to UniNaVid training.
    
    These parameters control the three-level memory architecture:
    - Long-term memory: Frames older than short_term_frames, merged by similarity
    - Short-term memory: Recent short_term_frames frames, compressed with Grid Pooling
    - Current frame: Full resolution for detailed observation
    
    Attributes:
        short_term_frames: Number of recent frames as short-term memory.
                          Frames older than this are classified as long-term memory.
                          Default: 64 (following Uni-NaVid paper)
        similarity_threshold: Cosine similarity threshold for merging long-term memory.
                             Higher values mean more aggressive merging.
                             Default: 0.985 (following Uni-NaVid paper)
        compress_stride: Pooling stride for history frame compression.
                        2 means 2x2 pooling (4x compression).
                        Default: 2
        drop_frame_prob: Probability of randomly dropping each history frame.
                        Used for data augmentation during training.
                        Default: 0.1 (10%)
        num_future_steps: Number of actions to predict per step.
                         Default: 4
        samples_per_episode: Number of time points to sample per episode.
                            Default: 8
    """
    
    short_term_frames: int = field(
        default=64,
        metadata={
            "help": "Number of recent frames as short-term memory. "
                    "Frames older than this are classified as long-term memory."
        }
    )
    
    similarity_threshold: float = field(
        default=0.985,
        metadata={
            "help": "Cosine similarity threshold for merging long-term memory frames. "
                    "Higher values mean more aggressive merging."
        }
    )
    
    compress_stride: int = field(
        default=2,
        metadata={
            "help": "Pooling stride for history frame compression. "
                    "2 means 2x2 pooling (4x compression)."
        }
    )
    
    drop_frame_prob: float = field(
        default=0.1,
        metadata={
            "help": "Probability of randomly dropping each history frame during training. "
                    "Set to 0 to disable frame dropping."
        }
    )
    
    num_future_steps: int = field(
        default=4,
        metadata={
            "help": "Number of actions to predict per step."
        }
    )
    
    samples_per_episode: int = field(
        default=8,
        metadata={
            "help": "Number of time points to sample per episode."
        }
    )
    
    image_resize_stride: float = field(
        default=1.0,
        metadata={
            "help": "Stride to resize images before feeding to VIT. "
                    "2 means resize to 1/2 size (e.g., 640x480 -> 320x240). "
                    "1.5 means resize to 2/3 size (e.g., 640x480 -> 427x320). "
                    "1.0 means no resize (default)."
        }
    )
    
    history_frame_stride: int = field(
        default=1,
        metadata={
            "help": "Temporal downsampling stride for history frames (not current frame). "
                    "2 means take every 2nd frame (50%% of history frames). "
                    "3 means take every 3rd frame (33%% of history frames). "
                    "1 means no downsampling (default). "
                    "Applied before random frame dropping."
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
    
    def __post_init__(self):
        """Validate argument values."""
        if self.short_term_frames < 1:
            raise ValueError(f"short_term_frames must be >= 1, got {self.short_term_frames}")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}")
        
        if self.compress_stride < 1:
            raise ValueError(f"compress_stride must be >= 1, got {self.compress_stride}")
        
        if not 0.0 <= self.drop_frame_prob <= 1.0:
            raise ValueError(f"drop_frame_prob must be in [0, 1], got {self.drop_frame_prob}")
        
        if self.num_future_steps < 1:
            raise ValueError(f"num_future_steps must be >= 1, got {self.num_future_steps}")
        
        if self.samples_per_episode < 1:
            raise ValueError(f"samples_per_episode must be >= 1, got {self.samples_per_episode}")
        
        if self.image_resize_stride < 1.0:
            raise ValueError(f"image_resize_stride must be >= 1.0, got {self.image_resize_stride}")
        
        if self.history_frame_stride < 1:
            raise ValueError(f"history_frame_stride must be >= 1, got {self.history_frame_stride}")