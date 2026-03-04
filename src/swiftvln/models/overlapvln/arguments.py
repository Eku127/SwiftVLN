# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Training Arguments

Extends StreamVLN training arguments with overlap and compression parameters.
QA mixed training parameters are inherited from BaseVLNTrainArguments.
"""

from dataclasses import dataclass, field

from swiftvln.common.base_arguments import BaseVLNTrainArguments


@dataclass
class OverlapVLNTrainArguments(BaseVLNTrainArguments):
    """
    OverlapVLN training arguments.
    
    Extends BaseVLNTrainArguments with:
    - History processor parameters (processor_type, compress_stride, etc.)
    - Overlap parameters (num_overlap)
    
    QA mixed training parameters (qa_dataset, qa_ratio, qa_max_samples) are
    inherited from BaseVLNTrainArguments.
    
    History Processing Options:
    - processor_type='per_frame': Per-frame compression (default)
      - log_base=1.0: Uniform sampling
      - log_base>1.0: Logarithmic sampling (more recent frames)
      - use_tome=False: Average pooling
      - use_tome=True: Grid-based Token Merging
    - processor_type='gtc': Global Token Clustering (cross-frame)
      - Clusters all history tokens into fixed output_tokens
    """
    
    # ==========================================================================
    # History Processor Selection
    # ==========================================================================
    history_processor_type: str = field(
        default='per_frame',
        metadata={
            "help": "Type of history information processor. Options: "
                    "'per_frame' (default): Per-frame compression with flexible sampling. "
                    "'gtc': Global Token Clustering, cross-frame clustering to fixed tokens."
        }
    )
    
    # ==========================================================================
    # Per-frame parameters (used when processor_type='per_frame')
    # ==========================================================================
    compress_stride: int = field(
        default=2,
        metadata={
            "help": "[Per-frame] Pooling stride for history frame compression. "
                    "stride=2 gives 4x compression (256->64 tokens), "
                    "stride=3 gives 9x compression (256->28 tokens), "
                    "stride=4 gives 16x compression (256->16 tokens)."
        }
    )
    
    use_tome: bool = field(
        default=False,
        metadata={
            "help": "[Per-frame] Use GridToMe (Grid-based Token Merging) instead of "
                    "average pooling. GridToMe provides better semantic preservation "
                    "for small objects at a slight speed cost. Default: False."
        }
    )
    
    log_base: float = field(
        default=1.0,
        metadata={
            "help": "[Per-frame] Sampling distribution for history frames. "
                    "1.0 = uniform sampling (default), "
                    ">1.0 = logarithmic sampling (more recent frames preserved). "
                    "2.0 = moderate, 3.0+ = aggressive concentration on recent frames."
        }
    )
    
    # ==========================================================================
    # GTC parameters (used when processor_type='gtc')
    # ==========================================================================
    gtc_output_tokens: int = field(
        default=512,
        metadata={
            "help": "[GTC] Fixed number of output tokens for Global Token Clustering. "
                    "All history frames are clustered into exactly this many tokens. "
                    "Default: 512."
        }
    )
    
    gtc_temperature: float = field(
        default=0.1,
        metadata={
            "help": "[GTC] Temperature for soft assignment in Soft K-Means. "
                    "Lower = sharper assignment, higher = smoother. Default: 0.1."
        }
    )
    
    gtc_num_iterations: int = field(
        default=1,
        metadata={
            "help": "[GTC] Number of Soft K-Means iterations. "
                    "1-2 iterations usually sufficient. Default: 1."
        }
    )
    
    # ==========================================================================
    # Overlap parameters for sliding window training
    # ==========================================================================
    num_overlap: int = field(
        default=16,
        metadata={
            "help": "Number of overlapping actions between consecutive windows. "
                    "When num_overlap > 0, the sliding window stride = num_frames - num_overlap. "
                    "The first num_overlap actions in non-first samples will have their loss masked. "
                    "Set to 0 to disable overlap (original behavior)."
        }
    )
    
    # ==========================================================================
    # System prompt setting
    # ==========================================================================
    system_prompt_setting: str = field(
        default='vanilla',
        metadata={
            "help": "System prompt strategy. Options: "
                    "'vanilla' (default): Standard prompt without initial view image. "
                    "'initial': Add the first frame of the episode (uncompressed) to the "
                    "system prompt as the initial observation at the starting point."
        }
    )
    
    # ==========================================================================
    # Embedding Enhancement
    # ==========================================================================
    use_pixel_embed: bool = field(
        default=False,
        metadata={
            "help": "Enable pixel coordinate embedding enhancement (MLP_xy). "
                    "When enabled, adds learnable Fourier-encoded pixel coordinate embeddings "
                    "to ViT features after visual encoding. Uses zero-initialization to ensure "
                    "no behavior change at the start of training. Default: False."
        }
    )

    use_pose_embed: bool = field(
        default=False,
        metadata={
            "help": "Enable pose embedding enhancement. "
                    "Injects per-image pose [delta_forward, delta_right, sin(dh), cos(dh)] "
                    "into ViT features after visual encoding. Default: False."
        }
    )

    pose_fusion_method: str = field(
        default='additive',
        metadata={
            "help": "Pose embedding fusion method: 'additive' or 'film'. "
                    "Default: additive."
        }
    )

    pose_norm_scale: float = field(
        default=100.0,
        metadata={
            "help": "Pose positional normalization scale for tanh(pos/scale). "
                    "Only affects delta_forward/delta_right. Default: 100.0."
        }
    )
