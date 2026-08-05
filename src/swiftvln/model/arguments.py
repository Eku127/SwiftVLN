# Copyright (c) Alibaba, Inc. and its affiliates.
"""Complete SwiftVLN training argument schema."""

from dataclasses import dataclass, field
from typing import Optional

from swift.arguments import SftArguments as TrainArguments

from swiftvln.experiment import EMBEDDING_MODES, SwiftVLNExperimentSpec


@dataclass
class SwiftVLNTrainArguments(TrainArguments):
    """
    SwiftVLN training arguments.

    Contains the complete train CLI surface instead of extending a single-use
    repository-local base dataclass.

    History Processing Options:
    - processor_type='per_frame': Per-frame compression (default)
      - log_base=1.0: Uniform sampling
      - log_base>1.0: Logarithmic sampling (more recent frames)
      - use_tome=False: Average pooling
      - use_tome=True: Grid-based Token Merging
    - processor_type='gtc': Global Token Clustering (cross-frame)
      - Clusters all history tokens into fixed output_tokens
    """

    num_frames: int = field(
        default=32,
        metadata={"help": "Window size for trajectory segmentation"},
    )
    num_history: int = field(
        default=8,
        metadata={"help": "Number of historical frames to sample"},
    )
    num_future_steps: int = field(
        default=4,
        metadata={"help": "Number of actions to predict per round"},
    )
    use_random: bool = field(
        default=False,
        metadata={"help": "Use random sampling for history (false=uniform)"},
    )
    vln_env_type: str = field(
        default="satnav",
        metadata={
            "help": "VLN environment type: 'habitat' (forward=0.25m) or "
            "'satnav' (forward=10m)."
        },
    )
    vln_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of VLN training samples"},
    )

    # ==========================================================================
    # History Processor Selection
    # ==========================================================================
    history_processor_type: str = field(
        default="per_frame",
        metadata={
            "help": "Type of history information processor. Options: "
            "'per_frame' (default): Per-frame compression with flexible sampling. "
            "'gtc': Global Token Clustering across all history frames. "
            "'segment_gtc': chronological segment-wise token clustering."
        },
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
        },
    )

    use_tome: bool = field(
        default=False,
        metadata={
            "help": "[Per-frame] Use GridToMe (Grid-based Token Merging) instead of "
            "average pooling. GridToMe provides better semantic preservation "
            "for small objects at a slight speed cost. Default: False."
        },
    )

    log_base: float = field(
        default=1.0,
        metadata={
            "help": "[Per-frame] Sampling distribution for history frames. "
            "1.0 = uniform sampling (default), "
            ">1.0 = logarithmic sampling (more recent frames preserved). "
            "2.0 = moderate, 3.0+ = aggressive concentration on recent frames."
        },
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
        },
    )

    gtc_temperature: float = field(
        default=0.1,
        metadata={
            "help": "[GTC] Temperature for soft assignment in Soft K-Means. "
            "Lower = sharper assignment, higher = smoother. Default: 0.1."
        },
    )

    gtc_num_iterations: int = field(
        default=1,
        metadata={
            "help": "[GTC] Number of Soft K-Means iterations. "
            "1-2 iterations usually sufficient. Default: 1."
        },
    )

    # ==========================================================================
    # Overlap parameters for sliding window training
    # ==========================================================================
    num_overlap: int = field(
        default=0,
        metadata={
            "help": "Number of overlapping actions between consecutive windows. "
            "When num_overlap > 0, the sliding window stride = num_frames - num_overlap. "
            "The first num_overlap actions in non-first samples will have their loss masked. "
            "Set to 0 to disable overlap (original behavior)."
        },
    )

    # ==========================================================================
    # System prompt setting
    # ==========================================================================
    system_prompt_setting: str = field(
        default="vanilla",
        metadata={
            "help": "System prompt strategy. Options: "
            "'vanilla' (default): Standard prompt without initial view image. "
            "'initial': Add the first frame of the episode (uncompressed) to the "
            "system prompt as the initial observation at the starting point."
        },
    )

    memory_method: str = field(
        default="history",
        metadata={
            "help": "History memory source. "
            "'history' (default): sample historical RGB frames. "
            "'map': replace history frames with SatNav explored-map memory."
        },
    )

    map_global_side_m: float = field(
        default=1000.0,
        metadata={"help": "[map] Global explored-map side length in true meters."},
    )

    map_local_side_m: float = field(
        default=400.0,
        metadata={"help": "[map] Local explored-map side length in true meters."},
    )

    map_render_px: int = field(
        default=448, metadata={"help": "[map] Render resolution for each map image."}
    )

    map_mask_method: str = field(
        default="dilate20",
        metadata={
            "help": "[map] Explored-area mask rule. "
            "Supported: 'strict', 'dilate20', or 'dilate<N>'."
        },
    )

    # ==========================================================================
    # Embedding Enhancement
    # ==========================================================================
    embedding_mode: str = field(
        default="none",
        metadata={
            "help": "Exactly one embedding enhancement mode: "
            "'none', 'pose' (additive), 'posefilm' (FiLM), or 'uav'.",
            "choices": EMBEDDING_MODES,
        },
    )

    uav_adapter_path: str = field(
        default="",
        metadata={
            "help": "Optional external Stage-A checkpoint (.pt or s2r output dir) used to "
            "initialize the UAV adapter."
        },
    )

    uav_adapter_type: str = field(
        default="transformer_v1",
        metadata={"help": "UAV adapter implementation type. Default: transformer_v1."},
    )

    uav_adapter_apply_scope: str = field(
        default="all_images",
        metadata={
            "help": "Where to apply the UAV adapter. Current Stage-B implementation only "
            "supports 'all_images'."
        },
    )

    pose_norm_scale: float = field(
        default=100.0,
        metadata={
            "help": "Pose positional normalization scale for tanh(pos/scale). "
            "Only affects delta_forward/delta_right. Default: 100.0."
        },
    )

    def __post_init__(self) -> None:
        SwiftVLNExperimentSpec(
            env_type=self.vln_env_type,
            model_family=(
                "qwen3_vl" if self.model_type == "swiftvln_qwen3_vl" else "qwen2_5_vl"
            ),
            num_frames=self.num_frames,
            num_future_steps=self.num_future_steps,
            num_overlap=self.num_overlap,
            memory_method=self.memory_method,
            history_processor_type=self.history_processor_type,
            num_history=self.num_history,
            log_base=self.log_base,
            use_random=self.use_random,
            compress_stride=self.compress_stride,
            use_tome=self.use_tome,
            gtc_output_tokens=self.gtc_output_tokens,
            gtc_temperature=self.gtc_temperature,
            gtc_num_iterations=self.gtc_num_iterations,
            map_global_side_m=self.map_global_side_m,
            map_local_side_m=self.map_local_side_m,
            map_render_px=self.map_render_px,
            map_mask_method=self.map_mask_method,
            system_prompt_setting=self.system_prompt_setting,
            embedding=self.embedding_mode,
        )
        super().__post_init__()
