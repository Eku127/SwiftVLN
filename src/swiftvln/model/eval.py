# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Multi-Environment Evaluation Entry Point

This script runs VLN evaluation using the trained OverlapVLN model
with history frame compression in multiple environments (Habitat or SatNav).

Key difference from StreamVLN:
- Uses OverlapVLN model and template with history compression
- Compression applies during inference as well

Usage:
    # Habitat evaluation (default)
    python -m swiftvln.model.eval --model_path /path/to/checkpoint --env-type habitat
    
    # SatNav evaluation
    python -m swiftvln.model.eval --model_path /path/to/checkpoint --env-type satnav \
        --satnav-config configs/satnav_task.yaml
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m swiftvln.model.eval \
        --model_path /path/to/checkpoint --env-type habitat --distributed
"""

# ============================================================================
# CRITICAL: Force NVIDIA EGL before ANY imports
# This MUST be set before importing habitat/habitat_sim to prevent Mesa fallback
# Without this, Habitat rendering can be 1000x slower on machines with Mesa installed
# ============================================================================
import os
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

import torch

from swiftvln.common import BaseVLNEval


class OverlapVLNEval(BaseVLNEval):
    """OverlapVLN evaluation implementation."""
    
    model_type = 'overlapvln_qwen2_5_vl'
    template_type = 'overlapvln_qwen2_5_vl'
    model_description = 'OverlapVLN'
    uses_compression = True
    uses_num_frames = True
    
    def add_model_specific_args(self, parser):
        """Add OverlapVLN-specific arguments."""
        parser.add_argument("--model_type", type=str, default=self.model_type,
                            choices=["overlapvln_qwen2_5_vl", "overlapvln_qwen3_vl"],
                            help="Registered OverlapVLN model type")
        parser.add_argument("--template_type", type=str, default=self.template_type,
                            choices=["overlapvln_qwen2_5_vl", "overlapvln_qwen3_vl"],
                            help="Registered OverlapVLN template type")
        parser.add_argument("--num_overlap", type=int, default=0,
                            help="Number of overlapping actions between windows")
        parser.add_argument("--use_tome", action="store_true",
                            help="Use GridToMe compression instead of average pooling (per_frame mode)")
        parser.add_argument("--verbose", action="store_true",
                            help="Enable verbose output during evaluation")
        parser.add_argument("--debug_landmark", action="store_true",
                            help="Enable detailed debug output for landmark episodes (saves frames, trajectory analysis, model outputs)")
        
        # System prompt setting
        parser.add_argument("--system_prompt_setting", type=str, default="vanilla",
                            choices=["vanilla", "initial"],
                            help="System prompt strategy: 'vanilla' (default) or 'initial' "
                                 "(add first frame as uncompressed initial observation)")
        parser.add_argument("--memory_method", type=str, default="history",
                            choices=["history", "map"],
                            help="History memory source: raw history frames or SatNav explored maps")
        parser.add_argument("--map_global_side_m", type=float, default=1000.0,
                            help="[map] Global explored-map side length in meters")
        parser.add_argument("--map_local_side_m", type=float, default=400.0,
                            help="[map] Local explored-map side length in meters")
        parser.add_argument("--map_render_px", type=int, default=448,
                            help="[map] Render resolution for each map image")
        parser.add_argument("--map_mask_method", type=str, default="dilate20",
                            help="[map] Explored-area mask rule, e.g. strict or dilate20")
        
        # History processor type
        parser.add_argument("--history_processor_type", type=str, default="per_frame",
                            choices=["per_frame", "gtc", "segment_gtc"],
                            help="History processing method: 'per_frame' (default), 'gtc' (Global Token Clustering), 'segment_gtc' (Segment-wise GTC)")
        
        # Per-frame specific arguments
        parser.add_argument("--log_base", type=float, default=1.0,
                            help="[Per-frame] Sampling distribution: 1.0=uniform, >1.0=logarithmic (more recent frames)")
        parser.add_argument("--use_random", action="store_true",
                            help="[Per-frame] Use random history sampling without replacement (must match training)")
        
        # GTC-specific arguments
        parser.add_argument("--gtc_output_tokens", type=int, default=512,
                            help="[GTC] Fixed number of output tokens for clustering")
        parser.add_argument("--gtc_temperature", type=float, default=0.1,
                            help="[GTC] Temperature for soft assignment (lower = sharper)")
        parser.add_argument("--gtc_num_iterations", type=int, default=1,
                            help="[GTC] Number of soft k-means iterations")
        parser.add_argument("--use_pixel_embed", action="store_true",
                            help="Enable pixel coordinate embedding enhancement (must match training)")
        parser.add_argument("--use_pose_embed", action="store_true",
                            help="Enable pose embedding enhancement (must match training)")
        parser.add_argument("--use_uav_adapter", action="store_true",
                            help="Enable Stage-A UAV adapter enhancement (must match training)")
        parser.add_argument("--uav_adapter_path", type=str, default="",
                            help="Optional external Stage-A checkpoint (.pt or s2r output dir)")
        parser.add_argument("--uav_adapter_type", type=str, default="transformer_v1",
                            help="UAV adapter implementation type")
        parser.add_argument("--uav_adapter_apply_scope", type=str, default="all_images",
                            help="Where to apply the UAV adapter. Current implementation uses all_images.")
        parser.add_argument("--pose_fusion_method", type=str, default="additive",
                            choices=["additive", "film"],
                            help="Pose embedding fusion method")
        parser.add_argument("--pose_norm_scale", type=float, default=100.0,
                            help="tanh normalization scale for pose position components")
    
    def get_summary_extras(self):
        """Add OverlapVLN-specific summary fields."""
        extras = super().get_summary_extras()
        if hasattr(self.args, 'model_type'):
            extras['model_type'] = self.args.model_type
        if hasattr(self.args, 'template_type'):
            extras['template_type'] = self.args.template_type
        if hasattr(self.args, 'num_overlap'):
            extras['num_overlap'] = self.args.num_overlap
        if hasattr(self.args, 'history_processor_type'):
            extras['history_processor_type'] = self.args.history_processor_type
        if hasattr(self.args, 'system_prompt_setting'):
            extras['system_prompt_setting'] = self.args.system_prompt_setting
        if hasattr(self.args, 'memory_method'):
            extras['memory_method'] = self.args.memory_method
        if getattr(self.args, 'memory_method', 'history') == 'map':
            extras['map_global_side_m'] = getattr(self.args, 'map_global_side_m', 1000.0)
            extras['map_local_side_m'] = getattr(self.args, 'map_local_side_m', 400.0)
            extras['map_render_px'] = getattr(self.args, 'map_render_px', 448)
            extras['map_mask_method'] = getattr(self.args, 'map_mask_method', 'dilate20')
        if hasattr(self.args, 'use_pixel_embed'):
            extras['use_pixel_embed'] = self.args.use_pixel_embed
        if hasattr(self.args, 'use_pose_embed'):
            extras['use_pose_embed'] = self.args.use_pose_embed
        if hasattr(self.args, 'use_uav_adapter'):
            extras['use_uav_adapter'] = self.args.use_uav_adapter
        if hasattr(self.args, 'uav_adapter_path') and self.args.uav_adapter_path:
            extras['uav_adapter_path'] = self.args.uav_adapter_path
        if hasattr(self.args, 'uav_adapter_type'):
            extras['uav_adapter_type'] = self.args.uav_adapter_type
        if hasattr(self.args, 'uav_adapter_apply_scope'):
            extras['uav_adapter_apply_scope'] = self.args.uav_adapter_apply_scope
        if hasattr(self.args, 'pose_fusion_method'):
            extras['pose_fusion_method'] = self.args.pose_fusion_method
        if hasattr(self.args, 'pose_norm_scale'):
            extras['pose_norm_scale'] = self.args.pose_norm_scale
        
        history_type = getattr(self.args, 'history_processor_type', 'per_frame')
        
        # Add processor-specific fields
        if history_type in ('gtc', 'segment_gtc'):
            if hasattr(self.args, 'gtc_output_tokens'):
                extras['gtc_output_tokens'] = self.args.gtc_output_tokens
            if hasattr(self.args, 'gtc_temperature'):
                extras['gtc_temperature'] = self.args.gtc_temperature
        else:
            # per_frame
            if hasattr(self.args, 'log_base'):
                extras['log_base'] = self.args.log_base
            if hasattr(self.args, 'use_random'):
                extras['use_random'] = self.args.use_random
            if hasattr(self.args, 'use_tome'):
                extras['use_tome'] = self.args.use_tome
        return extras
    
    def register_module(self):
        """Import OverlapVLN module to register model."""
        try:
            import swiftvln.model
        except ImportError:
            pass

    def load_model(self):
        """Load model and processor with optional pixel embedding module."""
        from swift.model import get_model_processor

        # Device mapping based on mode
        if self.world_size > 1:
            device_map = {'': self.local_rank}
        else:
            device_map = 'auto'

        model, processor = get_model_processor(
            model_id_or_path=self.args.model_path,
            model_type=self.args.model_type,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_impl='flash_attn',
            use_pixel_embed=getattr(self.args, 'use_pixel_embed', False),
            use_pose_embed=getattr(self.args, 'use_pose_embed', False),
            use_uav_adapter=getattr(self.args, 'use_uav_adapter', False),
            uav_adapter_path=getattr(self.args, 'uav_adapter_path', ''),
            uav_adapter_type=getattr(self.args, 'uav_adapter_type', 'transformer_v1'),
            uav_adapter_apply_scope=getattr(self.args, 'uav_adapter_apply_scope', 'all_images'),
            pose_fusion_method=getattr(self.args, 'pose_fusion_method', 'additive'),
            pose_norm_scale=getattr(self.args, 'pose_norm_scale', 100.0),
        )
        return model, processor
    
    def load_template(self, processor):
        """Load OverlapVLN template."""
        from swift.template import get_template
        
        template = get_template(
            template_type=self.args.template_type,
            processor=processor
        )
        
        return template
    
    @property
    def evaluator_class(self):
        """Lazy load evaluator class to avoid circular imports."""
        from swiftvln.model.evaluator import OverlapVLNEvaluator
        return OverlapVLNEvaluator


def main():
    eval_runner = OverlapVLNEval()
    eval_runner.run()


if __name__ == "__main__":
    main()
