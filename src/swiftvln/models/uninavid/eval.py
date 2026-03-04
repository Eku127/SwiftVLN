# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Multi-Environment Evaluation Entry Point

This script runs VLN evaluation using the trained UniNaVid model
with three-level memory compression and incremental feature caching.

Key features:
- Three-level memory architecture (long-term, short-term, current)
- Incremental feature caching (only encode new frame each step)
- Single-turn dialogue format (matching training)
- Auto-regressive evaluation

Usage:
    # Habitat evaluation (default)
    python -m swiftvln.models.uninavid.eval --model_path /path/to/checkpoint --env-type habitat
    
    # SatNav evaluation
    python -m swiftvln.models.uninavid.eval --model_path /path/to/checkpoint --env-type satnav \
        --satnav-config configs/satnav_task.yaml
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m swiftvln.models.uninavid.eval \
        --model_path /path/to/checkpoint --env-type habitat --distributed
"""

# ============================================================================
# CRITICAL: Force NVIDIA EGL before ANY imports
# This MUST be set before importing habitat/habitat_sim to prevent Mesa fallback
# Without this, Habitat rendering can be 1000x slower on machines with Mesa installed
# ============================================================================
import os
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

from swiftvln.common import BaseVLNEval


class UniNaVidEval(BaseVLNEval):
    """UniNaVid evaluation implementation with incremental feature caching."""
    
    model_type = 'uninavid_qwen2_5_vl'
    template_type = 'uninavid_qwen2_5_vl'
    model_description = 'UniNaVid'
    uses_compression = True
    uses_num_frames = False
    
    def add_model_specific_args(self, parser):
        """Add UniNaVid-specific arguments."""
        parser.add_argument("--short_term_frames", type=int, default=32,
                            help="Number of frames in short-term memory")
        parser.add_argument("--similarity_threshold", type=float, default=0.985,
                            help="Cosine similarity threshold for long-term merging")
        parser.add_argument("--image_resize_stride", type=float, default=1.5,
                            help="Image resize factor (1.0 = no resize)")
        parser.add_argument("--verbose", action="store_true",
                            help="Print verbose output during evaluation")
    
    def register_module(self):
        """Import UniNaVid module to register model."""
        import swiftvln.models.uninavid  # noqa: F401
    
    def load_template(self, processor):
        """Load UniNaVid template with eval_mode enabled."""
        from swift.llm import get_template
        
        template = get_template(
            template_type=self.template_type,
            processor=processor
        )
        
        # Enable eval mode for incremental processing
        if hasattr(template, 'eval_mode'):
            template.eval_mode = True
        
        return template
    
    @property
    def evaluator_class(self):
        """Lazy load evaluator class to avoid circular imports."""
        from swiftvln.models.uninavid.evaluator import UniNaVidEvaluator
        return UniNaVidEvaluator


def main():
    eval_runner = UniNaVidEval()
    eval_runner.run()


if __name__ == "__main__":
    main()
