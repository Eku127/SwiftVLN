# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Multi-Environment Evaluation Entry Point

This script runs VLN evaluation using the trained CompressVLN model
with history frame compression in multiple environments (Habitat or SatNav).

Key difference from StreamVLN:
- Uses CompressVLN model and template with history compression
- Compression applies during inference as well

Usage:
    # Habitat evaluation (default)
    python -m swiftvln.models.compressvln.eval --model_path /path/to/checkpoint --env-type habitat
    
    # SatNav evaluation
    python -m swiftvln.models.compressvln.eval --model_path /path/to/checkpoint --env-type satnav \
        --satnav-config configs/satnav_task.yaml
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m swiftvln.models.compressvln.eval \
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


class CompressVLNEval(BaseVLNEval):
    """CompressVLN evaluation implementation."""
    
    model_type = 'compressvln_qwen2_5_vl'
    template_type = 'compressvln_qwen2_5_vl'
    model_description = 'CompressVLN'
    uses_compression = True
    uses_num_frames = True
    
    def register_module(self):
        """Import CompressVLN module to register model."""
        try:
            import swiftvln.models.compressvln
        except ImportError:
            pass
    
    def load_template(self, processor):
        """Load CompressVLN template."""
        from swift.llm import get_template
        
        template = get_template(
            template_type=self.template_type,
            processor=processor
        )
        
        return template
    
    @property
    def evaluator_class(self):
        """Lazy load evaluator class to avoid circular imports."""
        from swiftvln.models.compressvln.evaluator import CompressVLNEvaluator
        return CompressVLNEvaluator


def main():
    eval_runner = CompressVLNEval()
    eval_runner.run()


if __name__ == "__main__":
    main()
