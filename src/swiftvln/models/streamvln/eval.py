# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Multi-Environment Evaluation Entry Point

This script runs VLN evaluation using the trained StreamVLNQwen25VL model
in multiple environments (Habitat or SatNav). Supports both single-process and distributed modes.

Usage:
    # Habitat evaluation (default)
    python -m swiftvln.models.streamvln.eval --model_path /path/to/checkpoint --env-type habitat
    
    # SatNav evaluation
    python -m swiftvln.models.streamvln.eval --model_path /path/to/checkpoint --env-type satnav \
        --satnav-config configs/satnav_task.yaml
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m swiftvln.models.streamvln.eval \
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


class StreamVLNEval(BaseVLNEval):
    """StreamVLN evaluation implementation."""
    
    model_type = 'streamvln_qwen2_5_vl'
    template_type = 'qwen2_5_vl'  # StreamVLN uses standard Qwen2.5-VL template
    model_description = 'StreamVLN'
    uses_compression = False
    uses_num_frames = True
    
    def register_module(self):
        """Import StreamVLN module to register model."""
        try:
            import swiftvln.models.streamvln
        except ImportError:
            pass
    
    def load_template(self, processor):
        """Load StreamVLN template (standard Qwen2.5-VL)."""
        from swift.llm import get_template
        from swift.llm.template import TemplateType
        
        template = get_template(
            template_type=TemplateType.qwen2_5_vl,
            processor=processor
        )
        
        return template
    
    def initialize_model(self, model):
        """Initialize StreamVLN model cache."""
        # StreamVLN requires model reset for streaming
        model.reset(env_num=1)
    
    @property
    def evaluator_class(self):
        """Lazy load evaluator class to avoid circular imports."""
        from swiftvln.models.streamvln.evaluator import StreamVLNEvaluator
        return StreamVLNEvaluator


def main():
    eval_runner = StreamVLNEval()
    eval_runner.run()


if __name__ == "__main__":
    main()
