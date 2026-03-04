# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Multi-Environment Evaluation Entry Point

This script runs VLN evaluation using the trained MonoVLN model
with history frame compression in multiple environments (Habitat or SatNav).

Key features:
- Single-turn dialogue format (matching training)
- History frame compression during inference
- Auto-regressive evaluation (each step is independent)

Usage:
    # Habitat evaluation (default)
    python -m swiftvln.models.monovln.eval --model_path /path/to/checkpoint --env-type habitat
    
    # SatNav evaluation
    python -m swiftvln.models.monovln.eval --model_path /path/to/checkpoint --env-type satnav \
        --satnav-config configs/satnav_task.yaml
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m swiftvln.models.monovln.eval \
        --model_path /path/to/checkpoint --env-type habitat --distributed
"""

# ============================================================================
# CRITICAL: Force NVIDIA EGL before ANY imports
# This MUST be set before importing habitat/habitat_sim to prevent Mesa fallback
# Without this, Habitat rendering can be 1000x slower on machines with Mesa installed
# ============================================================================
import os
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

import sys

# Setup paths for both module and direct execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_msswift_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
_vln_dir = os.path.dirname(_current_dir)

if _msswift_root not in sys.path:
    sys.path.insert(0, _msswift_root)
if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import base eval class
try:
    from swiftvln.common import BaseVLNEval
except ImportError:
    from common import BaseVLNEval


class MonoVLNEval(BaseVLNEval):
    """MonoVLN evaluation implementation."""
    
    model_type = 'monovln_qwen2_5_vl'
    template_type = 'monovln_qwen2_5_vl'
    model_description = 'MonoVLN'
    uses_compression = True
    uses_num_frames = False
    
    def register_module(self):
        """Import MonoVLN module to register model."""
        try:
            import swiftvln.models.monovln
        except ImportError:
            pass
    
    def load_template(self, processor):
        """Load MonoVLN template."""
        from swift.llm import get_template
        
        template = get_template(
            template_type=self.template_type,
            processor=processor
        )
        
        return template
    
    @property
    def evaluator_class(self):
        """Lazy load evaluator class to avoid circular imports."""
        try:
            from swiftvln.models.monovln.evaluator import MonoVLNEvaluator
            return MonoVLNEvaluator
        except ImportError:
            from evaluator import MonoVLNEvaluator
            return MonoVLNEvaluator


def main():
    eval_runner = MonoVLNEval()
    eval_runner.run()


if __name__ == "__main__":
    main()
