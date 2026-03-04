"""Common components for VLN training and evaluation.

This module provides unified abstractions for VLN that work across:
- Different model architectures (StreamVLN, CompressVLN, OverlapVLN)
- Different simulator backends (Habitat, SatNav)
- Mixed training with VLN + QA data

Training components (always available):
- BaseVLNTrainArguments: Base training arguments with QA support
- MixedVLNQADataset: Dataset for mixed VLN + QA training
- VLNMixedTrainingMixin: Mixin for trainers to support mixed training

Evaluation components (lazily imported to avoid simulator dependencies):
- EnvWrapper, HabitatEnvWrapper, SatNavEnvWrapper
- BaseVLNEvaluator, BaseVLNEval
"""

# Core training components (no external dependencies)
from .env_wrapper import EnvWrapper
from .compressor import HistoryTokenCompressor
from .history_processors import (
    HistoryProcessor,
    HistoryProcessorConfig,
    PerFrameCompressor,
    GlobalTokenClustering,
    create_history_processor,
)
from .base_arguments import BaseVLNTrainArguments
from .base_sft import BaseVLNSft
from .constants import (
    CURRENT_IMAGE_TOKEN,
    DEFAULT_ACTION_MAP,
    DEFAULT_CONJUNCTIONS,
    DEFAULT_IMAGE_TOKEN,
    HISTORY_MEMORY_TOKEN,
    PROMPT_TEMPLATE_HABITAT,
    PROMPT_TEMPLATE_SATNAV,
    format_navigation_prompt,
    get_navigation_prompt_template,
)
from .mixed_dataset import MixedVLNQADataset
from .trainer_mixin import VLNMixedTrainingMixin
from .base_dataset import BaseVLNDataset

# Lazy imports for simulator-dependent components
# These will only be imported when actually accessed
_lazy_imports = {
    'HabitatEnvWrapper': '.habitat_wrapper',
    'SatNavEnvWrapper': '.satnav_wrapper',
    'BaseVLNEvaluator': '.base_evaluator',
    'BaseVLNEval': '.base_eval',
    'append_text_to_image': '.utils',
    'init_distributed': '.utils',
    'gather_metrics': '.utils',
    'compress_videos': '.utils',
    'TrajectoryRecorder': '.utils',
    'ErrorAnalyzer': '.utils',
}

def __getattr__(name):
    """Lazy import for simulator-dependent components."""
    if name in _lazy_imports:
        module_name = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_name, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Training components
    'BaseVLNTrainArguments',
    'BaseVLNSft',
    'MixedVLNQADataset',
    'VLNMixedTrainingMixin',
    # Dataset components
    'BaseVLNDataset',
    'DEFAULT_IMAGE_TOKEN',
    'HISTORY_MEMORY_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    'DEFAULT_ACTION_MAP',
    'DEFAULT_CONJUNCTIONS',
    'PROMPT_TEMPLATE_HABITAT',
    'PROMPT_TEMPLATE_SATNAV',
    'get_navigation_prompt_template',
    'format_navigation_prompt',
    # Environment wrappers
    'EnvWrapper',
    'HabitatEnvWrapper',
    'SatNavEnvWrapper',
    # Base evaluator
    'BaseVLNEvaluator',
    # Base eval entry point
    'BaseVLNEval',
    # Compressor (legacy, used internally by PerFrameCompressor)
    'HistoryTokenCompressor',
    # History Processors (new unified interface)
    'HistoryProcessor',
    'HistoryProcessorConfig',
    'PerFrameCompressor',
    'GlobalTokenClustering',
    'create_history_processor',
    # Utils
    'append_text_to_image',
    'init_distributed',
    'gather_metrics',
    'compress_videos',
    'TrajectoryRecorder',
    'ErrorAnalyzer',
]
