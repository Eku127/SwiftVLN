"""Common components for VLN training and evaluation."""

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
from .env import EnvWrapper
from .history_processors import (
    GlobalTokenClustering,
    HistoryProcessor,
    HistoryProcessorConfig,
    PerFrameCompressor,
    create_history_processor,
)
from .history_processors.compressor import HistoryTokenCompressor
from .training import (
    BaseVLNSft,
    BaseVLNTrainArguments,
)

_lazy_imports = {
    'HabitatEnvWrapper': '.env',
    'SatNavEnvWrapper': '.env',
    'BaseVLNEvaluator': '.eval',
    'BaseVLNEval': '.eval',
    'compute_trajectory_type_stats': '.eval',
    'clean_results_for_output': '.eval',
    'save_timing_stats': '.eval',
    'get_swanlab_url': '.eval',
    'get_swanlab_url_from_train_metadata': '.eval',
    'append_text_to_image': '.utils',
    'init_distributed': '.utils',
    'compress_videos': '.utils',
    'TrajectoryRecorder': '.utils',
    'ErrorAnalyzer': '.utils',
}


def __getattr__(name):
    """Lazy import for simulator-dependent components."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'BaseVLNTrainArguments',
    'BaseVLNSft',
    'DEFAULT_IMAGE_TOKEN',
    'HISTORY_MEMORY_TOKEN',
    'CURRENT_IMAGE_TOKEN',
    'DEFAULT_ACTION_MAP',
    'DEFAULT_CONJUNCTIONS',
    'PROMPT_TEMPLATE_HABITAT',
    'PROMPT_TEMPLATE_SATNAV',
    'get_navigation_prompt_template',
    'format_navigation_prompt',
    'EnvWrapper',
    'HabitatEnvWrapper',
    'SatNavEnvWrapper',
    'BaseVLNEvaluator',
    'BaseVLNEval',
    'compute_trajectory_type_stats',
    'clean_results_for_output',
    'save_timing_stats',
    'get_swanlab_url',
    'get_swanlab_url_from_train_metadata',
    'HistoryTokenCompressor',
    'HistoryProcessor',
    'HistoryProcessorConfig',
    'PerFrameCompressor',
    'GlobalTokenClustering',
    'create_history_processor',
    'append_text_to_image',
    'init_distributed',
    'compress_videos',
    'TrajectoryRecorder',
    'ErrorAnalyzer',
]
