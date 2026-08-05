# Copyright (c) Alibaba, Inc. and its affiliates.
"""Evaluation-related common components."""

from .reporting import (
    clean_results_for_output,
    compute_trajectory_type_stats,
    get_swanlab_url,
    get_swanlab_url_from_train_metadata,
    save_timing_stats,
)
from .results import ResultRecorder

_lazy_imports = {
    "BaseVLNEval": ".runner",
    "BaseVLNEvaluator": ".evaluator",
}


def __getattr__(name):
    """Delay simulator-dependent imports until the component is requested."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseVLNEval",
    "BaseVLNEvaluator",
    "ResultRecorder",
    "compute_trajectory_type_stats",
    "clean_results_for_output",
    "save_timing_stats",
    "get_swanlab_url",
    "get_swanlab_url_from_train_metadata",
]
