"""Stage-A sim-to-real alignment utilities for SwiftVLN.

Public Stage-A objects are loaded lazily so the raw-data generation package can
run in a lightweight environment without importing PyTorch or Transformers.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PairRecord": (".dataset", "PairRecord"),
    "SatDronePairDataset": (".dataset", "SatDronePairDataset"),
    "build_manifest_records": (".dataset", "build_manifest_records"),
    "deduplicate_gta_rows": (".dataset", "deduplicate_gta_rows"),
    "load_manifest": (".dataset", "load_manifest"),
    "bidirectional_contrastive_loss": (
        ".losses",
        "bidirectional_contrastive_loss",
    ),
    "compute_retrieval_metrics": (".losses", "compute_retrieval_metrics"),
    "global_cosine_loss": (".losses", "global_cosine_loss"),
    "ProjectionHead": (".model", "ProjectionHead"),
    "Sim2RealAdapter": (".model", "Sim2RealAdapter"),
    "TeacherVisionTower": (".model", "TeacherVisionTower"),
    "masked_mean_pool": (".model", "masked_mean_pool"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve the historical top-level exports on first access."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
