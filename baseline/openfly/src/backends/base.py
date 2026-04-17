from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import accelerate.optimizer
import torch


CONTINUE_BACKEND = "continue"
SCRATCH_BACKEND = "scratch"
SUPPORTED_BACKENDS = {CONTINUE_BACKEND, SCRATCH_BACKEND}

# Legacy aliases kept for any external references
HF_BACKEND = CONTINUE_BACKEND
NATIVE_BACKEND = SCRATCH_BACKEND
LEGACY_BACKEND_ALIASES = {
    "hf": CONTINUE_BACKEND,
    "native": SCRATCH_BACKEND,
    HF_BACKEND: CONTINUE_BACKEND,
    NATIVE_BACKEND: SCRATCH_BACKEND,
}


def resolve_backend(requested: str) -> str:
    requested = LEGACY_BACKEND_ALIASES.get(requested, requested)
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported OpenFly backend: {requested}")
    return requested


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            return
        except Exception:
            pass
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "gradient_checkpointing_enable"):
        language_model.gradient_checkpointing_enable()


def patch_accelerate_optimizer_train_eval() -> None:
    def _safe_train(self):
        train_fn = getattr(self.optimizer, "train", None)
        if callable(train_fn):
            return train_fn()
        return None

    def _safe_eval(self):
        eval_fn = getattr(self.optimizer, "eval", None)
        if callable(eval_fn):
            return eval_fn()
        return None

    accelerate.optimizer.AcceleratedOptimizer.train = _safe_train
    accelerate.optimizer.AcceleratedOptimizer.eval = _safe_eval


@dataclass
class TrainBackendArtifacts:
    backend_name: str
    processor: Any
    model: Any
    data_collator: Any
    metadata: dict[str, Any]
    backend_meta: dict[str, Any] = field(default_factory=dict)


def build_default_backend_meta(
    backend_name: str,
    model_name_or_path: str,
    processor_source: str,
    action_format: str,
    grid_size: int,
) -> dict[str, Any]:
    return {
        "backend": backend_name,
        "model_name_or_path": os.path.abspath(model_name_or_path),
        "processor_source": os.path.abspath(processor_source),
        "action_format": action_format,
        "grid_size": int(grid_size),
    }
