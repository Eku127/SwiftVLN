# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Central model registry for SwiftVLN.
"""

from __future__ import annotations

from typing import Dict, Tuple

MODEL_TO_PACKAGE: Dict[str, str] = {
    "swiftvln": "swiftvln.model",
}

MODEL_TO_TRAINER: Dict[str, str] = {
    model: f"{package}.trainer"
    for model, package in MODEL_TO_PACKAGE.items()
}

MODEL_TO_EVAL: Dict[str, str] = {
    model: f"{package}.eval"
    for model, package in MODEL_TO_PACKAGE.items()
}

MODEL_CHOICES: Tuple[str, ...] = tuple(MODEL_TO_PACKAGE.keys())


def ensure_supported_model(model: str) -> None:
    if model not in MODEL_TO_PACKAGE:
        supported = ", ".join(MODEL_CHOICES)
        raise ValueError(f"Unsupported model: {model}. Supported: {supported}")
