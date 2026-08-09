"""Explicit backend factory without eager simulator imports."""

from __future__ import annotations

import importlib
from typing import Any

from .base import EvaluationBackend
from .specs import get_environment_spec


_BACKEND_CLASSES = {
    "habitat": "swiftvln.backends.habitat.backend:HabitatBackend",
    "satnav": "swiftvln.backends.satnav.backend:SatNavBackend",
}


def create_backend(name: str, config_path: str, args: Any) -> EvaluationBackend:
    spec = get_environment_spec(name)
    module_name, class_name = _BACKEND_CLASSES[spec.name].split(":", 1)
    module = importlib.import_module(module_name)
    backend_class = getattr(module, class_name)
    return backend_class(config_path, args)
