"""Simulator backend boundary with dependency-free root imports."""

from .specs import EnvironmentSpec, get_environment_spec

__all__ = [
    "BackendDependencyError",
    "EnvironmentSpec",
    "EvaluationBackend",
    "EnvWrapper",
    "create_backend",
    "get_environment_spec",
]


def __getattr__(name: str):
    if name == "create_backend":
        from .factory import create_backend

        return create_backend
    if name in {"BackendDependencyError", "EvaluationBackend", "EnvWrapper"}:
        from . import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
