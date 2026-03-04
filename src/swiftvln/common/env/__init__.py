# Copyright (c) Alibaba, Inc. and its affiliates.
"""Environment wrapper abstractions."""

from .base import EnvWrapper

__all__ = [
    'EnvWrapper',
    'HabitatEnvWrapper',
    'SatNavEnvWrapper',
]

_lazy_imports = {
    'HabitatEnvWrapper': '.habitat',
    'SatNavEnvWrapper': '.satnav',
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
