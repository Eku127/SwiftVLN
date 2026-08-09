"""Habitat backend integration."""

__all__ = ["HabitatEnvWrapper"]


def __getattr__(name: str):
    if name == "HabitatEnvWrapper":
        from .wrapper import HabitatEnvWrapper

        return HabitatEnvWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
