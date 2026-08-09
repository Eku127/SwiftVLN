"""SatNav backend integration."""

__all__ = ["SatNavEnvWrapper"]


def __getattr__(name: str):
    if name == "SatNavEnvWrapper":
        from .wrapper import SatNavEnvWrapper

        return SatNavEnvWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
