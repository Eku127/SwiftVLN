"""Unified command-line launcher for SatDronePair data generation."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .registry import get_registry, normalize_dataset_name, resolve_module


CONFIG_EXAMPLE_PATH = Path(__file__).resolve().parent / "config.example.yaml"
_PATH_KEYS = {
    "data_root",
    "dataset_dir",
    "dataset_root",
    "output",
    "output_dir",
    "sat_bounds_csv",
}


def _format_supported_commands() -> str:
    lines = ["Supported dataset commands:"]
    for dataset, commands in get_registry().items():
        lines.append(f"  - {dataset}: {', '.join(sorted(commands))}")
    lines.extend(
        [
            "",
            "Dataset aliases: gta-uav -> gta_uav, uav-visloc -> uavvisloc",
        ]
    )
    return "\n".join(lines)


def _usage() -> str:
    return (
        "Usage:\n"
        "  python -m swiftvln.s2r.data_generation "
        "[--config CONFIG] <dataset> <command> [args...]\n"
        "  swiftvln s2r-data "
        "[--config CONFIG] <dataset> <command> [args...]\n\n"
        "A config file is optional. Without one, pass the source command's "
        "required arguments explicitly.\n"
        f"Packaged config example: {CONFIG_EXAMPLE_PATH}\n\n"
        f"{_format_supported_commands()}"
    )


def run_command(dataset: str, command: str, forwarded_args: Sequence[str]) -> int:
    """Run a registered command without mutating process-global ``sys.argv``."""
    module_path = resolve_module(dataset, command)
    module = importlib.import_module(module_path)
    module_main = getattr(module, "main", None)
    if module_main is None:
        raise AttributeError(f"Module {module_path} does not define main()")

    result = module_main(list(forwarded_args))
    return int(result) if result is not None else 0


def _flag_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _is_flag_overridden(flag: str, forwarded_args: Sequence[str]) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in forwarded_args)


def _expand_config_value(key: str, value: Any, config_dir: Path) -> Any:
    """Expand environment/user paths and anchor relative configured paths."""
    if isinstance(value, list):
        return [_expand_config_value(key, item, config_dir) for item in value]
    if not isinstance(value, str):
        return value

    expanded = os.path.expandvars(os.path.expanduser(value))
    if key == "variant" and "=" in expanded:
        name, raw_path = expanded.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return f"{name}={path}"
    if key not in _PATH_KEYS or not expanded:
        return expanded

    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def _append_cli_value(
    parts: List[str],
    key: str,
    value: Any,
    config_dir: Path,
) -> None:
    if value is None or value is False:
        return

    flag = _flag_name(key)
    if isinstance(value, bool):
        parts.append(flag)
        return

    expanded = _expand_config_value(key, value, config_dir)
    if isinstance(expanded, (list, tuple)):
        if not expanded:
            return
        parts.append(flag)
        parts.extend(str(item) for item in expanded)
        return

    parts.extend([flag, str(expanded)])


def _extract_global_options(
    args: Sequence[str],
) -> Tuple[Optional[Path], List[str]]:
    """Extract ``--config`` from any position before command dispatch."""
    config_path: Optional[Path] = None
    remaining: List[str] = []
    index = 0
    raw_args = list(args)

    while index < len(raw_args):
        current = raw_args[index]
        if current == "--config":
            if index + 1 >= len(raw_args):
                raise ValueError("--config requires a path")
            if config_path is not None:
                raise ValueError("--config may only be provided once")
            config_path = Path(raw_args[index + 1]).expanduser()
            index += 2
            continue
        if current.startswith("--config="):
            if config_path is not None:
                raise ValueError("--config may only be provided once")
            config_path = Path(current.split("=", 1)[1]).expanduser()
            index += 1
            continue
        remaining.append(current)
        index += 1

    return config_path, remaining


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config {config_path}: {exc}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def _get_command_config(
    config: Mapping[str, Any],
    dataset: str,
    command: str,
) -> Mapping[str, Any]:
    datasets = config.get("DATASETS") or config.get("datasets") or {}
    if not isinstance(datasets, Mapping):
        raise ValueError("Config field DATASETS must be a mapping")

    normalized = {
        normalize_dataset_name(str(name)): value
        for name, value in datasets.items()
    }
    dataset_cfg = normalized.get(normalize_dataset_name(dataset), {})
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError(f"Dataset config for {dataset} must be a mapping")

    command_cfg = dataset_cfg.get(command, {})
    if not isinstance(command_cfg, Mapping):
        raise ValueError(f"Command config for {dataset}.{command} must be a mapping")
    return command_cfg


def _build_config_args(
    command_cfg: Mapping[str, Any],
    forwarded_args: Sequence[str],
    config_dir: Path,
) -> List[str]:
    if any(arg in {"-h", "--help"} for arg in forwarded_args):
        return []

    generated: List[str] = []
    for section_name in ("input", "output", "args"):
        section = (
            command_cfg.get(section_name)
            or command_cfg.get(section_name.upper())
            or {}
        )
        if not isinstance(section, Mapping):
            raise ValueError(f"Config section {section_name} must be a mapping")
        for key, value in section.items():
            flag = _flag_name(str(key))
            if _is_flag_overridden(flag, forwarded_args):
                continue
            _append_cli_value(
                generated,
                str(key),
                value,
                config_dir,
            )
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for SatDronePair source-data conversion commands."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    try:
        config_path, args = _extract_global_options(raw_args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if not args or args[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0

    if len(args) < 2:
        print("Expected <dataset> and <command>.", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    dataset, command = args[0], args[1]
    forwarded_args = args[2:]

    try:
        config = _load_config(config_path)
        command_config = _get_command_config(config, dataset, command)
        config_dir = config_path.resolve().parent if config_path else Path.cwd()
        config_args = _build_config_args(
            command_config,
            forwarded_args,
            config_dir,
        )
        return run_command(dataset, command, config_args + forwarded_args)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
