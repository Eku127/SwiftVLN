"""Static registry for SatDronePair data-generation commands."""

from __future__ import annotations

from typing import Dict, Mapping


_PACKAGE = "swiftvln.s2r.data_generation"

DATASET_COMMANDS: Dict[str, Dict[str, str]] = {
    "denseuav": {
        "build_pairs": f"{_PACKAGE}.denseuav.build_pairs",
        "sample_preview": f"{_PACKAGE}.sample_preview",
    },
    "gta_uav": {
        "build_pairs": f"{_PACKAGE}.gta_uav.build_pairs",
        "sample_preview": f"{_PACKAGE}.sample_preview",
    },
    "sues": {
        "build_pairs": f"{_PACKAGE}.sues.build_pairs",
        "pipeline": f"{_PACKAGE}.sues.pipeline",
        "center_recrop_pairs": f"{_PACKAGE}.sues.center_recrop_pairs",
        "sample_preview": f"{_PACKAGE}.sample_preview",
        "merge_variants_dense_style": (
            f"{_PACKAGE}.sues.merge_variants_dense_style"
        ),
        "merge_variants": f"{_PACKAGE}.merge_variants",
    },
    "uavvisloc": {
        "build_pairs": f"{_PACKAGE}.uavvisloc.build_pairs",
        "export_selected": f"{_PACKAGE}.uavvisloc.export_selected",
        "center_recrop_pairs": f"{_PACKAGE}.uavvisloc.center_recrop_pairs",
        "sample_preview": f"{_PACKAGE}.sample_preview",
        "merge_variants": f"{_PACKAGE}.merge_variants",
    },
}

DATASET_ALIASES = {
    "denseuav": "denseuav",
    "gta_uav": "gta_uav",
    "gta-uav": "gta_uav",
    "sues": "sues",
    "uavvisloc": "uavvisloc",
    "uav-visloc": "uavvisloc",
}


def normalize_dataset_name(name: str) -> str:
    """Normalize a dataset name accepted by the launcher."""
    key = name.strip().lower()
    return DATASET_ALIASES.get(key, key.replace("-", "_"))


def get_registry() -> Mapping[str, Mapping[str, str]]:
    """Return the immutable-by-convention command registry."""
    return DATASET_COMMANDS


def resolve_module(dataset: str, command: str) -> str:
    """Resolve ``dataset + command`` into an importable module path."""
    dataset_key = normalize_dataset_name(dataset)
    commands = DATASET_COMMANDS.get(dataset_key)
    if commands is None:
        raise KeyError(f"Unknown dataset: {dataset}")

    module_path = commands.get(command)
    if module_path is None:
        raise KeyError(f"Unknown command for {dataset_key}: {command}")
    return module_path
