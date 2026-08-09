"""SatNav configuration loading, isolated from the core package."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import OmegaConf


def load_satnav_config(config_path: str, args: Any):
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = args.eval_split

    raw_path = os.environ.get(
        "SWIFTVLN_SATNAV_EVAL_DATA_PATH", config.DATASET.DATA_PATH
    )
    scenes_dir = os.environ.get("SWIFTVLN_SATNAV_SCENES_DIR")
    if scenes_dir:
        config.DATASET.SCENES_DIR = scenes_dir
    if "{split}" in raw_path:
        resolved_path = raw_path.replace("{split}", args.eval_split)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"[SatNav] eval_split='{args.eval_split}' -> "
                f"'{resolved_path}' does not exist. Generate the requested "
                "SatNav episode split before evaluation."
            )
        config.DATASET.DATA_PATH = resolved_path
        print(
            f"[SatNav] eval_split='{args.eval_split}' -> "
            f"DATA_PATH: {resolved_path}"
        )
    else:
        config.DATASET.DATA_PATH = raw_path

    OmegaConf.set_struct(config, True)
    return config
