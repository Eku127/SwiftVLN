"""Habitat configuration loading, isolated from the core package."""

from __future__ import annotations

import os
from typing import Any

from swiftvln.backends.base import BackendDependencyError


def load_habitat_config(config_path: str, args: Any):
    try:
        import habitat
        from habitat.config.default_structured_configs import (
            CollisionsMeasurementConfig,
            FogOfWarConfig,
            TopDownMapMeasurementConfig,
        )
        from habitat_baselines.config.default import get_config
        from swiftvln.backends.habitat import measures  # noqa: F401
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "Habitat backend is unavailable because optional dependency "
            f"{exc.name!r} is not installed. Install the validated Habitat-Lab, "
            "Habitat-Sim, and habitat-baselines environment before using "
            "--env-type habitat."
        ) from exc

    config = get_config(config_path)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.eval_split
        data_path = os.environ.get("SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH")
        scenes_dir = os.environ.get("SWIFTVLN_HABITAT_SCENES_DIR")
        if data_path:
            config.habitat.dataset.data_path = data_path
        if scenes_dir:
            config.habitat.dataset.scenes_dir = scenes_dir
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(
            os.environ.get("LOCAL_RANK", 0)
        )
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return config
