"""Public facade for SatNav explored-map memory."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .cache import (
    CACHE_DISABLE_SENTINELS as _CACHE_DISABLE_SENTINELS,
    MAP_CACHE_FORMAT_VERSION as _MAP_CACHE_FORMAT_VERSION,
    MapCacheMixin,
    map_debug_dir as _map_debug_dir,
    map_debug_enabled as _map_debug_enabled,
    map_debug_limit as _map_debug_limit,
    resolve_cache_dir as _resolve_cache_dir,
    sanitize_debug_name as _sanitize_debug_name,
)
from .geometry import (
    MapGeometryMixin,
    parse_action as _parse_action,
    require_geo_dependencies as _require_geo_deps,
    wrap_heading_deg as _wrap_heading_deg,
)
from .metadata import (
    MapPose,
    SatNavEpisodeMetadata,
    SatNavTrajectoryMetadataResolver,
    _normalize_episode_list,
    normalize_scene_name as _normalize_scene_name,
)
from .render import MapRenderMixin


class SatNavMapMemoryBuilder(
    MapCacheMixin,
    MapGeometryMixin,
    MapRenderMixin,
):
    """Render fixed-size explored map-memory images for SatNav."""

    def __init__(
        self,
        scenes_dir: str,
        global_side_m: float = 1000.0,
        local_side_m: float = 400.0,
        render_px: int = 448,
        mask_method: str = "dilate20",
        hfov: float = 90.0,
        sensor_width: int = 448,
        sensor_height: int = 448,
        global_center_mode: str = "adaptive_start",
        cache_dir: Optional[str] = None,
    ):
        self.scenes_dir = os.path.abspath(scenes_dir)
        self.global_side_m = float(global_side_m)
        self.local_side_m = float(local_side_m)
        self.render_px = int(render_px)
        self.mask_method = str(mask_method).lower()
        self.hfov = float(hfov)
        self.sensor_width = int(sensor_width)
        self.sensor_height = int(sensor_height)
        self.global_center_mode = str(global_center_mode).lower()

        if self.global_side_m <= 0 or self.local_side_m <= 0:
            raise ValueError("map side lengths must be > 0.")
        if self.render_px <= 0:
            raise ValueError("map_render_px must be > 0.")
        if self.sensor_width <= 0 or self.sensor_height <= 0:
            raise ValueError("sensor dimensions must be > 0.")
        if self.global_center_mode not in {"start", "adaptive_start"}:
            raise ValueError(
                "global_center_mode must be 'start' or 'adaptive_start', "
                f"got {self.global_center_mode}."
            )

        self._wgs84_to_mercator = None
        self._scene_cache: Dict[str, Any] = {}
        self._debug_render_count = 0
        self._global_shift_margin_ratio = 0.10
        self._global_shift_quantize_m = 25.0
        self._configure_cache(cache_dir)


__all__ = [
    "MapPose",
    "SatNavEpisodeMetadata",
    "SatNavMapMemoryBuilder",
    "SatNavTrajectoryMetadataResolver",
]
