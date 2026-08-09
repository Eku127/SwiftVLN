"""Geospatial conversion, scene access, and footprint geometry for map memory."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .cache import map_debug_enabled
from .metadata import MapPose, normalize_scene_name


def require_geo_dependencies():
    """Import optional geo libraries only when map rendering needs them."""
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError(
            "memory_method=map requires optional geo dependencies: "
            "`rasterio` and `pyproj`."
        ) from exc
    return rasterio, Transformer, Resampling, WarpedVRT, from_bounds


def wrap_heading_deg(angle: float) -> float:
    return angle % 360.0


def parse_action(action: Any) -> Tuple[bool, bool, bool]:
    if isinstance(action, str):
        action_key = action.strip().upper()
        return (
            action_key in ("TURN_LEFT", "LEFT", "L", "2"),
            action_key in ("TURN_RIGHT", "RIGHT", "R", "3"),
            action_key in ("MOVE_FORWARD", "FORWARD", "F", "1"),
        )
    action_int = int(action)
    return action_int == 2, action_int == 3, action_int == 1


class MapGeometryMixin:
    """Own coordinate conversion, raster access, and camera footprints."""

    def _ensure_transformer(self) -> None:
        if self._wgs84_to_mercator is None:
            _, transformer_type, _, _, _ = require_geo_dependencies()
            self._wgs84_to_mercator = transformer_type.from_crs(
                "EPSG:4326",
                "EPSG:3857",
                always_xy=True,
            )

    def _wgs84_to_mercator_xy(
        self,
        lon: float,
        lat: float,
    ) -> Tuple[float, float]:
        self._ensure_transformer()
        x_coordinate, y_coordinate = self._wgs84_to_mercator.transform(
            lon,
            lat,
        )
        return float(x_coordinate), float(y_coordinate)

    def _true_meters_to_mercator(
        self,
        distance_m: float,
        mercator_y: float,
    ) -> float:
        earth_radius = 6378137.0
        latitude_radians = math.atan(
            math.sinh(mercator_y / earth_radius)
        )
        scale_factor = 1.0 / math.cos(latitude_radians)
        return float(distance_m) * scale_factor

    def _move_in_mercator(
        self,
        x_coordinate: float,
        y_coordinate: float,
        distance_m: float,
        heading_deg: float,
    ) -> Tuple[float, float]:
        distance_mercator = self._true_meters_to_mercator(
            distance_m,
            y_coordinate,
        )
        heading_radians = math.radians(heading_deg)
        delta_x = distance_mercator * math.sin(heading_radians)
        delta_y = distance_mercator * math.cos(heading_radians)
        return (
            float(x_coordinate + delta_x),
            float(y_coordinate + delta_y),
        )

    def integrate_poses_from_actions(
        self,
        start_position: Sequence[float],
        start_rotation: float,
        actions: Iterable[Any],
        step_size: float,
        turn_angle: float,
    ) -> List[MapPose]:
        if len(start_position) < 3:
            raise ValueError(
                "start_position must be [lon, lat, altitude]."
            )

        longitude = float(start_position[0])
        latitude = float(start_position[1])
        altitude = float(start_position[2])
        x_coordinate, y_coordinate = self._wgs84_to_mercator_xy(
            longitude,
            latitude,
        )
        heading_deg = wrap_heading_deg(float(start_rotation))

        poses: List[MapPose] = []
        for action in actions:
            is_left, is_right, is_forward = parse_action(action)
            if is_left:
                heading_deg = wrap_heading_deg(
                    heading_deg - turn_angle
                )
            elif is_right:
                heading_deg = wrap_heading_deg(
                    heading_deg + turn_angle
                )
            elif is_forward:
                x_coordinate, y_coordinate = self._move_in_mercator(
                    x_coordinate,
                    y_coordinate,
                    step_size,
                    heading_deg,
                )
            poses.append(
                MapPose(
                    x=x_coordinate,
                    y=y_coordinate,
                    altitude=altitude,
                    heading_deg=heading_deg,
                )
            )

        if not poses:
            poses.append(
                MapPose(
                    x=x_coordinate,
                    y=y_coordinate,
                    altitude=altitude,
                    heading_deg=heading_deg,
                )
            )
        return poses

    def _resolve_scene_path(self, scene_ref: str) -> str:
        candidate = str(scene_ref)
        if candidate.lower().endswith(".tif") and os.path.exists(candidate):
            return candidate

        scene_name = normalize_scene_name(candidate)
        tif_path = os.path.join(self.scenes_dir, f"{scene_name}.tif")
        if not os.path.exists(tif_path):
            raise FileNotFoundError(
                f"SatNav scene tif not found: {tif_path}"
            )
        return tif_path

    def _get_scene_dataset(self, scene_ref: str):
        tif_path = self._resolve_scene_path(scene_ref)
        cached = self._scene_cache.get(tif_path)
        if cached is not None:
            return cached

        rasterio, _, _, warped_vrt_type, _ = require_geo_dependencies()
        source = rasterio.open(tif_path)
        if source.crs is not None and source.crs.to_epsg() == 3857:
            dataset = source
        else:
            dataset = warped_vrt_type(source, crs="EPSG:3857")
        self._scene_cache[tif_path] = dataset
        if map_debug_enabled():
            print(
                "[MAP DEBUG][builder] opened scene dataset: "
                f"scene={normalize_scene_name(tif_path)} path={tif_path}"
            )
        return dataset

    def _read_rgb_crop(
        self,
        scene_ref: str,
        center_pose: MapPose,
        side_m: float,
    ) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        dataset = self._get_scene_dataset(scene_ref)
        _, _, resampling_type, _, from_bounds = require_geo_dependencies()
        half_side_mercator = self._true_meters_to_mercator(
            side_m / 2.0,
            center_pose.y,
        )
        left = center_pose.x - half_side_mercator
        right = center_pose.x + half_side_mercator
        bottom = center_pose.y - half_side_mercator
        top = center_pose.y + half_side_mercator

        window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=dataset.transform,
        )
        array = dataset.read(
            indexes=[1, 2, 3],
            window=window,
            out_shape=(3, self.render_px, self.render_px),
            boundless=True,
            fill_value=0,
            resampling=resampling_type.bilinear,
        )
        array = np.transpose(array, (1, 2, 0))
        array = np.clip(array, 0, 255).astype(
            np.uint8,
            copy=False,
        )
        return (
            Image.fromarray(array, mode="RGB"),
            (left, right, bottom, top),
        )

    def _mercator_to_pixel(
        self,
        x_coordinate: float,
        y_coordinate: float,
        bounds: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        left, right, bottom, top = bounds
        if right <= left or top <= bottom:
            return 0.0, 0.0
        pixel_x = (
            (x_coordinate - left) / (right - left) * self.render_px
        )
        pixel_y = (
            (top - y_coordinate) / (top - bottom) * self.render_px
        )
        return float(pixel_x), float(pixel_y)

    def _footprint_polygon_mercator(
        self,
        pose: MapPose,
    ) -> List[Tuple[float, float]]:
        aspect_ratio = self.sensor_width / max(1, self.sensor_height)
        half_x_true = pose.altitude * math.tan(
            math.radians(self.hfov / 2.0)
        )
        half_y_true = half_x_true / aspect_ratio
        scale = self._true_meters_to_mercator(1.0, pose.y)
        corners_true = [
            (+half_x_true, +half_y_true),
            (-half_x_true, +half_y_true),
            (-half_x_true, -half_y_true),
            (+half_x_true, -half_y_true),
        ]

        heading_radians = math.radians(pose.heading_deg)
        cosine = math.cos(heading_radians)
        sine = math.sin(heading_radians)
        points: List[Tuple[float, float]] = []
        for east_true, north_true in corners_true:
            east_rotated = east_true * cosine + north_true * sine
            north_rotated = -east_true * sine + north_true * cosine
            points.append(
                (
                    float(pose.x + east_rotated * scale),
                    float(pose.y + north_rotated * scale),
                )
            )
        return points

    def _footprint_polygon_px(
        self,
        pose: MapPose,
        bounds: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        return [
            self._mercator_to_pixel(x_coordinate, y_coordinate, bounds)
            for x_coordinate, y_coordinate in self._footprint_polygon_mercator(
                pose
            )
        ]


# Compatibility aliases for old private imports.
_require_geo_deps = require_geo_dependencies
_wrap_heading_deg = wrap_heading_deg
_parse_action = parse_action


__all__ = ["MapGeometryMixin"]
