"""Explored-mask composition and global/local map rendering."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .cache import (
    map_debug_dir,
    map_debug_enabled,
    map_debug_limit,
    sanitize_debug_name,
)
from .metadata import MapPose, normalize_scene_name


class MapRenderMixin:
    """Compose explored map images from geometry and cache services."""

    def _maybe_save_debug_maps(
        self,
        scene_id: str,
        window_start: int,
        current_index: int,
        observed_count: int,
        global_map: Image.Image,
        local_map: Image.Image,
    ) -> List[str]:
        debug_dir = map_debug_dir()
        if not debug_dir:
            return []

        scene_name = sanitize_debug_name(normalize_scene_name(scene_id))
        render_dir = Path(debug_dir) / scene_name
        render_dir.mkdir(parents=True, exist_ok=True)
        prefix = (
            f"render_{self._debug_render_count:03d}"
            f"_ws{int(window_start):04d}"
            f"_cur{int(current_index):04d}"
            f"_obs{int(observed_count):04d}"
        )
        global_path = render_dir / f"{prefix}_global.png"
        local_path = render_dir / f"{prefix}_local.png"
        global_map.save(global_path)
        local_map.save(local_path)
        return [str(global_path), str(local_path)]

    def _parse_dilate_meters(self) -> float:
        if self.mask_method == "strict":
            return 0.0
        if self.mask_method.startswith("dilate"):
            suffix = self.mask_method[len("dilate") :]
            if not suffix:
                return 20.0
            try:
                return float(suffix)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported map_mask_method: {self.mask_method}"
                ) from exc
        raise ValueError(
            f"Unsupported map_mask_method: {self.mask_method}"
        )

    def _build_explored_mask(
        self,
        observed_poses: Sequence[MapPose],
        side_m: float,
        bounds: Tuple[float, float, float, float],
    ) -> Image.Image:
        mask = Image.new("L", (self.render_px, self.render_px), 0)
        if not observed_poses:
            return mask

        draw = ImageDraw.Draw(mask)
        for pose in observed_poses:
            draw.polygon(
                self._footprint_polygon_px(pose, bounds),
                fill=255,
            )

        dilate_m = self._parse_dilate_meters()
        if dilate_m > 0:
            dilate_px = max(
                0,
                int(round((dilate_m / side_m) * self.render_px)),
            )
            if dilate_px > 0:
                kernel = max(3, 2 * dilate_px + 1)
                if kernel % 2 == 0:
                    kernel += 1
                mask = mask.filter(ImageFilter.MaxFilter(kernel))
        return mask

    def _compute_global_content_bounds(
        self,
        observed_poses: Sequence[MapPose],
        trajectory_poses: Sequence[MapPose],
        current_pose: MapPose,
    ) -> Tuple[float, float, float, float]:
        x_coordinates: List[float] = []
        y_coordinates: List[float] = []
        for pose in observed_poses:
            for x_coordinate, y_coordinate in (
                self._footprint_polygon_mercator(pose)
            ):
                x_coordinates.append(x_coordinate)
                y_coordinates.append(y_coordinate)

        for pose in trajectory_poses:
            x_coordinates.append(float(pose.x))
            y_coordinates.append(float(pose.y))
        x_coordinates.append(float(current_pose.x))
        y_coordinates.append(float(current_pose.y))

        if not x_coordinates:
            return (
                float(current_pose.x),
                float(current_pose.x),
                float(current_pose.y),
                float(current_pose.y),
            )
        return (
            min(x_coordinates),
            max(x_coordinates),
            min(y_coordinates),
            max(y_coordinates),
        )

    def _choose_adaptive_axis_center(
        self,
        start_coord: float,
        min_coord: float,
        max_coord: float,
        half_side_merc: float,
        margin_merc: float,
        quantize_merc: float,
    ) -> float:
        usable_half = max(1.0, half_side_merc - margin_merc)
        feasible_low = max_coord - usable_half
        feasible_high = min_coord + usable_half
        anchor_low = start_coord - half_side_merc
        anchor_high = start_coord + half_side_merc

        if feasible_low <= feasible_high:
            target = min(max(start_coord, feasible_low), feasible_high)
        else:
            target = 0.5 * (min_coord + max_coord)

        delta = target - start_coord
        if quantize_merc > 0:
            if abs(delta) < quantize_merc * 0.5:
                target = start_coord
            else:
                target = (
                    start_coord
                    + round(delta / quantize_merc) * quantize_merc
                )
                if feasible_low <= feasible_high:
                    target = min(
                        max(target, feasible_low),
                        feasible_high,
                    )
        target = min(max(target, anchor_low), anchor_high)
        return float(target)

    def _select_global_center_pose(
        self,
        start_pose: MapPose,
        observed_poses: Sequence[MapPose],
        trajectory_poses: Sequence[MapPose],
        current_pose: MapPose,
    ) -> Tuple[MapPose, Dict[str, float]]:
        if self.global_center_mode == "start":
            return start_pose, {
                "shift_x_m": 0.0,
                "shift_y_m": 0.0,
                "content_width_m": 0.0,
                "content_height_m": 0.0,
            }

        min_x, max_x, min_y, max_y = (
            self._compute_global_content_bounds(
                observed_poses=observed_poses,
                trajectory_poses=trajectory_poses,
                current_pose=current_pose,
            )
        )
        reference_y = start_pose.y
        half_side_merc = self._true_meters_to_mercator(
            self.global_side_m / 2.0,
            reference_y,
        )
        margin_merc = self._true_meters_to_mercator(
            self.global_side_m * self._global_shift_margin_ratio,
            reference_y,
        )
        quantize_merc = self._true_meters_to_mercator(
            self._global_shift_quantize_m,
            reference_y,
        )

        center_x = self._choose_adaptive_axis_center(
            start_coord=start_pose.x,
            min_coord=min_x,
            max_coord=max_x,
            half_side_merc=half_side_merc,
            margin_merc=margin_merc,
            quantize_merc=quantize_merc,
        )
        center_y = self._choose_adaptive_axis_center(
            start_coord=start_pose.y,
            min_coord=min_y,
            max_coord=max_y,
            half_side_merc=half_side_merc,
            margin_merc=margin_merc,
            quantize_merc=quantize_merc,
        )
        center_pose = MapPose(
            x=center_x,
            y=center_y,
            altitude=start_pose.altitude,
            heading_deg=start_pose.heading_deg,
        )
        scale = max(
            1e-6,
            self._true_meters_to_mercator(1.0, reference_y),
        )
        return center_pose, {
            "shift_x_m": float((center_x - start_pose.x) / scale),
            "shift_y_m": float((center_y - start_pose.y) / scale),
            "content_width_m": float((max_x - min_x) / scale),
            "content_height_m": float((max_y - min_y) / scale),
        }

    def _compose_map(
        self,
        scene_ref: str,
        center_pose: MapPose,
        side_m: float,
        observed_poses: Sequence[MapPose],
        trajectory_poses: Sequence[MapPose],
        start_pose: MapPose,
        current_pose: MapPose,
    ) -> Image.Image:
        base_image, bounds = self._read_rgb_crop(
            scene_ref,
            center_pose,
            side_m,
        )
        explored_mask = self._build_explored_mask(
            observed_poses,
            side_m,
            bounds,
        )

        base_array = np.asarray(base_image, dtype=np.uint8).copy()
        mask_array = np.asarray(explored_mask, dtype=np.uint8)
        base_array[mask_array == 0] = 0
        composed = Image.fromarray(base_array, mode="RGB")

        draw = ImageDraw.Draw(composed)
        line_width = max(2, self.render_px // 128)
        point_radius = max(4, self.render_px // 64)
        outline_radius = point_radius + 1
        arrow_length = max(12, self.render_px // 18)

        def point_in_bounds(point: Tuple[float, float]) -> bool:
            x_coordinate, y_coordinate = point
            return (
                -point_radius
                <= x_coordinate
                <= self.render_px + point_radius
                and -point_radius
                <= y_coordinate
                <= self.render_px + point_radius
            )

        trajectory_points = [
            self._mercator_to_pixel(pose.x, pose.y, bounds)
            for pose in trajectory_poses
        ]
        if len(trajectory_points) >= 2:
            draw.line(
                trajectory_points,
                fill=(220, 48, 48),
                width=line_width,
                joint="curve",
            )

        start_point = self._mercator_to_pixel(
            start_pose.x,
            start_pose.y,
            bounds,
        )
        if point_in_bounds(start_point):
            draw.ellipse(
                (
                    start_point[0] - outline_radius,
                    start_point[1] - outline_radius,
                    start_point[0] + outline_radius,
                    start_point[1] + outline_radius,
                ),
                fill=(255, 255, 255),
            )
            draw.ellipse(
                (
                    start_point[0] - point_radius,
                    start_point[1] - point_radius,
                    start_point[0] + point_radius,
                    start_point[1] + point_radius,
                ),
                fill=(40, 120, 255),
            )

        current_point = self._mercator_to_pixel(
            current_pose.x,
            current_pose.y,
            bounds,
        )
        if point_in_bounds(current_point):
            draw.ellipse(
                (
                    current_point[0] - outline_radius,
                    current_point[1] - outline_radius,
                    current_point[0] + outline_radius,
                    current_point[1] + outline_radius,
                ),
                fill=(24, 24, 24),
            )
            draw.ellipse(
                (
                    current_point[0] - point_radius,
                    current_point[1] - point_radius,
                    current_point[0] + point_radius,
                    current_point[1] + point_radius,
                ),
                fill=(244, 201, 32),
            )

            heading_radians = math.radians(current_pose.heading_deg)
            end_x = current_point[0] + arrow_length * math.sin(
                heading_radians
            )
            end_y = current_point[1] - arrow_length * math.cos(
                heading_radians
            )
            draw.line(
                [current_point, (end_x, end_y)],
                fill=(244, 201, 32),
                width=max(2, line_width),
            )
        return composed

    def render_from_actions(
        self,
        scene_id: str,
        start_position: Sequence[float],
        start_rotation: float,
        actions: Iterable[Any],
        window_start: int,
        step_size: float,
        turn_angle: float,
    ) -> List[Image.Image]:
        poses = self.integrate_poses_from_actions(
            start_position=start_position,
            start_rotation=start_rotation,
            actions=actions,
            step_size=step_size,
            turn_angle=turn_angle,
        )
        return self.render_from_poses(
            scene_id=scene_id,
            poses=poses,
            window_start=window_start,
        )

    def render_from_poses(
        self,
        scene_id: str,
        poses: Sequence[MapPose],
        window_start: int,
    ) -> List[Image.Image]:
        if not poses:
            raise ValueError(
                "render_from_poses requires at least one pose."
            )

        debug_enabled = map_debug_enabled()
        debug_start = time.perf_counter() if debug_enabled else 0.0
        cache_key: Optional[str] = None
        if self.cache_dir:
            cache_key = self._compute_cache_key(
                scene_id,
                poses,
                window_start,
            )
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                global_map, local_map = cached
                self._cache_hits += 1
                if (
                    debug_enabled
                    and self._debug_render_count < map_debug_limit()
                ):
                    elapsed_ms = (
                        time.perf_counter() - debug_start
                    ) * 1000.0
                    print(
                        "[MAP DEBUG][builder] "
                        f"cache_hit[{self._debug_render_count}] "
                        f"scene={normalize_scene_name(scene_id)} "
                        f"window_start={window_start} poses={len(poses)} "
                        f"key={cache_key[:12]} hits={self._cache_hits} "
                        f"elapsed_ms={elapsed_ms:.1f}"
                    )
                    self._debug_render_count += 1
                return [global_map, local_map]
            self._cache_misses += 1

        current_index = max(
            0,
            min(int(window_start), len(poses) - 1),
        )
        observed_until = max(
            0,
            min(int(window_start), len(poses)),
        )
        start_pose = poses[0]
        current_pose = poses[current_index]
        observed_poses = list(poses[:observed_until])
        trajectory_poses = list(poses[: current_index + 1])
        if not trajectory_poses:
            trajectory_poses = [current_pose]

        global_center_pose, global_center_meta = (
            self._select_global_center_pose(
                start_pose=start_pose,
                observed_poses=observed_poses,
                trajectory_poses=trajectory_poses,
                current_pose=current_pose,
            )
        )
        global_map = self._compose_map(
            scene_ref=scene_id,
            center_pose=global_center_pose,
            side_m=self.global_side_m,
            observed_poses=observed_poses,
            trajectory_poses=trajectory_poses,
            start_pose=start_pose,
            current_pose=current_pose,
        )
        local_map = self._compose_map(
            scene_ref=scene_id,
            center_pose=current_pose,
            side_m=self.local_side_m,
            observed_poses=observed_poses,
            trajectory_poses=trajectory_poses,
            start_pose=start_pose,
            current_pose=current_pose,
        )
        if self.cache_dir and cache_key is not None:
            try:
                self._save_cache(cache_key, global_map, local_map)
                self._cache_writes += 1
            except Exception as exc:
                warnings.warn(
                    f"[MapCache] failed to save key={cache_key[:12]}: {exc}"
                )

        if (
            debug_enabled
            and self._debug_render_count < map_debug_limit()
        ):
            elapsed_ms = (
                time.perf_counter() - debug_start
            ) * 1000.0
            saved_paths = self._maybe_save_debug_maps(
                scene_id=scene_id,
                window_start=window_start,
                current_index=current_index,
                observed_count=len(observed_poses),
                global_map=global_map,
                local_map=local_map,
            )
            cache_tag = (
                f"cache=miss({self._cache_misses}/w{self._cache_writes})"
                if self.cache_dir
                else "cache=off"
            )
            print(
                f"[MAP DEBUG][builder] render[{self._debug_render_count}] "
                f"scene={normalize_scene_name(scene_id)} "
                f"window_start={window_start} "
                f"current_index={current_index} "
                f"observed={len(observed_poses)}/{len(poses)} "
                f"trajectory={len(trajectory_poses)} "
                f"global_center_mode={self.global_center_mode} "
                f"shift=({global_center_meta['shift_x_m']:.1f}m,"
                f"{global_center_meta['shift_y_m']:.1f}m) "
                f"content=({global_center_meta['content_width_m']:.1f}m,"
                f"{global_center_meta['content_height_m']:.1f}m) "
                f"global={self.global_side_m:.0f}m "
                f"local={self.local_side_m:.0f}m "
                f"render_px={self.render_px} "
                f"heading={current_pose.heading_deg:.1f} "
                f"{cache_tag} elapsed_ms={elapsed_ms:.1f}"
            )
            if saved_paths:
                print(
                    "[MAP DEBUG][builder] saved_maps: "
                    f"global={saved_paths[0]} local={saved_paths[1]}"
                )
            self._debug_render_count += 1
        return [global_map, local_map]


__all__ = ["MapRenderMixin"]
