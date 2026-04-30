"""SatNav explored-map memory builder for OverlapVLN."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Bumping this invalidates *all* existing cache entries across hosts. Increment
# whenever the map rendering math changes in a way existing PNGs no longer
# reflect what `render_from_poses` would currently produce.
_MAP_CACHE_FORMAT_VERSION = 1

_CACHE_DISABLE_SENTINELS = {"off", "false", "none", "0", "disable", "disabled", "no"}


def _require_geo_deps():
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


def _normalize_scene_name(scene_ref: str) -> str:
    scene_name = os.path.basename(str(scene_ref).rstrip("/"))
    if scene_name.lower().endswith(".tif"):
        scene_name = scene_name[:-4]
    return scene_name


def _normalize_episode_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Unsupported SatNav episode JSON format for map metadata.")


def _map_debug_enabled() -> bool:
    return os.environ.get("OVERLAPVLN_DEBUG", "") != ""


def _map_debug_limit(default: int = 6) -> int:
    raw = os.environ.get("OVERLAPVLN_MAP_DEBUG_LIMIT", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _map_debug_dir() -> Optional[str]:
    raw = os.environ.get("OVERLAPVLN_MAP_DEBUG_DIR", "").strip()
    if raw:
        return os.path.abspath(raw)
    if not _map_debug_enabled():
        return None
    return os.path.abspath(os.path.join(os.getcwd(), "runtime", "debug", "overlapvln_map"))


def _resolve_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Resolve the effective on-disk cache directory.

    Precedence:
      1. ``OVERLAPVLN_MAP_CACHE_DIR`` env var (sentinel values disable cache).
      2. ``cache_dir`` argument supplied by the caller.
      3. ``None`` → cache disabled.
    """

    env_override = os.environ.get("OVERLAPVLN_MAP_CACHE_DIR", "").strip()
    if env_override:
        if env_override.lower() in _CACHE_DISABLE_SENTINELS:
            return None
        return os.path.abspath(env_override)
    if cache_dir:
        cache_dir = str(cache_dir).strip()
        if not cache_dir or cache_dir.lower() in _CACHE_DISABLE_SENTINELS:
            return None
        return os.path.abspath(cache_dir)
    return None


def _sanitize_debug_name(value: str) -> str:
    sanitized = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            sanitized.append(ch)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("._") or "unknown"


def _wrap_heading_deg(angle: float) -> float:
    return angle % 360.0


def _parse_action(action: Any) -> Tuple[bool, bool, bool]:
    if isinstance(action, str):
        action_key = action.strip().upper()
        return (
            action_key in ("TURN_LEFT", "LEFT", "L", "2"),
            action_key in ("TURN_RIGHT", "RIGHT", "R", "3"),
            action_key in ("MOVE_FORWARD", "FORWARD", "F", "1"),
        )
    action_int = int(action)
    return action_int == 2, action_int == 3, action_int == 1


@dataclass
class MapPose:
    """Absolute SatNav pose in EPSG:3857 meters."""

    x: float
    y: float
    altitude: float
    heading_deg: float


@dataclass
class SatNavEpisodeMetadata:
    """Episode metadata needed to reconstruct explored maps."""

    scene_id: str
    episode_id: str
    start_position: List[float]
    start_rotation: float


class SatNavTrajectoryMetadataResolver:
    """Resolve trajectory_data annotations to SatNav episode metadata."""

    def __init__(self, trajectory_data_dir: str):
        trajectory_data_dir = os.path.abspath(trajectory_data_dir)
        if not os.path.isdir(trajectory_data_dir):
            raise FileNotFoundError(f"trajectory_data dir not found: {trajectory_data_dir}")

        self.trajectory_data_dir = trajectory_data_dir
        self.dataset_root = os.path.dirname(trajectory_data_dir.rstrip("/"))
        self.dataset_parent = os.path.dirname(self.dataset_root)
        self.scenes_dir = os.path.join(self.dataset_parent, "scenes")
        self.summary_by_id = self._load_summary()
        self.episodes_by_key = self._load_episodes()

    def _load_summary(self) -> Dict[str, Dict[str, Any]]:
        summary_path = os.path.join(self.trajectory_data_dir, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"SatNav summary.json not found: {summary_path}")

        summary_by_id: Dict[str, Dict[str, Any]] = {}
        with open(summary_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                summary_by_id[str(record["id"])] = record
        return summary_by_id

    def _load_episodes(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        episodes_path = os.path.join(self.dataset_root, "episodes", "train", "all_episodes.json")
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"SatNav train episodes not found: {episodes_path}")

        with open(episodes_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        episodes = _normalize_episode_list(payload)

        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for episode in episodes:
            key = (
                _normalize_scene_name(episode.get("scene_id", "")),
                str(episode.get("episode_id")),
            )
            by_key[key] = episode
        return by_key

    def resolve(self, annotation: Dict[str, Any]) -> SatNavEpisodeMetadata:
        annotation_id = str(annotation.get("id"))
        summary = self.summary_by_id.get(annotation_id)
        if summary is None:
            raise KeyError(
                f"Annotation id={annotation_id} not found in {self.trajectory_data_dir}/summary.json"
            )

        scene_name = _normalize_scene_name(summary.get("scene_id", ""))
        episode_id = str(summary.get("episode_id"))
        episode = self.episodes_by_key.get((scene_name, episode_id))
        if episode is None:
            raise KeyError(
                f"Episode metadata not found for scene={scene_name}, episode_id={episode_id}"
            )

        return SatNavEpisodeMetadata(
            scene_id=summary.get("scene_id", episode.get("scene_id", scene_name)),
            episode_id=episode_id,
            start_position=list(episode["start_position"]),
            start_rotation=float(episode["start_rotation"]),
        )


class SatNavMapMemoryBuilder:
    """Render fixed-size explored map memory images for SatNav."""

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
        # Keep the start-centered view stable until explored content approaches
        # the boundary, then move in coarse steps to keep more route in-frame.
        self._global_shift_margin_ratio = 0.10
        self._global_shift_quantize_m = 25.0

        # On-disk cache for rendered (global, local) pairs. See _resolve_cache_dir
        # for precedence. When enabled, cache key is content-addressable over all
        # render-affecting parameters + scene_id + window_start + poses prefix.
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_writes = 0
        self._render_config_meta = {
            "format_version": _MAP_CACHE_FORMAT_VERSION,
            "scenes_dir": self.scenes_dir,
            "global_side_m": self.global_side_m,
            "local_side_m": self.local_side_m,
            "render_px": self.render_px,
            "mask_method": self.mask_method,
            "hfov": self.hfov,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
            "global_center_mode": self.global_center_mode,
            "global_shift_margin_ratio": self._global_shift_margin_ratio,
            "global_shift_quantize_m": self._global_shift_quantize_m,
        }
        self._render_config_digest = hashlib.blake2b(
            json.dumps(self._render_config_meta, sort_keys=True).encode("utf-8"),
            digest_size=16,
        ).digest()
        if self.cache_dir:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self._write_cache_meta_once()
            except OSError as exc:
                warnings.warn(
                    f"[MapCache] failed to prepare cache_dir={self.cache_dir}: {exc}; "
                    "disabling cache for this builder."
                )
                self.cache_dir = None

    def _maybe_save_debug_maps(
        self,
        scene_id: str,
        window_start: int,
        current_index: int,
        observed_count: int,
        global_map: Image.Image,
        local_map: Image.Image,
    ) -> List[str]:
        debug_dir = _map_debug_dir()
        if not debug_dir:
            return []

        scene_name = _sanitize_debug_name(_normalize_scene_name(scene_id))
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

    # ------------------------------------------------------------------
    # On-disk render cache (see __init__ for config + env precedence)
    # ------------------------------------------------------------------

    def _write_cache_meta_once(self) -> None:
        """Emit a human-readable README describing the render config.

        The cache key already encodes the full render config, so stale configs
        never produce stale hits; this file is informational only.
        """
        if not self.cache_dir:
            return
        meta_path = os.path.join(self.cache_dir, "_cache_meta.json")
        if os.path.exists(meta_path):
            return
        payload = {
            "note": (
                "OverlapVLN map-memory render cache. Files are content-addressable "
                "over render config + scene_id + window_start + poses prefix. Safe "
                "to `rm -rf` at any time; will be lazily repopulated."
            ),
            "format_version": _MAP_CACHE_FORMAT_VERSION,
            "render_config": self._render_config_meta,
        }
        try:
            self._atomic_write_json(meta_path, payload)
        except OSError:
            pass

    def _compute_cache_key(
        self,
        scene_id: str,
        poses: Sequence[MapPose],
        window_start: int,
    ) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self._render_config_digest)
        h.update(b"|scene|")
        h.update(_normalize_scene_name(scene_id).encode("utf-8"))
        h.update(b"|ws|")
        h.update(int(window_start).to_bytes(8, "little", signed=True))
        h.update(b"|poses|")
        # Only the prefix up to current_index affects the render output, but
        # keeping the full passed-in poses in the hash avoids accidental
        # collisions from upstream slicing changes.
        if poses:
            arr = np.asarray(
                [(p.x, p.y, p.altitude, p.heading_deg) for p in poses],
                dtype=np.float64,
            )
            # Round to 1 µm / 1e-4 deg — well within simulator repeatability.
            arr = np.round(arr, decimals=6)
            h.update(arr.tobytes())
            h.update(len(poses).to_bytes(4, "little", signed=False))
        return h.hexdigest()

    def _cache_paths(self, key: str) -> Tuple[str, str]:
        # Two-level sharding so a single directory never holds the full cache.
        subdir = os.path.join(self.cache_dir, key[:2], key[2:4])
        global_path = os.path.join(subdir, f"{key}_global.png")
        local_path = os.path.join(subdir, f"{key}_local.png")
        return global_path, local_path

    def _try_load_cache(
        self, key: str
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        if not self.cache_dir:
            return None
        global_path, local_path = self._cache_paths(key)
        if not (os.path.exists(global_path) and os.path.exists(local_path)):
            return None
        try:
            with Image.open(global_path) as img_g:
                img_g.load()
                global_img = img_g.copy()
            with Image.open(local_path) as img_l:
                img_l.load()
                local_img = img_l.copy()
            return global_img, local_img
        except Exception as exc:
            # Corrupted / partially written files: fall through to re-render.
            warnings.warn(f"[MapCache] failed to load key={key[:12]}: {exc}")
            return None

    def _save_cache(
        self,
        key: str,
        global_img: Image.Image,
        local_img: Image.Image,
    ) -> None:
        if not self.cache_dir:
            return
        global_path, local_path = self._cache_paths(key)
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        self._atomic_write_png(global_path, global_img)
        self._atomic_write_png(local_path, local_img)

    @staticmethod
    def _atomic_write_png(target_path: str, img: Image.Image) -> None:
        directory = os.path.dirname(target_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_", suffix=".png.part", dir=directory
        )
        os.close(fd)
        try:
            # optimize=False keeps write cost low; these caches are ephemeral.
            img.save(tmp_path, format="PNG", optimize=False)
            os.replace(tmp_path, target_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_json(target_path: str, payload: Dict[str, Any]) -> None:
        directory = os.path.dirname(target_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_", suffix=".json.part", dir=directory
        )
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, target_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _ensure_transformer(self):
        if self._wgs84_to_mercator is None:
            _, Transformer, _, _, _ = _require_geo_deps()
            self._wgs84_to_mercator = Transformer.from_crs(
                "EPSG:4326",
                "EPSG:3857",
                always_xy=True,
            )

    def _wgs84_to_mercator_xy(self, lon: float, lat: float) -> Tuple[float, float]:
        self._ensure_transformer()
        x, y = self._wgs84_to_mercator.transform(lon, lat)
        return float(x), float(y)

    def _true_meters_to_mercator(self, distance_m: float, mercator_y: float) -> float:
        earth_radius = 6378137.0
        lat_rad = math.atan(math.sinh(mercator_y / earth_radius))
        scale_factor = 1.0 / math.cos(lat_rad)
        return float(distance_m) * scale_factor

    def _move_in_mercator(
        self,
        x: float,
        y: float,
        distance_m: float,
        heading_deg: float,
    ) -> Tuple[float, float]:
        distance_mercator = self._true_meters_to_mercator(distance_m, y)
        heading_rad = math.radians(heading_deg)
        dx = distance_mercator * math.sin(heading_rad)
        dy = distance_mercator * math.cos(heading_rad)
        return float(x + dx), float(y + dy)

    def integrate_poses_from_actions(
        self,
        start_position: Sequence[float],
        start_rotation: float,
        actions: Iterable[Any],
        step_size: float,
        turn_angle: float,
    ) -> List[MapPose]:
        if len(start_position) < 3:
            raise ValueError("start_position must be [lon, lat, altitude].")

        lon, lat, altitude = float(start_position[0]), float(start_position[1]), float(start_position[2])
        x, y = self._wgs84_to_mercator_xy(lon, lat)
        heading_deg = _wrap_heading_deg(float(start_rotation))

        poses: List[MapPose] = []
        for action in actions:
            is_left, is_right, is_forward = _parse_action(action)
            if is_left:
                heading_deg = _wrap_heading_deg(heading_deg - turn_angle)
            elif is_right:
                heading_deg = _wrap_heading_deg(heading_deg + turn_angle)
            elif is_forward:
                x, y = self._move_in_mercator(x, y, step_size, heading_deg)
            poses.append(MapPose(x=x, y=y, altitude=altitude, heading_deg=heading_deg))

        if not poses:
            poses.append(MapPose(x=x, y=y, altitude=altitude, heading_deg=heading_deg))
        return poses

    def _resolve_scene_path(self, scene_ref: str) -> str:
        candidate = str(scene_ref)
        if candidate.lower().endswith(".tif") and os.path.exists(candidate):
            return candidate

        scene_name = _normalize_scene_name(candidate)
        tif_path = os.path.join(self.scenes_dir, f"{scene_name}.tif")
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"SatNav scene tif not found: {tif_path}")
        return tif_path

    def _get_scene_dataset(self, scene_ref: str):
        tif_path = self._resolve_scene_path(scene_ref)
        cached = self._scene_cache.get(tif_path)
        if cached is not None:
            return cached

        rasterio, _, _, WarpedVRT, _ = _require_geo_deps()
        src = rasterio.open(tif_path)
        if src.crs is not None and src.crs.to_epsg() == 3857:
            dataset = src
        else:
            dataset = WarpedVRT(src, crs="EPSG:3857")
        self._scene_cache[tif_path] = dataset
        if _map_debug_enabled():
            print(
                f"[MAP DEBUG][builder] opened scene dataset: "
                f"scene={_normalize_scene_name(tif_path)} path={tif_path}"
            )
        return dataset

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
                raise ValueError(f"Unsupported map_mask_method: {self.mask_method}") from exc
        raise ValueError(f"Unsupported map_mask_method: {self.mask_method}")

    def _read_rgb_crop(
        self,
        scene_ref: str,
        center_pose: MapPose,
        side_m: float,
    ) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        dataset = self._get_scene_dataset(scene_ref)
        _, _, Resampling, _, from_bounds = _require_geo_deps()

        half_side_merc = self._true_meters_to_mercator(side_m / 2.0, center_pose.y)
        left = center_pose.x - half_side_merc
        right = center_pose.x + half_side_merc
        bottom = center_pose.y - half_side_merc
        top = center_pose.y + half_side_merc

        window = from_bounds(left, bottom, right, top, transform=dataset.transform)
        arr = dataset.read(
            indexes=[1, 2, 3],
            window=window,
            out_shape=(3, self.render_px, self.render_px),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        )
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return Image.fromarray(arr, mode="RGB"), (left, right, bottom, top)

    def _mercator_to_pixel(
        self,
        x: float,
        y: float,
        bounds: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        left, right, bottom, top = bounds
        if right <= left or top <= bottom:
            return 0.0, 0.0
        px = (x - left) / (right - left) * self.render_px
        py = (top - y) / (top - bottom) * self.render_px
        return float(px), float(py)

    def _footprint_polygon_mercator(
        self,
        pose: MapPose,
    ) -> List[Tuple[float, float]]:
        aspect_ratio = self.sensor_width / max(1, self.sensor_height)
        half_x_true = pose.altitude * math.tan(math.radians(self.hfov / 2.0))
        half_y_true = half_x_true / aspect_ratio
        scale = self._true_meters_to_mercator(1.0, pose.y)

        corners_true = [
            (+half_x_true, +half_y_true),
            (-half_x_true, +half_y_true),
            (-half_x_true, -half_y_true),
            (+half_x_true, -half_y_true),
        ]

        theta = math.radians(pose.heading_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        points: List[Tuple[float, float]] = []
        for east_true, north_true in corners_true:
            east_rot = east_true * cos_t + north_true * sin_t
            north_rot = -east_true * sin_t + north_true * cos_t
            x = pose.x + east_rot * scale
            y = pose.y + north_rot * scale
            points.append((float(x), float(y)))
        return points

    def _footprint_polygon_px(
        self,
        pose: MapPose,
        bounds: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        return [
            self._mercator_to_pixel(x, y, bounds)
            for x, y in self._footprint_polygon_mercator(pose)
        ]

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
            polygon = self._footprint_polygon_px(pose, bounds)
            draw.polygon(polygon, fill=255)

        dilate_m = self._parse_dilate_meters()
        if dilate_m > 0:
            dilate_px = max(0, int(round((dilate_m / side_m) * self.render_px)))
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
        xs: List[float] = []
        ys: List[float] = []

        for pose in observed_poses:
            for x, y in self._footprint_polygon_mercator(pose):
                xs.append(x)
                ys.append(y)

        for pose in trajectory_poses:
            xs.append(float(pose.x))
            ys.append(float(pose.y))

        xs.append(float(current_pose.x))
        ys.append(float(current_pose.y))

        if not xs:
            return (
                float(current_pose.x),
                float(current_pose.x),
                float(current_pose.y),
                float(current_pose.y),
            )
        return min(xs), max(xs), min(ys), max(ys)

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
        # Hard anchor: the start point must remain inside the canvas even when
        # the explored region is larger than the available view.
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
                target = start_coord + round(delta / quantize_merc) * quantize_merc
                if feasible_low <= feasible_high:
                    target = min(max(target, feasible_low), feasible_high)
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

        min_x, max_x, min_y, max_y = self._compute_global_content_bounds(
            observed_poses=observed_poses,
            trajectory_poses=trajectory_poses,
            current_pose=current_pose,
        )
        reference_y = start_pose.y
        half_side_merc = self._true_meters_to_mercator(self.global_side_m / 2.0, reference_y)
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
        shift_x_m = (center_x - start_pose.x) / max(
            1e-6,
            self._true_meters_to_mercator(1.0, reference_y),
        )
        shift_y_m = (center_y - start_pose.y) / max(
            1e-6,
            self._true_meters_to_mercator(1.0, reference_y),
        )
        content_width_m = (max_x - min_x) / max(
            1e-6,
            self._true_meters_to_mercator(1.0, reference_y),
        )
        content_height_m = (max_y - min_y) / max(
            1e-6,
            self._true_meters_to_mercator(1.0, reference_y),
        )
        return center_pose, {
            "shift_x_m": float(shift_x_m),
            "shift_y_m": float(shift_y_m),
            "content_width_m": float(content_width_m),
            "content_height_m": float(content_height_m),
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
        base_image, bounds = self._read_rgb_crop(scene_ref, center_pose, side_m)
        explored_mask = self._build_explored_mask(observed_poses, side_m, bounds)

        base_arr = np.asarray(base_image, dtype=np.uint8).copy()
        mask_arr = np.asarray(explored_mask, dtype=np.uint8)
        base_arr[mask_arr == 0] = 0
        composed = Image.fromarray(base_arr, mode="RGB")

        draw = ImageDraw.Draw(composed)
        line_width = max(2, self.render_px // 128)
        point_radius = max(4, self.render_px // 64)
        outline_radius = point_radius + 1
        arrow_length = max(12, self.render_px // 18)

        def _point_in_bounds(point: Tuple[float, float]) -> bool:
            x, y = point
            return -point_radius <= x <= self.render_px + point_radius and -point_radius <= y <= self.render_px + point_radius

        trajectory_points = [self._mercator_to_pixel(pose.x, pose.y, bounds) for pose in trajectory_poses]
        if len(trajectory_points) >= 2:
            draw.line(trajectory_points, fill=(220, 48, 48), width=line_width, joint="curve")

        start_point = self._mercator_to_pixel(start_pose.x, start_pose.y, bounds)
        if _point_in_bounds(start_point):
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

        current_point = self._mercator_to_pixel(current_pose.x, current_pose.y, bounds)
        if _point_in_bounds(current_point):
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

            heading_rad = math.radians(current_pose.heading_deg)
            end_x = current_point[0] + arrow_length * math.sin(heading_rad)
            end_y = current_point[1] - arrow_length * math.cos(heading_rad)
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
            raise ValueError("render_from_poses requires at least one pose.")

        debug_enabled = _map_debug_enabled()
        debug_start = time.perf_counter() if debug_enabled else 0.0

        cache_key: Optional[str] = None
        if self.cache_dir:
            cache_key = self._compute_cache_key(scene_id, poses, window_start)
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                global_map, local_map = cached
                self._cache_hits += 1
                if debug_enabled and self._debug_render_count < _map_debug_limit():
                    elapsed_ms = (time.perf_counter() - debug_start) * 1000.0
                    print(
                        f"[MAP DEBUG][builder] cache_hit[{self._debug_render_count}] "
                        f"scene={_normalize_scene_name(scene_id)} "
                        f"window_start={window_start} poses={len(poses)} "
                        f"key={cache_key[:12]} hits={self._cache_hits} "
                        f"elapsed_ms={elapsed_ms:.1f}"
                    )
                    self._debug_render_count += 1
                return [global_map, local_map]
            self._cache_misses += 1

        current_index = max(0, min(int(window_start), len(poses) - 1))
        observed_until = max(0, min(int(window_start), len(poses)))

        start_pose = poses[0]
        current_pose = poses[current_index]
        observed_poses = list(poses[:observed_until])
        trajectory_poses = list(poses[: current_index + 1])
        if not trajectory_poses:
            trajectory_poses = [current_pose]

        global_center_pose, global_center_meta = self._select_global_center_pose(
            start_pose=start_pose,
            observed_poses=observed_poses,
            trajectory_poses=trajectory_poses,
            current_pose=current_pose,
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

        if debug_enabled and self._debug_render_count < _map_debug_limit():
            elapsed_ms = (time.perf_counter() - debug_start) * 1000.0
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
                f"scene={_normalize_scene_name(scene_id)} window_start={window_start} "
                f"current_index={current_index} observed={len(observed_poses)}/{len(poses)} "
                f"trajectory={len(trajectory_poses)} "
                f"global_center_mode={self.global_center_mode} "
                f"shift=({global_center_meta['shift_x_m']:.1f}m,{global_center_meta['shift_y_m']:.1f}m) "
                f"content=({global_center_meta['content_width_m']:.1f}m,{global_center_meta['content_height_m']:.1f}m) "
                f"global={self.global_side_m:.0f}m local={self.local_side_m:.0f}m "
                f"render_px={self.render_px} heading={current_pose.heading_deg:.1f} "
                f"{cache_tag} elapsed_ms={elapsed_ms:.1f}"
            )
            if saved_paths:
                print(
                    f"[MAP DEBUG][builder] saved_maps: "
                    f"global={saved_paths[0]} local={saved_paths[1]}"
                )
            self._debug_render_count += 1
        return [global_map, local_map]
