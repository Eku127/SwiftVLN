"""Content-addressed image caching and debug settings for map memory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .metadata import MapPose, normalize_scene_name


MAP_CACHE_FORMAT_VERSION = 1
CACHE_DISABLE_SENTINELS = {
    "off",
    "false",
    "none",
    "0",
    "disable",
    "disabled",
    "no",
}


def map_debug_enabled() -> bool:
    return os.environ.get("SWIFTVLN_DEBUG", "") != ""


def map_debug_limit(default: int = 6) -> int:
    raw = os.environ.get("SWIFTVLN_MAP_DEBUG_LIMIT", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def map_debug_dir() -> Optional[str]:
    raw = os.environ.get("SWIFTVLN_MAP_DEBUG_DIR", "").strip()
    if raw:
        return os.path.abspath(raw)
    if not map_debug_enabled():
        return None
    return os.path.abspath(
        os.path.join(
            os.getcwd(),
            "runtime",
            "debug",
            "swiftvln_map",
        )
    )


def resolve_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Resolve the environment override and caller cache preference."""
    env_override = os.environ.get("SWIFTVLN_MAP_CACHE_DIR", "").strip()
    if env_override:
        if env_override.lower() in CACHE_DISABLE_SENTINELS:
            return None
        return os.path.abspath(env_override)
    if cache_dir:
        cache_dir = str(cache_dir).strip()
        if not cache_dir or cache_dir.lower() in CACHE_DISABLE_SENTINELS:
            return None
        return os.path.abspath(cache_dir)
    return None


def sanitize_debug_name(value: str) -> str:
    sanitized = []
    for character in str(value):
        if character.isalnum() or character in ("-", "_", "."):
            sanitized.append(character)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("._") or "unknown"


class MapCacheMixin:
    """Own cache configuration, keying, atomic writes, and cache reads."""

    def _configure_cache(self, cache_dir: Optional[str]) -> None:
        self.cache_dir = resolve_cache_dir(cache_dir)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_writes = 0
        self._render_config_meta = {
            "format_version": MAP_CACHE_FORMAT_VERSION,
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
            json.dumps(
                self._render_config_meta,
                sort_keys=True,
            ).encode("utf-8"),
            digest_size=16,
        ).digest()
        if self.cache_dir:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self._write_cache_meta_once()
            except OSError as exc:
                warnings.warn(
                    f"[MapCache] failed to prepare cache_dir={self.cache_dir}: "
                    f"{exc}; disabling cache for this builder."
                )
                self.cache_dir = None

    def _write_cache_meta_once(self) -> None:
        """Emit a human-readable description of the render configuration."""
        if not self.cache_dir:
            return
        meta_path = os.path.join(self.cache_dir, "_cache_meta.json")
        if os.path.exists(meta_path):
            return
        payload = {
            "note": (
                "SwiftVLN map-memory render cache. Files are "
                "content-addressable over render config + scene_id + "
                "window_start + poses prefix. Safe to `rm -rf` at any time; "
                "will be lazily repopulated."
            ),
            "format_version": MAP_CACHE_FORMAT_VERSION,
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
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self._render_config_digest)
        digest.update(b"|scene|")
        digest.update(normalize_scene_name(scene_id).encode("utf-8"))
        digest.update(b"|ws|")
        digest.update(
            int(window_start).to_bytes(8, "little", signed=True)
        )
        digest.update(b"|poses|")
        if poses:
            pose_array = np.asarray(
                [
                    (pose.x, pose.y, pose.altitude, pose.heading_deg)
                    for pose in poses
                ],
                dtype=np.float64,
            )
            pose_array = np.round(pose_array, decimals=6)
            digest.update(pose_array.tobytes())
            digest.update(
                len(poses).to_bytes(4, "little", signed=False)
            )
        return digest.hexdigest()

    def _cache_paths(self, key: str) -> Tuple[str, str]:
        subdirectory = os.path.join(
            self.cache_dir,
            key[:2],
            key[2:4],
        )
        return (
            os.path.join(subdirectory, f"{key}_global.png"),
            os.path.join(subdirectory, f"{key}_local.png"),
        )

    def _try_load_cache(
        self,
        key: str,
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        if not self.cache_dir:
            return None
        global_path, local_path = self._cache_paths(key)
        if not (
            os.path.exists(global_path) and os.path.exists(local_path)
        ):
            return None
        try:
            with Image.open(global_path) as global_source:
                global_source.load()
                global_image = global_source.copy()
            with Image.open(local_path) as local_source:
                local_source.load()
                local_image = local_source.copy()
            return global_image, local_image
        except Exception as exc:
            warnings.warn(
                f"[MapCache] failed to load key={key[:12]}: {exc}"
            )
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
    def _atomic_write_png(target_path: str, image: Image.Image) -> None:
        directory = os.path.dirname(target_path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=".png.part",
            dir=directory,
        )
        os.close(descriptor)
        try:
            image.save(
                temporary_path,
                format="PNG",
                optimize=False,
            )
            os.replace(temporary_path, target_path)
        except Exception:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_json(
        target_path: str,
        payload: Dict[str, Any],
    ) -> None:
        directory = os.path.dirname(target_path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=".json.part",
            dir=directory,
        )
        os.close(descriptor)
        try:
            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temporary_path, target_path)
        except Exception:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            raise


# Compatibility aliases for old private imports.
_MAP_CACHE_FORMAT_VERSION = MAP_CACHE_FORMAT_VERSION
_CACHE_DISABLE_SENTINELS = CACHE_DISABLE_SENTINELS
_map_debug_enabled = map_debug_enabled
_map_debug_limit = map_debug_limit
_map_debug_dir = map_debug_dir
_resolve_cache_dir = resolve_cache_dir
_sanitize_debug_name = sanitize_debug_name


__all__ = ["MapCacheMixin"]
