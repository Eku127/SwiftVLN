"""SatNav episode metadata used by explored-map memory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def normalize_scene_name(scene_ref: str) -> str:
    """Return the stable scene name used by metadata and cache keys."""
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
    raise ValueError(
        "Unsupported SatNav episode JSON format for map metadata."
    )


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
    """Resolve trajectory annotations to SatNav episode metadata."""

    def __init__(self, trajectory_data_dir: str):
        trajectory_data_dir = os.path.abspath(trajectory_data_dir)
        if not os.path.isdir(trajectory_data_dir):
            raise FileNotFoundError(
                f"trajectory_data dir not found: {trajectory_data_dir}"
            )

        self.trajectory_data_dir = trajectory_data_dir
        self.dataset_root = os.path.dirname(
            trajectory_data_dir.rstrip("/")
        )
        self.dataset_parent = os.path.dirname(self.dataset_root)
        self.scenes_dir = os.path.join(self.dataset_parent, "scenes")
        self.summary_by_id = self._load_summary()
        self.episodes_by_key = self._load_episodes()

    def _load_summary(self) -> Dict[str, Dict[str, Any]]:
        summary_path = os.path.join(
            self.trajectory_data_dir,
            "summary.json",
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"SatNav summary.json not found: {summary_path}"
            )

        summary_by_id: Dict[str, Dict[str, Any]] = {}
        with open(summary_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                summary_by_id[str(record["id"])] = record
        return summary_by_id

    def _load_episodes(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        episodes_path = os.path.join(
            self.dataset_root,
            "episodes",
            "train",
            "all_episodes.json",
        )
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(
                f"SatNav train episodes not found: {episodes_path}"
            )

        with open(episodes_path, "r", encoding="utf-8") as handle:
            episodes = _normalize_episode_list(json.load(handle))

        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for episode in episodes:
            key = (
                normalize_scene_name(episode.get("scene_id", "")),
                str(episode.get("episode_id")),
            )
            by_key[key] = episode
        return by_key

    def resolve(
        self,
        annotation: Dict[str, Any],
    ) -> SatNavEpisodeMetadata:
        annotation_id = str(annotation.get("id"))
        summary = self.summary_by_id.get(annotation_id)
        if summary is None:
            raise KeyError(
                f"Annotation id={annotation_id} not found in "
                f"{self.trajectory_data_dir}/summary.json"
            )

        scene_name = normalize_scene_name(summary.get("scene_id", ""))
        episode_id = str(summary.get("episode_id"))
        episode = self.episodes_by_key.get((scene_name, episode_id))
        if episode is None:
            raise KeyError(
                "Episode metadata not found for "
                f"scene={scene_name}, episode_id={episode_id}"
            )

        return SatNavEpisodeMetadata(
            scene_id=summary.get(
                "scene_id",
                episode.get("scene_id", scene_name),
            ),
            episode_id=episode_id,
            start_position=list(episode["start_position"]),
            start_rotation=float(episode["start_rotation"]),
        )


# Compatibility for callers that imported the old private helper.
_normalize_scene_name = normalize_scene_name


__all__ = [
    "MapPose",
    "SatNavEpisodeMetadata",
    "SatNavTrajectoryMetadataResolver",
    "normalize_scene_name",
]
