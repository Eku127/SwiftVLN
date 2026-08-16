"""Validate SatNav and Habitat data used by online evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_EPISODES = {
    "satnav": {"val_seen": 4574, "val_unseen": 8756},
    "habitat": {"val_seen": 778, "val_unseen": 1839},
}
EXPECTED_SCENES = {"satnav": 59, "habitat": 90}
DATA_PATH_ENV = {
    "satnav": "SWIFTVLN_SATNAV_EVAL_DATA_PATH",
    "habitat": "SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH",
}
SCENES_DIR_ENV = {
    "satnav": "SWIFTVLN_SATNAV_SCENES_DIR",
    "habitat": "SWIFTVLN_HABITAT_SCENES_DIR",
}
DISPLAY_NAME = {"satnav": "SatNav", "habitat": "Habitat"}
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "scene_id",
    "instruction",
    "start_position",
    "start_rotation",
    "goals",
    "reference_path",
}


@dataclass(frozen=True)
class SplitValidation:
    episodes: int
    scenes: int


@dataclass(frozen=True)
class EvaluationDataValidation:
    env_type: str
    splits: Mapping[str, SplitValidation]
    scene_assets: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swiftvln validate-evaluation-data",
        description="Validate SatNav or Habitat online evaluation data.",
    )
    parser.add_argument(
        "--env-type",
        required=True,
        choices=sorted(EXPECTED_EPISODES),
        help="Evaluation environment to validate.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "Episode path template containing {split}. Defaults to the matching "
            "SWIFTVLN_*_EVAL_DATA_PATH environment variable."
        ),
    )
    parser.add_argument(
        "--scenes-dir",
        default=None,
        help=(
            "Scene root. Defaults to the matching SWIFTVLN_*_SCENES_DIR "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("val_seen", "val_unseen"),
        help="Split to validate. Repeat to select multiple splits; defaults to both.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for a JSON validation report.",
    )
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_split_path(template: str, split: str, split_count: int) -> Path:
    if "{split}" in template:
        return Path(template.replace("{split}", split)).expanduser()
    _require(
        split_count == 1,
        "--data-path must contain {split} when validating multiple splits",
    )
    return Path(template).expanduser()


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode file not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"Expected a JSON object in {path}")
    episodes = payload.get("episodes")
    _require(isinstance(episodes, list), f"Missing episodes list in {path}")
    return episodes


def _validate_episode_rows(
    episodes: Sequence[dict[str, Any]],
    *,
    split: str,
    expected_count: int,
) -> set[str]:
    _require(
        len(episodes) == expected_count,
        f"{split}: expected {expected_count} episodes, found {len(episodes)}",
    )
    seen_keys: set[str] = set()
    scene_ids: set[str] = set()
    for index, episode in enumerate(episodes):
        _require(isinstance(episode, dict), f"{split}[{index}] is not an object")
        missing = REQUIRED_EPISODE_FIELDS - set(episode)
        _require(
            not missing,
            f"{split}[{index}] is missing fields: {sorted(missing)}",
        )
        episode_id = str(episode["episode_id"])
        scene_id = str(episode["scene_id"])
        _require(episode_id != "", f"{split}[{index}] has an empty episode_id")
        _require(scene_id != "", f"{split}[{index}] has an empty scene_id")
        episode_key = f"{scene_id}::{episode_id}"
        _require(episode_key not in seen_keys, f"{split}: duplicate {episode_key}")
        seen_keys.add(episode_key)
        scene_ids.add(scene_id)
    return scene_ids


def _satnav_scene_path(scenes_dir: Path, scene_id: str) -> Path:
    name = Path(scene_id).name
    if not name.lower().endswith((".tif", ".tiff")):
        name = f"{name}.tif"
    return scenes_dir / name


def _habitat_scene_path(scenes_dir: Path, scene_id: str) -> Path:
    path = Path(scene_id)
    if path.is_absolute():
        return path
    parts = path.parts
    if "mp3d" in parts:
        path = Path(*parts[parts.index("mp3d") :])
    return scenes_dir / path


def _validate_satnav_scenes(
    scenes_dir: Path,
    referenced_scenes: set[str],
    expected_scene_count: int,
) -> int:
    assets = sorted(
        path
        for path in scenes_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    _require(
        len(assets) == expected_scene_count,
        f"expected {expected_scene_count} GeoTIFF scenes, found {len(assets)}",
    )
    missing = sorted(
        scene_id
        for scene_id in referenced_scenes
        if not _satnav_scene_path(scenes_dir, scene_id).is_file()
    )
    _require(not missing, f"missing referenced GeoTIFF scenes: {missing[:10]}")
    return len(assets)


def _validate_habitat_scenes(
    scenes_dir: Path,
    referenced_scenes: set[str],
    expected_scene_count: int,
) -> int:
    mp3d_dir = scenes_dir if scenes_dir.name == "mp3d" else scenes_dir / "mp3d"
    glb_assets = sorted(mp3d_dir.glob("*/*.glb"))
    navmesh_assets = sorted(mp3d_dir.glob("*/*.navmesh"))
    _require(
        len(glb_assets) == expected_scene_count,
        f"expected {expected_scene_count} MP3D .glb scenes, found {len(glb_assets)}",
    )
    _require(
        len(navmesh_assets) == expected_scene_count,
        "expected "
        f"{expected_scene_count} MP3D .navmesh scenes, found {len(navmesh_assets)}",
    )
    glb_keys = {path.relative_to(mp3d_dir).with_suffix("") for path in glb_assets}
    navmesh_keys = {
        path.relative_to(mp3d_dir).with_suffix("") for path in navmesh_assets
    }
    _require(
        glb_keys == navmesh_keys,
        "MP3D .glb and .navmesh scene names do not match",
    )
    missing: list[str] = []
    for scene_id in sorted(referenced_scenes):
        glb_path = _habitat_scene_path(scenes_dir, scene_id)
        navmesh_path = glb_path.with_suffix(".navmesh")
        if not glb_path.is_file() or not navmesh_path.is_file():
            missing.append(scene_id)
    _require(not missing, f"missing referenced MP3D scene pairs: {missing[:10]}")
    return len(glb_assets)


def validate_evaluation_data(
    env_type: str,
    data_path: str,
    scenes_dir: Path,
    *,
    splits: Sequence[str] = ("val_seen", "val_unseen"),
    expected_counts: Mapping[str, int] | None = None,
    expected_scene_count: int | None = None,
) -> EvaluationDataValidation:
    env_type = env_type.lower()
    _require(env_type in EXPECTED_EPISODES, f"unsupported environment: {env_type}")
    _require(bool(splits), "at least one split is required")
    scenes_dir = scenes_dir.expanduser()
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scenes_dir}")

    counts = dict(expected_counts or EXPECTED_EPISODES[env_type])
    split_results: dict[str, SplitValidation] = {}
    referenced_scenes: set[str] = set()
    for split in splits:
        _require(split in counts, f"no expected episode count for split {split}")
        path = _resolve_split_path(data_path, split, len(splits))
        episodes = _load_episodes(path)
        scene_ids = _validate_episode_rows(
            episodes,
            split=split,
            expected_count=counts[split],
        )
        referenced_scenes.update(scene_ids)
        split_results[split] = SplitValidation(
            episodes=len(episodes),
            scenes=len(scene_ids),
        )

    scene_count = (
        expected_scene_count
        if expected_scene_count is not None
        else EXPECTED_SCENES[env_type]
    )
    if env_type == "satnav":
        assets = _validate_satnav_scenes(
            scenes_dir,
            referenced_scenes,
            scene_count,
        )
    else:
        assets = _validate_habitat_scenes(
            scenes_dir,
            referenced_scenes,
            scene_count,
        )
    return EvaluationDataValidation(
        env_type=env_type,
        splits=split_results,
        scene_assets=assets,
    )


def _write_report(path: Path, result: EvaluationDataValidation) -> None:
    payload = {
        "status": "passed",
        "environment": result.env_type,
        "splits": {
            name: {"episodes": value.episodes, "scenes": value.scenes}
            for name, value in result.splits.items()
        },
        "scene_assets": result.scene_assets,
    }
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_path = args.data_path or os.environ.get(DATA_PATH_ENV[args.env_type])
    scenes_dir = args.scenes_dir or os.environ.get(SCENES_DIR_ENV[args.env_type])
    if not data_path:
        parser.error(
            f"--data-path or {DATA_PATH_ENV[args.env_type]} is required"
        )
    if not scenes_dir:
        parser.error(
            f"--scenes-dir or {SCENES_DIR_ENV[args.env_type]} is required"
        )

    try:
        result = validate_evaluation_data(
            args.env_type,
            data_path,
            Path(scenes_dir),
            splits=tuple(args.split or ("val_seen", "val_unseen")),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Validation failed: {exc}")
        return 1

    for split, value in result.splits.items():
        print(
            f"{split}: passed "
            f"({value.episodes} episodes, {value.scenes} scenes)"
        )
    asset_name = "GeoTIFF" if result.env_type == "satnav" else "MP3D"
    print(f"{asset_name} scenes: passed ({result.scene_assets})")
    if args.report is not None:
        _write_report(args.report, result)
        print(f"Report: {args.report}")
    print(f"{DISPLAY_NAME[result.env_type]} evaluation data is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
