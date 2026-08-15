"""Validate offline Habitat trajectories used by SwiftVLN training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HABITAT_ACTIONS = {-1, 1, 2, 3}


@dataclass
class ValidationResult:
    trajectories: int = 0
    frames: int = 0
    decoded_frames: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swiftvln validate-habitat-trajectories",
        description=(
            "Validate Habitat trajectory annotations, actions, RGB directories, "
            "and frame alignment."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Trajectory directory containing annotations.json and images/.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected number of trajectory annotations.",
    )
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Decode every JPEG in addition to checking frame counts.",
    )
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return annotations


def validate(
    data_dir: Path,
    *,
    expected_count: int | None = None,
    decode_images: bool = False,
) -> ValidationResult:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {data_dir}")

    annotations = _load_annotations(data_dir / "annotations.json")
    if expected_count is not None:
        _require(
            expected_count >= 0,
            "--expected-count must be non-negative",
        )
        _require(
            len(annotations) == expected_count,
            f"Expected {expected_count} trajectories, found {len(annotations)}",
        )

    result = ValidationResult(trajectories=len(annotations))
    seen_ids: set[int] = set()
    seen_videos: set[str] = set()
    referenced_dirs: set[Path] = set()

    for index, item in enumerate(annotations):
        _require(isinstance(item, dict), f"Annotation {index} is not an object")
        missing = {"id", "video", "instructions", "actions"} - set(item)
        _require(not missing, f"Annotation {index} is missing fields: {sorted(missing)}")

        episode_id = item["id"]
        _require(
            isinstance(episode_id, int),
            f"Annotation {index} has a non-integer id",
        )
        _require(
            episode_id not in seen_ids,
            f"Duplicate Episode ID: {episode_id}",
        )
        seen_ids.add(episode_id)

        video = item["video"]
        _require(
            isinstance(video, str) and video,
            f"Episode {episode_id} has an invalid video path",
        )
        video_path = Path(video)
        _require(
            not video_path.is_absolute() and ".." not in video_path.parts,
            f"Episode {episode_id} has a non-relative video path: {video}",
        )
        _require(video not in seen_videos, f"Duplicate video path: {video}")
        seen_videos.add(video)

        instructions = item["instructions"]
        _require(
            isinstance(instructions, list)
            and instructions
            and all(isinstance(value, str) and value for value in instructions),
            f"Episode {episode_id} has invalid instructions",
        )

        actions = item["actions"]
        _require(
            isinstance(actions, list) and actions,
            f"Episode {episode_id} has no actions",
        )
        _require(
            actions[0] == -1,
            f"Episode {episode_id} actions must begin with INIT (-1)",
        )
        _require(
            all(isinstance(action, int) and action in HABITAT_ACTIONS for action in actions),
            f"Episode {episode_id} contains an invalid Habitat action",
        )
        _require(
            -1 not in actions[1:],
            f"Episode {episode_id} contains INIT after the first action",
        )

        episode_dir = (data_dir / video_path).resolve()
        try:
            episode_dir.relative_to(data_dir)
        except ValueError as exc:
            raise ValueError(
                f"Episode {episode_id} video path leaves the data directory"
            ) from exc
        rgb_dir = episode_dir / "rgb"
        _require(rgb_dir.is_dir(), f"RGB directory not found: {rgb_dir}")
        referenced_dirs.add(episode_dir)

        frames = sorted(
            path
            for path in rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        )
        _require(
            len(frames) == len(actions),
            f"Episode {episode_id}: {len(frames)} JPEGs but "
            f"{len(actions)} actions",
        )
        result.frames += len(frames)

        if decode_images:
            from PIL import Image

            for frame in frames:
                with Image.open(frame) as image:
                    image.verify()
                result.decoded_frames += 1

    images_dir = data_dir / "images"
    _require(images_dir.is_dir(), f"Image root not found: {images_dir}")
    actual_dirs = {path.resolve() for path in images_dir.iterdir() if path.is_dir()}
    extra_dirs = sorted(str(path) for path in actual_dirs - referenced_dirs)
    missing_dirs = sorted(str(path) for path in referenced_dirs - actual_dirs)
    _require(not missing_dirs, f"Referenced trajectory directories missing: {missing_dirs[:5]}")
    _require(not extra_dirs, f"Unreferenced trajectory directories found: {extra_dirs[:5]}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(
        Path(args.data_dir),
        expected_count=args.expected_count,
        decode_images=args.decode_images,
    )
    print(
        "Validation complete: "
        f"trajectories={result.trajectories}, frames={result.frames}, "
        f"decoded_frames={result.decoded_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
