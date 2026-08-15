"""Generate offline Habitat trajectories for SwiftVLN training."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable


INITIAL_ACTION = -1
STOP_ACTION = 0
DEFAULT_MAX_ACTIONS = 498


@dataclass
class GenerationStats:
    selected: int = 0
    generated: int = 0
    resumed: int = 0
    skipped_max_actions: int = 0


def default_config_path() -> str:
    return str(files("swiftvln").joinpath("configs", "habitat", "r2r.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swiftvln generate-habitat-trajectories",
        description=(
            "Generate RGB observation-action trajectories from VLN-CE episodes "
            "with Habitat ShortestPathFollower."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("r2r", "rxr", "envdrop"),
        help="Dataset source used in generated trajectory directory names.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="VLN-CE episode JSON.GZ path.",
    )
    parser.add_argument(
        "--scenes-dir",
        required=True,
        help="Scene dataset root containing mp3d/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for annotations.json and images/.",
    )
    parser.add_argument(
        "--config-path",
        default=default_config_path(),
        help="Habitat task config. Defaults to SwiftVLN's packaged R2R config.",
    )
    parser.add_argument("--split", default="train", help="Habitat dataset split.")
    parser.add_argument(
        "--instruction-language",
        default=None,
        help=(
            "Keep only episodes whose instruction language matches this code, "
            "for example 'en' matches 'en-US'."
        ),
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help=(
            "Habitat-Sim GPU device index. Defaults to LOCAL_RANK under "
            "torchrun and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_MAX_ACTIONS,
        help="Discard trajectories longer than this action count.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Generate only the first N selected episodes.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="Generate only this Episode ID. May be supplied more than once.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed records in an existing output directory.",
    )
    return parser


def _ordered_episodes(episodes: Iterable[Any]) -> list[Any]:
    by_scene: dict[str, list[Any]] = defaultdict(list)
    for episode in episodes:
        by_scene[str(episode.scene_id)].append(episode)
    return [
        episode
        for scene_id in sorted(by_scene)
        for episode in by_scene[scene_id]
    ]


def _language_matches(language: str, requested: str) -> bool:
    language = language.lower()
    requested = requested.lower()
    return language == requested or language.startswith(requested + "-")


def _select_episodes(
    episodes: Iterable[Any],
    episode_ids: set[str],
    max_episodes: int | None,
    instruction_language: str | None,
) -> list[Any]:
    selected = _ordered_episodes(episodes)
    if instruction_language:
        selected = [
            episode
            for episode in selected
            if _language_matches(
                str(getattr(episode.instruction, "language", "")),
                instruction_language,
            )
        ]
        if not selected:
            raise ValueError(
                f"No episodes found for instruction language "
                f"{instruction_language!r}"
            )
    if episode_ids:
        selected = [
            episode
            for episode in selected
            if str(episode.episode_id) in episode_ids
        ]
        found = {str(episode.episode_id) for episode in selected}
        missing = sorted(episode_ids - found)
        if missing:
            raise ValueError(f"Episode IDs not found: {', '.join(missing)}")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be greater than zero")
        selected = selected[:max_episodes]
    return selected


def _shard_episodes(
    episodes: Iterable[Any],
    rank: int,
    world_size: int,
) -> list[Any]:
    by_scene: dict[str, list[Any]] = defaultdict(list)
    for episode in episodes:
        by_scene[str(episode.scene_id)].append(episode)
    return [
        episode
        for scene_id in sorted(by_scene)
        for episode in by_scene[scene_id][rank::world_size]
    ]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    with path.open(encoding="utf-8") as handle:
        return list(json.load(handle))


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _scene_name(episode: Any) -> str:
    scene_path = Path(str(episode.scene_id))
    if not scene_path.parent.name:
        raise ValueError(f"Cannot resolve scene name from {episode.scene_id!r}")
    return scene_path.parent.name


def _video_id(episode: Any, dataset: str) -> str:
    try:
        episode_id = int(episode.episode_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Habitat trajectory generation requires a numeric Episode ID, "
            f"got {episode.episode_id!r}"
        ) from exc
    return f"{_scene_name(episode)}_{dataset}_{episode_id:06d}"


def _instruction_list(episode: Any) -> list[str]:
    instruction = episode.instruction.instruction_text
    if isinstance(instruction, list):
        return [str(item) for item in instruction]
    return [str(instruction)]


def _generate_episode(
    env: Any,
    episode: Any,
    *,
    dataset: str,
    output_dir: Path,
    max_actions: int,
    rank: int,
) -> dict[str, Any] | None:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from PIL import Image

    reference_path = episode.reference_path
    if len(reference_path) < 2:
        raise ValueError(
            f"Episode {episode.episode_id} has fewer than two reference waypoints"
        )

    video_id = _video_id(episode, dataset)
    final_episode_dir = output_dir / "images" / video_id
    partial_episode_dir = (
        output_dir / ".partial" / f"rank_{rank:03d}" / video_id
    )
    if partial_episode_dir.exists():
        shutil.rmtree(partial_episode_dir)
    rgb_dir = partial_episode_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    env.current_episode = episode
    observation = env.reset()
    follower = ShortestPathFollower(
        sim=env.sim,
        goal_radius=0.5,
        return_one_hot=False,
    )
    actions = [INITIAL_ACTION]
    next_waypoint_id = 1
    frame_count = 0

    while not env.episode_over:
        frame_count += 1
        Image.fromarray(observation["rgb"]).convert("RGB").save(
            rgb_dir / f"{frame_count:03d}.jpg"
        )

        reached_reference_end = False
        next_action = follower.get_next_action(reference_path[next_waypoint_id])
        if next_action is None:
            raise RuntimeError(
                f"Episode {episode.episode_id}: shortest path is unavailable"
            )
        while next_action == STOP_ACTION:
            next_waypoint_id += 1
            if next_waypoint_id == len(reference_path) - 1:
                follower = ShortestPathFollower(
                    sim=env.sim,
                    goal_radius=0.25,
                    return_one_hot=False,
                )
            if next_waypoint_id >= len(reference_path):
                reached_reference_end = True
                break
            next_action = follower.get_next_action(
                reference_path[next_waypoint_id]
            )
            if next_action is None:
                raise RuntimeError(
                    f"Episode {episode.episode_id}: shortest path is unavailable"
                )

        if reached_reference_end:
            break

        observation = env.step(next_action)
        actions.append(int(next_action))

    if len(actions) != frame_count:
        raise RuntimeError(
            f"Episode {episode.episode_id}: {len(actions)} actions but "
            f"{frame_count} RGB frames"
        )
    if len(actions) > max_actions:
        shutil.rmtree(partial_episode_dir)
        return None

    final_episode_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_episode_dir.exists():
        shutil.rmtree(final_episode_dir)
    os.replace(partial_episode_dir, final_episode_dir)
    return {
        "id": int(episode.episode_id),
        "video": f"images/{video_id}",
        "instructions": _instruction_list(episode),
        "actions": actions,
    }


def _distributed_identity() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be greater than zero")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    return rank, world_size, local_rank


def _prepare_output_dir(
    output_dir: Path,
    *,
    resume: bool,
    rank: int,
    process_group: Any | None,
) -> None:
    error: str | None = None
    if rank == 0:
        try:
            if output_dir.exists() and any(output_dir.iterdir()) and not resume:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. "
                    "Use --resume or select a new directory."
                )
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # propagated to every torchrun rank below
            error = f"{type(exc).__name__}: {exc}"

    if process_group is not None:
        payload = [error]
        process_group.broadcast_object_list(payload, src=0)
        error = payload[0]
        process_group.barrier()
    if error is not None:
        raise RuntimeError(error)


def _prepare_language_episode_data(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    rank: int,
    process_group: Any | None,
) -> Path | None:
    if not args.instruction_language:
        return None

    language = args.instruction_language
    safe_language = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in language.lower()
    )
    prepared_root = output_dir / ".prepared_episodes"
    prepared_path = prepared_root / f"{args.dataset}_{safe_language}.json.gz"
    error: str | None = None

    if rank == 0:
        try:
            source_path = Path(args.data_path).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Episode data not found: {source_path}")
            if not prepared_path.is_file():
                with gzip.open(source_path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)

                episodes = []
                for episode in payload.get("episodes", []):
                    instruction = episode.get("instruction", {})
                    if not _language_matches(
                        str(instruction.get("language", "")),
                        language,
                    ):
                        continue
                    prepared_episode = dict(episode)
                    prepared_episode["instruction"] = {
                        "instruction_text": instruction["instruction_text"],
                        "instruction_tokens": instruction.get(
                            "instruction_tokens",
                            [],
                        ),
                    }
                    episodes.append(prepared_episode)

                if not episodes:
                    raise ValueError(
                        f"No episodes found for instruction language {language!r}"
                    )

                prepared_root.mkdir(parents=True, exist_ok=True)
                temporary_path = prepared_path.with_suffix(".json.gz.tmp")
                with gzip.open(
                    temporary_path,
                    "wt",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "episodes": episodes,
                            "instruction_vocab": {"word_list": []},
                        },
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                os.replace(temporary_path, prepared_path)
                print(
                    f"Prepared {len(episodes)} {language} episodes: "
                    f"{prepared_path}"
                )
        except Exception as exc:  # propagated to every torchrun rank below
            error = f"{type(exc).__name__}: {exc}"

    if process_group is not None:
        payload = [error]
        process_group.broadcast_object_list(payload, src=0)
        error = payload[0]
        process_group.barrier()
    if error is not None:
        raise RuntimeError(error)

    args.data_path = str(prepared_path)
    args.instruction_language = None
    return prepared_root


def _rank_paths(
    output_dir: Path,
    rank: int,
    world_size: int,
) -> tuple[Path, Path, Path]:
    if world_size == 1:
        return (
            output_dir / "annotations.json",
            output_dir / ".annotations.jsonl",
            output_dir / "generation_summary.json",
        )
    return (
        output_dir / f"annotations_{rank}.json",
        output_dir / f".annotations_{rank}.jsonl",
        output_dir / f"generation_summary_{rank}.json",
    )


def generate(
    args: argparse.Namespace,
    *,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> tuple[GenerationStats, list[int]]:
    try:
        import habitat
        from habitat.config import read_write
        from habitat_baselines.config.default import get_config
        from swiftvln.backends.habitat import measures  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Habitat trajectory generation requires the swiftvln-eval environment"
        ) from exc

    data_path = Path(args.data_path).expanduser().resolve()
    scenes_dir = Path(args.scenes_dir).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for label, path in (
        ("Episode data", data_path),
        ("Scene directory", scenes_dir),
        ("Habitat config", config_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.max_actions <= 0:
        raise ValueError("--max-actions must be greater than zero")

    annotations_path, journal_path, summary_path = _rank_paths(
        output_dir,
        rank,
        world_size,
    )
    existing_records: list[dict[str, Any]] = []
    if args.resume:
        existing_records = _load_records(annotations_path)
        journal_records = _load_records(journal_path)
        existing_by_id = {int(item["id"]): item for item in existing_records}
        existing_by_id.update(
            {int(item["id"]): item for item in journal_records}
        )
        existing_records = list(existing_by_id.values())
    config = get_config(str(config_path))
    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            args.gpu_device_id
            if args.gpu_device_id is not None
            else local_rank
        )

    env = habitat.Env(config=config)
    try:
        all_selected = _select_episodes(
            env.episodes,
            set(args.episode_id),
            args.max_episodes,
            args.instruction_language,
        )
        selected = _shard_episodes(all_selected, rank, world_size)
        stats = GenerationStats(selected=len(selected))
        records_by_id = {int(item["id"]): item for item in existing_records}
        with journal_path.open("a", encoding="utf-8") as journal:
            for index, episode in enumerate(selected, start=1):
                episode_id = int(episode.episode_id)
                existing = records_by_id.get(episode_id)
                if existing is not None:
                    rgb_dir = output_dir / existing["video"] / "rgb"
                    if rgb_dir.is_dir():
                        stats.resumed += 1
                        print(
                            f"[rank {rank}][{index}/{len(selected)}] resume episode "
                            f"{episode_id}"
                        )
                        continue

                record = _generate_episode(
                    env,
                    episode,
                    dataset=args.dataset,
                    output_dir=output_dir,
                    max_actions=args.max_actions,
                    rank=rank,
                )
                if record is None:
                    stats.skipped_max_actions += 1
                    print(
                        f"[rank {rank}][{index}/{len(selected)}] skip episode "
                        f"{episode_id}: "
                        f"more than {args.max_actions} actions"
                    )
                    continue

                records_by_id[episode_id] = record
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
                stats.generated += 1
                print(
                    f"[rank {rank}][{index}/{len(selected)}] generated episode "
                    f"{episode_id}: "
                    f"{len(record['actions'])} frames"
                )
    finally:
        env.close()

    records = list(records_by_id.values())
    _write_json_atomic(annotations_path, records)
    _write_json_atomic(
        summary_path,
        {
            **asdict(stats),
            "rank": rank,
            "world_size": world_size,
            "written": len(records),
        },
    )
    if journal_path.exists():
        journal_path.unlink()
    rank_partial_root = output_dir / ".partial" / f"rank_{rank:03d}"
    if rank_partial_root.is_dir() and not any(rank_partial_root.iterdir()):
        rank_partial_root.rmdir()
    return stats, [int(episode.episode_id) for episode in all_selected]


def _merge_rank_outputs(
    output_dir: Path,
    *,
    world_size: int,
    ordered_episode_ids: list[int],
) -> GenerationStats:
    records_by_id: dict[int, dict[str, Any]] = {}
    combined = GenerationStats()
    rank_summaries = []

    for rank in range(world_size):
        annotations_path, _, summary_path = _rank_paths(
            output_dir,
            rank,
            world_size,
        )
        if not annotations_path.is_file():
            raise FileNotFoundError(f"Annotation shard not found: {annotations_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary shard not found: {summary_path}")

        for record in _load_records(annotations_path):
            episode_id = int(record["id"])
            if episode_id in records_by_id:
                raise RuntimeError(
                    f"Episode {episode_id} appears in multiple annotation shards"
                )
            records_by_id[episode_id] = record

        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        rank_summaries.append(summary)
        combined.selected += int(summary["selected"])
        combined.generated += int(summary["generated"])
        combined.resumed += int(summary["resumed"])
        combined.skipped_max_actions += int(summary["skipped_max_actions"])

    records = [
        records_by_id[episode_id]
        for episode_id in ordered_episode_ids
        if episode_id in records_by_id
    ]
    if len(records) != len(records_by_id):
        raise RuntimeError("Merged annotation shards contain unexpected Episode IDs")

    _write_json_atomic(output_dir / "annotations.json", records)
    _write_json_atomic(
        output_dir / "generation_summary.json",
        {
            **asdict(combined),
            "world_size": world_size,
            "written": len(records),
            "ranks": rank_summaries,
        },
    )
    return combined


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, world_size, local_rank = _distributed_identity()
    process_group = None
    if world_size > 1:
        import torch.distributed as distributed

        distributed.init_process_group(backend="gloo")
        process_group = distributed

    output_dir = Path(args.output_dir).expanduser().resolve()
    prepared_root: Path | None = None
    completed = False
    try:
        _prepare_output_dir(
            output_dir,
            resume=args.resume,
            rank=rank,
            process_group=process_group,
        )
        prepared_root = _prepare_language_episode_data(
            args,
            output_dir,
            rank=rank,
            process_group=process_group,
        )
        stats, ordered_episode_ids = generate(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

        if process_group is None:
            print(
                "Generation complete: "
                f"selected={stats.selected}, generated={stats.generated}, "
                f"resumed={stats.resumed}, "
                f"skipped_max_actions={stats.skipped_max_actions}"
            )
        else:
            process_group.barrier()
            merge_error: str | None = None
            if rank == 0:
                try:
                    combined = _merge_rank_outputs(
                        output_dir,
                        world_size=world_size,
                        ordered_episode_ids=ordered_episode_ids,
                    )
                    partial_root = output_dir / ".partial"
                    if partial_root.is_dir() and not any(partial_root.iterdir()):
                        partial_root.rmdir()
                    print(
                        "Parallel generation complete: "
                        f"selected={combined.selected}, "
                        f"generated={combined.generated}, "
                        f"resumed={combined.resumed}, "
                        "skipped_max_actions="
                        f"{combined.skipped_max_actions}"
                    )
                except Exception as exc:
                    merge_error = f"{type(exc).__name__}: {exc}"
            payload = [merge_error]
            process_group.broadcast_object_list(payload, src=0)
            if payload[0] is not None:
                raise RuntimeError(payload[0])
        completed = True
    finally:
        if rank == 0 and completed and prepared_root is not None:
            shutil.rmtree(prepared_root)
        if process_group is not None and process_group.is_initialized():
            process_group.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
