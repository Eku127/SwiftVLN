"""Evaluate a primitive checkpoint on balanced category samples.

The regular :mod:`primitive_metrics` command evaluates a contiguous slice of
``annotations.json``.  This companion command reads the dataset's
``sample_manifest.jsonl``, draws a reproducible sample for every source
category, and evaluates them through the exact same autoregressive inference
session.  It also writes short diagnostic MP4 files: these replay the expert
RGB observations with the gold and predicted action overlaid.  They are useful
for inspecting model decisions, but are not simulator closed-loop rollouts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .primitive_metrics import (
    ACTION_NAMES,
    PrimitiveMetricConfig,
    _accumulate,
    _empty_summary,
    _evaluate_episode,
    _finish_summary,
    _gold_actions,
    _load_annotations,
    _load_inference_session,
    _normalise_annotation,
    _positive_int,
    _nonnegative_int,
)


CATEGORIES = ("boundary", "landmark", "road")


def _read_manifest(dataset_path: Path) -> dict[int, dict[str, Any]]:
    path = dataset_path / "sample_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            "category evaluation requires sample_manifest.jsonl: " f"{path}"
        )
    manifest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            sample_id = record.get("id") if isinstance(record, dict) else None
            category = record.get("category") if isinstance(record, dict) else None
            if not isinstance(sample_id, int) or not isinstance(category, str):
                raise ValueError(f"{path}:{line_number}: requires integer id and category")
            if sample_id in manifest:
                raise ValueError(f"{path}:{line_number}: duplicate sample id {sample_id}")
            manifest[sample_id] = record
    if not manifest:
        raise ValueError(f"manifest is empty: {path}")
    return manifest


def _select_balanced_items(
    dataset_path: Path,
    *,
    samples_per_category: int | None,
    seed: int,
    categories: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    manifest = _read_manifest(dataset_path)
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    for position, record in enumerate(_load_annotations(dataset_path)):
        item = _normalise_annotation(record, position, dataset_path)
        sample_id = item["id"]
        if not isinstance(sample_id, int) or sample_id not in manifest:
            raise ValueError(f"annotation {position}: missing matching manifest id")
        category = manifest[sample_id]["category"]
        if category in grouped:
            item["manifest"] = manifest[sample_id]
            grouped[category].append(item)

    selected: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        pool = grouped[category]
        if samples_per_category is None:
            selected[category] = sorted(pool, key=lambda item: int(item["id"]))
            continue
        if len(pool) < samples_per_category:
            raise ValueError(
                f"category {category!r} has only {len(pool)} valid samples; "
                f"{samples_per_category} are required"
            )
        chooser = random.Random(f"{seed}:{category}")
        selected[category] = chooser.sample(pool, samples_per_category)
    return selected


def _episode_directory_name(item: Mapping[str, Any]) -> str:
    manifest = item["manifest"]
    scene_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(manifest["source_scene_id"]))
    trajectory_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(manifest["source_trajectory_id"]))
    return f"{scene_id}__{trajectory_id}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _draw_video_frame(
    image_path: Path,
    *,
    category: str,
    instruction: str,
    step: int,
    total_steps: int,
    gold: str,
    predicted: str,
) -> Any:
    """Create one readable RGB frame with a metadata header."""

    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    font = ImageFont.load_default()
    lines = [
        f"{category} | step {step + 1}/{total_steps}",
        f"gold: {gold}    prediction: {predicted}",
        *textwrap.wrap(instruction, width=max(36, width // 10))[:3],
    ]
    line_height = 16
    header_height = 10 + line_height * len(lines)
    canvas = Image.new("RGB", (width, height + header_height), "black")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(lines):
        draw.text((6, 5 + line_index * line_height), line, fill="white", font=font)
    return canvas


def _write_diagnostic_video(
    path: Path,
    item: Mapping[str, Any],
    predicted: Sequence[int],
    *,
    category: str,
) -> None:
    """Write an expert-frame replay with per-step gold/predicted labels."""

    import imageio.v2 as imageio
    import numpy as np

    gold = _gold_actions(item["raw_actions"])
    if len(gold) != len(predicted) or len(gold) != len(item["frame_paths"]):
        raise ValueError("cannot render a video with misaligned action/frame lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=2, codec="libx264", macro_block_size=1) as writer:
        for step, (frame_path, expected, actual) in enumerate(
            zip(item["frame_paths"], gold, predicted)
        ):
            frame = _draw_video_frame(
                frame_path,
                category=category,
                instruction=str(item["instruction"]),
                step=step,
                total_steps=len(gold),
                gold=ACTION_NAMES[expected],
                predicted=ACTION_NAMES[actual],
            )
            writer.append_data(np.asarray(frame))


def _category_record(
    item: Mapping[str, Any],
    predicted: Sequence[int],
    turns: Sequence[Mapping[str, Any]],
    timing: Mapping[str, float],
    exact_match: bool,
    video_path: Path | None,
) -> dict[str, Any]:
    gold = _gold_actions(item["raw_actions"])
    return {
        "sample_id": item["id"],
        "trajectory_id": item["trajectory_id"],
        "category": item["manifest"]["category"],
        "source_segment_index": item["manifest"].get("source_segment_index"),
        "instruction": item["instruction"],
        "source_full_instruction": item["manifest"].get("source_full_instruction"),
        "gold_actions": [ACTION_NAMES[action] for action in gold],
        "predicted_actions": [ACTION_NAMES[action] for action in predicted],
        "exact_match": exact_match,
        "turns": list(turns),
        "timing_seconds": dict(timing),
        "diagnostic_video": str(video_path) if video_path is not None else None,
    }


def _build_config(args: argparse.Namespace) -> PrimitiveMetricConfig:
    return PrimitiveMetricConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        num_frames=args.num_frames,
        num_history=args.num_history,
        num_future_steps=args.num_future_steps,
        num_overlap=args.num_overlap,
        history_processor_type=args.history_processor_type,
        compress_stride=args.compress_stride,
        log_base=args.log_base,
        use_random=args.use_random,
        use_tome=args.use_tome,
        system_prompt_setting=args.system_prompt_setting,
        embedding_mode=args.embedding_mode,
        max_samples=None,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        attn_impl=args.attn_impl,
    )


def evaluate_categories(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_future_steps != 1:
        raise ValueError("category primitive evaluation requires --num-future-steps=1")
    if args.num_overlap != 0:
        raise ValueError("category primitive evaluation requires --num-overlap=0")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_balanced_items(
        args.dataset_path,
        samples_per_category=None if args.all_samples else args.samples_per_category,
        seed=args.seed,
        categories=args.categories,
    )
    selected_rows = [
        {
            "sample_id": item["id"],
            "category": category,
            "trajectory_id": item["trajectory_id"],
            "source_segment_index": item["manifest"].get("source_segment_index"),
        }
        for category in args.categories
        for item in selected[category]
    ]
    _write_jsonl(args.output_dir / "selected_samples.jsonl", selected_rows)

    config = _build_config(args)
    _, session = _load_inference_session(config)
    started_at = time.time()
    category_summaries: dict[str, dict[str, Any]] = {}
    category_prediction_paths: dict[str, str] = {}
    aggregate = _empty_summary()
    for category in args.categories:
        state = _empty_summary()
        rows: list[dict[str, Any]] = []
        for position, item in enumerate(selected[category], start=1):
            predicted, turns, timing = _evaluate_episode(session, item)
            gold = _gold_actions(item["raw_actions"])
            exact = _accumulate(state, gold, predicted)
            _accumulate(aggregate, gold, predicted)
            video_path: Path | None = None
            if position <= args.videos_per_category:
                video_directory = args.output_dir / "diagnostic_videos" / category
                if args.group_videos_by_episode:
                    video_directory = video_directory / _episode_directory_name(item)
                video_path = video_directory / f"{int(item['id']):06d}_{item['trajectory_id']}.mp4"
                _write_diagnostic_video(video_path, item, predicted, category=category)
            rows.append(_category_record(item, predicted, turns, timing, exact, video_path))
            print(
                f"[primitive category eval] {category} {position}/{len(selected[category])} "
                f"exact={int(exact)} actions={len(gold)}",
                flush=True,
            )
        prediction_path = args.output_dir / f"predictions_{category}.jsonl"
        _write_jsonl(prediction_path, rows)
        category_prediction_paths[category] = str(prediction_path.resolve())
        category_summaries[category] = _finish_summary(state)
        _write_json(args.output_dir / f"metrics_{category}.json", category_summaries[category])

    result = {
        "record_type": "primitive_category_action_metrics",
        "created_at_unix": time.time(),
        "checkpoint_name": args.model_path.name,
        "checkpoint_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "categories": list(args.categories),
        "samples_per_category": args.samples_per_category,
        "videos_per_category": args.videos_per_category,
        "selected_sample_count": sum(len(items) for items in selected.values()),
        "category_prediction_paths": category_prediction_paths,
        "metrics": {
            "overall": _finish_summary(aggregate),
            "by_category": category_summaries,
        },
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
            "samples_per_category": args.samples_per_category,
            "all_samples": args.all_samples,
            "videos_per_category": args.videos_per_category,
            "group_videos_by_episode": args.group_videos_by_episode,
        },
        "elapsed_seconds": time.time() - started_at,
        "video_note": (
            "Diagnostic videos replay expert RGB frames with gold/predicted action overlays; "
            "they are not closed-loop simulator rollouts."
        ),
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balanced per-category autoregressive evaluation for primitive SwiftVLN datasets."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-type", choices=("swiftvln_qwen2_5_vl", "swiftvln_qwen3_vl"), required=True
    )
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--samples-per-category", type=_positive_int, default=50)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate every selected sample for each category, without balanced subsampling.",
    )
    parser.add_argument("--videos-per-category", type=_nonnegative_int, default=50)
    parser.add_argument(
        "--group-videos-by-episode",
        action="store_true",
        help="Store diagnostic videos below category/scene_id__trajectory_id/.",
    )
    parser.add_argument("--num-frames", type=_positive_int, default=8)
    parser.add_argument("--num-history", type=_nonnegative_int, default=8)
    parser.add_argument("--num-future-steps", type=_positive_int, default=1)
    parser.add_argument("--num-overlap", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--history-processor-type", choices=("per_frame", "gtc", "segment_gtc"), default="per_frame"
    )
    parser.add_argument("--compress-stride", type=_positive_int, default=2)
    parser.add_argument("--log-base", type=float, default=1.0)
    parser.add_argument("--use-random", action="store_true")
    parser.add_argument("--use-tome", action="store_true")
    parser.add_argument("--system-prompt-setting", choices=("vanilla", "initial"), default="vanilla")
    parser.add_argument("--embedding-mode", choices=("none", "pose", "posefilm", "uav"), default="none")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-impl", choices=("flash_attn", "sdpa", "eager"), default="flash_attn")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_categories(args)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"Primitive category evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
