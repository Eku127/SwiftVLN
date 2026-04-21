#!/usr/bin/env python3
"""Export standalone SatNav map-memory examples for papers and reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_map_memory_module():
    repo_root = _repo_root()
    module_path = repo_root / "src/swiftvln/model/map_memory.py"
    spec = importlib.util.spec_from_file_location("swiftvln_map_memory_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load map_memory module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-data-dir",
        required=True,
        help="SatNav trajectory_data directory containing annotations.json and summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store exported map examples.",
    )
    parser.add_argument(
        "--annotation-ids",
        required=True,
        nargs="+",
        type=int,
        help="Annotation ids to export.",
    )
    parser.add_argument("--global-side-m", type=float, default=1000.0)
    parser.add_argument("--local-side-m", type=float, default=400.0)
    parser.add_argument("--render-px", type=int, default=448)
    parser.add_argument("--mask-method", type=str, default="dilate20")
    parser.add_argument(
        "--stage-spec",
        type=str,
        default="start:0,mid:0.5,end:1.0",
        help="Comma-separated stage_name:fraction in [0,1]. Example: start:0,mid:0.5,end:1.0",
    )
    return parser.parse_args()


def _load_annotations(path: Path) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return {int(item["id"]): item for item in annotations}


def _load_summary(path: Path) -> Dict[int, dict]:
    summary = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary[int(record["id"])] = record
    return summary


def _normalize_scene_name(scene_ref: str) -> str:
    return Path(str(scene_ref)).stem


def _parse_stage_spec(raw: str) -> List[Tuple[str, float]]:
    stages: List[Tuple[str, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid stage spec chunk: {chunk}")
        name, value = chunk.split(":", 1)
        frac = float(value)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"Stage fraction must be in [0,1], got {chunk}")
        stages.append((name.strip(), frac))
    if not stages:
        raise ValueError("At least one stage must be specified.")
    return stages


def _window_start_from_fraction(num_actions: int, fraction: float) -> int:
    if num_actions <= 0:
        return 0
    last_index = max(0, num_actions - 1)
    return max(0, min(last_index, int(round(last_index * fraction))))


def _episode_lookup(resolver, scene_ref: str, episode_id: str) -> dict:
    key = (_normalize_scene_name(scene_ref), str(episode_id))
    episode = resolver.episodes_by_key.get(key)
    if episode is None:
        raise KeyError(f"Episode metadata not found for {key}")
    return episode


def _route_metrics(actions: Iterable[int], start_rotation: float) -> dict:
    heading = float(start_rotation)
    x = 0.0
    y = 0.0
    points = [(x, y)]
    left_turns = 0
    right_turns = 0
    forward_steps = 0
    for action in actions:
        action = int(action)
        if action == 2:
            heading = (heading - 15.0) % 360.0
            left_turns += 1
        elif action == 3:
            heading = (heading + 15.0) % 360.0
            right_turns += 1
        elif action == 1:
            heading_rad = math.radians(heading)
            x += 10.0 * math.sin(heading_rad)
            y += 10.0 * math.cos(heading_rad)
            points.append((x, y))
            forward_steps += 1

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width_m = max(xs) - min(xs)
    height_m = max(ys) - min(ys)
    displacement_m = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    forward_m = 10.0 * forward_steps
    return {
        "forward_m": float(forward_m),
        "turns": int(left_turns + right_turns),
        "left_turns": int(left_turns),
        "right_turns": int(right_turns),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "displacement_m": float(displacement_m),
    }


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _compose_pair(global_img: Image.Image, local_img: Image.Image, title: str, subtitle: str) -> Image.Image:
    margin = 18
    gutter = 18
    header_h = 56
    width = global_img.width + local_img.width + gutter + margin * 2
    height = max(global_img.height, local_img.height) + header_h + margin * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    draw.text((margin, 10), title, fill=(16, 16, 16), font=font)
    draw.text((margin, 30), subtitle, fill=(96, 96, 96), font=font)

    y0 = header_h
    canvas.paste(global_img, (margin, y0))
    canvas.paste(local_img, (margin + global_img.width + gutter, y0))
    draw.rectangle((margin - 1, y0 - 1, margin + global_img.width, y0 + global_img.height), outline=(220, 220, 220), width=1)
    lx = margin + global_img.width + gutter
    draw.rectangle((lx - 1, y0 - 1, lx + local_img.width, y0 + local_img.height), outline=(220, 220, 220), width=1)
    return canvas


def _compose_strip(pairs: List[Tuple[str, Image.Image]], title: str, subtitle: str) -> Image.Image:
    margin = 18
    gutter = 18
    header_h = 56
    width = margin * 2 + sum(img.width for _, img in pairs) + gutter * max(0, len(pairs) - 1)
    height = header_h + margin * 2 + max(img.height for _, img in pairs)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((margin, 10), title, fill=(16, 16, 16), font=font)
    draw.text((margin, 30), subtitle, fill=(96, 96, 96), font=font)

    x = margin
    y = header_h
    for _, img in pairs:
        canvas.paste(img, (x, y))
        draw.rectangle((x - 1, y - 1, x + img.width, y + img.height), outline=(220, 220, 220), width=1)
        x += img.width + gutter
    return canvas


def _save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    args = _parse_args()
    module = _load_map_memory_module()

    trajectory_data_dir = Path(args.trajectory_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _load_annotations(trajectory_data_dir / "annotations.json")
    summary_by_id = _load_summary(trajectory_data_dir / "summary.json")
    stages = _parse_stage_spec(args.stage_spec)

    resolver = module.SatNavTrajectoryMetadataResolver(str(trajectory_data_dir))
    builder = module.SatNavMapMemoryBuilder(
        scenes_dir=resolver.scenes_dir,
        global_side_m=args.global_side_m,
        local_side_m=args.local_side_m,
        render_px=args.render_px,
        mask_method=args.mask_method,
        cache_dir="off",
    )

    index_lines = [
        "# Map Memory Paper Examples",
        "",
        f"trajectory_data_dir: `{trajectory_data_dir}`",
        f"render_config: global={args.global_side_m}m, local={args.local_side_m}m, render={args.render_px}px, mask={args.mask_method}",
        "",
    ]

    for rank, annotation_id in enumerate(args.annotation_ids, start=1):
        annotation = annotations.get(annotation_id)
        if annotation is None:
            raise KeyError(f"Annotation id not found: {annotation_id}")
        summary = summary_by_id.get(annotation_id)
        if summary is None:
            raise KeyError(f"summary.json is missing annotation id: {annotation_id}")

        meta = resolver.resolve(annotation)
        episode = _episode_lookup(resolver, meta.scene_id, meta.episode_id)
        scene_name = _normalize_scene_name(meta.scene_id)
        metrics = _route_metrics(annotation["actions"], meta.start_rotation)
        case_name = f"{rank:02d}_{scene_name}_ann{annotation_id}"
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        pair_images: List[Tuple[str, Image.Image]] = []
        stage_records: List[dict] = []
        num_actions = len(annotation["actions"])
        instruction = annotation["instructions"][0] if annotation.get("instructions") else ""

        for stage_name, fraction in stages:
            window_start = _window_start_from_fraction(num_actions, fraction)
            global_img, local_img = builder.render_from_actions(
                scene_id=meta.scene_id,
                start_position=meta.start_position,
                start_rotation=meta.start_rotation,
                actions=annotation["actions"],
                window_start=window_start,
                step_size=10.0,
                turn_angle=15.0,
            )

            global_path = case_dir / f"{stage_name}_global.png"
            local_path = case_dir / f"{stage_name}_local.png"
            _save_image(global_path, global_img)
            _save_image(local_path, local_img)

            pair_title = f"{case_name} | {stage_name}"
            pair_subtitle = (
                f"window_start={window_start} / {max(0, num_actions - 1)} | "
                f"global {args.global_side_m:.0f}m | local {args.local_side_m:.0f}m"
            )
            pair_img = _compose_pair(global_img, local_img, pair_title, pair_subtitle)
            pair_path = case_dir / f"{stage_name}_pair.png"
            _save_image(pair_path, pair_img)
            pair_images.append((stage_name, pair_img))

            stage_records.append(
                {
                    "stage_name": stage_name,
                    "fraction": fraction,
                    "window_start": window_start,
                    "global_png": global_path.name,
                    "local_png": local_path.name,
                    "pair_png": pair_path.name,
                }
            )

        strip_title = case_name
        strip_subtitle = (
            f"scene={scene_name} episode={meta.episode_id} | "
            f"type={episode.get('trajectory_type', '?')} subtype={episode.get('trajectory_subtype', '?')} | "
            f"forward={metrics['forward_m']:.0f}m turns={metrics['turns']}"
        )
        strip_img = _compose_strip(pair_images, strip_title, strip_subtitle)
        strip_path = case_dir / "overview_strip.png"
        _save_image(strip_path, strip_img)

        payload = {
            "annotation_id": annotation_id,
            "scene_name": scene_name,
            "episode_id": str(meta.episode_id),
            "video": annotation.get("video"),
            "instruction": instruction,
            "trajectory_type": episode.get("trajectory_type", "?"),
            "trajectory_subtype": episode.get("trajectory_subtype", "?"),
            "start_position": list(meta.start_position),
            "start_rotation": float(meta.start_rotation),
            "render_config": {
                "global_side_m": args.global_side_m,
                "local_side_m": args.local_side_m,
                "render_px": args.render_px,
                "mask_method": args.mask_method,
            },
            "metrics": metrics,
            "summary_record": summary,
            "stages": stage_records,
            "overview_strip": strip_path.name,
        }
        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        index_lines.extend(
            [
                f"## {case_name}",
                "",
                f"- scene: `{scene_name}`",
                f"- annotation_id: `{annotation_id}`",
                f"- episode_id: `{meta.episode_id}`",
                f"- trajectory_type: `{episode.get('trajectory_type', '?')}`",
                f"- trajectory_subtype: `{episode.get('trajectory_subtype', '?')}`",
                (
                    f"- metrics: forward={metrics['forward_m']:.0f}m, turns={metrics['turns']}, "
                    f"width={metrics['width_m']:.1f}m, height={metrics['height_m']:.1f}m, "
                    f"displacement={metrics['displacement_m']:.1f}m"
                ),
                f"- overview: [{strip_path.name}]({case_name}/{strip_path.name})",
                f"- metadata: [{case_name}/metadata.json]({case_name}/metadata.json)",
                "",
            ]
        )

    with open(output_dir / "index.md", "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines).rstrip() + "\n")

    print(f"Exported {len(args.annotation_ids)} map-memory examples to {output_dir}")


if __name__ == "__main__":
    main()
