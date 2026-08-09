"""Schema-aware preview renderer for all SatDronePair source datasets."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .image_utils import load_rgb_image


PAD = 20
GAP = 8
TITLE_H = 20
GROUP_GAP = 24
BACKGROUND = (20, 20, 24)
LABEL_BACKGROUND = (38, 38, 44)
SATELLITE_BORDER = (80, 140, 200)
DRONE_BORDER = (80, 180, 100)


@dataclass(frozen=True)
class PreviewItem:
    satellite_path: Path
    drone_path: Path
    label: str
    rotation_ccw_deg: float | None = None


@dataclass(frozen=True)
class PreviewGroup:
    name: str
    color: tuple[int, int, int]
    items: Sequence[PreviewItem]


@dataclass(frozen=True)
class PreviewPlan:
    title: str
    panel: int
    label_height: int
    row_gap: int
    groups: Sequence[PreviewGroup]
    show_group_headers: bool = True
    satellite_label: str = "Satellite north-up"
    drone_label: str = "Drone north-up"


def _required(row: Mapping[str, str], key: str, dataset: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{dataset} pairs.csv row is missing required field {key!r}")
    return value


def _read_rows(dataset_dir: Path) -> List[Dict[str, str]]:
    csv_path = dataset_dir / "pairs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{csv_path} has no rows")
    return rows


def _sample(
    rows: Sequence[Mapping[str, str]],
    count: int,
    rng: random.Random,
) -> List[Mapping[str, str]]:
    if count <= 0:
        raise ValueError("--rows must be positive")
    return rng.sample(list(rows), min(count, len(rows)))


def _rotate_north_up(image: Image.Image, degrees: float) -> Image.Image:
    """Mirror-pad an image before rotation, then restore its original extent."""
    width, height = image.size
    horizontal = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    vertical = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    both = horizontal.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    padded = Image.new(image.mode, (width * 3, height * 3))
    tiles = (
        (both, vertical, both),
        (horizontal, image, horizontal),
        (both, vertical, both),
    )
    for row_index, row in enumerate(tiles):
        for column_index, tile in enumerate(row):
            padded.paste(tile, (column_index * width, row_index * height))
    rotated = padded.rotate(degrees, resample=Image.Resampling.BICUBIC)
    center_x = rotated.width // 2
    center_y = rotated.height // 2
    return rotated.crop(
        (
            center_x - width // 2,
            center_y - height // 2,
            center_x - width // 2 + width,
            center_y - height // 2 + height,
        )
    )


def _denseuav_plan(
    dataset_dir: Path,
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> PreviewPlan:
    colors = {
        "H80": (70, 130, 200),
        "H90": (60, 180, 100),
        "H100": (200, 110, 50),
    }
    gsd = {"H80": 18.5, "H90": 20.8, "H100": 23.1}
    groups: List[PreviewGroup] = []
    for altitude in colors:
        candidates = [row for row in rows if row.get("altitude") == altitude]
        if not candidates:
            continue
        items = []
        for row in _sample(candidates, args.rows, rng):
            sample_id = _required(row, "sample_id", "denseuav")
            split = _required(row, "split", "denseuav")
            rotation = float(_required(row, "north_up_rot", "denseuav"))
            original_id = sample_id.replace(f"_{altitude}", "")
            items.append(
                PreviewItem(
                    satellite_path=dataset_dir / "satellite" / f"{sample_id}.jpg",
                    drone_path=dataset_dir / "drone" / f"{sample_id}.jpg",
                    label=f"{original_id} [{split}] rot={rotation:g}deg",
                    rotation_ccw_deg=rotation,
                )
            )
        meters = altitude.removeprefix("H")
        groups.append(
            PreviewGroup(
                name=(
                    f"{altitude} (altitude {meters}m | "
                    f"drone GSD ~ {gsd[altitude]}cm/px)"
                ),
                color=colors[altitude],
                items=items,
            )
        )
    if not groups:
        raise RuntimeError("pairs.csv contains no supported DenseUAV altitudes")
    return PreviewPlan(
        title="DenseUAV - Satellite (north-up) | Drone (north-up)",
        panel=args.panel or 256,
        label_height=38,
        row_gap=6,
        groups=groups,
    )


def _gta_uav_plan(
    dataset_dir: Path,
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> PreviewPlan:
    colors = {
        "confirmed": (60, 160, 100),
        "corrected": (200, 120, 50),
        "ambiguous": (180, 80, 80),
        "unchecked": (90, 120, 200),
    }
    groups: List[PreviewGroup] = []
    for status in args.groups or ["confirmed"]:
        candidates = [row for row in rows if row.get("heading_status") == status]
        if not candidates:
            continue
        items = []
        for row in _sample(candidates, args.rows, rng):
            sample_id = _required(row, "sample_id", "gta_uav")
            split = _required(row, "split", "gta_uav")
            area = row.get("source_area_modes") or _required(
                row, "area_mode", "gta_uav"
            )
            rotation = _required(row, "north_up_rot", "gta_uav")
            iou = _required(row, "iou", "gta_uav")
            items.append(
                PreviewItem(
                    satellite_path=dataset_dir
                    / _required(row, "export_satellite_path", "gta_uav"),
                    drone_path=dataset_dir
                    / _required(row, "export_drone_path", "gta_uav"),
                    label=(
                        f"{sample_id} [{split} | {area}] "
                        f"iou={iou} rot={rotation}deg"
                    ),
                )
            )
        groups.append(
            PreviewGroup(
                name=f"{status} ({len(items)} samples)",
                color=colors.get(status, (80, 80, 120)),
                items=items,
            )
        )
    if not groups:
        raise RuntimeError("No matching rows found for the requested preview groups")
    return PreviewPlan(
        title="GTA-UAV - Satellite (north-up) | Drone (north-up)",
        panel=args.panel or 256,
        label_height=38,
        row_gap=6,
        groups=groups,
    )


def _sues_plan(
    dataset_dir: Path,
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> PreviewPlan:
    filtered = list(rows)
    if any("nadir_conf" in row for row in filtered):
        filtered = [
            row
            for row in filtered
            if float(row.get("nadir_conf", "0") or 0.0) >= args.min_nadir_conf
        ]
    if args.heights:
        heights = {value.strip() for value in args.heights.split(",") if value.strip()}
        filtered = [row for row in filtered if row.get("height", "").strip() in heights]
    if not filtered:
        raise RuntimeError("SUES pairs.csv has no rows after filtering")

    items = []
    for row in _sample(filtered, args.rows, rng):
        sample_id = _required(row, "sample_id", "sues")
        nadir = f" | nadir={row['nadir_conf']}" if "nadir_conf" in row else ""
        items.append(
            PreviewItem(
                satellite_path=dataset_dir
                / "satellite"
                / _required(row, "satellite_file", "sues"),
                drone_path=dataset_dir
                / "drone"
                / _required(row, "drone_file", "sues"),
                label=(
                    f"{sample_id} | frame={_required(row, 'drone_frame', 'sues')} "
                    f"| rot={_required(row, 'drone_rotation_ccw_deg', 'sues')}deg "
                    f"| crop={_required(row, 'sat_crop_frac', 'sues')} "
                    f"| score={_required(row, 'match_score', 'sues')}{nadir}"
                ),
            )
        )
    return PreviewPlan(
        title="SUES Pair Preview | Left: Satellite Right: Drone",
        panel=args.panel or 320,
        label_height=42,
        row_gap=12,
        groups=[PreviewGroup(name="", color=BACKGROUND, items=items)],
        show_group_headers=False,
        satellite_label="Satellite",
        drone_label="Drone",
    )


def _uavvisloc_plan(
    dataset_dir: Path,
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> PreviewPlan:
    items = []
    for row in _sample(rows, args.rows, rng):
        sample_id = _required(row, "sample_id", "uavvisloc")
        items.append(
            PreviewItem(
                satellite_path=dataset_dir
                / _required(row, "export_satellite_path", "uavvisloc"),
                drone_path=dataset_dir
                / _required(row, "export_drone_path", "uavvisloc"),
                label=(
                    f"{sample_id} [seq={_required(row, 'seq_id', 'uavvisloc')} "
                    f"| heading={_required(row, 'heading_source', 'uavvisloc')}] "
                    f"crop={_required(row, 'crop_side_m', 'uavvisloc')}m "
                    f"rot={_required(row, 'north_up_rot', 'uavvisloc')}deg"
                ),
            )
        )
    return PreviewPlan(
        title="UAV-VisLoc - Satellite (north-up) | Drone (north-up)",
        panel=args.panel or 256,
        label_height=40,
        row_gap=8,
        groups=[PreviewGroup(name="", color=BACKGROUND, items=items)],
        show_group_headers=False,
        satellite_label="Satellite",
        drone_label="Drone",
    )


_PLAN_BUILDERS: Dict[
    str,
    Callable[
        [Path, Sequence[Mapping[str, str]], argparse.Namespace, random.Random],
        PreviewPlan,
    ],
] = {
    "denseuav": _denseuav_plan,
    "gta_uav": _gta_uav_plan,
    "sues": _sues_plan,
    "uavvisloc": _uavvisloc_plan,
}


def render_preview(plan: PreviewPlan, output: Path) -> None:
    """Render a normalized two-column grid from a dataset-specific plan."""
    if plan.panel <= 0:
        raise ValueError("--panel must be positive")
    group_header_height = plan.label_height if plan.show_group_headers else 0
    group_heights = [
        group_header_height
        + len(group.items) * (plan.label_height + plan.panel + plan.row_gap)
        for group in plan.groups
    ]
    width = PAD * 2 + plan.panel * 2 + GAP
    height = (
        PAD * 2
        + TITLE_H
        + sum(group_heights)
        + max(0, len(group_heights) - 1) * GROUP_GAP
    )
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((PAD, 6), plan.title, fill=(220, 220, 220), font=font)

    y = PAD + TITLE_H
    for group_index, group in enumerate(plan.groups):
        if plan.show_group_headers:
            draw.rectangle(
                (PAD, y, width - PAD, y + plan.label_height - 4),
                fill=group.color,
            )
            draw.text(
                (PAD + 6, y + 10),
                group.name,
                fill=(255, 255, 255),
                font=font,
            )
            y += plan.label_height

        for item in group.items:
            draw.rectangle(
                (PAD, y, width - PAD, y + plan.label_height - 4),
                fill=LABEL_BACKGROUND,
            )
            draw.text((PAD + 6, y + 10), item.label, fill=(185, 185, 185), font=font)
            draw.text((PAD + 6, y + 2), plan.satellite_label, fill=(160, 210, 255), font=font)
            draw.text(
                (PAD + plan.panel + GAP + 6, y + 2),
                plan.drone_label,
                fill=(160, 255, 160),
                font=font,
            )
            y += plan.label_height

            satellite = load_rgb_image(item.satellite_path)
            drone = load_rgb_image(item.drone_path)
            if item.rotation_ccw_deg is not None:
                satellite = _rotate_north_up(satellite, item.rotation_ccw_deg)
                drone = _rotate_north_up(drone, item.rotation_ccw_deg)
            satellite = satellite.resize(
                (plan.panel, plan.panel), Image.Resampling.LANCZOS
            )
            drone = drone.resize((plan.panel, plan.panel), Image.Resampling.LANCZOS)
            canvas.paste(satellite, (PAD, y))
            canvas.paste(drone, (PAD + plan.panel + GAP, y))
            draw.rectangle(
                (PAD, y, PAD + plan.panel - 1, y + plan.panel - 1),
                outline=SATELLITE_BORDER,
                width=1,
            )
            draw.rectangle(
                (
                    PAD + plan.panel + GAP,
                    y,
                    PAD + plan.panel * 2 + GAP - 1,
                    y + plan.panel - 1,
                ),
                outline=DRONE_BORDER,
                width=1,
            )
            y += plan.panel + plan.row_gap

        if group_index + 1 < len(plan.groups):
            y += GROUP_GAP

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=95)
    print(f"Saved preview: {output} ({canvas.width}x{canvas.height})")


def create_parser(*, fixed_dataset: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a schema-aware SatDronePair QA preview."
    )
    if fixed_dataset is None:
        parser.add_argument("--dataset", choices=sorted(_PLAN_BUILDERS), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel", type=int, default=None)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="GTA heading-status groups (default: confirmed).",
    )
    parser.add_argument(
        "--min-nadir-conf",
        type=float,
        default=0.0,
        help="SUES minimum nadir confidence.",
    )
    parser.add_argument(
        "--heights",
        default="",
        help="SUES comma-separated height filter.",
    )
    return parser


def main(argv=None, *, dataset: str | None = None) -> None:
    if dataset is not None and dataset not in _PLAN_BUILDERS:
        raise ValueError(f"Unknown preview dataset: {dataset}")
    parser = create_parser(fixed_dataset=dataset)
    args = parser.parse_args(argv)
    args.dataset = dataset or args.dataset
    default_rows = {"denseuav": 4, "gta_uav": 4, "sues": 8, "uavvisloc": 6}
    if args.rows is None:
        args.rows = default_rows[args.dataset]

    rows = _read_rows(args.dataset_dir)
    plan = _PLAN_BUILDERS[args.dataset](
        args.dataset_dir,
        rows,
        args,
        random.Random(args.seed),
    )
    output = args.output or args.dataset_dir / "samples_preview.jpg"
    render_preview(plan, output)


if __name__ == "__main__":
    main()
