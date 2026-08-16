"""Synchronously center-recrop both images in an exported pair dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .image_utils import center_square_recrop, load_rgb_image


@dataclass(frozen=True)
class PairSchema:
    name: str
    satellite_key: str
    drone_key: str
    image_prefixes: tuple[str, str]
    fallback_dataset_name: str

    def image_paths(self, row: Dict[str, str]) -> tuple[Path, Path]:
        paths = []
        for key, prefix in zip(
            (self.satellite_key, self.drone_key),
            self.image_prefixes,
        ):
            value = row.get(key, "").strip()
            if not value:
                raise ValueError(f"pairs.csv has an empty {key!r} value")
            relative_path = Path(prefix) / value if prefix else Path(value)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"pairs.csv image path must be relative: {value!r}")
            paths.append(relative_path)
        return paths[0], paths[1]


_SCHEMAS = (
    PairSchema(
        name="sues",
        satellite_key="satellite_file",
        drone_key="drone_file",
        image_prefixes=("satellite", "drone"),
        fallback_dataset_name="sues_pair_export",
    ),
    PairSchema(
        name="uavvisloc",
        satellite_key="export_satellite_path",
        drone_key="export_drone_path",
        image_prefixes=("", ""),
        fallback_dataset_name="UAV-VisLoc",
    ),
)


def detect_schema(fieldnames: Sequence[str]) -> PairSchema:
    fields = set(fieldnames)
    for schema in _SCHEMAS:
        if {schema.satellite_key, schema.drone_key}.issubset(fields):
            return schema
    expected = " or ".join(
        f"{schema.satellite_key}/{schema.drone_key}" for schema in _SCHEMAS
    )
    raise ValueError(f"Unsupported pairs.csv schema; expected {expected}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center-recrop both views while preserving the pair schema."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args(argv)


def _load_rows(csv_path: Path) -> tuple[List[Dict[str, str]], PairSchema]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        schema = detect_schema(reader.fieldnames or [])
    return rows, schema


def _write_metadata(
    source: Path,
    destination: Path,
    schema: PairSchema,
    args: argparse.Namespace,
) -> None:
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = {"name": schema.fallback_dataset_name}
    payload["postprocess"] = {
        "type": "center_recrop",
        "crop_size_px": args.crop_size,
        "output_size_px": args.output_size,
    }
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]
    notes.append(
        "Pairs were synchronously center-cropped to "
        f"{args.crop_size}x{args.crop_size} and resized to {args.output_size}."
    )
    payload["notes"] = notes
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == dataset_dir:
        raise ValueError("--output-dir must differ from --dataset-dir")

    pairs_csv = dataset_dir / "pairs.csv"
    rows, schema = _load_rows(pairs_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for relative_path in schema.image_paths(row):
            source = dataset_dir / relative_path
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            center_square_recrop(
                load_rgb_image(source),
                args.crop_size,
                args.output_size,
            ).save(destination, "JPEG", quality=args.jpeg_quality)

    shutil.copy2(pairs_csv, output_dir / "pairs.csv")
    _write_metadata(
        dataset_dir / "dataset_info.json",
        output_dir / "dataset_info.json",
        schema,
        args,
    )
    print(f"Center-recrops written to {output_dir} (schema={schema.name})")


if __name__ == "__main__":
    main()
