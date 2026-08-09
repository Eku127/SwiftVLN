"""Merge exported view variants into one Stage-A dataset directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .center_recrop_pairs import PairSchema, detect_schema


DEFAULT_VARIANT_MAP = (
    "orig:satellite:drone,"
    "crop384:satellite_crop384:drone_crop384,"
    "crop256:satellite_crop256:drone_crop256"
)


@dataclass(frozen=True)
class Variant:
    name: str
    dataset_dir: Path
    satellite_root: Path | None = None
    drone_root: Path | None = None
    skip_missing: bool = False


def _check_unique_name(name: str, seen: set[str]) -> None:
    if not name:
        raise ValueError("Variant name cannot be empty")
    if name in seen:
        raise ValueError(f"Duplicate variant name: {name}")
    seen.add(name)


def parse_variants(values: Sequence[str]) -> List[Variant]:
    """Parse the existing multi-root ``NAME=DATASET_DIR`` form."""
    variants: List[Variant] = []
    seen: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not raw_path:
            raise ValueError(
                f"Invalid --variant {value!r}; expected NAME=DATASET_DIR"
            )
        _check_unique_name(name, seen)
        variants.append(Variant(name=name, dataset_dir=Path(raw_path).resolve()))
    if not variants:
        raise ValueError("At least one --variant is required")
    return variants


def _relative_directory(value: str) -> Path:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Variant image directory must be relative: {value!r}")
    return path


def parse_variant_map(dataset_dir: Path, text: str) -> List[Variant]:
    """Parse one-root ``NAME:SATELLITE_DIR:DRONE_DIR`` specifications."""
    dataset_dir = dataset_dir.resolve()
    variants: List[Variant] = []
    seen: set[str] = set()
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --variant-map item {item!r}; expected NAME:SAT_DIR:DRONE_DIR"
            )
        name, satellite_dir, drone_dir = parts
        _check_unique_name(name, seen)
        variants.append(
            Variant(
                name=name,
                dataset_dir=dataset_dir,
                satellite_root=dataset_dir / _relative_directory(satellite_dir),
                drone_root=dataset_dir / _relative_directory(drone_dir),
                skip_missing=True,
            )
        )
    if not variants:
        raise ValueError("No variants parsed from --variant-map")
    return variants


def _load_rows(dataset_dir: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    csv_path = dataset_dir / "pairs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing pairs.csv: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"pairs.csv is empty: {csv_path}")
    return rows, fieldnames


def _source_paths(
    variant: Variant,
    schema: PairSchema,
    row: Dict[str, str],
) -> tuple[Path, Path]:
    satellite_relative, drone_relative = schema.image_paths(row)
    if variant.satellite_root is not None and variant.drone_root is not None:
        if schema.name != "sues":
            raise ValueError("--dataset-dir/--variant-map currently requires SUES schema")
        satellite_name = Path(row[schema.satellite_key])
        drone_name = Path(row[schema.drone_key])
        return (
            variant.satellite_root / satellite_name,
            variant.drone_root / drone_name,
        )
    return (
        variant.dataset_dir / satellite_relative,
        variant.dataset_dir / drone_relative,
    )


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing variant image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _output_row(
    row: Dict[str, str],
    *,
    sample_id: str,
    variant_name: str,
    schema: PairSchema,
) -> Dict[str, str]:
    result: Dict[str, str] = {
        "sample_id": sample_id,
        "source_sample_id": row.get("sample_id", ""),
        "variant": variant_name,
    }
    for key, value in row.items():
        if key not in {"sample_id", schema.satellite_key, schema.drone_key}:
            result[key] = value
    result[f"source_{schema.satellite_key}"] = row.get(schema.satellite_key, "")
    result[f"source_{schema.drone_key}"] = row.get(schema.drone_key, "")
    if schema.name == "sues":
        result[schema.satellite_key] = f"{sample_id}.jpg"
        result[schema.drone_key] = f"{sample_id}.jpg"
    else:
        result[schema.satellite_key] = f"satellite/{sample_id}.jpg"
        result[schema.drone_key] = f"drone/{sample_id}.jpg"
    return result


def _ordered_fields(rows: Sequence[Dict[str, str]]) -> List[str]:
    preferred = ["sample_id", "source_sample_id", "variant"]
    fields: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def _portable_metadata(value):
    """Drop machine-local absolute paths from copied dataset metadata."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(item, str) and Path(item).is_absolute():
                continue
            result[key] = _portable_metadata(item)
        return result
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten exported orig/crop variants into one Stage-A dataset. "
            "Use --variant for multiple dataset roots, or --dataset-dir with "
            "--variant-map for multiple image directories under one SUES root."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--variant",
        action="extend",
        nargs="+",
        metavar="NAME=DATASET_DIR",
        help="One or more independently exported variant datasets.",
    )
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="One SUES export containing several image variant directories.",
    )
    parser.add_argument(
        "--variant-map",
        default=None,
        help="Comma list of NAME:SAT_DIR:DRONE_DIR for one-root mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-start", type=int, default=1)
    parser.add_argument("--id-width", type=int, default=6)
    parser.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="Prefer hard links (with copy fallback), or always copy.",
    )
    return parser.parse_args(argv)


def _resolve_variants(args: argparse.Namespace) -> List[Variant]:
    if args.dataset_dir is not None:
        return parse_variant_map(
            args.dataset_dir,
            args.variant_map or DEFAULT_VARIANT_MAP,
        )
    if args.variant_map is not None:
        raise ValueError("--variant-map requires --dataset-dir")
    return parse_variants(args.variant)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.id_start < 0:
        raise ValueError("--id-start must be non-negative")
    if args.id_width <= 0:
        raise ValueError("--id-width must be positive")

    variants = _resolve_variants(args)
    output_dir = args.output_dir.resolve()
    if any(output_dir == variant.dataset_dir for variant in variants):
        raise ValueError("--output-dir must differ from every source dataset directory")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_rows: List[Dict[str, str]] = []
    variant_counts: Counter[str] = Counter({variant.name: 0 for variant in variants})
    source_metadata: Dict[str, object] = {}
    row_cache: Dict[Path, Tuple[List[Dict[str, str]], List[str]]] = {}
    schema_name = ""
    schema: PairSchema | None = None
    next_id = args.id_start

    for variant in variants:
        if (
            variant.satellite_root is not None
            and variant.drone_root is not None
            and (
                not variant.satellite_root.is_dir()
                or not variant.drone_root.is_dir()
            )
        ):
            print(
                f"[skip variant] {variant.name}: missing "
                f"{variant.satellite_root} or {variant.drone_root}"
            )
            continue

        if variant.dataset_dir not in row_cache:
            row_cache[variant.dataset_dir] = _load_rows(variant.dataset_dir)
        rows, fieldnames = row_cache[variant.dataset_dir]
        detected_schema = detect_schema(fieldnames)
        if schema_name and detected_schema.name != schema_name:
            raise ValueError(
                "All variants must use the same pairs.csv schema; got "
                f"{schema_name} and {detected_schema.name}"
            )
        schema_name = detected_schema.name
        schema = detected_schema

        info_path = variant.dataset_dir / "dataset_info.json"
        if info_path.is_file():
            source_metadata[variant.name] = _portable_metadata(
                json.loads(info_path.read_text(encoding="utf-8"))
            )

        for row in rows:
            try:
                satellite_path, drone_path = _source_paths(
                    variant,
                    detected_schema,
                    row,
                )
                if not satellite_path.is_file() or not drone_path.is_file():
                    raise FileNotFoundError(
                        f"Missing variant images: {satellite_path}, {drone_path}"
                    )
            except (ValueError, FileNotFoundError):
                if variant.skip_missing:
                    continue
                raise

            sample_id = f"{next_id:0{args.id_width}d}"
            next_id += 1
            _materialize(
                satellite_path,
                output_dir / "satellite" / f"{sample_id}.jpg",
                args.mode,
            )
            _materialize(
                drone_path,
                output_dir / "drone" / f"{sample_id}.jpg",
                args.mode,
            )
            merged_rows.append(
                _output_row(
                    row,
                    sample_id=sample_id,
                    variant_name=variant.name,
                    schema=detected_schema,
                )
            )
            variant_counts[variant.name] += 1

    if not merged_rows or schema is None:
        raise RuntimeError("No merged rows generated")

    with (output_dir / "pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=_ordered_fields(merged_rows))
        writer.writeheader()
        writer.writerows(merged_rows)

    dataset_label = "UAV-VisLoc" if schema.name == "uavvisloc" else "SUES"
    info = {
        "name": f"{dataset_label} merged variants (Stage-A layout)",
        "source_variants": [variant.name for variant in variants],
        "merged_total_pairs": len(merged_rows),
        "id_start": args.id_start,
        "id_width": args.id_width,
        "variants": dict(variant_counts),
        "output_structure": {
            "satellite_dir": "satellite",
            "drone_dir": "drone",
            "pairs_csv": "pairs.csv",
        },
        "source_dataset_info": source_metadata,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Merged pairs: {len(merged_rows)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
