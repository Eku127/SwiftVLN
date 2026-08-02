"""Merge multiple exported view variants into one Stage-A dataset directory."""

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


@dataclass(frozen=True)
class Variant:
    name: str
    dataset_dir: Path


def parse_variants(values: Sequence[str]) -> List[Variant]:
    """Parse ``NAME=DATASET_DIR`` specifications with duplicate checks."""
    variants: List[Variant] = []
    seen = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(
                f"Invalid --variant {value!r}; expected NAME=DATASET_DIR"
            )
        if name in seen:
            raise ValueError(f"Duplicate variant name: {name}")
        seen.add(name)
        variants.append(Variant(name=name, dataset_dir=Path(raw_path).resolve()))
    if not variants:
        raise ValueError("At least one --variant is required")
    return variants


def _load_rows(variant: Variant) -> Tuple[List[Dict[str, str]], List[str]]:
    csv_path = variant.dataset_dir / "pairs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing pairs.csv for variant {variant.name}: {csv_path}"
        )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"pairs.csv is empty for variant {variant.name}")
    return rows, fieldnames


def _detect_schema(fieldnames: Sequence[str]) -> Tuple[str, str, str]:
    fields = set(fieldnames)
    if {"export_satellite_path", "export_drone_path"}.issubset(fields):
        return "uavvisloc", "export_satellite_path", "export_drone_path"
    if {"satellite_file", "drone_file"}.issubset(fields):
        return "sues", "satellite_file", "drone_file"
    raise ValueError(
        "Unsupported pairs.csv schema: expected either export_satellite_path/"
        "export_drone_path or satellite_file/drone_file"
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
    satellite_key: str,
    drone_key: str,
) -> Dict[str, str]:
    result: Dict[str, str] = {
        "sample_id": sample_id,
        "source_sample_id": row.get("sample_id", ""),
        "variant": variant_name,
    }
    for key, value in row.items():
        if key not in {"sample_id", satellite_key, drone_key}:
            result[key] = value
    result[f"source_{satellite_key}"] = row.get(satellite_key, "")
    result[f"source_{drone_key}"] = row.get(drone_key, "")

    satellite_path = f"satellite/{sample_id}.jpg"
    drone_path = f"drone/{sample_id}.jpg"
    if satellite_key == "satellite_file":
        satellite_path = f"{sample_id}.jpg"
        drone_path = f"{sample_id}.jpg"
    result[satellite_key] = satellite_path
    result[drone_key] = drone_path
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
            "All variants must use the same pairs.csv schema."
        )
    )
    parser.add_argument(
        "--variant",
        action="extend",
        nargs="+",
        required=True,
        metavar="NAME=DATASET_DIR",
        help="Variant specification; pass one or more values.",
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


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.id_start < 0:
        raise ValueError("--id-start must be non-negative")
    if args.id_width <= 0:
        raise ValueError("--id-width must be positive")

    variants = parse_variants(args.variant)
    output_dir = args.output_dir.resolve()
    if any(output_dir == variant.dataset_dir for variant in variants):
        raise ValueError("--output-dir must differ from every variant dataset directory")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_rows: List[Dict[str, str]] = []
    variant_counts: Counter[str] = Counter()
    source_metadata: Dict[str, object] = {}
    schema_name = ""
    satellite_key = ""
    drone_key = ""
    next_id = args.id_start

    for variant in variants:
        rows, fieldnames = _load_rows(variant)
        detected_schema, detected_satellite_key, detected_drone_key = _detect_schema(
            fieldnames
        )
        if schema_name and detected_schema != schema_name:
            raise ValueError(
                "All variants must use the same pairs.csv schema; got "
                f"{schema_name} and {detected_schema}"
            )
        schema_name = detected_schema
        satellite_key = detected_satellite_key
        drone_key = detected_drone_key

        info_path = variant.dataset_dir / "dataset_info.json"
        if info_path.is_file():
            source_metadata[variant.name] = _portable_metadata(
                json.loads(info_path.read_text(encoding="utf-8"))
            )

        for row in rows:
            source_satellite = row.get(satellite_key, "").strip()
            source_drone = row.get(drone_key, "").strip()
            if not source_satellite or not source_drone:
                raise ValueError(
                    f"Variant {variant.name} has an empty image path in pairs.csv"
                )

            sample_id = f"{next_id:0{args.id_width}d}"
            next_id += 1
            satellite_path = variant.dataset_dir / source_satellite
            drone_path = variant.dataset_dir / source_drone
            if satellite_key == "satellite_file":
                satellite_path = variant.dataset_dir / "satellite" / source_satellite
                drone_path = variant.dataset_dir / "drone" / source_drone
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
                    satellite_key=satellite_key,
                    drone_key=drone_key,
                )
            )
            variant_counts[variant.name] += 1

    pairs_csv = output_dir / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ordered_fields(merged_rows))
        writer.writeheader()
        writer.writerows(merged_rows)

    dataset_label = "UAV-VisLoc" if schema_name == "uavvisloc" else "SUES"
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
