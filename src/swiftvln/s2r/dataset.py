"""Manifest building and dataset loading for Stage-A sim-to-real alignment."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .split import assign_splits, build_group_id


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _join_if_relative(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    direct = root / path
    if direct.exists():
        return str(direct)
    parent_join = root.parent / path
    if parent_join.exists():
        return str(parent_join)
    return str(direct)


def _resolve_image_path(
    dataset_dir: Path,
    *,
    rel_path: Optional[str] = None,
    subdir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    if rel_path:
        return _join_if_relative(dataset_dir, rel_path)
    if subdir and filename:
        return str(dataset_dir / subdir / filename)
    raise ValueError("Cannot resolve image path without rel_path or (subdir, filename).")


@dataclass
class PairRecord:
    """One paired UAV/satellite sample."""

    dataset: str
    pair_id: str
    sample_id: str
    split: str
    group_id: str
    uav_image: str
    sat_image: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    height_m: Optional[float] = None
    north_up_rot: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _denseuav_record(dataset_dir: Path, row: Dict[str, str]) -> PairRecord:
    sample_id = row["sample_id"]
    dataset = "denseuav"
    pair_id = f"{dataset}:{sample_id}"
    altitude = row.get("altitude", "")
    height_m = _safe_float(altitude.replace("H", "")) if altitude else None
    return PairRecord(
        dataset=dataset,
        pair_id=pair_id,
        sample_id=sample_id,
        split="train",
        group_id=f"{dataset}:{build_group_id(dataset, row)}",
        uav_image=_resolve_image_path(dataset_dir, subdir="drone", filename=f"{sample_id}.jpg"),
        sat_image=_resolve_image_path(dataset_dir, subdir="satellite", filename=f"{sample_id}.jpg"),
        lat=_safe_float(row.get("lat")),
        lon=_safe_float(row.get("lon")),
        height_m=height_m,
        north_up_rot=_safe_float(row.get("north_up_rot")),
        meta={"source_row_split": row.get("split", ""), "altitude": altitude},
    )


def _gta_record(dataset_dir: Path, row: Dict[str, str]) -> PairRecord:
    sample_id = row["sample_id"]
    dataset = "gta"
    pair_id = f"{dataset}:{sample_id}"
    return PairRecord(
        dataset=dataset,
        pair_id=pair_id,
        sample_id=sample_id,
        split="train",
        group_id=f"{dataset}:{build_group_id(dataset, row)}",
        uav_image=_resolve_image_path(dataset_dir, rel_path=row.get("export_drone_path")),
        sat_image=_resolve_image_path(dataset_dir, rel_path=row.get("export_satellite_path")),
        height_m=_safe_float(row.get("height")),
        north_up_rot=_safe_float(row.get("north_up_rot")),
        meta={
            "area_mode": row.get("area_mode", ""),
            "source_area_modes": row.get("_source_area_modes", []),
            "source_pair_ids": row.get("_source_pair_ids", []),
            "source_row_splits": row.get("_source_row_splits", []),
            "satellite_img_name": row.get("satellite_img_name", ""),
            "cam_yaw": _safe_float(row.get("cam_yaw")),
            "iou": _safe_float(row.get("iou")),
            "source_row_split": row.get("split", ""),
        },
    )


def _sues_record(dataset_dir: Path, row: Dict[str, str]) -> PairRecord:
    sample_id = row["sample_id"]
    dataset = "sues"
    pair_id = f"{dataset}:{sample_id}"
    drone_file = row.get("drone_file") or f"{sample_id}.jpg"
    sat_file = row.get("satellite_file") or f"{sample_id}.jpg"
    return PairRecord(
        dataset=dataset,
        pair_id=pair_id,
        sample_id=sample_id,
        split="train",
        group_id=f"{dataset}:{build_group_id(dataset, row)}",
        uav_image=_resolve_image_path(dataset_dir, subdir="drone", filename=drone_file),
        sat_image=_resolve_image_path(dataset_dir, subdir="satellite", filename=sat_file),
        height_m=_safe_float(row.get("height")),
        north_up_rot=_safe_float(row.get("drone_rotation_ccw_deg")),
        meta={
            "scene_id": row.get("scene_id", ""),
            "variant": row.get("variant", ""),
            "match_score": _safe_float(row.get("match_score")),
            "nadir_conf": _safe_float(row.get("nadir_conf")),
        },
    )


def _uavvisloc_record(dataset_dir: Path, row: Dict[str, str]) -> PairRecord:
    sample_id = row["sample_id"]
    dataset = "uavvisloc"
    pair_id = f"{dataset}:{sample_id}"
    return PairRecord(
        dataset=dataset,
        pair_id=pair_id,
        sample_id=sample_id,
        split="train",
        group_id=f"{dataset}:{build_group_id(dataset, row)}",
        uav_image=_resolve_image_path(dataset_dir, rel_path=row.get("export_drone_path")),
        sat_image=_resolve_image_path(dataset_dir, rel_path=row.get("export_satellite_path")),
        lat=_safe_float(row.get("lat")),
        lon=_safe_float(row.get("lon")),
        height_m=_safe_float(row.get("height_m")),
        north_up_rot=_safe_float(row.get("north_up_rot")),
        meta={
            "seq_id": row.get("seq_id", ""),
            "heading_deg": _safe_float(row.get("heading_deg")),
            "heading_source": row.get("heading_source", ""),
        },
    )


SOURCE_BUILDERS = {
    "denseuav": _denseuav_record,
    "gta": _gta_record,
    "sues": _sues_record,
    "uavvisloc": _uavvisloc_record,
}


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _gta_pair_identity(row: Dict[str, str]) -> tuple[str, str]:
    """Return the protocol-independent identity of one GTA image pair."""
    drone_identity = (
        row.get("drone_img_name")
        or row.get("drone_img_path")
        or row.get("source_drone_path")
        or row.get("export_drone_path")
        or row.get("sample_id")
        or "unknown"
    )
    satellite_identity = (
        row.get("satellite_img_name")
        or row.get("satellite_img_path")
        or row.get("source_satellite_path")
        or row.get("export_satellite_path")
        or "unknown"
    )
    return str(drone_identity), str(satellite_identity)


def deduplicate_gta_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Collapse duplicated same-area/cross-area exports into physical pairs.

    The released GTA ``pairs.csv`` contains two byte-identical exports for
    every physical UAV/satellite pair.  We retain one deterministic row
    (preferring ``same_area`` only as a stable path choice) and preserve all
    source protocol metadata for auditability.
    """
    by_identity: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        by_identity.setdefault(_gta_pair_identity(row), []).append(row)

    canonical_rows: List[Dict[str, Any]] = []
    for identity in sorted(by_identity):
        duplicates = by_identity[identity]
        canonical = min(
            duplicates,
            key=lambda item: (
                0 if str(item.get("area_mode", "")).strip() == "same_area" else 1,
                str(item.get("sample_id", "")),
            ),
        ).copy()

        canonical_sample_id = Path(identity[0]).stem
        if canonical_sample_id and canonical_sample_id != "unknown":
            canonical["sample_id"] = canonical_sample_id
        canonical["_source_area_modes"] = sorted({
            str(item.get("area_mode", "")).strip()
            for item in duplicates
            if str(item.get("area_mode", "")).strip()
        })
        canonical["_source_pair_ids"] = sorted({
            str(item.get("sample_id", "")).strip()
            for item in duplicates
            if str(item.get("sample_id", "")).strip()
        })
        canonical["_source_row_splits"] = sorted({
            str(item.get("split", "")).strip()
            for item in duplicates
            if str(item.get("split", "")).strip()
        })
        canonical_rows.append(canonical)
    return canonical_rows


def build_manifest_records(
    data_root: str,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
    skip_missing: bool = True,
) -> List[Dict[str, Any]]:
    """Build manifest records from the Stage-A SatDronePair root."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"SatDronePair root not found: {data_root}")

    records: List[Dict[str, Any]] = []
    for dataset in ("denseuav", "gta", "sues", "uavvisloc"):
        dataset_dir = root / dataset
        csv_path = dataset_dir / "pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"pairs.csv missing for {dataset}: {csv_path}")

        builder = SOURCE_BUILDERS[dataset]
        rows: Sequence[Dict[str, str]] = _load_rows(csv_path)
        if dataset == "gta":
            rows = deduplicate_gta_rows(rows)
        for row in rows:
            record = builder(dataset_dir, row).to_dict()
            if skip_missing:
                if not os.path.exists(record["uav_image"]) or not os.path.exists(record["sat_image"]):
                    continue
            else:
                if not os.path.exists(record["uav_image"]):
                    raise FileNotFoundError(f"Missing UAV image: {record['uav_image']}")
                if not os.path.exists(record["sat_image"]):
                    raise FileNotFoundError(f"Missing satellite image: {record['sat_image']}")
            records.append(record)

    return assign_splits(records, val_ratio=val_ratio, seed=seed)


def write_manifest(records: Sequence[Dict[str, Any]], output_path: str) -> None:
    """Write manifest records to jsonl."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_manifest(manifest_path: str, split: Optional[str] = None) -> List[PairRecord]:
    """Load manifest jsonl into PairRecord objects."""
    records: List[PairRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if split and item.get("split") != split:
                continue
            records.append(PairRecord(**item))
    return records


def limit_records_across_datasets(records: Sequence[PairRecord], max_samples: int) -> List[PairRecord]:
    """Limit records with deterministic cross-dataset round-robin sampling."""
    if max_samples <= 0 or len(records) <= max_samples:
        return list(records)

    by_dataset: Dict[str, List[PairRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)

    dataset_names = list(by_dataset.keys())
    offsets = {dataset: 0 for dataset in dataset_names}
    active = list(dataset_names)
    selected: List[PairRecord] = []

    while active and len(selected) < max_samples:
        next_active: List[str] = []
        for dataset in active:
            bucket = by_dataset[dataset]
            offset = offsets[dataset]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[dataset] = offset + 1
            if offsets[dataset] < len(bucket):
                next_active.append(dataset)
            if len(selected) >= max_samples:
                break
        active = next_active

    return selected


class SatDronePairDataset(Dataset):
    """Simple paired-image dataset for Stage-A alignment."""

    def __init__(
        self,
        manifest_path: str,
        *,
        split: str,
        max_samples: int = 0,
    ) -> None:
        self.records = load_manifest(manifest_path, split=split)
        if max_samples and max_samples > 0:
            self.records = limit_records_across_datasets(self.records, max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        with Image.open(record.uav_image) as uav_handle:
            uav_image = uav_handle.convert("RGB").copy()
        with Image.open(record.sat_image) as sat_handle:
            sat_image = sat_handle.convert("RGB").copy()

        return {
            "pair_id": record.pair_id,
            "dataset": record.dataset,
            "uav_image": uav_image,
            "sat_image": sat_image,
            "record": record,
        }


def collate_pair_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of paired images."""
    return {
        "pair_id": [item["pair_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "uav_image": [item["uav_image"] for item in batch],
        "sat_image": [item["sat_image"] for item in batch],
        "record": [item["record"] for item in batch],
    }
