"""Split utilities for Stage-A sim-to-real alignment."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Set, Tuple


def denseuav_group_id(sample_id: str) -> str:
    """Group DenseUAV rows by base position id, not altitude."""
    return sample_id.split("_", 1)[0]


def gta_group_id(row: Dict[str, str]) -> str:
    """Group GTA rows by satellite tile, independent of benchmark protocol.

    GTA-UAV exports the same physical pair once for ``same_area`` and once for
    ``cross_area``.  Including ``area_mode`` in the group id allowed identical
    images to land in different Stage-A splits.  The satellite tile is the
    shared geographic identity and therefore the correct split boundary.
    """
    sat_name = (
        row.get("satellite_img_name")
        or row.get("satellite_img_path")
        or row.get("export_satellite_path")
        or row.get("satellite_file")
        or "unknown"
    )
    return os.path.basename(sat_name)


def sues_group_id(row: Dict[str, str]) -> str:
    """Group SUES rows by scene id."""
    return str(row.get("scene_id") or row.get("source_sample_id") or row.get("sample_id") or "unknown")


def uavvisloc_group_id(row: Dict[str, str]) -> str:
    """Group UAV-VisLoc rows by sequence id."""
    return str(row.get("seq_id") or row.get("sample_id") or "unknown")


def build_group_id(dataset_name: str, row: Dict[str, str]) -> str:
    """Build a deterministic per-source group id."""
    dataset = dataset_name.lower()
    if dataset == "denseuav":
        return denseuav_group_id(str(row["sample_id"]))
    if dataset == "gta":
        return gta_group_id(row)
    if dataset == "sues":
        return sues_group_id(row)
    if dataset == "uavvisloc":
        return uavvisloc_group_id(row)
    raise ValueError(f"Unsupported dataset for group split: {dataset_name}")


def split_groups(
    group_ids: Sequence[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Split a set of group ids into train/val partitions."""
    unique_groups = sorted(set(group_ids))
    if not unique_groups:
        return set(), set()

    if len(unique_groups) == 1:
        return {unique_groups[0]}, set()

    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    num_val = int(round(len(unique_groups) * val_ratio))
    num_val = max(1, num_val)
    num_val = min(num_val, len(unique_groups) - 1)

    val_groups = set(unique_groups[:num_val])
    train_groups = set(unique_groups[num_val:])
    return train_groups, val_groups


def assign_splits(
    records: List[Dict[str, object]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Assign train/val split per source using precomputed group ids."""
    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        dataset = str(record["dataset"])
        by_dataset.setdefault(dataset, []).append(record)

    output: List[Dict[str, object]] = []
    for dataset, dataset_records in by_dataset.items():
        groups = [str(record["group_id"]) for record in dataset_records]
        train_groups, val_groups = split_groups(groups, val_ratio=val_ratio, seed=seed)
        for record in dataset_records:
            item = dict(record)
            item["split"] = "val" if str(item["group_id"]) in val_groups else "train"
            output.append(item)

    output.sort(key=lambda item: (str(item["dataset"]), str(item["pair_id"])))
    return output
