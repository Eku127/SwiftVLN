#!/usr/bin/env python3
"""Backup then filter SatNav eval ``all_episodes.json`` by keep-list CSVs.

Each row in ``<split>_keep.csv`` must contain ``scene_episode_key`` in the form
``scene_id::episode_id`` (same convention as ``generate_satnav_keep_lists.py``).

Episodes are matched using string-normalized ``(scene_id, episode_id)`` keys.
**Episode payload fields are not rewritten** (do not renumber ``episode_id`` for
"continuity"): SatNav ``episode_id`` repeats across scenes and ties into
``trajectory_data`` / image directory naming.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Set, Tuple


SCENE_EPISODE_KEY_SEP = "::"


def load_keep_keys(csv_path: Path) -> Set[str]:
    keys: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "scene_episode_key" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}: missing column scene_episode_key")
        for row in reader:
            k = (row.get("scene_episode_key") or "").strip()
            if k:
                keys.add(k)
    return keys


def episode_key(ep: dict) -> str:
    scene = str(ep.get("scene_id", "")).strip()
    eid = str(ep.get("episode_id", "")).strip()
    return f"{scene}{SCENE_EPISODE_KEY_SEP}{eid}"


def filter_episodes(episodes: List[dict], keep: Set[str]) -> Tuple[list, int, int]:
    kept: list = []
    for ep in episodes:
        k = episode_key(ep)
        if k in keep:
            kept.append(ep)
    n_src = len(episodes)
    return kept, len(kept), n_src - len(kept)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(
        prefix=".all_episodes_", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(data)
        Path(tmp).replace(path)
    finally:
        p = Path(tmp)
        if p.exists():
            p.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-version-dir",
        type=Path,
        required=True,
        help="e.g. /mnt/data3/.../satnav_datasets/ver_260404",
    )
    p.add_argument(
        "--keep-list-dir",
        type=Path,
        required=True,
        help="Directory containing val_seen_keep.csv and val_unseen_keep.csv",
    )
    p.add_argument(
        "--backup-subdir",
        type=str,
        default="backup_full_eval_before_subset",
        help="Created under episodes/eval/<backup-subdir>/",
    )
    args = p.parse_args()

    version_dir: Path = args.dataset_version_dir.resolve()
    keep_dir: Path = args.keep_list_dir.resolve()
    eval_root = version_dir / "episodes" / "eval"
    backup_root = eval_root / args.backup_subdir

    for split in ("val_seen", "val_unseen"):
        src = eval_root / split / "all_episodes.json"
        keep_csv = keep_dir / f"{split}_keep.csv"
        if not src.is_file():
            raise FileNotFoundError(src)
        if not keep_csv.is_file():
            raise FileNotFoundError(keep_csv)

        keep_keys = load_keep_keys(keep_csv)
        with src.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        episodes = doc.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{src}: expected top-level object with 'episodes' list")

        backup_dst = backup_root / split / "all_episodes.json"
        backup_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup_dst)
        print(f"[backup] {src} -> {backup_dst}")

        filtered, n_kept, n_dropped = filter_episodes(episodes, keep_keys)
        filtered_keys = {episode_key(ep) for ep in filtered}
        if filtered_keys != keep_keys:
            only_keep = sorted(keep_keys - filtered_keys)[:10]
            only_json = sorted(filtered_keys - keep_keys)[:10]
            raise ValueError(
                f"{split}: keep set mismatch vs filtered output. "
                f"only_in_keep_csv(sample)={only_keep} only_in_json(sample)={only_json}"
            )

        new_doc = dict(doc)
        new_doc["episodes"] = filtered
        atomic_write_json(src, new_doc)
        print(
            f"[trim] {split}: kept={n_kept} dropped={n_dropped} "
            f"(keep_csv_rows={len(keep_keys)}) -> {src}"
        )


if __name__ == "__main__":
    main()
