#!/usr/bin/env python3
"""Apply a center square recrop to exported sat-drone pairs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from ..image_utils import center_square_recrop, load_rgb_image


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = args.dataset_dir / "pairs.csv"
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Missing {pairs_csv}")

    with pairs_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key in ("export_satellite_path", "export_drone_path"):
            src = args.dataset_dir / row[key]
            dst = args.output_dir / row[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            image = load_rgb_image(src)
            center_square_recrop(image, args.crop_size, args.output_size).save(
                dst, "JPEG", quality=args.jpeg_quality
            )

    shutil.copy2(pairs_csv, args.output_dir / "pairs.csv")

    info_path = args.dataset_dir / "dataset_info.json"
    if info_path.is_file():
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        payload = {"name": "UAV-VisLoc"}
    payload["postprocess"] = {
        "type": "center_recrop",
        "crop_size_px": args.crop_size,
        "output_size_px": args.output_size,
    }
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]
    notes.append(
        f"Pairs were postprocessed by synchronously center-cropping both images to {args.crop_size}x{args.crop_size} and resizing to {args.output_size}."
    )
    payload["notes"] = notes
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Center-recrops written to {args.output_dir}")


if __name__ == "__main__":
    main()
