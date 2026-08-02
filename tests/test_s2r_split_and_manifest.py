import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.s2r.dataset import (
    build_manifest_records,
    deduplicate_gta_rows,
    write_manifest,
)


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(123, 10, 200)).save(path)


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_denseuav(root: Path):
    rows = [
        {"sample_id": "000001_H80", "altitude": "H80", "split": "train", "lon": "1.0", "lat": "2.0", "north_up_rot": "10"},
        {"sample_id": "000001_H90", "altitude": "H90", "split": "train", "lon": "1.0", "lat": "2.0", "north_up_rot": "10"},
        {"sample_id": "000002_H80", "altitude": "H80", "split": "train", "lon": "3.0", "lat": "4.0", "north_up_rot": "20"},
        {"sample_id": "000003_H80", "altitude": "H80", "split": "test", "lon": "5.0", "lat": "6.0", "north_up_rot": "30"},
    ]
    for row in rows:
        _write_image(root / "denseuav" / "drone" / f"{row['sample_id']}.jpg")
        _write_image(root / "denseuav" / "satellite" / f"{row['sample_id']}.jpg")
    _write_csv(root / "denseuav" / "pairs.csv", rows[0].keys(), rows)


def _prepare_gta(root: Path):
    rows = [
        {
            "sample_id": "g1",
            "split": "train",
            "area_mode": "same_area",
            "drone_img_name": "g1.png",
            "satellite_img_name": "tile_a.png",
            "export_drone_path": "drone/g1.jpg",
            "export_satellite_path": "satellite/g1.jpg",
            "height": "100",
            "north_up_rot": "15",
            "cam_yaw": "90",
            "iou": "0.5",
        },
        {
            "sample_id": "g2",
            "split": "test",
            "area_mode": "same_area",
            "drone_img_name": "g2.png",
            "satellite_img_name": "tile_a.png",
            "export_drone_path": "drone/g2.jpg",
            "export_satellite_path": "satellite/g2.jpg",
            "height": "110",
            "north_up_rot": "25",
            "cam_yaw": "80",
            "iou": "0.4",
        },
        {
            "sample_id": "g3",
            "split": "train",
            "area_mode": "cross_area",
            "drone_img_name": "g3.png",
            "satellite_img_name": "tile_b.png",
            "export_drone_path": "drone/g3.jpg",
            "export_satellite_path": "satellite/g3.jpg",
            "height": "120",
            "north_up_rot": "35",
            "cam_yaw": "70",
            "iou": "0.6",
        },
    ]
    protocol_duplicates = []
    for row in rows:
        duplicate = dict(row)
        duplicate["sample_id"] = f"{row['sample_id']}_other_protocol"
        duplicate["split"] = "test" if row["split"] == "train" else "train"
        duplicate["area_mode"] = (
            "cross_area" if row["area_mode"] == "same_area" else "same_area"
        )
        duplicate["export_drone_path"] = f"drone/{duplicate['sample_id']}.jpg"
        duplicate["export_satellite_path"] = f"satellite/{duplicate['sample_id']}.jpg"
        protocol_duplicates.append(duplicate)
    rows.extend(protocol_duplicates)

    for row in rows:
        _write_image(root / "gta" / row["export_drone_path"])
        _write_image(root / "gta" / row["export_satellite_path"])
    _write_csv(root / "gta" / "pairs.csv", rows[0].keys(), rows)


def _prepare_sues(root: Path):
    rows = [
        {
            "sample_id": "s1",
            "scene_id": "scene_a",
            "height": "150",
            "drone_rotation_ccw_deg": "5",
            "match_score": "0.2",
            "nadir_conf": "0.7",
            "variant": "orig",
            "satellite_file": "s1.jpg",
            "drone_file": "s1.jpg",
        },
        {
            "sample_id": "s2",
            "scene_id": "scene_a",
            "height": "160",
            "drone_rotation_ccw_deg": "6",
            "match_score": "0.3",
            "nadir_conf": "0.8",
            "variant": "orig",
            "satellite_file": "s2.jpg",
            "drone_file": "s2.jpg",
        },
        {
            "sample_id": "s3",
            "scene_id": "scene_b",
            "height": "170",
            "drone_rotation_ccw_deg": "7",
            "match_score": "0.4",
            "nadir_conf": "0.9",
            "variant": "orig",
            "satellite_file": "s3.jpg",
            "drone_file": "s3.jpg",
        },
    ]
    for row in rows:
        _write_image(root / "sues" / "drone" / row["drone_file"])
        _write_image(root / "sues" / "satellite" / row["satellite_file"])
    _write_csv(root / "sues" / "pairs.csv", rows[0].keys(), rows)


def _prepare_uavvisloc(root: Path):
    rows = [
        {
            "sample_id": "u1",
            "seq_id": "seq_a",
            "lat": "10",
            "lon": "20",
            "height_m": "300",
            "north_up_rot": "40",
            "heading_deg": "120",
            "heading_source": "imu",
            "export_satellite_path": "satellite/u1.jpg",
            "export_drone_path": "drone/u1.jpg",
        },
        {
            "sample_id": "u2",
            "seq_id": "seq_a",
            "lat": "11",
            "lon": "21",
            "height_m": "310",
            "north_up_rot": "41",
            "heading_deg": "121",
            "heading_source": "imu",
            "export_satellite_path": "satellite/u2.jpg",
            "export_drone_path": "drone/u2.jpg",
        },
        {
            "sample_id": "u3",
            "seq_id": "seq_b",
            "lat": "12",
            "lon": "22",
            "height_m": "320",
            "north_up_rot": "42",
            "heading_deg": "122",
            "heading_source": "imu",
            "export_satellite_path": "satellite/u3.jpg",
            "export_drone_path": "drone/u3.jpg",
        },
    ]
    for row in rows:
        _write_image(root / "uavvisloc" / row["export_drone_path"])
        _write_image(root / "uavvisloc" / row["export_satellite_path"])
    _write_csv(root / "uavvisloc" / "pairs.csv", rows[0].keys(), rows)


class S2RSplitManifestTest(unittest.TestCase):
    def test_build_manifest_records_grouped_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prepare_denseuav(root)
            _prepare_gta(root)
            _prepare_sues(root)
            _prepare_uavvisloc(root)

            records = build_manifest_records(str(root), val_ratio=0.34, seed=7, skip_missing=False)
            self.assertEqual(len(records), 13)
            by_pair = {record["pair_id"]: record for record in records}

            self.assertEqual(by_pair["denseuav:000001_H80"]["split"], by_pair["denseuav:000001_H90"]["split"])
            self.assertEqual(by_pair["gta:g1"]["split"], by_pair["gta:g2"]["split"])
            self.assertEqual(by_pair["sues:s1"]["split"], by_pair["sues:s2"]["split"])
            self.assertEqual(by_pair["uavvisloc:u1"]["split"], by_pair["uavvisloc:u2"]["split"])
            self.assertEqual({record["split"] for record in records}, {"train", "val"})

            gta_records = [record for record in records if record["dataset"] == "gta"]
            self.assertEqual(len(gta_records), 3)
            self.assertEqual(
                {tuple(record["meta"]["source_area_modes"]) for record in gta_records},
                {("cross_area", "same_area")},
            )
            self.assertTrue(all(len(record["meta"]["source_pair_ids"]) == 2 for record in gta_records))

    def test_compact_gta_csv_preserves_protocol_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prepare_denseuav(root)
            _prepare_gta(root)
            _prepare_sues(root)
            _prepare_uavvisloc(root)

            csv_path = root / "gta/pairs.csv"
            with csv_path.open(encoding="utf-8") as handle:
                legacy_rows = list(csv.DictReader(handle))
            compact_rows = deduplicate_gta_rows(legacy_rows)
            for row in compact_rows:
                for public, internal in (
                    ("source_area_modes", "_source_area_modes"),
                    ("source_pair_ids", "_source_pair_ids"),
                    ("source_row_splits", "_source_row_splits"),
                    ("source_meta_files", "_source_meta_files"),
                ):
                    row[public] = "|".join(row.pop(internal))
            _write_csv(csv_path, compact_rows[0].keys(), compact_rows)

            records = build_manifest_records(
                str(root), val_ratio=0.34, seed=7, skip_missing=False
            )
            gta_records = [record for record in records if record["dataset"] == "gta"]
            self.assertEqual(len(gta_records), 3)
            self.assertTrue(
                all(
                    record["meta"]["source_area_modes"]
                    == ["cross_area", "same_area"]
                    for record in gta_records
                )
            )
            self.assertTrue(
                all(len(record["meta"]["source_pair_ids"]) == 2 for record in gta_records)
            )

    def test_write_manifest_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "manifest.jsonl"
            records = [
                {
                    "dataset": "denseuav",
                    "pair_id": "denseuav:sample",
                    "sample_id": "sample",
                    "split": "train",
                    "group_id": "denseuav:sample",
                    "uav_image": "/tmp/uav.jpg",
                    "sat_image": "/tmp/sat.jpg",
                    "lat": 1.0,
                    "lon": 2.0,
                    "height_m": 80.0,
                    "north_up_rot": 10.0,
                    "meta": {"source_row_split": "train"},
                }
            ]
            write_manifest(records, str(output_path))
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            item = json.loads(lines[0])
            self.assertEqual(item["pair_id"], "denseuav:sample")


if __name__ == "__main__":
    unittest.main()
