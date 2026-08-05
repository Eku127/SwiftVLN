from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from swiftvln.s2r.data_generation.main import run_command
from swiftvln.s2r.data_generation.registry import resolve_module


REPO_ROOT = Path(__file__).resolve().parents[1]


class S2RPreviewContractTest(unittest.TestCase):
    def _write_dataset(self, root: Path, dataset: str) -> None:
        (root / "satellite").mkdir(parents=True)
        (root / "drone").mkdir(parents=True)
        Image.new("RGB", (24, 24), (40, 80, 120)).save(
            root / "satellite/sample.jpg"
        )
        Image.new("RGB", (24, 24), (120, 80, 40)).save(
            root / "drone/sample.jpg"
        )

        rows = {
            "denseuav": {
                "sample_id": "sample",
                "altitude": "H80",
                "split": "train",
                "north_up_rot": "15",
            },
            "gta_uav": {
                "sample_id": "sample",
                "split": "train",
                "area_mode": "same-area",
                "heading_status": "confirmed",
                "north_up_rot": "0",
                "iou": "0.8",
                "export_satellite_path": "satellite/sample.jpg",
                "export_drone_path": "drone/sample.jpg",
            },
            "sues": {
                "sample_id": "sample",
                "height": "150",
                "drone_frame": "1",
                "drone_rotation_ccw_deg": "0",
                "sat_crop_frac": "0.8",
                "match_score": "0.9",
                "nadir_conf": "0.7",
                "satellite_file": "sample.jpg",
                "drone_file": "sample.jpg",
            },
            "uavvisloc": {
                "sample_id": "sample",
                "seq_id": "01",
                "heading_source": "pose",
                "crop_side_m": "120",
                "north_up_rot": "0",
                "export_satellite_path": "satellite/sample.jpg",
                "export_drone_path": "drone/sample.jpg",
            },
        }
        row = rows[dataset]
        with (root / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_all_dataset_commands_share_one_renderer(self):
        modules = {
            resolve_module(dataset, "sample_preview")
            for dataset in ("denseuav", "gta_uav", "sues", "uavvisloc")
        }
        self.assertEqual(
            modules,
            {"swiftvln.s2r.data_generation.sample_preview"},
        )

    def test_each_schema_renders_a_valid_jpeg_through_the_launcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dataset in ("denseuav", "gta_uav", "sues", "uavvisloc"):
                with self.subTest(dataset=dataset):
                    dataset_dir = root / dataset
                    self._write_dataset(dataset_dir, dataset)
                    output = dataset_dir / "preview.jpg"
                    result = run_command(
                        dataset,
                        "sample_preview",
                        [
                            "--dataset-dir",
                            str(dataset_dir),
                            "--rows",
                            "1",
                            "--panel",
                            "32",
                            "--output",
                            str(output),
                        ],
                    )
                    self.assertEqual(result, 0)
                    with Image.open(output) as preview:
                        self.assertEqual(preview.format, "JPEG")
                        self.assertGreater(preview.width, 64)
                        self.assertGreater(preview.height, 32)

    def test_dataset_specific_preview_modules_are_removed(self):
        for dataset in ("denseuav", "gta_uav", "sues", "uavvisloc"):
            path = (
                REPO_ROOT
                / "src/swiftvln/s2r/data_generation"
                / dataset
                / "sample_preview.py"
            )
            self.assertFalse(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
