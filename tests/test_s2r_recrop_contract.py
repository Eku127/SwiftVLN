from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from swiftvln.s2r.data_generation.main import run_command
from swiftvln.s2r.data_generation.registry import resolve_module


REPO_ROOT = Path(__file__).resolve().parents[1]


class S2RRecropContractTest(unittest.TestCase):
    def _write_dataset(self, root: Path, schema: str) -> None:
        (root / "satellite").mkdir(parents=True)
        (root / "drone").mkdir(parents=True)
        Image.new("RGB", (24, 20), (20, 60, 100)).save(
            root / "satellite/sample.jpg"
        )
        Image.new("RGB", (20, 24), (100, 60, 20)).save(
            root / "drone/sample.jpg"
        )
        if schema == "sues":
            row = {
                "sample_id": "sample",
                "satellite_file": "sample.jpg",
                "drone_file": "sample.jpg",
            }
        else:
            row = {
                "sample_id": "sample",
                "export_satellite_path": "satellite/sample.jpg",
                "export_drone_path": "drone/sample.jpg",
            }
        with (root / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (root / "dataset_info.json").write_text(
            json.dumps({"name": schema, "notes": "source"}),
            encoding="utf-8",
        )

    def test_both_commands_share_one_schema_aware_module(self):
        modules = {
            resolve_module(dataset, "center_recrop_pairs")
            for dataset in ("sues", "uavvisloc")
        }
        self.assertEqual(
            modules,
            {"swiftvln.s2r.data_generation.center_recrop_pairs"},
        )

    def test_each_schema_preserves_paths_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dataset in ("sues", "uavvisloc"):
                with self.subTest(dataset=dataset):
                    source = root / f"{dataset}_source"
                    output = root / f"{dataset}_crop"
                    self._write_dataset(source, dataset)
                    result = run_command(
                        dataset,
                        "center_recrop_pairs",
                        [
                            "--dataset-dir",
                            str(source),
                            "--output-dir",
                            str(output),
                            "--crop-size",
                            "16",
                            "--output-size",
                            "8",
                        ],
                    )
                    self.assertEqual(result, 0)
                    for relative_path in (
                        "satellite/sample.jpg",
                        "drone/sample.jpg",
                    ):
                        with Image.open(output / relative_path) as image:
                            self.assertEqual(image.size, (8, 8))
                    self.assertEqual(
                        (output / "pairs.csv").read_text(encoding="utf-8"),
                        (source / "pairs.csv").read_text(encoding="utf-8"),
                    )
                    metadata = json.loads(
                        (output / "dataset_info.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(metadata["postprocess"]["crop_size_px"], 16)
                    self.assertEqual(metadata["postprocess"]["output_size_px"], 8)
                    self.assertIsInstance(metadata["notes"], list)

    def test_dataset_specific_recrop_modules_are_removed(self):
        for dataset in ("sues", "uavvisloc"):
            path = (
                REPO_ROOT
                / "src/swiftvln/s2r/data_generation"
                / dataset
                / "center_recrop_pairs.py"
            )
            self.assertFalse(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
