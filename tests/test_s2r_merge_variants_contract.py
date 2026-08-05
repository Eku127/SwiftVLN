from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from swiftvln.s2r.data_generation.main import run_command
from swiftvln.s2r.data_generation.registry import resolve_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def _write_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class S2RMergeVariantsContractTest(unittest.TestCase):
    def test_multi_root_mode_remains_available_for_uavvisloc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            variants = []
            for index, name in enumerate(("orig", "crop384")):
                variant = root / name
                _write_image(variant / "satellite/sample.jpg", (10 + index, 20, 30))
                _write_image(variant / "drone/sample.jpg", (30, 20, 10 + index))
                _write_csv(
                    variant / "pairs.csv",
                    {
                        "sample_id": "sample",
                        "seq_id": "01",
                        "export_satellite_path": "satellite/sample.jpg",
                        "export_drone_path": "drone/sample.jpg",
                    },
                )
                variants.append(f"{name}={variant}")

            output = root / "merged"
            run_command(
                "uavvisloc",
                "merge_variants",
                [
                    "--variant",
                    *variants,
                    "--output-dir",
                    str(output),
                    "--mode",
                    "copy",
                ],
            )
            with (output / "pairs.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["variant"] for row in rows], ["orig", "crop384"])
            self.assertEqual(
                [row["export_satellite_path"] for row in rows],
                ["satellite/000001.jpg", "satellite/000002.jpg"],
            )

    def test_one_root_variant_map_replaces_sues_special_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sues"
            _write_csv(
                source / "pairs.csv",
                {
                    "sample_id": "scene_H150",
                    "scene_id": "scene",
                    "satellite_file": "sample.jpg",
                    "drone_file": "sample.jpg",
                },
            )
            for index, (satellite_dir, drone_dir) in enumerate(
                (("satellite", "drone"), ("satellite_crop384", "drone_crop384"))
            ):
                _write_image(source / satellite_dir / "sample.jpg", (20 + index, 30, 40))
                _write_image(source / drone_dir / "sample.jpg", (40, 30, 20 + index))

            output = root / "merged"
            run_command(
                "sues",
                "merge_variants",
                [
                    "--dataset-dir",
                    str(source),
                    "--variant-map",
                    "orig:satellite:drone,crop384:satellite_crop384:drone_crop384",
                    "--output-dir",
                    str(output),
                    "--mode",
                    "copy",
                ],
            )
            with (output / "pairs.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["variant"] for row in rows], ["orig", "crop384"])
            self.assertEqual(
                [row["satellite_file"] for row in rows],
                ["000001.jpg", "000002.jpg"],
            )
            for sample_id in ("000001", "000002"):
                self.assertTrue((output / f"satellite/{sample_id}.jpg").is_file())
                self.assertTrue((output / f"drone/{sample_id}.jpg").is_file())

    def test_old_sues_command_and_module_are_removed(self):
        with self.assertRaises(KeyError):
            resolve_module("sues", "merge_variants_dense_style")
        self.assertFalse(
            (
                REPO_ROOT
                / "src/swiftvln/s2r/data_generation/sues/merge_variants_dense_style.py"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
