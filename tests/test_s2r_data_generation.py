import csv
import importlib
import inspect
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from swiftvln import cli as swiftvln_cli
from swiftvln.s2r.data_generation.image_utils import center_square_recrop
from swiftvln.s2r.data_generation.main import main as generation_main
from swiftvln.s2r.data_generation.main import run_command
from swiftvln.s2r.data_generation import merge_variants
from swiftvln.s2r.data_generation.gta_uav import build_pairs as gta_build_pairs
from swiftvln.s2r.data_generation.registry import (
    get_registry,
    normalize_dataset_name,
    resolve_module,
)
from swiftvln.s2r.data_generation.sues import center_recrop_pairs

generation_module = importlib.import_module("swiftvln.s2r.data_generation.main")


class S2RDataGenerationCliTest(unittest.TestCase):
    def test_registry_modules_are_importable_and_accept_argv(self):
        for commands in get_registry().values():
            for module_path in commands.values():
                module = importlib.import_module(module_path)
                self.assertTrue(callable(module.main), module_path)
                self.assertIn("argv", inspect.signature(module.main).parameters)

    def test_dataset_aliases(self):
        self.assertEqual(normalize_dataset_name("gta-uav"), "gta_uav")
        self.assertEqual(normalize_dataset_name("uav-visloc"), "uavvisloc")
        self.assertEqual(
            resolve_module("gta-uav", "build_pairs"),
            "swiftvln.s2r.data_generation.gta_uav.build_pairs",
        )

    def test_run_command_forwards_argv_without_global_mutation(self):
        captured = []

        def fake_main(argv):
            captured.extend(argv)
            return 7

        fake_module = SimpleNamespace(main=fake_main)
        with patch.object(
            generation_module.importlib,
            "import_module",
            return_value=fake_module,
        ):
            result = run_command("denseuav", "build_pairs", ["--workers", "2"])

        self.assertEqual(result, 7)
        self.assertEqual(captured, ["--workers", "2"])

    def test_config_paths_expand_relative_to_config_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "generation.yaml"
            override_output = root / "override"
            config_path.write_text(
                """
DATASETS:
  denseuav:
    build_pairs:
      input:
        dataset_root: ${S2R_TEST_RAW}
      output:
        output_dir: generated/denseuav
      args:
        workers: 12
        skip_ncc: true
""".lstrip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"S2R_TEST_RAW": "raw/DenseUAV"}):
                with patch.object(
                    generation_module,
                    "run_command",
                    return_value=0,
                ) as mocked_run:
                    result = generation_main(
                        [
                            "denseuav",
                            "build_pairs",
                            "--config",
                            str(config_path),
                            "--workers",
                            "3",
                            "--output-dir",
                            str(override_output),
                        ]
                    )

            self.assertEqual(result, 0)
            dataset, command, forwarded = mocked_run.call_args.args
            self.assertEqual((dataset, command), ("denseuav", "build_pairs"))
            self.assertEqual(
                forwarded[:2],
                ["--dataset-root", str((root / "raw/DenseUAV").resolve())],
            )
            self.assertIn("--skip-ncc", forwarded)
            self.assertNotIn(str((root / "generated/denseuav").resolve()), forwarded)
            self.assertEqual(forwarded[-4:], [
                "--workers",
                "3",
                "--output-dir",
                str(override_output),
            ])

    def test_missing_config_and_unknown_command_are_user_errors(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            missing_result = generation_main(
                ["--config", "/definitely/missing.yaml", "denseuav", "build_pairs"]
            )
        self.assertEqual(missing_result, 2)
        self.assertIn("Config file not found", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            unknown_result = generation_main(["denseuav", "not-a-command"])
        self.assertEqual(unknown_result, 2)
        self.assertIn("Unknown command", stderr.getvalue())

    def test_swiftvln_cli_forwards_s2r_help(self):
        with patch("swiftvln.s2r.data_generation.main", return_value=11) as run:
            result = swiftvln_cli.main(["s2r-data", "--help"])
        self.assertEqual(result, 11)
        run.assert_called_once_with(["--help"])

    def test_packaged_config_is_portable_yaml(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "src/swiftvln/s2r/data_generation/config.example.yaml"
        )
        text = config_path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        self.assertEqual(
            set(payload["DATASETS"]),
            {"denseuav", "gta_uav", "sues", "uavvisloc"},
        )
        self.assertNotIn("/mnt/", text)


class S2RDataGenerationImageTest(unittest.TestCase):
    def test_gta_protocol_duplicates_export_once_with_portable_metadata(self):
        common = {
            "drone_img_name": "100_0001_0000000001.png",
            "drone_img_path": Path("/raw/gta/drone.png"),
            "satellite_img_name": "7_0_1_2.png",
            "satellite_img_path": Path("/raw/gta/satellite.png"),
            "drone_loc_x": 1.0,
            "drone_loc_y": 2.0,
            "satellite_loc_x": 3.0,
            "satellite_loc_y": 4.0,
            "height": 100.0,
            "cam_roll": -90.0,
            "cam_pitch": 0.0,
            "cam_yaw": 30.0,
            "iou": 0.5,
            "num_positive_tiles": 1,
            "heading_status": "confirmed",
            "north_up_rot": 30.0,
        }
        pairs = [
            gta_build_pairs.Pair(
                sample_id="cross_area__100_0001_0000000001",
                source_meta_file="cross-area-drone2sate-train.json",
                split="train",
                area_mode="cross_area",
                **common,
            ),
            gta_build_pairs.Pair(
                sample_id="same_area__100_0001_0000000001",
                source_meta_file="same-area-drone2sate-test.json",
                split="test",
                area_mode="same_area",
                **common,
            ),
        ]

        compact = gta_build_pairs.deduplicate_protocol_pairs(pairs)
        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0].sample_id, "100_0001_0000000001")
        self.assertEqual(compact[0].source_area_modes, ("cross_area", "same_area"))
        self.assertEqual(compact[0].source_row_splits, ("test", "train"))

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            gta_build_pairs.write_pairs_csv(compact, output)
            text = (output / "pairs.csv").read_text(encoding="utf-8")
            with (output / "pairs.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertNotIn("/raw/", text)
            self.assertNotIn("drone_img_path", row)
            self.assertEqual(row["source_area_modes"], "cross_area|same_area")
            self.assertEqual(
                row["export_drone_path"], "drone/100_0001_0000000001.jpg"
            )

    def test_center_square_recrop(self):
        image = Image.new("RGB", (100, 80), color=(20, 30, 40))
        result = center_square_recrop(image, crop_size=60, output_size=32)
        self.assertEqual(result.size, (32, 32))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            center_square_recrop(image, crop_size=0, output_size=32)
        with self.assertRaisesRegex(ValueError, "exceeds image size"):
            center_square_recrop(image, crop_size=101, output_size=32)

    def test_sues_recrop_command_preserves_pair_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            (source / "satellite").mkdir(parents=True)
            (source / "drone").mkdir(parents=True)
            Image.new("RGB", (96, 80), color=(255, 0, 0)).save(
                source / "satellite/pair.jpg"
            )
            Image.new("RGB", (96, 80), color=(0, 255, 0)).save(
                source / "drone/pair.jpg"
            )
            with (source / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "satellite_file", "drone_file"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "pair",
                        "satellite_file": "pair.jpg",
                        "drone_file": "pair.jpg",
                    }
                )

            with redirect_stdout(io.StringIO()):
                center_recrop_pairs.main(
                    [
                        "--dataset-dir",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--crop-size",
                        "64",
                        "--output-size",
                        "32",
                    ]
                )

            self.assertEqual(
                (source / "pairs.csv").read_bytes(),
                (output / "pairs.csv").read_bytes(),
            )
            with Image.open(output / "satellite/pair.jpg") as image:
                self.assertEqual(image.size, (32, 32))
            with Image.open(output / "drone/pair.jpg") as image:
                self.assertEqual(image.size, (32, 32))
            info = yaml.safe_load(
                (output / "dataset_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["postprocess"]["crop_size_px"], 64)

    def test_unified_cli_runs_minimal_sues_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            output = root / "SatDronePair/sues"
            satellite_dir = raw / "satellite-view/0001"
            drone_dir = raw / "drone_view_512/0001/150"
            satellite_dir.mkdir(parents=True)
            drone_dir.mkdir(parents=True)

            pattern = Image.effect_noise((96, 96), 48).convert("RGB")
            pattern.save(satellite_dir / "0.png")
            pattern.save(drone_dir / "0.jpg", quality=100)

            with redirect_stdout(io.StringIO()):
                result = generation_main(
                    [
                        "sues",
                        "pipeline",
                        "--data-root",
                        str(raw),
                        "--output-dir",
                        str(output),
                        "--heights",
                        "150",
                        "--max-scenes",
                        "1",
                        "--match-size",
                        "32",
                        "--coarse-angle-step",
                        "360",
                        "--coarse-fracs",
                        "1.0",
                        "--top-k-frames",
                        "1",
                        "--fine-angle-radius",
                        "0",
                        "--fine-frac-radius",
                        "0",
                        "--output-size",
                        "64",
                        "--nadir-eval-size",
                        "64",
                        "--nadir-grid",
                        "2",
                        "--nadir-patch-size",
                        "16",
                        "--nadir-min-conf",
                        "0",
                        "--skip-recrops",
                        "--skip-preview",
                    ]
                )

            self.assertEqual(result, 0)
            with (output / "pairs.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "0001_H150")
            with Image.open(output / "satellite/0001_H150.jpg") as image:
                self.assertEqual(image.size, (64, 64))
            with Image.open(output / "drone/0001_H150.jpg") as image:
                self.assertEqual(image.size, (64, 64))

    def test_merge_uavvisloc_variants_produces_manifest_compatible_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_dirs = [root / "orig", root / "crop384"]
            for index, variant_dir in enumerate(variant_dirs):
                (variant_dir / "satellite").mkdir(parents=True)
                (variant_dir / "drone").mkdir(parents=True)
                Image.new("RGB", (32, 32), color=(index * 50, 10, 20)).save(
                    variant_dir / "satellite/source.jpg"
                )
                Image.new("RGB", (32, 32), color=(10, index * 50, 20)).save(
                    variant_dir / "drone/source.jpg"
                )
                with (variant_dir / "pairs.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "sample_id",
                            "seq_id",
                            "export_satellite_path",
                            "export_drone_path",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sample_id": "source",
                            "seq_id": "01",
                            "export_satellite_path": "satellite/source.jpg",
                            "export_drone_path": "drone/source.jpg",
                        }
                    )
                (variant_dir / "dataset_info.json").write_text(
                    '{"data_root": "/private/machine/path", "size": 1}\n',
                    encoding="utf-8",
                )

            output = root / "merged"
            with redirect_stdout(io.StringIO()):
                merge_variants.main(
                    [
                        "--variant",
                        f"orig={variant_dirs[0]}",
                        f"crop384={variant_dirs[1]}",
                        "--output-dir",
                        str(output),
                        "--mode",
                        "copy",
                    ]
                )

            with (output / "pairs.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["variant"] for row in rows], ["orig", "crop384"])
            self.assertEqual(
                [row["export_satellite_path"] for row in rows],
                ["satellite/000001.jpg", "satellite/000002.jpg"],
            )
            self.assertTrue((output / "drone/000001.jpg").is_file())
            self.assertTrue((output / "satellite/000002.jpg").is_file())
            info = yaml.safe_load(
                (output / "dataset_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["source_variants"], ["orig", "crop384"])
            self.assertNotIn(
                "data_root", info["source_dataset_info"]["orig"]
            )


if __name__ == "__main__":
    unittest.main()
