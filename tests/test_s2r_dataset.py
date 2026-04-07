import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.s2r.dataset import SatDronePairDataset, collate_pair_batch


def _write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path)


class S2RDatasetTest(unittest.TestCase):
    def test_dataset_loads_images_and_collates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uav_path = root / "drone" / "a.jpg"
            sat_path = root / "sat" / "a.jpg"
            _write_image(uav_path, (255, 0, 0))
            _write_image(sat_path, (0, 255, 0))

            manifest_path = root / "manifest.jsonl"
            record = {
                "dataset": "denseuav",
                "pair_id": "denseuav:a",
                "sample_id": "a",
                "split": "train",
                "group_id": "denseuav:a",
                "uav_image": str(uav_path),
                "sat_image": str(sat_path),
                "lat": None,
                "lon": None,
                "height_m": None,
                "north_up_rot": None,
                "meta": {},
            }
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = SatDronePairDataset(str(manifest_path), split="train")
            sample = dataset[0]
            self.assertEqual(sample["pair_id"], "denseuav:a")
            self.assertEqual(sample["uav_image"].size, (12, 10))
            self.assertEqual(sample["sat_image"].size, (12, 10))

            batch = collate_pair_batch([sample])
            self.assertEqual(batch["pair_id"], ["denseuav:a"])
            self.assertEqual(len(batch["uav_image"]), 1)

    def test_dataset_max_samples_round_robins_across_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            rows = []
            datasets = ("denseuav", "gta", "sues", "uavvisloc")
            for dataset_index, dataset in enumerate(datasets):
                for item_index in range(3):
                    uav_path = root / dataset / "drone" / f"{item_index}.jpg"
                    sat_path = root / dataset / "sat" / f"{item_index}.jpg"
                    _write_image(uav_path, (dataset_index * 30, item_index * 20, 10))
                    _write_image(sat_path, (10, dataset_index * 30, item_index * 20))
                    rows.append(
                        {
                            "dataset": dataset,
                            "pair_id": f"{dataset}:{item_index}",
                            "sample_id": f"{item_index}",
                            "split": "val",
                            "group_id": f"{dataset}:{item_index}",
                            "uav_image": str(uav_path),
                            "sat_image": str(sat_path),
                            "lat": None,
                            "lon": None,
                            "height_m": None,
                            "north_up_rot": None,
                            "meta": {},
                        }
                    )
            manifest_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            dataset = SatDronePairDataset(str(manifest_path), split="val", max_samples=8)
            counts = Counter(record.dataset for record in dataset.records)

            self.assertEqual(len(dataset), 8)
            self.assertEqual(set(counts.keys()), set(datasets))
            self.assertEqual(counts["denseuav"], 2)
            self.assertEqual(counts["gta"], 2)
            self.assertEqual(counts["sues"], 2)
            self.assertEqual(counts["uavvisloc"], 2)


if __name__ == "__main__":
    unittest.main()
