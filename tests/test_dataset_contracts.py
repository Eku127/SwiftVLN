from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from swiftvln.model.dataset import SwiftVLNDataset  # noqa: E402


class SwiftVLNDatasetContractTest(unittest.TestCase):
    def _build_dataset_root(self, actions_len: int) -> Path:
        root = Path(self.tmpdir.name)
        (root / "rgb").mkdir(exist_ok=True)
        annotations = [
            {
                "id": "ep0",
                "trajectory_id": "traj0",
                "steps": [],
                "video": "rgb",
                "instructions": ["go to the goal"],
                "actions": list(range(actions_len)),
            }
        ]
        (root / "annotations.json").write_text(
            json.dumps(annotations),
            encoding="utf-8",
        )
        return root

    def _window_starts(self, *, actions_len: int, num_overlap: int):
        dataset = SwiftVLNDataset(
            data_path=str(self._build_dataset_root(actions_len)),
            num_frames=32,
            num_history=0,
            num_future_steps=4,
            num_overlap=num_overlap,
            env_type="satnav",
        )
        return [start_idx for _, _, start_idx in dataset.data_list]

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_overlap_keeps_stride_aligned_tail(self):
        self.assertEqual(
            self._window_starts(actions_len=35, num_overlap=16),
            [0, 16],
        )

    def test_zero_overlap_keeps_tail_coverage(self):
        self.assertEqual(
            self._window_starts(actions_len=35, num_overlap=0),
            [0, 3],
        )

    def test_map_memory_rejects_unsupported_environment(self):
        with self.assertRaisesRegex(ValueError, "only satnav"):
            SwiftVLNDataset(
                data_path=str(self._build_dataset_root(8)),
                env_type="habitat",
                memory_method="map",
            )

    def test_overlap_must_be_smaller_than_window(self):
        with self.assertRaisesRegex(ValueError, "must be < num_frames"):
            SwiftVLNDataset(
                data_path=str(self._build_dataset_root(8)),
                num_frames=32,
                num_overlap=32,
            )


if __name__ == "__main__":
    unittest.main()
