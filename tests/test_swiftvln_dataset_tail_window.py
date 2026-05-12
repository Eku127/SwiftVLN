import json
import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.model.dataset import SwiftVLNDataset


class TestSwiftVLNDatasetTailWindow(unittest.TestCase):
    def _build_dataset_root(self, actions_len: int) -> Path:
        root = Path(self.tmpdir.name)
        (root / "rgb").mkdir()
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
        (root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
        return root

    def _starts(self, *, actions_len: int, num_overlap: int):
        root = self._build_dataset_root(actions_len)
        dataset = SwiftVLNDataset(
            data_path=str(root),
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
            self._starts(actions_len=35, num_overlap=16),
            [0, 16],
        )

    def test_overlap_zero_keeps_tail_coverage(self):
        self.assertEqual(
            self._starts(actions_len=35, num_overlap=0),
            [0, 3],
        )


if __name__ == "__main__":
    unittest.main()
