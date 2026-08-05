from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from swift.arguments import SftArguments  # noqa: E402
from swift.pipelines.train.sft import SwiftSft  # noqa: E402
from swiftvln.model.arguments import SwiftVLNTrainArguments  # noqa: E402
from swiftvln.model.trainer import SwiftVLNSft  # noqa: E402


class TrainingStructureContractTest(unittest.TestCase):
    def test_repository_local_single_use_base_classes_are_gone(self):
        self.assertIs(SwiftVLNTrainArguments.__bases__[0], SftArguments)
        self.assertIs(SwiftVLNSft.__bases__[0], SwiftSft)

    def test_dataset_path_normalization_preserves_cli_forms(self):
        trainer = object.__new__(SwiftVLNSft)
        trainer.args = SimpleNamespace(dataset=["/data/a", "/data/b"])
        self.assertEqual(
            trainer._dataset_paths(),
            ("/data/a,/data/b", ["/data/a", "/data/b"]),
        )

        trainer.args = SimpleNamespace(dataset=" /data/a, /data/b ")
        self.assertEqual(
            trainer._dataset_paths(),
            (" /data/a, /data/b ", ["/data/a", "/data/b"]),
        )


if __name__ == "__main__":
    unittest.main()
