from __future__ import annotations

import sys
import unittest
from dataclasses import fields
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

    def test_embedding_configuration_is_one_four_way_field(self):
        field_names = {item.name for item in fields(SwiftVLNTrainArguments)}
        self.assertIn("embedding_mode", field_names)
        self.assertNotIn("use_pose_embed", field_names)
        self.assertNotIn("use_uav_adapter", field_names)
        self.assertNotIn("pose_fusion_method", field_names)

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
